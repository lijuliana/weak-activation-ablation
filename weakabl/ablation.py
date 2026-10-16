"""Forward-pre-hooks on `down_proj` that (a) calibrate per-neuron max |activation|
and (b) zero weak activations, or a count-matched random set of activations.

The "neuron activation" is the SwiGLU output act(gate(x)) * up(x), i.e. the input
to down_proj, of shape (batch, seq, d_ff).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def mlp_down_projs(model) -> list[torch.nn.Module]:
    """Return the down_proj module of every decoder layer (Qwen3 / Llama layout)."""
    return [layer.mlp.down_proj for layer in model.model.layers]


@dataclass
class AblationStats:
    zeroed: list[float] = field(default_factory=list)  # per layer, count of zeroed entries
    total: list[float] = field(default_factory=list)   # per layer, count of entries seen

    def fraction_per_layer(self) -> list[float]:
        return [z / t if t else 0.0 for z, t in zip(self.zeroed, self.total)]

    def fraction(self) -> float:
        z, t = sum(self.zeroed), sum(self.total)
        return z / t if t else 0.0


class Ablator:
    """Attach to a model; switch between modes:

    - "off":       no-op.
    - "calibrate": accumulate per-neuron running max |activation| (positions >= 1 only,
                   to avoid first-token massive activations dominating the max).
    - "threshold": zero entries with |a| < frac * amax[neuron].
    - "random":    for each (layer, token) zero the *same number* of entries as the
                   threshold rule would, but chosen uniformly at random.
    """

    def __init__(self, model, seed: int = 0):
        self.modules = mlp_down_projs(model)
        self.n_layers = len(self.modules)
        self.mode = "off"
        self.frac = 0.0
        self.amax: list[torch.Tensor] | None = None
        self.calib_max: list[torch.Tensor | None] = [None] * self.n_layers
        self.calib_max_incl0: list[torch.Tensor | None] = [None] * self.n_layers
        self.track = False
        self.stats = AblationStats([0.0] * self.n_layers, [0.0] * self.n_layers)
        self.seed = seed
        self._gen: torch.Generator | None = None
        self._handles = [m.register_forward_pre_hook(self._make_hook(i)) for i, m in enumerate(self.modules)]

    # ---- lifecycle -----------------------------------------------------------------
    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def set(self, mode: str, frac: float = 0.0, seed: int | None = None, track: bool = False):
        assert mode in ("off", "calibrate", "threshold", "random"), mode
        if mode in ("threshold", "random"):
            assert self.amax is not None, "load or compute calibration first"
        self.mode, self.frac, self.track = mode, float(frac), track
        if seed is not None:
            self.seed = seed
        self._gen = None  # re-seeded lazily on the right device
        self.stats = AblationStats([0.0] * self.n_layers, [0.0] * self.n_layers)

    def load_calibration(self, amax: torch.Tensor, device):
        """amax: (n_layers, d_ff) tensor of per-neuron max |activation|."""
        assert amax.shape[0] == self.n_layers
        self.amax = [amax[i].to(device=device, dtype=torch.float32) for i in range(self.n_layers)]

    def calibration_tensor(self, include_pos0: bool = False) -> torch.Tensor:
        src = self.calib_max_incl0 if include_pos0 else self.calib_max
        assert all(t is not None for t in src), "calibration not run"
        return torch.stack([t.float().cpu() for t in src])  # (n_layers, d_ff)

    # ---- the hook ------------------------------------------------------------------
    def _generator(self, device) -> torch.Generator:
        if self._gen is None or self._gen.device != torch.device(device):
            self._gen = torch.Generator(device=device).manual_seed(self.seed)
        return self._gen

    def _make_hook(self, i: int):
        def hook(module, args):
            h = args[0]
            if self.mode == "off" or (self.mode in ("threshold", "random") and self.frac == 0.0):
                return None
            if self.mode == "calibrate":
                m_all = h.detach().abs().amax(dim=(0, 1)).float()
                self.calib_max_incl0[i] = m_all if self.calib_max_incl0[i] is None else torch.maximum(self.calib_max_incl0[i], m_all)
                if h.shape[1] > 1:
                    m = h[:, 1:].detach().abs().amax(dim=(0, 1)).float()
                    self.calib_max[i] = m if self.calib_max[i] is None else torch.maximum(self.calib_max[i], m)
                return None

            thr = (self.amax[i] * self.frac).to(h.dtype)
            mask = h.abs() < thr  # (B, T, D)
            if self.mode == "random":
                k = mask.sum(dim=-1, keepdim=True)  # (B, T, 1) count to match
                r = torch.rand(h.shape, generator=self._generator(h.device), device=h.device)
                kth = r.sort(dim=-1).values.gather(-1, (k - 1).clamp(min=0))
                mask = (r <= kth) & (k > 0)
            if self.track:
                self.stats.zeroed[i] += float(mask.sum())
                self.stats.total[i] += float(mask.numel())
            return (h.masked_fill(mask, 0),)

        return hook

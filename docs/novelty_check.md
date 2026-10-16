# Novelty check: do weak MLP activations carry important (long-tail) information?

*Literature scan, 2026-10-16. Every citation below has a verified entry in `refs.bib`.*

## Scope

**Question.** When the low-magnitude SwiGLU activations of an LLM's MLP neurons are zeroed, is the
damage concentrated on rare (long-tail) factual knowledge, as a superposition account predicts, or is
it small and roughly uniform, as the "clean neurons" account predicts?

**In scope.** (i) The claim under dispute and its direct replies; (ii) activation-sparsity methods that
zero weak activations at inference time and what they measured; (iii) how facts are stored in MLPs;
(iv) long-tail knowledge benchmarks and compression studies that split knowledge by popularity.
**Out of scope.** Weight pruning as a method (only cited where it measured long-tail knowledge),
SAE architecture details, vision-model interpretability beyond the essay itself.

## The claim and the counter-claim

Cammarata's essay argues that polysemanticity mostly dissolves under closer analysis and that
superposition is the least-testable explanation for what remains \cite{cammarata2026clean}. The
weak-activation evidence in the essay is a vision experiment: on an 18-block ConvNeXt, removing the
weakest 75% of activations lowered ImageNet accuracy from 72% to 71% with changes confined to near-ties,
while removing the strongest 7% dropped it to 4%. The author concludes that "rare things ... have grown
on related circuits" rather than hiding in weak activations. For language models the essay reports
activation distributions and attribution graphs on Qwen3-32B and Kimi-K3, but no weak-activation
ablation, no LM-loss measurement and nothing about factual knowledge; a companion technical paper on
language neurons is "forthcoming" \cite{cammarata2026clean}. The essay is a non-peer-reviewed working
hypothesis.

Nanda's reply (LessWrong comment, 2026-10-16) states that the work "doesn't really affect my cruxes":
models store far more facts than they have neurons, facts are "an unstructured database of key-value
pairs" that superposition packs efficiently, he "expects weak neuron activations to carry a lot of
superposed semantic information" so that ablating them would meaningfully hurt performance, and the
essay does not address this for LLMs; he also doubts that small vision models memorize enough for the
ConvNeXt result to transfer \cite{nanda2026comment}. This is exactly the experiment proposed here, and
neither side has run it on a language model.

## Theme 1: activation-sparsity methods already zero weak activations, but only measure aggregates

Deja Vu showed that contextual sparsity (neurons that are near-zero for a given input) can be predicted
and skipped at inference time with little loss on OPT models \cite{liu2023dejavu}. ReLU Strikes Back
argued for re-introducing ReLU to get high activation sparsity at little cost \cite{mirzadeh2023relu},
and Sparsing Law studied how activation sparsity scales with model size and data
\cite{luo2024sparsing}. For SwiGLU models, CATS zeros SwiGLU outputs whose magnitude falls below a
per-layer threshold chosen from the activation distribution \cite{lee2024cats}, and TEAL extends
magnitude thresholding to every linear layer's input, reaching 40-50% model-wide sparsity "with minimal
performance degradation" on Llama-2/3 and Mistral \cite{liu2025teal}; a 2026 follow-up robustifies
TEAL-style thresholds \cite{xu2026spon}. These papers are strong evidence for the Goodfire side on
*aggregate* metrics: perplexity and standard zero-shot benchmarks barely move at substantial sparsity.

What none of them measures is the *distribution* of the damage. TEAL reports averages over tasks such
as ARC, HellaSwag, PIQA, WinoGrande and MMLU; CATS reports perplexity and similar task averages. No
activation-sparsity paper we found evaluates closed-book factual recall, and none splits any metric by
entity popularity or token frequency. A small average drop is fully compatible with a large drop on a
rare minority.

## Theme 2: how facts are stored in MLPs (why the superposition side expects a long-tail effect)

Geva et al. showed MLP layers behave as key-value memories \cite{geva2021kv}, and Dai et al. proposed
that specific "knowledge neurons" store facts \cite{dai2022knowledge}. Niu et al. found that
knowledge-neuron attributions largely track token-expression patterns rather than knowledge, and that
editing them generalizes poorly \cite{niu2024knowledgeneuron}. Nanda et al.'s fact-finding sequence on
Pythia-2.8B found that factual lookup was "not localisable to specific neurons or even a single layer",
that different facts used different neurons with little overlap, and that the result was "dense and
uninterpretable" \cite{nanda2023factfinding}. Sparse-probing work found some neuron-aligned features
but also many features spread across neurons \cite{gurnee2023haystack, gurnee2024universal}, consistent
with the superposition picture \cite{elhage2022toy}. The SoLU paper raised the specific worry that an
architecture can "hide some non-neuron-aligned features by decreasing their magnitude and then later
recovering it with LayerNorm" \cite{elhage2022solu}, i.e. that small activations can carry features;
Cammarata reads the same evidence as distributed representation rather than packing
\cite{cammarata2026clean}. JumpReLU SAEs explicitly treat small pre-activations as the main source of
false positives and zero them below a learned threshold \cite{rajamanoharan2024jumprelu}, an indirect
sign that at least in dictionary space, weak activations are mostly noise.

Together these give the superposition account a concrete prediction: if facts are a dense,
distributed lookup table, each fact's signal on any one neuron is small, and the facts with the least
training signal (rare entities) should have the smallest activations and be the first to go.

## Theme 3: long-tail knowledge is fragile, and weight compression has been split by popularity

Rare facts are learned worse: QA accuracy tracks how many pre-training documents mention the entity
\cite{kandpal2023longtail}, and PopQA was built precisely to measure this, binning questions by the
subject entity's Wikipedia page views \cite{mallen2023popqa}. Compression studies on *weights* have
started to ask whether the damage is uniform. Jaiswal et al. found pruning decimates knowledge-intensive
QA even when perplexity barely moves \cite{jaiswal2024compressing}; Namburi et al. found parametric
knowledge is sensitive to compression \cite{namburi2023cost}; the Junk DNA hypothesis found that
pruning *small-magnitude weights* monotonically impairs "difficult" tasks, the weight-space analogue of
the question asked here \cite{yin2024junkdna}. Most directly, Wu et al. (2026) compressed Llama-3.1-8B,
Qwen3-8B and Gemma-2-9B with 4-bit quantization, unstructured/N:M pruning and layer dropping, split
PopQA into equal-count head/middle/tail groups by subject popularity, and found that *head* knowledge
loses a larger fraction of its base accuracy than tail knowledge in 104 of 105 configurations
\cite{wu2026asymmetric}. That result cuts against the naive "long tail goes first" intuition, but it is
about weight compression, and it does not include TEAL/CATS-style activation thresholding.

## Gap analysis: what is new here

1. **No LLM weak-activation ablation exists on either side of the dispute.** The essay's evidence is a
   ConvNeXt experiment; the LLM companion paper is unpublished \cite{cammarata2026clean}. Nanda's
   prediction is stated, not tested \cite{nanda2026comment}.
2. **Activation-sparsity papers never split damage by popularity.** TEAL/CATS/Deja Vu report averages
   \cite{liu2025teal, lee2024cats, liu2023dejavu}; none measures closed-book factual recall at all.
3. **The one popularity-split compression study is weight-level**, and its head-loses-more finding
   makes the activation-level answer a genuine open question rather than a foregone conclusion
   \cite{wu2026asymmetric}.
4. **No study uses a count-matched random-ablation control** to separate "weak activations matter" from
   "tail facts are fragile to any perturbation". Without it, a tail-heavy drop under thresholding would
   be uninterpretable.

The experiment in this repository fills exactly that gap: per-neuron max-calibrated thresholding of
SwiGLU activations in Qwen3-1.7B-Base, swept from 0 to 30% of each neuron's max, against a
count-matched random control, read out on held-out LM loss and PopQA accuracy by popularity bin.

## Annotated bibliography (short)

- \cite{cammarata2026clean}: Essay; ConvNeXt weakest-75% ablation 72→71%, strongest-7% 72→4%;
  LLM claims rest on neuron audits of Qwen3-32B, no LLM ablation. Non-peer-reviewed.
- \cite{nanda2026comment}: The counter-claim; predicts ablating weak activations hurts, facts are a
  key-value store in superposition. Non-peer-reviewed.
- \cite{liu2025teal}: Magnitude thresholding of all linear inputs; 40-50% sparsity, small average
  benchmark drop; no factual-recall or long-tail split.
- \cite{lee2024cats}: SwiGLU-output thresholding (closest mechanically to this experiment); ~25%
  sparsity with little average loss; same gap.
- \cite{liu2023dejavu}: Contextual sparsity is predictable and cheap to exploit; aggregate metrics only.
- \cite{wu2026asymmetric}: PopQA head/middle/tail under weight quantization/pruning; head loses
  relatively more. Weight-level, not activation-level.
- \cite{nanda2023factfinding}: Factual recall in Pythia-2.8B is distributed and dense across MLP
  layers 2-6; no clean neuron mechanism.
- \cite{mallen2023popqa}: PopQA; accuracy rises with subject popularity; defines the bins used here.
- \cite{kandpal2023longtail}: Fact recall tracks pre-training document count.
- \cite{yin2024junkdna}: Small weights are not junk; pruning them hurts hard tasks monotonically.
- \cite{jaiswal2024compressing}, \cite{namburi2023cost}: Compression hurts knowledge tasks more than
  perplexity suggests.
- \cite{geva2021kv}, \cite{dai2022knowledge}, \cite{niu2024knowledgeneuron}: Facts in MLPs as
  key-value memory; knowledge-neuron thesis and its critique.
- \cite{elhage2022toy}, \cite{elhage2022solu}, \cite{gurnee2023haystack}, \cite{gurnee2024universal},
  \cite{rajamanoharan2024jumprelu}: Superposition background and the "features hidden in small
  activations" worry.
- \cite{mirzadeh2023relu}, \cite{luo2024sparsing}, \cite{xu2026spon}: Activation sparsity is natural
  and scalable; all aggregate-metric.

## Search log

Queries run on 2026-10-16 (web search, arXiv API for verification, direct fetch of primary pages):
"Goodfire Neural Networks Are Already Clean weak activations ablation"; "Neel Nanda weak activations
superposition rare facts already clean"; "activation sparsity LLM factual knowledge long-tail TEAL CATS
magnitude thresholding"; "LLM compression pruning quantization long-tail knowledge PopQA"; "low magnitude
neuron activations superposition small activations MLP"; "activation sparsity threshold ablation knowledge
recall"; "Junk DNA hypothesis small-magnitude weights"; "activation sparsity rare tokens frequency";
"activation sparsity PopQA TriviaQA entity popularity"; "ablating weak activations language model KL
superposition"; "activation sparsity memorization factual recall knowledge-intensive". Secondary
reports of Jake Mendel's and Leo Gao's replies on X were seen in a news roundup but the posts themselves
could not be fetched, so they are not cited.

## Verification note

`refs.bib` was run through the citation-checker script (CrossRef, Semantic Scholar, OpenAlex). It
confirmed 13 entries and reported 10 as not found: the three web sources (the Goodfire essay, Nanda's
LessWrong comment and the Alignment Forum fact-finding post, none of which those databases index) and
seven arXiv papers (CATS, Niu et al., Jaiswal et al., Wu et al., Gurnee et al. 2024, Mirzadeh et al.,
and the SoLU post). Each of the arXiv entries was instead verified by ID against the arXiv API, which
returned the title and author list recorded in the .bib; the three web sources were fetched directly.
The checker's "chimeric" flag on the Junk DNA entry comes from it truncating the title at the colon and
matching a biology paper; the full title and authors match arXiv:2310.02277 (ICML 2024).

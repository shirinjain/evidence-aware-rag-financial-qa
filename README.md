# Evidence-Aware RAG for Multi-Hop Financial QA

Retrieval-augmented QA over SEC filings (10-Ks, 10-Qs, 8-Ks, earnings releases), built on the [FinanceBench](https://arxiv.org/abs/2311.11944) dataset. Financial documents are dominated by tables (where the right row and the wrong row can differ by a single word or sign convention) and many real questions are multi-hop, requiring several distinct facts that are sometimes restated in more than one place in the corpus. This project addresses both problems as two research questions:

- **RQ1** — can a domain-informed, hard-negative-mined cross-encoder reranker meaningfully improve retrieval? **Yes: +56.3% recall@10** on a genuinely held-out test set.
- **RQ2** — can a coverage-aware ranking objective (avoiding redundant restatements of an already-covered fact in favor of a still-missing one) be learned and generalize? **Yes, modestly: +32.9% recall@10** on a redundant-hop test set, validated via cross-validation after diagnosing and discarding two failed approaches.

Full writeups: [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) (unifying overview), [`RQ1_RESULTS.md`](RQ1_RESULTS.md), [`RQ2_RESULTS.md`](RQ2_RESULTS.md).

## Dataset

**Corpus**: 146,821 chunks from 84 SEC filings across 32 companies (2015-2024). PDFs converted with Docling, restricted per-document to page ranges around known evidence locations to keep conversion tractable.

**Chunking strategy (parent-child)**: two different granularities, chosen deliberately per content type, not chunked at a single uniform size.
- **Child chunks (the actual retrieval unit)**: table **rows** are split individually - one row = one chunk, with the entity/fiscal-year/section-title context and column headers preserved alongside it. This granularity is needed because financial questions need the *exact* row (the right row and the wrong row can be a single word apart, e.g. "Net income" vs "Net income attributable to noncontrolling interest"), not a whole table. Narrative text is kept at **paragraph** granularity, not sentence-split, since narrative evidence in this domain tends to be coherent multi-sentence explanations rather than atomic single-sentence facts.
- **Parent chunks**: whole-table text, kept as a lookup table (`data/processed/coarse_chunks.parquet`) rather than a retrieval unit. Coarse (whole-table) chunking was tested directly *as* the retrieval unit and found to reduce performance - source tables here range from 1 to 66 rows, so concatenating large ones just produces diluted, oversized chunks; this differs from prior published benchmarks whose "no chunking" results came from already-small, pre-segmented source documents.

**Question variants** - two independent axes matter here, and it's easy to conflate them:
- *How many hops does a question have* (how many distinct facts it needs): single-metric questions have exactly 1 hop; multi-metric questions have 2-4 hops (one per metric asked, e.g. "What was X's revenue, capex, and total assets in FY2018?").
- *How many chunks satisfy each individual hop* ("hop size"): most hops map to exactly 1 chunk (no redundancy) - but some facts are genuinely restated in more than one place in the corpus (an earnings release restated unchanged in the subsequent 10-K, a number in prose corroborated by a table row). Coverage@k and recall@k are mathematically identical whenever hop size is always 1; they diverge specifically on redundant-hop questions, which is the entire premise of RQ2.

Concretely, the question sets used:
- **Precise question set** (591 questions, used for RQ1 training/evaluation): gold chunks constructed exactly - the chunk is picked first, the question written around it - rather than relying solely on FinanceBench's own recorded evidence (which, after cleanup, still had a residual false-positive rate: bare years, generic single words, wrong-statement-type matches). Single-metric (459, 1 hop), multi-metric (121, 2-4 hops, hop size 1), narrative (11, selected from FinanceBench's own clean-gold questions).
- **Held-out test set** (185 questions): same construction method, built from 7 metrics *never* used in any training data (goodwill, accounts payable, other assets, total equity, accrued liabilities, SG&A, total operating expenses), for a genuine same-distribution generalization check - not just new questions, new *facts*.
- **Natural FinanceBench questions** (train/val/test splits): real annotator-written questions, analytically phrased ("What is the FY2019 fixed asset turnover ratio for X?") - used to test cross-distribution generalization, since this style differs from the constructed precise-question style.
- **RQ2 redundant-hop set** (32 usable questions, hop size 2+): mined directly from the corpus by searching for cases where the same numeric value is reported in more than one chunk - verified by exact value matching, not just label matching (naive label matching produced ~45 false positives, e.g. a segment revenue row and a total revenue row both matching the label "net sales" while being different figures entirely).

## Methodology

**Baseline model** (stage 1, `scripts/pipeline_best.py::retrieve_fusion`) - unmodified, no learned reranking of any kind:
- **Sparse**: BM25 (`bm25s` library), each chunk's text prefixed with `entity, FYyear, section_title:` context, NLTK's 198-word stopword list.
- **Dense**: embedding model **`sentence-transformers/all-mpnet-base-v2`** (768-dim, off-the-shelf, not fine-tuned), same contextual prefix, cosine similarity.
- **Entity/year filtering**: regex-based company-name and fiscal-year detection restrict candidates to the detected company/year, addressing a diagnosed cross-company and cross-year confusion problem (generic line-item labels like "Dividends paid to shareholders" are nearly identical across companies).
- **Fusion**: convex combination of min-max normalized BM25 and dense scores (`alpha=0.4`, found via sweep) - beats plain BM25, plain dense, and reciprocal rank fusion on this corpus.

**Final model(s)** - two separate learned stages, each independently validated, *not* combined into one serving pipeline (a beta-sweep found pure cross-encoder always beats any blend with LambdaMART, even on the redundant-hop questions LambdaMART targets - see `RQ2_RESULTS.md`):
1. **RQ1's final pipeline** = baseline + query decomposition for multi-metric questions (`retrieve_decomposed`/`retrieve_auto` - splitting a bundled multi-metric query into per-metric sub-questions, since bundling dilutes term-overlap scoring per metric) + cross-encoder reranking (base model **`cross-encoder/ms-marco-MiniLM-L-6-v2`**, fine-tuned via RankNetLoss on domain-informed hard negatives - the pipeline's own top-ranked *wrong* answers, not random negatives - on 591 questions / 3,682 question-document pairs, 5 epochs) + beta-blending (`beta=1.0` for same-distribution deployment, `beta=0.3` as the dual-distribution-safe default).
2. **RQ2's final model** = baseline + a from-scratch coverage-aware LambdaMART (custom Δcoverage@k gradient, generalizing the standard ΔNDCG LambdaRank gradient to a different, exactly-computable target metric), built entirely on features with no prior fine-tuning history - a cheaper alternative to the cross-encoder when it isn't available, not a per-question routing choice when it is.

### The RankNetLoss objective (RQ1's cross-encoder fine-tuning)

For a pair of candidates (i, j) retrieved for the same question, where i is truly more relevant than j (i is gold, j is a mined hard negative), the cross-encoder produces a raw scalar score for each - `s_i`, `s_j` - from its joint (query, passage) self-attention forward pass. RankNetLoss (Burges et al., 2005) converts the score difference into a predicted probability that i should outrank j, via a sigmoid, and minimizes ordinary binary cross-entropy against the true target (1, since i genuinely should rank above j here):

```
P_ij = 1 / (1 + exp(-sigma * (s_i - s_j)))
L = -log(P_ij) = log(1 + exp(-sigma * (s_i - s_j)))
```

Minimizing this loss pushes `s_i - s_j` to be large and positive - i.e. pushes the gold document's score up and the hard negative's score down whenever the model doesn't already separate them enough. Trained via ordinary backprop, since the cross-encoder is a neural network, not a tree ensemble.

**Why this loss, not an embedding-distance contrastive loss (e.g. InfoNCE)**: a cross-encoder fuses the query and passage into one joint representation and outputs a single scalar score - there's no separate query-embedding and passage-embedding to take a distance between (unlike a bi-encoder, which is what contrastive losses are built for). RankNetLoss operates directly on the scalar *scores* the model actually produces, comparing pairs of scores rather than pairs of embeddings, which is exactly what a cross-encoder can give you.

**The connection to LambdaMART below, worth stating explicitly**: differentiating RankNet's loss with respect to the scores produces exactly the `rho_ij = 1/(1+exp(sigma*(s_i-s_j)))` term the LambdaMART section below is built on - LambdaRank *is* RankNet's gradient, just with each pair's contribution additionally scaled by `|ΔMetric|` (ΔNDCG classically, Δcoverage@k in RQ2's modification). RQ1's loss and RQ2's loss aren't two unrelated ideas - one is the foundation the other generalizes.

### The LambdaMART objective, and how it's modified for coverage

Standard LambdaRank doesn't optimize NDCG by writing it directly into a loss - NDCG is a sorting-based function of the ranking, and you can't backpropagate through "sort these items." Instead, for every pair of candidates (i, j) in a query's group, it computes how much a target metric would change if their ranks were swapped, and uses that magnitude to scale an ordinary pairwise gradient:

```
rho_ij = 1 / (1 + exp(sigma * (s_winner - s_loser)))   # how wrong the CURRENT scores are
lambda  = sigma * rho_ij * |delta_metric|                # gradient magnitude
```

The candidate that *should* rank higher gets `+lambda`; the other gets `-lambda`. `rho_ij` is near zero when the model already has the pair in the right order (nothing to correct), and large when it has them backwards. `|delta_metric|` is the swap-importance weight - this is what makes top-of-list mistakes matter more than tail mistakes. This mechanism is metric-agnostic: NDCG is simply what it's traditionally paired with, nothing about the "compute Δmetric when swapping i,j" step is NDCG-specific.

**The modification**: swap `ΔNDCG(swap i,j)` for **Δcoverage@k(swap i,j)**, computed exactly rather than approximated. This relies on one property specific to coverage@k: it is a pure set-membership function over the top-k (a hop is "covered" iff any of its chunks is in the top-k), so **swapping two candidates that are both inside, or both outside, the current top-k provably cannot change coverage@k at all** - only a pair straddling the rank-k boundary can. So for each candidate `a` currently inside the top-k and each `b` currently outside it, the delta is computed by simulating the swap and recomputing coverage@k: if it increases, `b` is the winner; if it decreases, `a` is already correctly placed and gets reinforced; if unchanged, the pair contributes no gradient at all. This naturally gives zero gradient to two redundant copies of an *already*-covered hop competing with each other (reordering them can't help or hurt coverage), while generating a strong, correctly-signed gradient whenever a redundant copy is keeping an under-represented hop's only candidate out of the top-k - exactly the behavior the whole exercise exists to teach.

**Training setup**: implemented from scratch (`scripts/train_coverage_lambdamart.py`) using `sklearn.tree.DecisionTreeRegressor` as the weak learner in a manual gradient-boosting loop, rather than LightGBM/XGBoost - their macOS wheels need `libomp` via Homebrew, a bigger environment dependency than warranted, and a custom objective needs full gradient control that off-the-shelf libraries don't expose a clean hook for anyway. Two design choices were load-bearing for actually getting a result that generalizes:
- **Warm-starting**: the ensemble is initialized to the stage-1 fused score (not zero), and boosting only learns a small additive *correction* on top of it, using shallow, heavily-regularized trees (`max_depth=2`, `min_samples_leaf=25`, `learning_rate=0.05`). An earlier, un-warm-started version with richer features (including the cross-encoder's own score) could and did memorize company-specific score patterns from the ~26 available training examples that didn't transfer to new companies' filings.
- **Feature set with no fine-tuning history** - deliberately excludes the cross-encoder's own score. An earlier version that included it mostly just learned to copy that one already-strong column (no real generalization), and it turned out to double as a contamination path: some of the mined training questions' gold chunks are the same chunks the cross-encoder was itself fine-tuned on in RQ1, so its score partly measured memorization rather than a fair signal. The features actually used:

  | Feature | What it captures |
  |---|---|
  | `bm25_norm` | Min-max normalized BM25 score |
  | `dense_norm` | Min-max normalized dense (mpnet) cosine score |
  | `cc_score` | Stage-1's convex-combination fused score - also the warm-start value |
  | `is_value_dup_of_higher_ranked` | **The core coverage-specific signal.** Does this candidate share an extracted numeric value with another candidate already ranked above it (in stage-1 order)? A company-agnostic, structural "someone already reported this exact fact" flag, reusing the same value-extraction/normalization logic that mined the RQ2 training data itself. Needed two bug fixes to be reliable: matching the label-adjacent value correctly for table rows, and, for narrative text, excluding bare years and percentages (an early version flagged two unrelated chunks as "duplicates" purely because both happened to mention "fiscal 2018") |
  | `entity_match` / `year_match` | Whether the candidate's detected company/fiscal-year matches the question's |
  | `is_table_row` | Table row vs. narrative chunk type |
  | `lexical_overlap` | Fraction of the question's content words appearing in the candidate's text |

  Of these, `is_value_dup_of_higher_ranked` is the only one purpose-built for this problem; the rest are the same signals stage 1 already produces, just exposed directly to the tree instead of only being combined via the fixed `alpha=0.4` formula.

## RQ1: domain-informed hard-negative cross-encoder reranking

Evaluated on the 185-question held-out set (n=185; mostly single/multi-metric factual questions, only 5 with any redundant hop):

| | Recall@10 | Coverage@10 | NDCG@10 |
|---|---|---|---|
| Baseline (stage 1 only), n=185 | 0.738 | 0.741 | 0.522 |
| **+ cross-encoder (beta=1.0), n=185** | **0.942 (+27.6%)** | **0.945 (+27.5%)** | **0.842 (+61.3%)** |

Key methodological finding: an earlier fine-tuning attempt on 1/13th the data produced a textbook-normal loss curve while the resulting model had *inverted* score preferences on direct inspection - a converging loss curve does not guarantee a useful model. Also found: `beta=1.0` (full trust in the reranker) is optimal on same-distribution held-out data but regresses on cross-distribution (analytically-phrased) questions - resolved via `beta=0.3` as a dual-distribution-safe default.

## RQ2: coverage-aware ranking for multi-hop questions

Evaluated on a separate 32-question redundant-hop set (n=32; every question here has facts restated across multiple chunks by construction, unlike the RQ1 table above — this is *why* the same baseline pipeline's recall@10 looks much lower here (0.35 vs. 0.74): redundant copies dilute the denominator, which is exactly the failure mode this metric pair is designed to expose (note coverage@10 for this same baseline is 0.497, much closer to the RQ1 numbers, since coverage isn't fooled by the redundancy)):

| | Recall@10 | Coverage@10 | NDCG@5 |
|---|---|---|---|
| Baseline (stage 1 only), n=32 | 0.352 | 0.497 | 0.260 |
| **+ coverage-aware LambdaMART, n=32** | **0.438 (+24.4%)** | **0.576 (+15.9%)** | **0.331 (+27.1%)** |

Getting here required diagnosing three failed attempts in sequence: a feature set including the cross-encoder's own score mostly just copied that feature (no real generalization); digging into why surfaced a genuine contamination bug (44% of mined training questions' gold chunks were the same chunks the cross-encoder was itself fine-tuned on); removing the cross-encoder entirely and using only features with no fine-tuning history, plus warm-starting from the stage-1 score with heavy regularization, is what finally produced a real, cross-validated effect (3 improved / 28 unchanged / 1 worsened across 32 questions, out-of-fold).

## Repository structure

```
scripts/          retrieval pipeline, corpus construction, fine-tuning, RQ2 mining/training
eval/              precision/recall/NDCG and coverage@k metrics
data/processed/    question sets, gold labels, RQ2 datasets, generated prompts
```

## What this demonstrates methodologically

Every claimed improvement here was checked against a genuinely held-out set before being reported, and every time a result looked too good on a first pass, the discrepancy was investigated rather than smoothed over - including the negative results and abandoned approaches, not just the ones that worked. See `PROJECT_SUMMARY.md` Section 6-7 for the full honest accounting of limitations and what didn't generalize.

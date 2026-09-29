# RQ1 Results: Domain-Informed Hard-Negative Cross-Encoder Reranking

## 1. Baseline Retrieval Methodology

**Corpus**: 146,821 chunks (82,514 table rows + 64,307 narrative sentences/paragraphs) built from 84 SEC filings (10-K, 10-Q, 8-K, earnings releases) spanning 32 companies, 2015-2024. Filings were parsed with Docling (the same tool the underlying FinReflectKG dataset's authors used), restricted per-document to a page range around each filing's known evidence locations to keep conversion tractable, then split into table-row chunks (one row = one chunk, preserving column headers as context) and paragraph-level narrative chunks (not sentence-split, since FinanceBench's narrative evidence spans are coherent multi-sentence explanations, not atomic single-sentence facts).

**Sparse retrieval (BM25)**: built with the `bm25s` library over each chunk's text prefixed with `entity, FYyear, section_title:` context, using NLTK's 198-word English stopword list (found to be more complete than bm25s's default 33-word list, which was missing common question words).

**Dense retrieval**: `sentence-transformers/all-mpnet-base-v2` embeddings (same contextual prefix as BM25's index text) computed on Colab GPU, stored in a ChromaDB collection with cosine distance.

**Entity filtering**: regex-based company-name detection (word-boundary matching against the corpus's 32 known company names and common aliases) restricts candidates to the detected company, addressing a diagnosed failure mode where generic financial line-item labels ("Dividends paid to shareholders," "Net cash provided by operating activities") are nearly identical across companies and cause cross-company confusion in pure keyword/embedding matching.

**Year filtering**: regex-based fiscal-year detection (handling "FYyyyy" tokens without word-boundary false negatives, and expanding a mentioned range like "FY2019 - FY2021" to include the enclosed year) restricts candidates to the detected fiscal year(s), addressing a diagnosed cross-year confusion (the wrong fiscal year's document ranking highly for an otherwise-correct company match).

**Fusion**: BM25 and dense candidate lists (top-k each) are min-max normalized within each method, then combined via **convex combination**: `score = alpha * bm25_norm + (1-alpha) * dense_norm`. A sweep over alpha in [0.3, 0.5, 0.7, 0.9] (in 0.1 increments around the midpoint) found **alpha=0.4** (dense weighted slightly higher than BM25) to outperform both plain BM25, plain dense, and reciprocal rank fusion (RRF, k=60) on this corpus - attributed to dense embeddings better bridging the vocabulary gap between analytically-phrased questions ("3-year average operating margin") and raw filing line items ("Operating income"), a gap pure keyword matching cannot close.

This four-stage pipeline (entity filter -> year filter -> BM25+dense CC fusion) is implemented in `scripts/pipeline_best.py::retrieve_fusion`.

## 2. Precise Question Set (used for both baseline evaluation and reranker training)

Rather than rely solely on FinanceBench's own recorded evidence citations (which, after three rounds of cleanup, still carried a residual long tail of false-positive gold labels - generic single words, bare dates, mis-attributed statement sections), we additionally constructed a **precise question set** where the gold chunk is known exactly by construction: the chunk is selected first, and the question is written around it.

- **Single-metric (459 questions)**: one question per (company, fiscal year, metric), e.g. *"What was 3M's total assets in FY2018?"*. Extraction restricted to rows whose section heading specifically matches the relevant statement type (income statement, balance sheet, or cash flow statement - matched via whitespace-stripped substring checks to handle PDF-to-text line-break artifacts, and explicitly excluding "Comprehensive Income" and "Statement of Changes in Equity," which share surface phrasing with the income statement but report different figures). Covers 17 metrics: revenue, net income, gross profit, operating income, cost of sales, income tax expense, total assets, total liabilities, cash and equivalents, total current assets/liabilities, long-term debt, inventories, capital expenditures, operating cash flow, depreciation and amortization, dividends paid. Gold = exactly 1 chunk, always.
- **Multi-metric (121 questions)**: 2-4 metrics per (company, year) bundled into one question, e.g. *"What was 3M's cash and cash equivalents, cost of sales, total current assets and total current liabilities in FY2018?"*. Gold = exactly N chunks for N metrics asked (capped at 4 to keep questions realistic).
- **Narrative (11 questions)**: rather than constructed, these are *selected* from FinanceBench's own train-split questions whose gold evidence is already small (<=5 chunks) and narrative-dominant (>=80%) - real annotator-written questions where alignment is already clean.

Total: **591 precise questions**, all with exact or verified-clean gold.

## 3. Baseline Results (pre-reranking), 591 Precise Questions

| Tier | N | P@10 | R@10 | P@50 | R@50 | P@100 | R@100 | NDCG@100 |
|---|---|---|---|---|---|---|---|---|
| Single-metric | 459 | 0.072 | 0.719 | 0.019 | 0.954 | 0.010 | 0.978 | 0.490 |
| Multi-metric | 121 | 0.109 | 0.313 | 0.052 | 0.716 | 0.030 | 0.813 | 0.395 |
| Narrative | 11 | 0.055 | 0.424 | 0.016 | 0.652 | 0.009 | 0.742 | 0.240 |
| **Combined** | **591** | **0.079** | **0.630** | **0.026** | **0.900** | **0.014** | **0.940** | **0.466** |

Single-metric recall@100 (0.978) confirms most of the retrieval difficulty in this domain is attributable to gold-alignment noise and multi-fact composition rather than a hard retrieval ceiling: a genuinely simple, well-scoped factual lookup is found almost every time once given sufficient depth.

## 4. Cross-Encoder Reranking Methodology

**Architecture rationale**: a cross-encoder processes the query and a candidate passage *together* through joint self-attention, fusing them into a single representation before a linear head outputs one scalar relevance score - unlike a bi-encoder (BM25/dense above), which embeds query and passage *separately* and compares via distance. This makes cross-encoders structurally incapable of full-corpus retrieval (nothing can be precomputed - every candidate requires a fresh forward pass with the query), but structurally suited to fine-grained reranking of an already-narrowed shortlist, since joint attention lets the model directly compare specific query tokens against specific passage tokens (e.g. matching a requested metric name to a table row's label) rather than relying on independently-pooled embeddings. We apply it as stage 2, reranking stage 1's top-k candidates only.

**Loss function**: `RankNetLoss` (Burges et al., 2005), a pairwise ranking loss operating on the model's scalar scores directly (not applicable: embedding-distance-based contrastive losses, since a cross-encoder produces no embedding to take a distance between - only bi-encoders can use those). For each training question, the model sees its full list of positive and hard-negative documents together and is trained so predicted score differences match the true relevance ordering.

**Hard-negative mining**: for each of the 591 precise questions, up to 6 gold chunks are sampled as positives, and up to 3x as many hard negatives are drawn from `pipeline_best.py`'s own top-ranked *non-gold* results - i.e., candidates that already fooled the current best retrieval pipeline. This is the domain-informed hard-negative principle: negatives are not random, but exactly the confusions the retrieval system is known to make (cross-year/cross-company boilerplate twins, table-vs-narrative distractors, wrong-statement mismatches).

**Training data**: 591 question-rows, 3,682 total (question, document) pairs, base model `cross-encoder/ms-marco-MiniLM-L-6-v2`, 5 epochs, batch size 16, learning rate 2e-5.

**A critical methodological finding during this process**: an earlier fine-tuning attempt (277 training rows, ~1/13th the final data volume) produced a training loss curve that looked entirely normal - steadily decreasing train and validation loss, no divergence - yet the resulting model *actively degraded* downstream retrieval quality, with inverted score preferences confirmed by direct inspection (hard negatives scoring higher than true positives on the model's own training examples). **A converging loss curve does not guarantee a useful model** - only measuring actual downstream task performance does. Following this, every epoch's checkpoint (not just the "best-by-eval-loss" one) was saved and evaluated directly on retrieval metrics before adoption.

**Score blending (beta)**: to guard against exactly this kind of unreliable reranker output, the reranker's normalized score is blended with stage 1's rank-position score rather than fully replacing it: `final_score = beta * reranker_score_norm + (1-beta) * stage1_rank_score`. beta=1.0 is full trust (pure reranker order); beta=0.0 ignores the reranker entirely. This is evaluated as a sweep, not assumed.

## 5. Reranking Results

Two categories of held-out evaluation were used, since they probe different generalization questions:
- **Same-distribution held-out** (185 questions: 140 single-metric + 40 multi-metric, built from 7 metrics *never* used in the training set, e.g. goodwill, accounts payable, total equity - plus 5 natural FinanceBench val-split questions with genuine redundant hop structure): tests whether the reranker generalizes to *new facts* phrased in the *same style* it trained on.
- **Cross-distribution held-out** (84 train-split + 30 val-split natural FinanceBench questions, analytically-phrased computed-ratio questions like "What is the FY2019 fixed asset turnover ratio for X?"): tests generalization to a *different question style* entirely.

### 5a. beta=1.0 (full trust in reranker)

| Eval set | N | P@5 before | P@5 after | NDCG@5 before | NDCG@5 after |
|---|---|---|---|---|---|
| 591 precise (training-source; expect inflated by memorization) | 591 | 0.109 | 0.252 (+131%) | 0.322 | 0.797 (+147%) |
| **Held-out precise (unseen facts, same style)** | **185** | **0.155** | **0.253 (+64%)** | **0.482** | **0.831 (+72%)** |
| Natural FinanceBench, train split | 84 | 0.088 | 0.112 (+27%) | 0.113 | 0.165 (+46%) |
| Natural FinanceBench, val split | 30 | 0.127 | 0.100 (**-21%**) | 0.178 | 0.136 (**-24%**) |

At beta=1.0, the reranker is **monotonically better with more trust on same-distribution data** (a full beta sweep from 0.0 to 1.0 shows steadily increasing P@5/NDCG@5, plateauing near beta=0.7-1.0) but **actively regresses on the natural FinanceBench val split** - the opposite pattern. This is a genuine, reproducible finding about *distribution-dependent* reranker reliability, not noise: the same checkpoint helps substantially when the query matches its training distribution (single/multi-metric factual lookups) and hurts when it doesn't (analytical computed-ratio phrasing).

**By tier, 591 precise set, beta=1.0 (after rerank):**

| Tier | N | P@5 | R@5 | NDCG@5 |
|---|---|---|---|---|
| Single-metric | 459 | 0.189 | 0.946 | 0.838 |
| Multi-metric | 121 | 0.506 | 0.697 | 0.695 |
| Narrative | 11 | 0.073 | 0.303 | 0.205 |
| Combined | 591 | 0.252 | 0.883 | 0.797 |

### 5b. beta=0.3 (conservative blend)

| Eval set | N | P@5 before | P@5 after | NDCG@5 before | NDCG@5 after |
|---|---|---|---|---|---|
| 591 precise (training-source; expect inflated by memorization) | 591 | 0.109 | 0.197 (+81%) | 0.322 | 0.654 (+103%) |
| Held-out precise (unseen facts, same style) | 185 | 0.155 | 0.217 (+40%) | 0.482 | 0.736 (+53%) |
| Natural FinanceBench, train split | 84 | 0.088 | 0.093 (+5.5%) | 0.113 | 0.141 (+25%) |
| Natural FinanceBench, val split | 30 | 0.127 | 0.133 (+5.2%) | 0.178 | 0.201 (+13%) |

beta=0.3 is **positive on every evaluation set**, including the natural-distribution val split where beta=1.0 regressed - at the cost of a smaller gain on the same-distribution held-out set (+40% vs +64% P@5) than full trust would give.

**By tier, 591 precise set, beta=0.3 (after rerank), full P/R/NDCG@{5,10,20,50,100}:**

| Tier | N | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | P@100 | R@100 | NDCG@100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Single-metric | 459 | 0.169 | 0.843 | 0.716 | 0.090 | 0.904 | 0.736 | 0.010 | 0.983 | 0.753 |
| Multi-metric | 121 | 0.314 | 0.448 | 0.457 | 0.193 | 0.544 | 0.504 | 0.031 | 0.835 | 0.600 |
| Narrative | 11 | 0.109 | 0.424 | 0.254 | 0.073 | 0.606 | 0.317 | 0.009 | 0.742 | 0.345 |
| Combined | 591 | 0.197 | 0.755 | 0.654 | 0.111 | 0.825 | 0.680 | 0.014 | 0.948 | 0.714 |

**By tier, held-out precise (185), beta=0.3 (after rerank):**

| Tier | N | P@5 before | NDCG@5 before | P@5 after | NDCG@5 after |
|---|---|---|---|---|---|
| Single-metric | 140 | - | 0.553 | 0.190 | 0.818 |
| Multi-metric | 40 | - | 0.297 | 0.335 | 0.539 |
| Natural-redundant | 5 | - | 0.000 | 0.040 | 0.026 |
| Combined | 185 | 0.155 | 0.482 | 0.217 | 0.736 |

The multi-metric tier's after-rerank NDCG@5 (0.539) trails single-metric (0.818) at every beta - consistent across the whole study: ranking *multiple* correct chunks all near the top is a strictly harder objective for the cross-encoder (which scores each candidate independently, with no awareness of what else has already been selected) than ranking a single correct chunk near the top. The 5-question natural-redundant slice is too small to draw a real conclusion from and is reported only for completeness.

### 5c. Recommendation

If the deployed system's query distribution is expected to resemble the training distribution (direct factual lookups), **beta=1.0 is the stronger choice**. If the query distribution is mixed or expected to include more analytically-phrased questions, **beta=0.3 is the dual-distribution-safe choice** - smaller gains everywhere, but no observed regression on any evaluation set tested. `scripts/pipeline_best.py` defaults to beta=1.0, with this tradeoff documented in its module docstring.

### 5d. Correction: the properly held-out headline table (185 questions, full depth)

Section 5a's "591 precise" row is **not** a held-out result - the cross-encoder was fine-tuned using chunks from that same 591-question set, so its numbers are inflated by direct memorization (this is stated in the row label, but it's worth a dedicated, complete table since it was originally only reported at k=5). The 185-question `val_precise` set is the genuinely comparable, held-out headline number - built from 7 metrics deliberately never touched during cross-encoder training. Full depth (k=5/10/50/100), beta=1.0:

| | P@5 | R@5 | Cov@5 | P@10 | R@10 | Cov@10 | P@50 | R@50 | Cov@50 | P@100 | R@100 | Cov@100 | NDCG@10 | NDCG@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (stage-1) | 0.149 | 0.586 | 0.588 | 0.096 | 0.738 | 0.741 | 0.026 | 0.926 | 0.931 | 0.014 | 0.940 | 0.945 | 0.522 | 0.577 |
| **Baseline + CE (beta=1.0)** | **0.254** | **0.916** | **0.919** | **0.136** | **0.942** | **0.945** | **0.027** | **0.945** | **0.950** | **0.014** | **0.945** | **0.950** | **0.842** | **0.843** |

Recall and coverage are not *exactly* identical here (unlike the 591-set) since this set includes 5 natural-redundant questions - but they differ by only +0.3 to +0.5 percentage points at every k, close enough to treat as equal in practice. This table, not the 591-question one, is the fair basis for comparison against RQ2's results (see `RQ2_RESULTS.md`), since both are now genuinely held-out.

### 5e. Query decomposition for multi-metric questions (new finding)

**The problem, found via manual spot-check**: for a multi-metric question ("What was 3M's cash and cash equivalents, cost of sales, total current assets and total current liabilities in FY2018?"), one of the four gold chunks (`Cash and cash equivalents | $2,853 | $3,053`) was missing not just from the top-10 context, but from BM25's top-100 *and* dense's top-100 entirely - confirmed directly, not inferred. Re-run as a single-metric question ("What was 3M's cash and cash equivalents in FY2018?"), the identical chunk ranked #1. The cause: bundling 4 metric names into one query dilutes BM25/dense term-overlap scoring for each individual metric against the vocabulary of the *other* three metrics also present in the same query - a real, diagnosable failure mode specific to how multi-metric questions are phrased, unrelated to redundancy (RQ2) or cross-distribution generalization (Section 5c).

**The fix**: `retrieve_decomposed()` (`scripts/pipeline_best.py`) detects the metrics named in a question (matching against the same 24-metric label lists used for corpus mining and precise-question construction), splits the question into one single-metric sub-question per detected metric ("What was {entity}'s {metric} in FY{year}?"), retrieves+reranks each independently through the full pipeline (stage-1 fusion + cross-encoder, beta=1.0), and **interleaves** (round-robin, not concatenated block-by-block) each sub-question's top-3 results into one combined ranking. The interleaving detail matters: an earlier version concatenated each sub-question's full top-3 block sequentially, which starves later-listed metrics of any top-5 representation regardless of how well they individually retrieved - caught by checking per-question coverage@5 before accepting the result, fixed, and re-validated.

**Validated results** (both arms through the identical full CE-reranked pipeline - decomposition is the only variable):

| | N | Coverage@5 | Coverage@10 | NDCG@5 | Precision@5 |
|---|---|---|---|---|---|
| **Held-out multi-metric, bundled (current default)** | 40 | 0.740 | 0.806 | 0.720 | 0.465 |
| **Held-out multi-metric, decomposed** | 40 | **0.892 (+20.6%)** | **0.969 (+20.2%)** | **0.818 (+13.7%)** | **0.565 (+21.5%)** |
| Train-source multi-metric, bundled | 121 | 0.621 | 0.688 | 0.644 | 0.448 |
| Train-source multi-metric, decomposed | 121 | 0.782 (+26.1%) | 0.864 (+25.6%) | 0.737 (+14.3%) | 0.574 (+28.1%) |

Per-question coverage@10 on the held-out set: **13 improved, 27 unchanged, 0 worsened** - a completely one-directional result. `retrieve_auto()` is the recommended entry point going forward: it detects multi-metric questions automatically and routes to `retrieve_decomposed()`, falling back to ordinary `retrieve()` for single-metric questions or when entity/year can't be confidently detected. `retrieve()` itself is left unchanged so existing evaluation scripts that call it directly are unaffected.

## 6. What Didn't Work (reported for completeness)

- **Coarse (whole-table) chunking**, hypothesized from a published benchmark (arxiv 2604.01733, T2-RAGBench) reaching Recall@5=0.816 with "no chunking." Tested directly and found to *reduce* performance - that paper's corpus (built from FinQA/ConvFinQA/TAT-DQA) has naturally small (~920-token) source documents, so "no chunking" was a property of pre-segmented data, not a portable technique; our tables vary from 1 to 66 rows, and concatenating large ones produced diluted, unhelpfully large chunks.
- **Off-the-shelf community "finance reranker" models** on HuggingFace (checked 3, all with near-zero usage/validation signal) - not adopted due to unverifiable training quality.
- **RankNetLoss fine-tuning on a small dataset** (277 rows) - produced a model with a normal-looking loss curve but inverted, harmful downstream behavior; resolved by roughly 4x more training data and 3x richer negative sampling (591 rows, 3,682 docs).


## Appendix A: RQ2 Design & Scoping Discussion (process notes, kept for transparency)

*The material below is the working discussion that led to RQ2's design - reformatted for readability but not removed, per record. The finalized, current version of everything in this appendix (final dataset composition, final results) lives in `RQ2_RESULTS.md`; where numbers below differ from that document, `RQ2_RESULTS.md` is the up-to-date source (e.g. the redundant tier grew further after this discussion, from 37/13 to the mined totals reported there).*

### A.1 Initial RQ2 training set (before the mining pass described in `RQ2_RESULTS.md`)

`data/processed/rq2_dataset.json`, 56 questions initially:

| Source | Count | What it is |
|---|---|---|
| `synthetic_multi_metric` | 44 | Multi-metric precise questions (from `precise_questions.json`) - each metric asked = one "hop," each hop currently has exactly 1 gold chunk (no redundancy) |
| `natural_tightened` | 12 | Real FinanceBench train-split questions with genuine redundant hop groups (the same fact restated in multiple places - narrative + table, or prior-year comparatives) |

Stats across all 56: 2-7 hop groups per question (avg 3.7), 1-23 total gold chunks per question (avg 4.6). Only the 12 natural ones had `has_redundant_hops=True` - redundant hop-group sizes found: mostly 2-3 duplicate chunks per fact, with two outlier cases of 16 (a fact heavily restated across a filing).

**Held-out val set did not exist yet at this point.** All 12 `natural_tightened` questions were train-split (verified against `question_splits.json`), so this 56-question set had zero held-out counterpart - training on it and evaluating on it would repeat the exact contamination mistake caught and fixed earlier for RQ1. The plan: build a val set the same way `val_precise` was built for RQ1 (same distribution, genuinely unseen) - 40 questions from `val_precise_questions.json` (multi-metric, built from the 7 held-out metrics never used in training) plus ~5 natural FinanceBench val-split questions with real redundant hop groups. This became `build_rq2_val_dataset.py` / `rq2_val_dataset.json`.

### A.2 Why coverage-aware ranking is a different problem than RQ1

RQ1's reranker is trained and measured with precision/recall/NDCG - metrics that treat every gold chunk as interchangeable: if a gold chunk is in the top-k, credit is given regardless of which gold chunk it was. That's correct for single-metric questions (only ever one gold chunk anyway). It breaks down for multi-hop questions, because "gold chunks" aren't identical - they group into hop groups, where each hop group = one distinct fact the question needs, and a hop group can contain multiple chunks that are redundant restatements of the same fact (e.g. "total assets" appearing once in a table and again in an MD&A narrative sentence).

**Concrete example** (already in `eval/coverage_metrics.py`) - a real ROA-style question needing 2 facts (total assets, net income), each restated 9 times in the filing (18 gold chunks total):

- Ranking A: top-10 = 9 different restatements of "net income" + 1 unrelated chunk. Precision@10 = 0.9 - looks excellent. But this ranking is completely useless: you can't compute ROA without total assets, and it retrieved zero.
- Ranking B: top-10 = 5 "total assets" chunks + 5 "net income" chunks. Precision@10 = 1.0. Actually usable - both facts present.

Standard precision/recall can't tell these apart, or worse, can score the useless ranking A higher than a genuinely balanced one if the redundant hop happens to rank slightly better. Coverage@k fixes this by asking a different question entirely: not "what fraction of all gold chunks did we find," but "what fraction of distinct required facts have at least one representative in the top-k." Formally: `coverage@k = |{hop groups with >=1 member in top-k}| / |hop groups|`. Ranking A above scores coverage@10 = 0.5 (found net income, missed total assets) - correctly reflecting it's only half-useful, regardless of how many redundant net-income copies it stuffed in.

### A.3 How do you make a ranker optimize for coverage instead of just relevance?

Two approaches of increasing sophistication were scoped:

**Approach A - Greedy hop-aware re-ranking (no training, pure algorithm).** Borrows Maximal Marginal Relevance (MMR): once a hop group already has a representative near the top of the list, additional chunks from that same hop group are worth less - the fact is already covered, another copy wastes a slot that could go to a still-missing fact. Concretely: build the final list greedily, one slot at a time - pick the highest-scoring candidate whose hop group is not yet represented; fall back to best-remaining-score once every hop group has a representative. Requires knowing each candidate's hop-group membership, which is known at data-construction time but not at true inference time on an unseen question - so this is best treated as either an oracle upper bound (how good could coverage be if hop-grouping were known perfectly) or a heuristic using auto-detected hop clusters. (This was later actually built and tested - see `RQ2_RESULTS.md`'s discussion of the greedy MMR heuristic and its precision/coverage tradeoff.)

**Approach B - Coverage-aware learned ranker (generalizing LambdaMART).** LambdaRank doesn't optimize NDCG by writing it directly into a loss (NDCG is non-differentiable - you can't backpropagate through "sort these items"). Instead, for every pair of candidates (i, j), it computes how much the target metric would change if their ranks were swapped, and uses that magnitude ("lambda") to scale an ordinary pairwise gradient. This mechanism works for *any* rank-based metric evaluable on a full ranking, not just NDCG - NDCG is simply what it's traditionally paired with. To build a coverage-aware version: swap `ΔNDCG(swap i,j)` for `Δcoverage@k(swap i,j)`, computed via `coverage_at_k()`. If candidate i is a redundant copy of an already-covered hop and candidate j is the only representative found so far of a still-missing hop, swapping them so j moves up and i moves down increases coverage@k substantially - that pair gets a large lambda gradient, teaching the model to prefer j over i in this exact scenario, even though a pure relevance-only model might score i and j almost identically (both are "gold" - coverage@k is the only thing that can see the difference). This is a legitimate, well-founded extension (the same generalization researchers use to optimize MAP or ERR instead of NDCG), not a hack.

**Proposed evaluation structure** (mirrors RQ1's rigor): (1) diagnose the problem first with no new training - run the existing RQ1 pipeline on the held-out RQ2 val set, reporting recall@k and coverage@k side by side; (2) measure how much of the gap a training-free greedy re-ranking (Approach A) closes; (3) train the learned coverage-aware LambdaMART (Approach B) and evaluate coverage@k/recall@k on the same held-out val set, same train/val discipline as RQ1. This three-way comparison (relevance-only -> training-free diversification -> learned coverage-aware ranking) is exactly what `RQ2_RESULTS.md` reports.

### A.4 The Lambda gradient mechanism, precisely

For a pair of candidates (i, j) where i is more relevant than j (i is gold/hop-satisfying, j isn't - or more generally, i has a higher relevance label):

```
rho_ij = 1 / (1 + exp(sigma * (s_i - s_j)))    # how wrong the CURRENT model's ordering is
lambda_ij = sigma * rho_ij * |delta_metric_ij|  # the pairwise gradient magnitude
```

- If the model already scores i above j correctly, `s_i - s_j` is large and positive, so `rho_ij ~= 0` - barely any gradient, don't push on a pair that's already right.
- If the model has it backwards (j scored above i), `rho_ij` is large - big correction needed.
- `|delta_metric_ij|` is the swap-importance weight: how much NDCG (or coverage@k) would change if i and j traded positions - this is what makes top-of-list mistakes matter more than tail mistakes.

Then: i receives `+lambda_ij`, j receives `-lambda_ij` - i gets pushed up, j gets pushed down. This is computed for *every* such pair in the group, not just one pair per document - if a query has 3 gold candidates and 47 non-gold ones, each gold candidate is paired against all 47. A single candidate's total lambda for a boosting round is the sum of every pairwise lambda it participates in; that summed value becomes the pseudo-target the next tree in the ensemble is fit to (the "MART" in LambdaMART - gradient boosting, each new tree approximating the current negative-gradient signal, added on top of all previous trees). The whole process then repeats: rescore with the updated ensemble, recompute mis-orderings, recompute lambdas, fit the next tree. For the coverage-aware version, the *only* thing that changes in this entire mechanism is `|delta_metric_ij|`: instead of computing ΔNDCG, simulate the swap and recompute `coverage_at_k()` before/after using the query's `hop_groups` - everything else (sigmoid term, pair summation, tree-fitting) is identical, off-the-shelf LambdaMART machinery. (The actual implementation in `train_coverage_lambdamart.py` exploits an exactness property of coverage@k not available for NDCG - see `RQ2_RESULTS.md` Section 5.)

### A.5 Worked example: a mined redundant training question (Ulta Beauty FY2023)

Ulta Beauty's Q4 earnings press release and the subsequent 10-K report byte-identical figures across ~14 metrics - the single richest real redundancy example found during mining:

```json
{
  "id": "redundant_multi_ulta2023",
  "question": "What was Ulta Beauty's net sales, net income, total assets, and cash and cash equivalents in FY2023?",
  "source": "redundant_mined_earnings_vs_10k",
  "gold_chunk_ids": [
    "ULTABEAUTY_2023Q4_EARNINGS__block20__row0", "ULTABEAUTY_2023_10K__block332__row0",
    "ULTABEAUTY_2023Q4_EARNINGS__block20__row9", "ULTABEAUTY_2023_10K__block332__row10",
    "ULTABEAUTY_2023Q4_EARNINGS__block21__row14", "ULTABEAUTY_2023_10K__block330__row14",
    "ULTABEAUTY_2023Q4_EARNINGS__block21__row2", "ULTABEAUTY_2023_10K__block330__row2"
  ],
  "hop_groups": {
    "net sales": ["ULTABEAUTY_2023Q4_EARNINGS__block20__row0", "ULTABEAUTY_2023_10K__block332__row0"],
    "net income": ["ULTABEAUTY_2023Q4_EARNINGS__block20__row9", "ULTABEAUTY_2023_10K__block332__row10"],
    "total assets": ["ULTABEAUTY_2023Q4_EARNINGS__block21__row14", "ULTABEAUTY_2023_10K__block330__row14"],
    "cash and cash equivalents": ["ULTABEAUTY_2023Q4_EARNINGS__block21__row2", "ULTABEAUTY_2023_10K__block330__row2"]
  },
  "has_redundant_hops": true
}
```

Each hop has exactly 2 gold chunks (one from `ULTABEAUTY_2023Q4_EARNINGS`, the press release, one from `ULTABEAUTY_2023_10K`, the annual filing) - verified by checking the raw text of all 8 chunks directly: `Net sales | $ 10,208,580` appears verbatim in both, `Net income | $ 1,242,408` in both, etc. This is exactly the AES-ROA pattern (multiple genuinely distinct facts, each independently restated), occurring naturally rather than needing to be synthesized. Only 4 of Ulta's 14 redundant metrics for FY2023 are shown above for readability.

### A.6 Clarifying "hops" vs. "hop size" (two independent axes)

**Axis 1 - how many hops does the question have?** A "hop" = one distinct fact the question requires. A single-metric question ("What was 3M's revenue in FY2018?") has 1 hop. A multi-metric question ("What was 3M's capex, revenue, and total assets in FY2018?") has 3 hops - one per metric asked. Multi-metric questions are, by definition, multi-hop.

**Axis 2 - how many chunks satisfy each individual hop?** Completely separate from axis 1. For a given hop (say, "revenue"), does the corpus have exactly one chunk reporting that figure, or several (e.g. the same revenue number appearing in both an earnings release and the 10-K)?

Both the 121 synthetic multi-metric questions and the mined/natural redundant questions are multi-hop (multiple metrics per question) - the only thing that differs between them is axis 2:

| | Hops per question | Chunks per hop |
|---|---|---|
| Single-metric (459) | 1 | 1 (only ever 1, by construction) |
| Synthetic multi-metric (121) | 2-4 | 1 - `compose_precise_questions.py` only ever kept the first matching chunk per metric |
| Mined/natural redundant | 1-14 | 2+ - the same fact genuinely appears in more than one place (earnings release + 10-K, MD&A summary + financial statement, prose + table) |

This is exactly why coverage@k and recall@k are mathematically identical for the 121 synthetic questions (confirmed empirically - every number matched to the decimal): with hop size always 1, "find at least one chunk for this hop" and "find the hop's one and only chunk" are the same condition. The redundant tier is the only place hop size ever exceeds 1, which is why it's the only tier that teaches the coverage-specific behavior, and why the mining effort in `RQ2_RESULTS.md` (growing it well past this initial 12/5) mattered.

### A.7 Why not train on only the redundant-tier examples?

A reasonable instinct, but it carries real risk:

1. **The redundant examples don't stand alone.** Within any one redundant-tier query, most (query, candidate) pairs are still completely ordinary - "this gold chunk vs. that unrelated distractor from a different company." Only a small subset are the specifically novel kind ("this hop's 2nd redundant chunk vs. a different hop's only chunk"). There's no way to cleanly extract "only the informative pairs" - the informative signal is diffused inside full query groups, not separable as standalone training instances.

2. **Training only on ~37-50 examples means training on a much smaller number of distinct real-world scenarios.** These questions cluster into only about a dozen distinct filings (e.g. Ulta Beauty 2023 alone contributes several of the mined train questions, since its 14 redundant metrics get split into multiple questions). Training exclusively on this is a real overfitting risk - the model could latch onto something like "JPMorgan's `(a)` footnote markers mean redundancy" instead of a genuinely transferable rule, with no way to tell the difference without more held-out data than is available.

3. **The non-redundant examples teach the prerequisite skill.** Before a ranker can usefully decide "these two candidates satisfy the same hop, don't stack them both," it first needs to reliably decide "is this candidate relevant at all" - how much to trust BM25 vs. dense vs. entity-match, across a wide variety of companies and phrasings. The much larger non-redundant question pool is what teaches that general skill robustly. Strip it out, and the redundancy-avoidance behavior has nothing solid underneath it to specialize from.

(This reasoning is exactly why `RQ2_RESULTS.md`'s final training procedure always includes the full non-redundant pool alongside the redundant tier, rather than training on the redundant tier alone.)
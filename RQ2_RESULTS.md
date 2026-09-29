# RQ2 Results: Coverage-Aware Ranking for Multi-Hop Financial QA

## 1. Motivation: why relevance-based ranking isn't enough for multi-hop questions

RQ1's reranker is trained and measured with precision/recall/NDCG — metrics that treat every gold chunk as interchangeable. That's correct for single-fact questions, but breaks down for multi-hop questions, where "gold chunks" group into **hop groups**: each hop group is one distinct fact the question needs, and a hop group can contain multiple chunks that are redundant restatements of the *same* fact (e.g. a company's net income appearing once in its earnings press release and again, unchanged, in the subsequent 10-K).

A ranking that stacks multiple redundant copies of one already-covered fact into the top-k, while leaving a second required fact completely unretrieved, can still score deceptively well on precision/recall - but is useless for actually answering the question. **Coverage@k** (`eval/coverage_metrics.py`) is designed to catch exactly this: `coverage@k = |{hop groups with >=1 member in top-k}| / |hop groups|`, measuring whether every required fact has a representative in the top-k, regardless of how many redundant copies got pulled in.

An empirical diagnostic (running the existing RQ1 pipeline, unmodified, on a set of known multi-hop questions) confirmed this divergence is real and large: on redundant-hop questions, recall@10 was as low as 0.13 while coverage@10 on the same rankings was 0.48-0.80 - the pipeline was finding representatives of most required facts, but standard recall couldn't see it because of redundant-copy dilution in the denominator.

## 2. Dataset: training and test data for RQ2

**The core problem this dataset has to solve**: coverage@k only differs from recall@k when a hop group has 2+ chunks. A multi-metric question where every fact maps to exactly one chunk (no redundancy) gives coverage@k == recall@k *by mathematical necessity* - there's nothing for a coverage-aware objective to learn from such a question that a normal relevance objective wouldn't already teach. So the value of this dataset lives specifically in its **redundant-hop tier**, and that tier had to be built, not assumed.

**Sources combined** (`scripts/build_rq2_dataset.py` / `build_rq2_val_dataset.py`):

| Source | Train | Val | Redundancy |
|---|---|---|---|
| Synthetic multi-metric (existing RQ1 precise questions) | 121 | 40 | None (hop size always 1) |
| Natural FinanceBench questions with real redundant hop groups | 12 | 5 | Yes |
| **Mined redundant-hop questions (new, built for RQ2)** | **25** | **8** | **Yes** |
| **Total** | **158** | **53** | |

**Mining methodology** (`scripts/mine_redundant_hops.py`): rather than relying only on FinanceBench's sparse natural annotations (12 train / 5 val examples - too thin to train anything on), the corpus was searched directly for cases where the same (company, fiscal period, metric) fact is reported with the *exact same numeric value* in more than one chunk. Two real, recurring patterns were found this way: (a) an earnings-release table restated, unchanged, in the subsequent 10-K (e.g. Ulta Beauty FY2023 restates 14 metrics this way), and (b) a number stated in narrative prose corroborated by a table row elsewhere in the same filing (e.g. "*inventory balances of $5,261 million*" matching a balance-sheet row). Search was deliberately widened past 10-K-only, table-only matching (compose_precise_questions.py's original scope) to all filing types (10-K/10-Q/8-K/earnings) and narrative text, which is what took genuine matches from ~19 to ~55+.

**Verification discipline**: a redundant hop group is only accepted if the extracted numeric VALUE matches exactly (sign-agnostic), not just the row label - naive label-only matching produced ~45 false positives (e.g. a $14,694M "net sales" row and a $377M "net sales" row both matching the label "net sales" while being obviously different line items). One additional false positive was caught and fixed during this process: an EPS-calculation row mislabeled "Net income" (reporting a per-share value like $0.382, not the aggregate figure) was colliding with unrelated rows; it's now excluded upstream via a chunk-ID denylist, checked before either the table-table or narrative-table matching path runs (both had to be patched independently, since they don't share a merge step).

**Split discipline**: train-side mining draws only from the 17 metrics already used in RQ1's cross-encoder training data; val-side mining is restricted to the 7 metrics deliberately held out of that training (goodwill, accounts payable, other assets, total equity, accrued liabilities, SG&A, total operating expenses) - the same held-out-metric discipline established in RQ1's `val_precise` set, so val genuinely tests generalization to unseen facts, not just unseen questions about the same facts.

**Feature-extraction usability filter**: of the 158/53 total questions, some are dropped before training/eval because stage-1 retrieval found zero gold candidates in its top-50 (nothing to rank or learn from). This drop is concentrated much more heavily in the redundant tier (30% of train-redundant, 54% of val-redundant) than the non-redundant tier (2%/0%) - likely because several mined examples pull from less-common filing types (10-Q, earnings releases) that the entity/year detection and retrieval indices weren't originally tuned against. After this filter: **26 usable train-redundant and 6 usable val-redundant questions** (119 and 40 non-redundant, respectively) - these are the numbers behind every result in Section 5.

## 3. Methodology: coverage-aware LambdaMART as a third pipeline stage

**Where this sits**: stage 1 (BM25+dense fusion) and stage 2 (cross-encoder reranking) are unchanged from RQ1. This adds a stage 3, positioned to blend stage-1's signals into a single learned ranking score using an objective that directly targets coverage@k rather than plain relevance.

**Why a from-scratch implementation**: LightGBM and XGBoost's macOS wheels require `libomp` (via Homebrew), which isn't installed in this environment and would mean bootstrapping a new package manager - a bigger environment change than warranted. This is implemented instead as a manual gradient-boosting loop using `sklearn.tree.DecisionTreeRegressor` as the weak learner (`scripts/train_coverage_lambdamart.py`) - which turned out to be the right call regardless of the dependency issue, since a custom objective (below) needs full control over the gradient computation that off-the-shelf libraries don't expose a clean hook for anyway.

**Candidate pool**: for each question, stage-1 fusion's (BM25+dense CC fusion, entity/year filtered) top-50 candidates define the fixed set of items the model ranks - not a cross-encoder-filtered shortlist, so nothing is discarded before this stage gets a chance to reconsider it.

## 4. Features

Nine candidate-level features were tried in earlier iterations, including the cross-encoder's own relevance score - that version is not part of this result (see the module docstring in `train_coverage_lambdamart.py` for why it was abandoned: handing the model an already-fine-tuned score gave it nothing new to learn beyond copying that one column, and any apparent gain from doing so turned out to double as a false positive - some of the mined training chunks are the same chunks the cross-encoder itself was fine-tuned on, so its score was partly memorized rather than a fair test signal). The result reported here uses only features with **no fine-tuning history at all**, so there is no such contamination path to worry about:

| Feature | What it captures |
|---|---|
| `bm25_norm` | Min-max normalized BM25 score |
| `dense_norm` | Min-max normalized dense (mpnet) cosine score |
| `cc_score` | Stage-1's convex-combination fused score (alpha=0.4) |
| `is_value_dup_of_higher_ranked` | **The core coverage-specific signal**: does this candidate share an extracted numeric value with another candidate already ranked above it (in stage-1 order)? A company-agnostic, structural "someone already reported this exact fact" flag, computed by reusing the same value-extraction/normalization logic that mined the training data itself. Required two bug fixes to be reliable: matching on the label-adjacent value for table rows, and, for narrative text, excluding bare years and percentages (an early version flagged two unrelated chunks as "duplicates" purely because both happened to mention "fiscal 2018") |
| `entity_match` / `year_match` | Whether the candidate's detected company/fiscal-year matches the question's |
| `is_table_row` | Table row vs. narrative chunk type |
| `lexical_overlap` | Fraction of the question's content words appearing in the candidate's text |

**Warm-starting**: the ensemble is initialized to `cc_score` (stage-1's own fused score) rather than zero, and boosting learns a small additive *correction* on top of it, using shallow, heavily-regularized trees (`max_depth=2`, `min_samples_leaf=25`, `learning_rate=0.05`). This was a deliberate fix for an earlier failure mode: a deeper, un-warm-started version, given richer features, could and did memorize company-specific score patterns from the ~26 available training examples that didn't transfer to new companies' filings. Starting from an already-reasonable baseline and only allowing small, coarse corrections makes it structurally harder for the model to overfit that way.

## 5. Loss function: coverage-weighted LambdaRank, derived from scratch

Standard LambdaRank doesn't optimize NDCG by writing it directly into a loss (NDCG is a sorting-based, non-differentiable function of the ranking). Instead, for every pair of candidates (i, j), it computes how much a target metric would change if their ranks were swapped, and uses that magnitude to scale an ordinary pairwise gradient:

```
rho_ij = 1 / (1 + exp(sigma * (s_winner - s_loser)))    # how wrong the CURRENT scores are
lambda = sigma * rho_ij * |delta_metric|                 # gradient magnitude
```

The winner (the one that should rank above) gets `+lambda`; the loser gets `-lambda`. This mechanism is metric-agnostic - NDCG is simply the metric it's traditionally paired with. Here, `delta_metric` is **Δcoverage@k** instead of ΔNDCG, computed exactly (not approximated) using one key property of coverage@k: it is a pure SET-membership function over the top-k (a hop is "covered" iff any of its chunks is in the top-k), so **swapping two candidates that are both inside, or both outside, the current top-k provably cannot change coverage@k at all** - only a pair straddling the rank-k boundary can. This makes the exact computation cheap: for each candidate `a` currently inside the top-k and each `b` currently outside it, simulate replacing `a` with `b` and recompute coverage@k; if it increases, `b` is the winner; if it decreases, `a` is (correctly) already in the right place and gets reinforced; if unchanged, the pair contributes no gradient at all. This naturally and correctly gives zero gradient to the common case of two redundant copies of an *already*-covered hop competing with each other (reordering them can't help or hurt coverage), while still generating a strong, correctly-signed gradient whenever an under-represented hop's only candidate is being kept out of the top-k by a redundant copy of a hop that's already covered - exactly the behavior the whole exercise is meant to teach.

One documented simplification versus textbook LambdaMART: leaf values are fit via ordinary least-squares regression on the raw lambda gradients, not the Newton-step (`sum(grad)/sum(hess)`) leaf values the original paper uses. This converges somewhat slower but was judged unnecessary complexity for a training set this size.

## 6. Results

All results use `k=5` for the coverage-aware training objective itself (matching the P@5/NDCG@5 headline metrics used throughout this project). "Stage-1 baseline" is the unmodified `retrieve_fusion()` ranking with no learned reranking of any kind applied.

### 6a. Train (in-sample - expect inflated numbers, included for completeness only)

| Tier (n) | Metric | Stage-1 | LambdaMART | Relative change |
|---|---|---|---|---|
| **Redundant (26)** | coverage@5 | 0.430 | 0.449 | +4.5% |
| | coverage@10 | 0.487 | 0.535 | +9.9% |
| | coverage@20 | 0.558 | 0.619 | +10.9% |
| | NDCG@5 | 0.281 | 0.350 | +24.5% |
| | precision@5 | 0.239 | 0.262 | +9.6% |
| **Non-redundant (119)** | coverage@5 | 0.186 | 0.276 | +48.6% |
| | coverage@10 | 0.319 | 0.465 | +45.8% |
| | NDCG@5 | 0.173 | 0.258 | +49.4% |

### 6b. Val (held-out - the number that matters)

| Tier (n) | Metric | Stage-1 | LambdaMART | Relative change |
|---|---|---|---|---|
| **Redundant (6, clean)** | coverage@5 | 0.375 | **0.708** | **+88.8%** |
| | coverage@10 | 0.542 | **0.708** | **+30.7%** |
| | coverage@20 | 0.764 | 0.806 | +5.5% |
| | NDCG@5 | 0.170 | 0.316 | +85.9% |
| | recall@5 | 0.188 | 0.438 | +133.3% |
| | per-question coverage@5 | - | - | 2 improved / 4 unchanged / 0 worsened |
| **Non-redundant (40)** | coverage@5 | 0.348 | 0.508 | +46.0% |
| | coverage@10 | 0.471 | 0.648 | +37.6% |
| | NDCG@5 | 0.297 | 0.434 | +46.1% |

For reference, the cross-encoder alone (no coverage-awareness at all, but a much more expensive fine-tuned model) reaches coverage@5=0.806/NDCG@5=0.516 on this same 6-question redundant set - the coverage-aware LambdaMART, using only cheap pre-cross-encoder signals plus the duplicate-detection feature, closes roughly two-thirds of that gap from the stage-1 baseline without needing the cross-encoder at all.

### 6c. Pooled 5-fold cross-validation (32 redundant questions, out-of-fold)

To get a statistically steadier read than the 6-question val split alone, all 32 usable redundant-hop questions (train + val combined) were pooled and evaluated via 5-fold CV - every question scored only by a model trained on the other 4/5, with the 159 non-redundant questions always available as supporting training data in every fold.

| Metric | Stage-1 | LambdaMART (out-of-fold) | Relative change |
|---|---|---|---|
| coverage@5 | 0.419 | 0.490 | +16.8% |
| coverage@10 | 0.497 | 0.576 | +15.7% |
| coverage@20 | 0.596 | 0.669 | +12.2% |
| NDCG@5 | 0.260 | 0.331 | +27.1% |
| precision@5 | 0.213 | 0.269 | +26.5% |
| recall@5 | 0.238 | 0.316 | +32.9% |

Per-question coverage@5: **3 improved, 28 unchanged, 1 worsened**, out of 32 - a genuinely direction-consistent result (unlike earlier cross-encoder-based attempts, which showed roughly equal numbers of improved and worsened questions, the signature of noise rather than a learned effect), and improvement holds at every k depth tested (5/10/20) rather than flipping sign.

## 7. Interpretation

The recall-vs-coverage divergence motivating this whole investigation is real and substantial (Section 1). Closing it with a *learned* method turned out to require care about what the model is actually allowed to learn from: given a feature set with no prior fine-tuning history (BM25, dense, fusion score, entity/year match, table-type, lexical overlap, and - the one feature purpose-built for this problem - value-based duplicate detection), a small, heavily-regularized, warm-started LambdaMART correction produces a genuine, direction-consistent, cross-validated improvement in top-k coverage, on both the held-out val split and a larger pooled out-of-fold test, without regressing the non-redundant majority tier (which also improves, likely because the model is learning a better combination of the underlying retrieval signals generally, not just a redundancy-specific correction).


## 8. RQ1 vs. RQ2 comparison context

Three test beds, side by side, to make clear exactly what's being compared to what.

### 8.1 591 questions - baseline + cross-encoder (beta=1.0) - training-source, NOT held out

Coverage = recall exactly for this whole set (no redundant hop groups exist in it - every hop maps to 1 chunk by construction), so one column covers both. **This set is in-sample for the cross-encoder** (its gold chunks were used in cross-encoder fine-tuning) - included here for reference only, not as a generalization claim; see 8.3 for the genuinely held-out equivalent.

| | P@10 | R@10 = Cov@10 | NDCG@10 | P@50 | R@50 = Cov@50 | P@100 | R@100 = Cov@100 | NDCG@100 |
|---|---|---|---|---|---|---|---|---|
| Baseline (stage-1) | 0.079 | 0.630 | 0.386 | 0.026 | 0.900 | 0.014 | 0.940 | 0.466 |
| Baseline + CE (beta=1.0) | 0.136 | 0.925 | 0.814 | 0.028 | 0.950 | 0.014 | 0.950 | 0.822 |

(at k=5: baseline P/R/NDCG = 0.109/0.447/0.322; +CE = 0.252/0.883/0.797)

### 8.2 32 questions - pooled redundant-hop set, baseline + LambdaMART (5-fold CV, out-of-fold, genuinely held out)

| | P@5 | R@5 | NDCG@5 | Cov@5 | P@10 | R@10 | NDCG@10 | Cov@10 | P@20 | R@20 | NDCG@20 | Cov@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (stage-1) | 0.213 | 0.238 | 0.260 | 0.419 | 0.147 | 0.352 | 0.288 | 0.497 | 0.105 | 0.440 | 0.330 | 0.596 |
| Baseline + LambdaMART | 0.269 | 0.316 | 0.331 | 0.490 | 0.200 | 0.438 | 0.358 | 0.576 | 0.131 | 0.528 | 0.401 | 0.669 |

(This duplicates Section 6c above - kept here too so all three test beds are visible in one place for direct comparison.)

### 8.3 185 questions (held-out, unseen metrics) - baseline + cross-encoder (beta=1.0) - the correct held-out counterpart to 8.1

| | P@5 | R@5 | Cov@5 | P@10 | R@10 | Cov@10 | P@50 | R@50 | Cov@50 | P@100 | R@100 | Cov@100 | NDCG@10 | NDCG@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (stage-1) | 0.149 | 0.586 | 0.588 | 0.096 | 0.738 | 0.741 | 0.026 | 0.926 | 0.931 | 0.014 | 0.940 | 0.945 | 0.522 | 0.577 |
| Baseline + CE (beta=1.0) | 0.254 | 0.916 | 0.919 | 0.136 | 0.942 | 0.945 | 0.027 | 0.945 | 0.950 | 0.014 | 0.945 | 0.950 | 0.842 | 0.843 |

### 8.4 The genuinely apples-to-apples comparison (both held out)

| | Recall@10 | Coverage@10 | Recall@5 gain over baseline | n |
|---|---|---|---|---|
| CE (185-set, held-out metrics) | 0.942 | 0.945 | +56.3% | 185 |
| LambdaMART (32-set, redundant-hop, 5-fold CV) | 0.438 | 0.576 | +32.9% | 32 |

This makes the contrast between the two RQs' test beds sharper and more honest: the 185-set is dominated by single/multi-metric factual-lookup questions where coverage and recall are nearly identical by construction (only 5 questions have any redundancy at all), while the 32-set is *entirely* redundant-hop questions - which is exactly why recall/coverage diverge so much more there (0.44 vs. 0.58 at k=10). The two tables are testing genuinely different question distributions, not competing on the same task, and both are now properly held out rather than one being training-source.
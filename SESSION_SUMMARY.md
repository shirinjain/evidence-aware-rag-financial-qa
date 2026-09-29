# Session Summary — Retrieval Pipeline Improvements

All numbers below are measured on the 84-question natural-FinanceBench **train** split (test remains untouched). Gold evidence uses the cleaned alignment (value-matching + justification-tightening applied where possible).

## Results progression

| Config | P@5 | NDCG@5 | R@20 |
|---|---|---|---|
| BM25 alone | 0.038 | 0.061 | 0.072 |
| Dense (mpnet) alone | 0.052 | 0.075 | 0.164 |
| RRF fusion (k=60) | 0.057 | 0.083 | 0.161 |
| CC fusion (alpha=0.5) | 0.062 | 0.075 | 0.164 |
| CC fusion + entity filter | 0.069 | 0.087 | 0.184 |
| CC fusion + entity + year filter | 0.081 | 0.101 | 0.213 |
| **CC fusion (alpha=0.4) + entity + year filter (current best)** | **0.088** | **0.113** | ~0.21 |

Recall depth check (best config, k up to 200): **recall@200 = 0.481** — close to the 50% target discussed, up from BM25 alone's 0.216 at the same depth.

Current best pipeline is `scripts/pipeline_best.py`.

## What worked, and why

1. **Entity filtering** (`scripts/entity_detection.py`) — generic financial line-item labels ("Dividends paid to shareholders," "Net cash provided by operating activities") are nearly identical across companies, so both BM25 and dense pull in wrong-company chunks. Filtering to the detected company removes this.

2. **Year filtering** (`scripts/year_detection.py`) — even within the right company, the wrong fiscal year's document sometimes wins (e.g. a 3M FY2018 question pulling a `3M_2023Q2_10Q` chunk). Filtering to detected year(s) fixes this. Handles ranges ("FY2019 - FY2021") by including all years in between, not just the endpoints, since multi-year-average questions need all of them.

3. **CC (convex combination) fusion beats RRF here** — weighted, min-max-normalized score averaging of BM25 + dense outperformed rank-based RRF fusion in direct comparison. Alpha sweep found 0.4 (dense weighted slightly higher than BM25) beats the default 0.5.

4. **Gold-alignment cleanup (3+ rounds)** — the original loose "does this row's label+number appear anywhere in the evidence_text blob" matching had a long tail of false positives (generic single words like "Total"/"Rate"/"Interest", bare dates, company-address rows, duplicated table captions). Added: (a) a blocklist + bare-year/date pattern rejection, (b) rejecting labels that duplicate their own section heading, (c) a new **value-matching tier**: when the question's `answer` field is a single clean number, mark a row gold only if it's inside the annotator's evidence_text AND contains that exact value (sign-agnostic, since financial statements show outflows as negative/parenthesized while answers are stated as positive magnitudes). This is now the highest-confidence gold tier (22/150 questions), with justification-based tightening as tier 2 (16/150), and the broad fallback for the rest.

## What didn't work (tested, not adopted)

- **Coarse (whole-table) chunking** — hypothesized this would help based on a published paper (arxiv 2604.01733) reaching Recall@5=0.816 with "no chunking." Tested directly: made things *worse* (P@5 0.038→0.026 unfiltered). Root cause found: that paper's dataset (T2-RAGBench, built from FinQA/ConvFinQA/TAT-DQA) has naturally small (~920 token) source documents — theirs is "no chunking needed" because the documents were already small, not "coarse chunking helps." Our corpus starts from full 10-Ks; concatenating a 66-row table into one block creates huge, diluted chunks with no equivalent uniformity. Reverted to row-level chunking.

- **Off-the-shelf community "finance reranker" models on HuggingFace** — checked `jefffreyli/financial-reranker`, `tolivert/financial-reranker-v1`, `cindy415/financial_reranker`. All have essentially zero usage/validation signal (12, 21, 0 downloads respectively) — too risky to trust without independent verification. Not adopted.

- **Cross-encoder fine-tuning (RankNetLoss, hard-negative mining) — NOT COMPLETED, two attempts.** Built the hard-negative data construction pipeline (`scripts/build_finetune_data.py`), using this project's own best pipeline's top-ranked-but-wrong results as hard negatives (83 question-rows, 874 total docs). Both attempts to fine-tune `ms-marco-MiniLM-L-6-v2` via `RankNetLoss` locally on this machine's MPS backend showed the same **escalating per-step slowdown** pattern:
  - Attempt 1 (batch_size=8, 5 epochs): 82s → 251s/step, killed after 2 steps.
  - Attempt 2 (batch_size=4, explicit `mini_batch_size=8`, 2 epochs — tighter memory controls to rule out a batch-content-size cause): started fast (2-8s/step), but by step 22/36 had climbed past 60s/step and was still rising. Killed at that point.
  - This reproducing across two differently-configured attempts suggests a genuine MPS-backend degradation under sustained load on this hardware (thermal throttling or memory fragmentation), not a hyperparameter issue - hyperparameter changes didn't fix the pattern, only delayed its onset.
  - **No fine-tuned model was produced or evaluated this session.** `scripts/finetune_cross_encoder.py` and the hard-negative data are ready to use — **recommend running this via Colab GPU** (same pattern used successfully for embeddings and in the original project) rather than local MPS, next time this is attempted.

## What the paper (arxiv 2604.01733, T2-RAGBench) confirmed independently

- Published baseline FinanceBench-style RAG accuracy is genuinely low (~10-19% in early setups) — our low numbers are consistent with the documented literature, not a sign of a broken pipeline.
- Even the best open embedding models only reach ~29.4% Recall@1 on this style of benchmark.
- Cross-encoder reranking was their single biggest lever (+12.1pp Recall@5) — bigger than fusion (+5.1pp) or contextual retrieval (+2.2pp). They used a paid Cohere Rerank v4.0 Pro model, finance-domain-benchmarked — a domain-tuned reranker, not a generic one, which likely explains part of the gap.

## Validated on val (not just train) — the gains hold, even more strongly

| | P@5 | NDCG@5 | R@20 |
|---|---|---|---|
| BM25 alone (30 val questions) | 0.087 | 0.123 | 0.116 |
| **Best pipeline (CC a=0.4 + entity + year filter)** | **0.127** | **0.178** | **0.185** |
| Relative improvement | **+46%** | **+45%** | **+60%** |

This confirms the improvements found while iterating on train aren't overfit to train-specific quirks — the same pipeline, unchanged, generalizes to val with an even larger relative gain. This is real, not a train-set artifact.

## Precise question set — built our own QA pairs with exact, small gold sets

Rather than continue patching alignment noise on FinanceBench's own recorded evidence (diminishing returns after 3+ cleanup rounds), built a companion dataset where **gold is exact by construction** — the chunk is picked first, the question is written around it, so there's no fuzzy matching at all:

- **`scripts/compose_precise_questions.py`** (30 questions): for each (company, year) with 2+ of {revenue, net income, total assets, total liabilities} available, asks for all of them in one question (e.g. "What was 3M's revenue, total assets and total liabilities in FY2018?"). Gold = exactly the N chunks for the N metrics asked. Extraction requires the row's section heading to specifically match "Consolidated Statement(s) of Income/Operations" (revenue, net income) or "Consolidated Balance Sheet(s)" (assets, liabilities) — **found and fixed a real bug** during this: a 3M "Net income" row was initially pulled from the *Statement of Changes in Equity* (an equity-allocation breakdown, wrong statement) because the looser "consolidated statement" substring check doesn't distinguish between statement types, and a second bug where the whitespace-broken heading text ("Consolidated Statement of Incom e" — Docling split "Income" across a line break) needed the same whitespace-stripped matching fix used earlier in the project. Verified: 3M FY2018 revenue ($32,765M), total assets ($36,500M), total liabilities ($26,652M) all check out.
- **`scripts/select_precise_narrative_questions.py`** (17 questions): prose doesn't have a clean label/value structure to construct fresh questions from, so instead this *selects* real FinanceBench questions whose already-cleaned gold is small (≤5) and narrative-dominant (≥80%) — a real annotator already wrote a focused question citing 1-2 paragraphs, so these are naturally precise once alignment is clean.

**Result: 47 total precise questions, mean gold count 2.5 (vs 21.9-37.4 for the FinanceBench-derived set across cleanup rounds).**

| | P@5 | NDCG@5 | R@20 |
|---|---|---|---|
| BM25 alone | 0.038 | 0.054 | 0.369 |
| **Best pipeline (CC a=0.4 + entity + year filter)** | **0.089** | **0.146** | **0.532** |

**Recall@20 crosses 50% for the first time this project** (0.532), more than double the FinanceBench-derived set's R@20 (~0.21) on the same pipeline. This directly confirms the user's hypothesis: much of the previously-low recall was an artifact of noisy/inflated gold counts, not purely a retrieval-quality ceiling.

## Full precise-question set grown further, combined evaluation

Expanded `compose_precise_questions.py` with more metrics (capital expenditures, operating cash flow, depreciation and amortization, alongside revenue/net income/total assets/total liabilities) and added **single-metric question generation** (one question per available metric, gold=1 always) alongside the existing multi-metric compound questions - this directly reproduces the "3M capex" pattern (a sharp, single-fact lookup) that scored 0.8 precision in the natural set.

Combined evaluation across all tiers (258 questions total, same best pipeline throughout):

| Tier | Questions | Gold size | R@20 | R@50 | R@100 | NDCG@100 |
|---|---|---|---|---|---|---|
| Single-metric | 185 | 1 (always) | 0.741 | 0.886 | **0.946** | 0.429 |
| Multi-metric | 44 | 2-7 | 0.434 | 0.544 | 0.689 | 0.411 |
| Narrative | 17 | 1-3 | 0.627 | 0.686 | 0.745 | 0.269 |
| Natural-redundant (original FinanceBench style, train-only) | 12 | varies, redundant | 0.115 | 0.170 | 0.174 | 0.107 |
| **Combined** | **258** | — | **0.652** | **0.782** | **0.853** | **0.400** |

Single-metric questions reach 94.6% recall@100 - close to the "100% recall" target discussed, confirming that most of the earlier low recall was inflated-gold-count noise, not a hard retrieval ceiling, for genuinely simple lookups. Multi-metric and narrative remain harder (multiple distinct facts, or prose without a clean value to match), and the original FinanceBench-style computed-ratio questions remain hardest by a wide margin - consistent with the vocabulary-mismatch diagnosis from earlier in the session.

## RQ2 dataset — combining synthetic (clean) and natural (redundant) hop structure

`scripts/build_rq2_dataset.py` produces `data/processed/rq2_dataset.json` (56 questions: 44 synthetic multi-metric + 12 natural train-split questions with real redundant hop_groups, like AES ROA's ~9x-restated "net income").

Key finding: **coverage@k == recall@k exactly on the synthetic set** (each hop maps to exactly 1 chunk, no redundancy to diverge on) but **coverage@k is ~4x higher than recall@k on the natural-redundant set** (P@5=0.067, recall@5=0.068, coverage@5=0.278) - a second, complementary failure mode to the earlier AES ROA ranking example: recall *understates* true task completion when hops have redundant restatements (finding 1 of 9 "net income" copies is enough to have the fact, but recall penalizes for not finding the other 8). Both directions - deceptively high precision/recall with low coverage (ranking-level), and deceptively low recall despite full coverage (redundancy-level) - are real, demonstrated blind spots in standard IR metrics that motivate RQ2's coverage-aware loss.

## Next steps, in priority order

1. **Run `scripts/colab_finetune_cross_encoder.py` on Colab GPU** (prepared and ready — same cell-by-cell pattern used successfully for embeddings earlier). Local MPS training was not viable in two attempts. This is the highest-value remaining lever per both our own diagnosis (right-table-wrong-row errors) and the published paper's ablation (+12.1pp Recall@5, the single biggest effect they measured).
2. Once a fine-tuned reranker exists, rerank `pipeline_best.py`'s top-50 candidates with it and re-measure P@5/NDCG@5 against the current best (0.088/0.113 train, 0.127/0.178 val).
3. Consider extending value-matching gold-tightening to a broader regex/answer-format coverage (currently only clean single-number answers qualify — 52/150 questions) to further reduce the residual gold-alignment noise in the remaining ~110 questions.
4. Test the alpha=0.4 fusion weight more broadly once a reranker is added, since reranking may shift which upstream fusion weight is optimal.

## Known remaining limitation

Diagnosed directly on a concrete example (3M FY2018 capex question): the pipeline correctly narrows down to the *right table* (the cash flow statement) but doesn't always rank the *specific correct row* within it above other rows in the same table (e.g. "Proceeds from sale of PP&E" outranking "Purchases of property, plant and equipment"). This is precisely the fine-grained discrimination problem cross-encoder reranking (joint attention over query+passage) is meant to solve, and which BM25/dense fusion structurally cannot — reinforcing that reranking remains the right next lever once a working fine-tuning setup is available (Colab GPU, given local MPS training has not converged in reasonable time in two attempts).

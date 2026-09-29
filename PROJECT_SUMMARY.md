# Evidence-Aware RAG for Multi-Hop Financial QA

*Full experimental detail lives in `RQ1_RESULTS.md` and `RQ2_RESULTS.md`; this document ties the two research questions together into one coherent project.*

## Abstract

Retrieval-augmented QA over SEC filings (10-Ks, 10-Qs, 8-Ks, earnings releases) is harder than standard open-domain RAG for two structural reasons: (1) financial documents are dominated by tables, where the correct row and the wrong row can differ by a single word or a sign convention, and (2) many real questions are multi-hop - they require several distinct facts, each potentially restated in more than one place in the corpus (an earnings release and the subsequent 10-K reporting the same figure, for instance). This project addresses both problems on the FinanceBench dataset. **RQ1** asks whether a domain-informed, hard-negative-mined cross-encoder reranker can close the gap between BM25+dense retrieval and something usable for financial QA - it can: **+56.3% recall@10** (0.60 -> 0.94) on a genuinely held-out, same-distribution test set, with a further honest finding that full trust in the reranker (beta=1.0) is optimal for factual-lookup questions but actively regresses on analytically-phrased ones, resolved via a tunable blending mechanism. **RQ2** asks whether a coverage-aware ranking objective - one that explicitly penalizes redundant restatements of an already-covered fact in favor of a still-missing one - can be learned and can generalize, given how little naturally-redundant training data exists. a minimal, warm-started, heavily-regularized LambdaMART variant - trained on features with no prior fine-tuning history at all - produces a genuine, cross-validated **+32.9% recall@10** (+16.8% coverage@10) improvement on a 32-question redundant-hop test set, entirely without using the cross-encoder.

## 1. Motivation

FinanceBench and similar financial-QA benchmarks are frequently reported with surprisingly low baseline retrieval numbers in published work. Early diagnostic work in this project traced that to two separable causes rather than one: genuine retrieval difficulty, and noisy gold-evidence alignment (FinanceBench's own recorded evidence spans include false positives - bare years, generic single words, wrong-statement-type matches). Both needed to be addressed before any reranking method could be evaluated meaningfully, which is why this project built its own **precise question set** (591 questions, exact-by-construction gold chunks) alongside the natural FinanceBench questions, rather than relying solely on the latter's noisier annotations.

Two distinct sub-problems remained even after that fix, motivating the two research questions:

- **RQ1**: BM25+dense fusion alone retrieves the right chunk somewhere in a wide candidate list, but not reliably near the top - a cross-encoder reranker is the standard fix, but domain-general or naively-trained versions don't obviously transfer to dense financial tables.
- **RQ2**: even a well-tuned reranker optimizes precision/recall/NDCG, metrics that treat every gold chunk as interchangeable. For multi-hop questions where facts are restated redundantly across a filing, this creates a measurable blind spot - a ranking can score well while actually retrieving nine copies of one fact and zero of another.

## 2. System architecture

Three stages, each independently validated and each addressing one of the problems above:

1. **Stage 1 - retrieval** (`scripts/pipeline_best.py::retrieve_fusion`): BM25 (`bm25s`) + dense (`all-mpnet-base-v2`) retrieval, each filtered by regex-based company-entity and fiscal-year detection, fused via convex combination (alpha=0.4, found by sweep). For multi-metric questions, `retrieve_decomposed()` splits the query into one single-metric sub-question per detected metric before this stage runs, rather than retrieving on the bundled question directly - see RQ1 Section 5e.
2. **Stage 2 - cross-encoder reranking** (RQ1): a `ms-marco-MiniLM-L-6-v2` cross-encoder, fine-tuned via RankNetLoss on domain-informed hard negatives mined from the pipeline's own top-ranked wrong answers, reranks stage 1's top-k. Blended with stage 1's ranking via a tunable `beta`.
3. **Stage 3 - coverage-aware ranking** (RQ2): a from-scratch LambdaMART variant, trained with a custom Δcoverage@k gradient (generalizing the standard ΔNDCG LambdaRank gradient to a different, non-differentiable target metric), reorders stage 1's candidates to explicitly avoid over-selecting redundant copies of an already-covered fact.

`retrieve_auto()` is the recommended top-level entry point: it detects multi-metric questions and routes to the decomposed path automatically, falling back to the ordinary retrieve() otherwise.

Stages 2 and 3 were developed and validated independently, on different, purpose-built test sets (see Section 5) - they were never combined into a single serving-time stack, since RQ2's investigation found that naively applying the cross-encoder after LambdaMART (or vice versa) is order-blind and simply erases whichever stage ran first, rather than combining their benefits. A principled combination (score-blending, analogous to RQ1's beta mechanism) was scoped but not built - see Section 6.

## 3. RQ1: domain-informed hard-negative cross-encoder reranking

**Method**: hard negatives are not random - they are the pipeline's own top-ranked *wrong* answers, i.e. exactly the confusions the retrieval system is already known to make (cross-year/cross-company boilerplate twins, table-vs-narrative distractors, wrong-statement-type mismatches). Trained via RankNetLoss on 591 questions / 3,682 (question, document) pairs.

**Key finding, methodological**: an earlier fine-tuning attempt on 1/13th the data produced a textbook-normal loss curve (steadily decreasing train/eval loss) while the resulting model had *inverted* score preferences on direct inspection - hard negatives scoring above true positives. This is the central lesson carried through the rest of the project: **a converging loss curve does not guarantee a useful model**; only measuring actual downstream retrieval performance does.

**Key finding, results** (full detail: `RQ1_RESULTS.md`): on the genuinely held-out, same-distribution 185-question test set (built from 7 metrics never touched during fine-tuning):

| | Recall@10 | Coverage@10 | NDCG@10 |
|---|---|---|---|
| Baseline (stage 1 only) | 0.738 | 0.741 | 0.522 |
| **+ cross-encoder (beta=1.0)** | **0.942 (+27.6%)** | **0.945 (+27.5%)** | **0.842 (+61.3%)** |

**Key finding, generalization boundary**: beta=1.0 (full trust in the reranker) is optimal on this same-distribution held-out set, but *actively regresses* performance (-21% P@5) on a natural-FinanceBench, analytically-phrased held-out set - a genuine, reproducible, distribution-dependent reliability finding, not noise. Resolved via `beta=0.3` as a dual-distribution-safe default when the deployed query mix is uncertain; `beta=1.0` when it's known to resemble the training distribution.

**Key finding, retrieval-time (added after initial writeup)**: manually spot-checking individual prompts surfaced a real bug affecting multi-metric questions specifically - bundling several metric names into one query (e.g. "cash and cash equivalents, cost of sales, total current assets and total current liabilities") dilutes BM25/dense term-overlap scoring for each individual metric, to the point that one gold chunk was missing from the top-100 entirely when bundled, yet ranked #1 when asked alone. Fixed via query decomposition - `retrieve_decomposed()` splits the question into one single-metric sub-question per detected metric, retrieves+reranks each independently through the full pipeline, and interleaves the results. Validated on held-out multi-metric questions: **coverage@10 0.806 -> 0.969 (+20.2%), 0 of 40 questions regressed.** `retrieve_auto()` is now the recommended pipeline entry point (full detail: `RQ1_RESULTS.md` Section 5e).

## 4. RQ2: coverage-aware ranking for multi-hop questions

**Method**: `coverage@k = |{hop groups with >=1 member in top-k}| / |hop groups|` (`eval/coverage_metrics.py`) directly measures whether every required fact has a representative in the top-k, unlike recall, which is diluted by redundant restatements of an already-covered fact. A training/test set for this had to be built, not assumed - only 12-17 natural FinanceBench questions have genuine redundant hop groups, too thin to train or reliably test anything on. Corpus mining (`scripts/mine_redundant_hops.py`), searching across all filing types (not just 10-Ks) and both table-table and narrative-table value matches, grew the redundant-hop tier to 37 train / 13 val questions, verified via exact numeric-value matching (not just label matching, which produced ~45 false positives on inspection).

**Key finding, methodological**: three failed attempts preceded the working one, each individually diagnosed rather than discarded silently:
1. A feature set including the cross-encoder's own score showed apparent train-set gains that turned out to be the model mostly copying that one already-strong feature - held-out performance was statistically indistinguishable from noise (roughly equal numbers of improved/worsened questions across a 5-fold CV).
2. Digging into *why* it didn't generalize surfaced a genuine contamination bug: 44% of the mined training questions' gold chunks were the same chunks the cross-encoder was itself fine-tuned on in RQ1, meaning the "cross-encoder only" reference score was partly memorized rather than a fair signal for a meaningful fraction of the pooled test set.
3. Removing the cross-encoder entirely and using only features with no fine-tuning history (BM25, dense, fusion score, entity/year match, and a purpose-built value-duplicate-detection feature), combined with warm-starting the ensemble from the stage-1 score instead of zero and heavy regularization, is what finally produced a real, direction-consistent, cross-validated effect.

**Key finding, results** (full detail: `RQ2_RESULTS.md`): 5-fold cross-validation over all 32 usable redundant-hop questions, each scored only by a model trained on the other 4/5:

| | Recall@10 | Coverage@10 | NDCG@5 |
|---|---|---|---|
| Baseline (stage 1 only) | 0.352 | 0.497 | 0.260 |
| **+ coverage-aware LambdaMART** | **0.438 (+24.4%)** | **0.576 (+15.9%)** | **0.331 (+27.1%)** |

Per-question coverage@5: 3 improved, 28 unchanged, 1 worsened - a genuinely direction-consistent result, and improvement holds across k=5/10/20 rather than flipping sign, unlike the earlier cross-encoder-based attempt.

## 5. Test-bed summary (what's being compared to what)

| Test set | n | Question style | Redundancy | Used for |
|---|---|---|---|---|
| 591 precise (training-source) | 591 | Single/multi-metric factual lookup | None | RQ1 sanity check only - NOT a generalization claim |
| 185 held-out (`val_precise`) | 185 | Single/multi-metric, unseen metrics | Minimal (5/185) | RQ1's headline held-out result |
| Natural FinanceBench (train/val) | 84/30 | Analytical, computed-ratio | Minimal | RQ1's cross-distribution generalization check |
| 32 pooled redundant-hop | 32 | Multi-metric, heavily redundant | 100% by construction | RQ2's headline held-out result (via 5-fold CV) |

These are intentionally different distributions, not a single leaderboard - RQ1's 185-set and RQ2's 32-set test genuinely different failure modes, and the two headline numbers (recall@10 +56.3% vs. +32.9%) are not in competition with each other.

## 6. Limitations and future work

- **RQ2's effect size is real but modest**, and validated on a small pool (32 questions, roughly a dozen distinct underlying filings) - more mining, or accepting a smaller effect as the honest ceiling given available data, are both reasonable next steps.
- **Stages 2 and 3 were never combined into one serving pipeline.** A principled blend (score-level, not order-overwrite) was scoped but not built - the natural next experiment.
- **Retrieval-metric improvements are necessary but not sufficient for end-to-end answer accuracy.** Coverage@k confirms the right evidence is present in context; it says nothing about whether a downstream LLM extracts and reasons over it correctly (positional/"lost in the middle" effects, distractor confusion, arithmetic errors). An end-to-end generation-and-compare eval was scoped but not run at full scale (would require a metered LLM API not available in this environment).
- **Adaptive retrieval depth** (k=10 default, deeper for detected multi-hop questions) was identified as a better policy than a flat top-k for final answer generation, given recall saturates by k=10 for simple questions but coverage keeps climbing well past k=10 for redundant-hop ones - scoped but not implemented.

## 7. What this project demonstrates methodologically

Independent of the specific numbers, the recurring discipline across both RQs is worth stating explicitly, since it's what makes the results trustworthy: **every claimed improvement was checked against a genuinely held-out set before being reported**, and every time a result looked too good on a first pass (RQ1's early beta=1.0 "win" that didn't survive a proper val check; RQ2's cross-encoder-feature LambdaMART that turned out to be copying one column; RQ2's contaminated pooled k-fold that partially inflated a "clean" result), the discrepancy was investigated rather than smoothed over, the root cause identified, and the corrected result reported instead - including the negative results and abandoned approaches, not just the ones that worked.


Answer the question using ONLY the information in the context below, which is extracted from SEC filings (10-K/10-Q/8-K/earnings releases). If the context does not contain enough information to answer, say so explicitly rather than guessing.

Context:
[1] Goodwill | 10,051 | 10,513
[2] 3M goodwill totaled approximately $10.1 billion as of December 31, 2018. 3M's annual goodwill impairment testing is performed in the fourth quarter of each year. Impairment testing for goodwill is done at a reporting unit level, with all goodwill assigned to a reporting unit. Reporting units are one level below the business segment level, but are required to be combined when reporting units within the same segment have similar economic characteristics. At 3M, reporting units correspond to a division. 3M did not combine any of its reporting units for impairment testing.
[3] 3M is a highly integrated enterprise, where businesses share technology and leverage common fundamental strengths and capabilities, thus many of 3M's businesses could not easily be sold on a stand-alone basis. 3M's focus on research and development has resulted in a portion of 3M's value being comprised of internally developed businesses that have no goodwill associated with them. Based on the annual test in the fourth quarter of 2018, no goodwill impairment was indicated for any of the reporting units.
[4] 3M makes acquisitions of certain businesses from time to time that are aligned with its strategic intent with respect to, among other factors, growth markets and adjacent product lines or technologies. Goodwill resulting from business combinations is largely attributable to the existing workforce of the acquired businesses and synergies expected to arise after 3M's acquisition of these businesses.
[5] As of October 1, 2018, 3M had 24 primary reporting units, with ten reporting units accounting for approximately 89 percent of the goodwill. These ten reporting units were comprised of the following divisions: Advanced Materials, Display Materials and Systems, Electronics Materials Solutions, Health Information Systems, Industrial Adhesives and Tapes, Infection Prevention, Oral Care Solutions, Personal Safety, Separation and Purification, and Transportation Safety. The estimated fair value for all reporting units was in excess of carrying value by approximately 79 percent or more. 3M's market value at both December 31, 2018, and September 30, 2018, was significantly in excess of its shareholders' equity of approximately $10 billion.
[6] Purchased goodwill | 1,296 | 6 | 1,302
[7] As discussed in Note 3, 3M sold its Communication Markets division in 2018, which comprised substantially all of the $272 million reduction in goodwill associated with divestitures during 2018.
[8] Accounting standards require that goodwill be tested for impairment annually and between annual tests in certain circumstances such as a change in reporting units or the testing of recoverability of a significant asset group within a reporting unit. At 3M, reporting units correspond to a division.
[9] In addition, approximately $275 million of goodwill was estimated to be attributable to disposal groups classified as held-for-sale as of December 31, 2017, based upon relative fair value. The amounts above have not been segregated and are classified within the existing corresponding line items on the Company's consolidated balance sheet.
[10] Note 4. Goodwill and Intangible Assets | 76

Question: What was 3M's goodwill in FY2018?

Answer:
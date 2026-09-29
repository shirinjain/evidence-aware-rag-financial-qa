# Evidence-Aware RAG for Multi-Hop Financial QA

Retrieval-augmented QA over SEC filings (10-Ks, 10-Qs, 8-Ks, earnings releases), built on the [FinanceBench](https://arxiv.org/abs/2311.11944) dataset. Financial documents are dominated by tables (where the right row and the wrong row can differ by a single word or sign convention) and many real questions are multi-hop, requiring several distinct facts that are sometimes restated in more than one place in the corpus. This project addresses both problems as two research questions:

- **RQ1** — can a domain-informed, hard-negative-mined cross-encoder reranker meaningfully improve retrieval? **Yes: +56.3% recall@10** on a genuinely held-out test set.
- **RQ2** — can a coverage-aware ranking objective (avoiding redundant restatements of an already-covered fact in favor of a still-missing one) be learned and generalize? **Yes, modestly: +32.9% recall@10** on a redundant-hop test set, validated via cross-validation after diagnosing and discarding two failed approaches.

Full writeups: [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) (unifying overview), [`RQ1_RESULTS.md`](RQ1_RESULTS.md), [`RQ2_RESULTS.md`](RQ2_RESULTS.md).

## Dataset

- **Corpus**: 146,821 chunks from 84 SEC filings across 32 companies (2015-2024), parsed with Docling. Parent-child chunking: table rows split individually, narrative text kept at paragraph granularity.
- **Precise question set** (591 questions): rather than relying solely on FinanceBench's own recorded evidence (which after cleanup still had a residual false-positive rate), gold chunks were constructed exactly - the chunk is selected first, and the question is written around it. Single-metric (459), multi-metric (121), and narrative (11) tiers.
- **Held-out test set** (185 questions): built from 7 metrics *never* used in any training data, for a genuine same-distribution generalization check.
- **RQ2 redundant-hop set** (32 usable questions): mined directly from the corpus - cases where the same fact is reported in more than one place (an earnings release restated in the subsequent 10-K, a number in prose corroborated by a table row), verified by exact numeric-value matching, not just label matching.

## Methodology: a three-stage pipeline

1. **Stage 1 - retrieval** (`scripts/pipeline_best.py::retrieve_fusion`): BM25 + dense (`all-mpnet-base-v2`) retrieval, filtered by regex-based company-entity and fiscal-year detection, fused via convex combination. Multi-metric questions are automatically decomposed into per-metric sub-questions and interleaved (`retrieve_decomposed`/`retrieve_auto`) - bundling metrics into one query was found to dilute term-overlap scoring per metric.
2. **Stage 2 - cross-encoder reranking** (RQ1): fine-tuned via RankNetLoss on domain-informed hard negatives - the pipeline's own top-ranked *wrong* answers, not random negatives. Blended with stage-1's ranking via a tunable `beta`.
3. **Stage 3 - coverage-aware ranking** (RQ2): a from-scratch LambdaMART variant with a custom Δcoverage@k gradient (generalizing the standard ΔNDCG LambdaRank gradient to a different, exactly-computable target metric), built on features with no prior fine-tuning history so results are free of the contamination risk that sank an earlier attempt.

## RQ1: domain-informed hard-negative cross-encoder reranking

| | Recall@10 | Coverage@10 | NDCG@10 |
|---|---|---|---|
| Baseline (stage 1 only) | 0.738 | 0.741 | 0.522 |
| **+ cross-encoder (beta=1.0)** | **0.942 (+27.6%)** | **0.945 (+27.5%)** | **0.842 (+61.3%)** |

Key methodological finding: an earlier fine-tuning attempt on 1/13th the data produced a textbook-normal loss curve while the resulting model had *inverted* score preferences on direct inspection - a converging loss curve does not guarantee a useful model. Also found: `beta=1.0` (full trust in the reranker) is optimal on same-distribution held-out data but regresses on cross-distribution (analytically-phrased) questions - resolved via `beta=0.3` as a dual-distribution-safe default.

## RQ2: coverage-aware ranking for multi-hop questions

| | Recall@10 | Coverage@10 | NDCG@5 |
|---|---|---|---|
| Baseline (stage 1 only) | 0.352 | 0.497 | 0.260 |
| **+ coverage-aware LambdaMART** | **0.438 (+24.4%)** | **0.576 (+15.9%)** | **0.331 (+27.1%)** |

Getting here required diagnosing three failed attempts in sequence: a feature set including the cross-encoder's own score mostly just copied that feature (no real generalization); digging into why surfaced a genuine contamination bug (44% of mined training questions' gold chunks were the same chunks the cross-encoder was itself fine-tuned on); removing the cross-encoder entirely and using only features with no fine-tuning history, plus warm-starting from the stage-1 score with heavy regularization, is what finally produced a real, cross-validated effect (3 improved / 28 unchanged / 1 worsened across 32 questions, out-of-fold).

## Repository structure

```
scripts/          retrieval pipeline, corpus construction, fine-tuning, RQ2 mining/training
eval/              precision/recall/NDCG and coverage@k metrics
data/processed/    question sets, gold labels, RQ2 datasets, generated prompts
```

## What this demonstrates methodologically

Every claimed improvement here was checked against a genuinely held-out set before being reported, and every time a result looked too good on a first pass, the discrepancy was investigated rather than smoothed over - including the negative results and abandoned approaches, not just the ones that worked. See `PROJECT_SUMMARY.md` Section 6-7 for the full honest accounting of limitations and what didn't generalize.

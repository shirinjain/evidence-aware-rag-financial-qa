"""
Final prompt construction for answer generation, using each question's
already-known tier (not runtime complexity detection - these are all
pre-built, pre-labeled evaluation questions, so there's nothing to
infer) to pick retrieval depth: k=10 for single/multi-metric
(non-redundant) questions, k=20 for redundant-hop questions - matching
the depth-vs-recall/coverage tradeoff established in RQ1/RQ2 (recall
saturates by k=10 for simple questions but coverage keeps climbing to
k=20 for multi-hop ones). Cross-encoder (beta=1.0) is used as the
scorer in all cases - the beta sweep in RQ2 showed it beats every
blend with LambdaMART, including on the redundant-hop set LambdaMART
was built for, so there's no evidence for switching scorers by tier.

Output: one JSONL file per source set, each line:
{id, question, k_used, tier, context (top-k chunk texts, in rank
order), prompt (the fully formatted string ready for an LLM call)}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_best import retrieve_auto

PROMPT_TEMPLATE = """Answer the question using ONLY the information in the context below, which is extracted from SEC filings (10-K/10-Q/8-K/earnings releases). Table rows list multiple years of data separated by "|" - values are ordered most-recent-fiscal-year-first, per this corpus's consistent reporting convention. If the context does not contain enough information to answer, say so explicitly rather than guessing.

Retrieved evidence:
{context}

Question: {question}

Answer:"""


def _load_id_to_text():
    """Despite the name, returns id -> metadata dict (entity, fiscal_period,
    section_title, raw_text, parent_id) - each context line needs
    entity/year/section to disambiguate which company, fiscal period, and
    statement a row belongs to (a bare "Cost of sales | 3,882 | 4,020 | ..."
    row doesn't say which of its columns is which year without this)."""
    chunks = pd.read_parquet(ROOT / "data" / "processed" / "child_chunks.parquet")
    return {
        row["chunk_id"]: {
            "entity": row["entity"],
            "fiscal_period": row["fiscal_period"],
            "section_title": row["section_title"],
            "raw_text": row["raw_text"],
            "parent_id": row["parent_id"],
            "chunk_type": row["chunk_type"],
        }
        for _, row in chunks.iterrows()
    }


def _format_context(chunk_ids: list[str], id_to_meta: dict) -> str:
    lines = []
    for i, cid in enumerate(chunk_ids, 1):
        meta = id_to_meta.get(cid, {})
        header = f"({meta.get('entity', '?')}, FY{meta.get('fiscal_period', '?')}, {meta.get('section_title', '?')})"
        lines.append(f"[{i}] {header} {meta.get('raw_text', '')}")
    return "\n".join(lines)


def build_prompts_for_set(questions: list[dict], k: int, tier_label: str, id_to_meta: dict) -> list[dict]:
    """k here is the final NUMBER OF CONTEXT CHUNKS wanted - retrieve()'s
    own `k` argument controls per-method candidate breadth BEFORE fusion,
    not the final truncated size (fusing two top-50 lists can return up
    to ~100 unique candidates), so retrieval always runs at the standard
    wide breadth and gets explicitly sliced down to `k` here instead."""
    records = []
    for q in questions:
        ranked = retrieve_auto(q["question"], k=50, beta=1.0)  # auto-decomposes multi-metric questions - see RQ1_RESULTS.md Section 5e
        chunk_ids = ranked[:k]
        prompt = PROMPT_TEMPLATE.format(
            context=_format_context(chunk_ids, id_to_meta),
            question=q["question"],
        )
        records.append({
            "id": q.get("id") or q.get("precise_id") or q.get("val_precise_id") or q.get("financebench_id"),
            "question": q["question"],
            "k_used": k,
            "tier": tier_label,
            "context_chunk_ids": chunk_ids,
            "prompt": prompt,
        })
    return records


def _load_natural_redundant_val() -> list[dict]:
    """The 5 natural FinanceBench val-split questions with real redundant
    hop groups - same construction as build_rq2_val_dataset.py, needed
    here to make the 185-set a true 185 (140 single + 40 multi + 5 natural)."""
    splits = json.load(open(ROOT / "data" / "processed" / "question_splits.json"))
    gold = {r["financebench_id"]: r for r in json.load(open(ROOT / "data" / "processed" / "gold_relevance.json"))}
    natural_qs = {q["financebench_id"]: q for q in [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]}
    out = []
    for qid, r in gold.items():
        if splits.get(qid) != "val":
            continue
        hop_groups = r.get("hop_groups", {})
        if len(hop_groups) >= 2 and any(len(v) >= 2 for v in hop_groups.values()):
            out.append({"id": qid, "question": natural_qs[qid]["question"], "gold_chunk_ids": r["gold_chunk_ids"]})
    return out


def _load_usable_redundant_32_ids() -> set[str]:
    """The exact 32 redundant-hop questions RQ2's results are validated
    on - i.e. the subset where stage-1 retrieval found >=1 gold
    candidate at feature-extraction time (see extract_lambdamart_features.py).
    Read directly from the already-computed feature pickles so this is
    guaranteed identical to the validated test bed, not a re-derivation
    that could drift from it."""
    import pickle
    ids = set()
    for path in ["lambdamart_train_features.pkl", "lambdamart_val_features.pkl"]:
        with open(ROOT / "data" / "processed" / path, "rb") as f:
            groups = pickle.load(f)
        for g in groups:
            if any(len(v) >= 2 for v in g["hop_groups"].values()):
                ids.add(g["id"])
    return ids


def main():
    id_to_meta = _load_id_to_text()

    # 185-set (held-out, mostly non-redundant) - k=10 - 140 single + 40 multi + 5 natural-redundant = 185
    single = json.load(open(ROOT / "data" / "processed" / "val_precise_single_metric_questions.json"))
    multi = json.load(open(ROOT / "data" / "processed" / "val_precise_questions.json"))
    natural_redundant_val = _load_natural_redundant_val()
    val_185 = single + multi + natural_redundant_val
    prompts_185 = build_prompts_for_set(val_185, k=10, tier_label="185_set_k10", id_to_meta=id_to_meta)

    # 591-set (training-source, non-redundant) - k=10
    single_591 = json.load(open(ROOT / "data" / "processed" / "precise_single_metric_questions.json"))
    multi_591 = json.load(open(ROOT / "data" / "processed" / "precise_questions.json"))
    narrative_591 = json.load(open(ROOT / "data" / "processed" / "precise_narrative_questions.json"))
    set_591 = single_591 + multi_591 + narrative_591
    prompts_591 = build_prompts_for_set(set_591, k=10, tier_label="591_set_k10", id_to_meta=id_to_meta)

    # 32-set (pooled redundant-hop, RESTRICTED to the exact validated 32) - k=20
    rq2_train = json.load(open(ROOT / "data" / "processed" / "rq2_dataset.json"))
    rq2_val = json.load(open(ROOT / "data" / "processed" / "rq2_val_dataset.json"))
    usable_32_ids = _load_usable_redundant_32_ids()

    def is_redundant(r):
        return any(len(v) >= 2 for v in r["hop_groups"].values())

    redundant_32 = [r for r in (rq2_train + rq2_val) if is_redundant(r) and r["id"] in usable_32_ids]
    print(f"32-set filter: {len(redundant_32)} of {len(usable_32_ids)} expected usable IDs matched")
    prompts_32 = build_prompts_for_set(redundant_32, k=20, tier_label="redundant_hop_k20", id_to_meta=id_to_meta)

    out_dir = ROOT / "data" / "processed" / "prompts"
    out_dir.mkdir(exist_ok=True)
    for name, records in [("prompts_185.jsonl", prompts_185), ("prompts_591.jsonl", prompts_591), ("prompts_32.jsonl", prompts_32)]:
        with open(out_dir / name, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(records)} prompts to {out_dir / name}")


if __name__ == "__main__":
    main()

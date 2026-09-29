"""
Feature extraction for the coverage-aware LambdaMART stage.

For every RQ2 question (train: rq2_dataset.json, val: rq2_val_dataset.json),
runs stage-1 fusion (retrieve_fusion, the SAME baseline used everywhere
else in this project) to get a fixed top-k candidate pool per query -
NOT the cross-encoder's reranked output, so a candidate the cross-
encoder mis-scored low is still available for LambdaMART to reconsider
(see conversation: candidates must come from the wider baseline pool,
cross-encoder score enters only as a feature).

One row per (query, candidate) with:
  - bm25_norm, dense_norm, cc_score, stage1_rank_norm: stage-1 signals
  - ce_score_norm: cross-encoder score (checkpoint-160), normalized
    per-query the same way rerank() does
  - entity_match, year_match: 0/1 filter-agreement flags
  - is_table_row: 0/1 chunk-type flag
  - lexical_overlap: fraction of question content-words appearing
    verbatim in the candidate's raw_text (cheap proxy for "does this
    row's label match what was asked about")
  - label: 1 if candidate is gold for this query, else 0
  - hop_label: which hop_group name this candidate satisfies, or None
    (needed by the training script's coverage@k gradient computation,
    not used as a model input feature)

Output: data/processed/lambdamart_train_features.pkl,
        data/processed/lambdamart_val_features.pkl
Each is a dict: {"groups": [(query_id, question, hop_groups, group_df), ...]}
where group_df is a pandas DataFrame of that query's candidate rows.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bm25_retrieve import retrieve_with_scores as bm25_scores
from dense_retrieve import retrieve_with_scores as dense_scores
from entity_detection import detect_entity
from year_detection import detect_years
from pipeline_best import _load_reranker
from mine_redundant_hops import _first_value, _normalize_value

K = 50
FEATURE_COLS = [
    "bm25_norm", "dense_norm", "cc_score", "stage1_rank_norm",
    "ce_score_norm", "entity_match", "year_match", "is_table_row", "lexical_overlap",
    "is_value_dup_of_higher_ranked",
]

_STOPWORDS = {"the", "a", "an", "of", "in", "for", "and", "was", "what", "is", "to", "on", "at", "'s"}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _lexical_overlap(question: str, chunk_text: str) -> float:
    q_words = _content_words(question)
    if not q_words:
        return 0.0
    c_words = _content_words(chunk_text)
    return len(q_words & c_words) / len(q_words)


_YEAR_RANGE = range(1990, 2036)


def _extract_values(raw_text: str) -> set[float]:
    """Table rows: first value after the label (matches mine_redundant_hops
    convention). Narrative text (no "|" delimiter): scan the whole text,
    since prose states a value inline ("increased to $X million") rather
    than in a fixed label|value position - returns every plausible match,
    not just one, since which number is "the" value is ambiguous in prose.

    Narrative scanning excludes bare years ("fiscal 2018") and percentages
    ("increased 10%") - found via a real bug: two totally unrelated
    narrative chunks that both happen to mention the same fiscal year
    were getting flagged as "duplicates" purely by that coincidence,
    which measurably hurt non-redundant-tier performance (see
    conversation - this caused a real regression before being caught)."""
    if "|" in raw_text:
        v = _first_value(raw_text)
        norm = _normalize_value(v)
        return {norm} if norm is not None else set()

    out = set()
    for m in re.finditer(r"(?<![\d.])(\$?)\s?-?\(?(\d[\d,]*(?:\.\d+)?)\)?(%?)", raw_text):
        has_dollar, num_str, has_percent = m.groups()
        if has_percent:
            continue
        num_str = num_str.rstrip(",")  # regex can swallow a trailing sentence comma, not a thousands-separator
        norm = _normalize_value(num_str)
        if norm is None:
            continue
        if not has_dollar and "," not in num_str and int(norm) in _YEAR_RANGE and norm == int(norm):
            continue  # bare year mention, not a dollar figure
        out.add(norm)
    return out


def _load_metadata():
    chunks = pd.read_parquet(ROOT / "data" / "processed" / "child_chunks.parquet")
    id_to_entity = dict(zip(chunks["chunk_id"], chunks["entity"]))
    id_to_year = dict(zip(chunks["chunk_id"], chunks["fiscal_period"]))
    id_to_text = dict(zip(chunks["chunk_id"], chunks["raw_text"]))
    id_to_type = dict(zip(chunks["chunk_id"], chunks["chunk_type"]))
    return id_to_entity, id_to_year, id_to_text, id_to_type


def extract_features_for_question(question, gold_ids, hop_groups, id_to_entity, id_to_year, id_to_text, id_to_type, reranker, alpha=0.4):
    bm25_pairs = bm25_scores(question, k=K)
    dense_pairs = dense_scores(question, k=K)

    entity = detect_entity(question)
    years = detect_years(question)

    def _filter(pairs):
        out = pairs
        if entity:
            out = [(c, s) for c, s in out if id_to_entity.get(c) == entity]
        if years:
            out = [(c, s) for c, s in out if id_to_year.get(c) in years]
        return out

    bm25_f = _filter(bm25_pairs)
    dense_f = _filter(dense_pairs)

    def _norm_map(pairs):
        if not pairs:
            return {}
        scores = [s for _, s in pairs]
        lo, hi = min(scores), max(scores)
        rng = hi - lo if hi > lo else 1.0
        return {c: (s - lo) / rng for c, s in pairs}

    bm25_norm = _norm_map(bm25_f)
    dense_norm = _norm_map(dense_f)

    cc_score = {}
    for c in set(bm25_norm) | set(dense_norm):
        cc_score[c] = alpha * bm25_norm.get(c, 0.0) + (1 - alpha) * dense_norm.get(c, 0.0)

    if not cc_score:
        return None

    stage1_ranked = sorted(cc_score, key=cc_score.get, reverse=True)
    n = len(stage1_ranked)
    stage1_rank_norm = {c: (n - i) / n for i, c in enumerate(stage1_ranked)}

    pairs_for_ce = [(question, id_to_text.get(c, "")) for c in stage1_ranked]
    ce_raw = reranker.predict(pairs_for_ce, show_progress_bar=False)
    lo, hi = min(ce_raw), max(ce_raw)
    rng = hi - lo if hi > lo else 1.0
    ce_norm = {c: (s - lo) / rng for c, s in zip(stage1_ranked, ce_raw)}

    chunk_to_hop = {}
    for hop_name, cids in hop_groups.items():
        for cid in cids:
            chunk_to_hop[cid] = hop_name

    # value-duplicate detection: does this candidate share a numeric value
    # with any candidate ranked ABOVE it in stage-1 order? This is the
    # explicit, company-agnostic "someone already reported this fact"
    # signal that per-row score features can't express on their own -
    # see conversation: this is the feature added specifically to try to
    # fix LambdaMART's zero-generalization result on held-out redundant
    # questions (it had memorized train-specific score patterns instead
    # of a transferable redundancy rule, with no feature to express one).
    values_by_chunk = {c: _extract_values(id_to_text.get(c, "")) for c in stage1_ranked}
    seen_values = set()
    is_dup = {}
    for c in stage1_ranked:
        my_values = values_by_chunk[c]
        is_dup[c] = 1.0 if (my_values & seen_values) else 0.0
        seen_values |= my_values

    rows = []
    for c in stage1_ranked:
        rows.append({
            "chunk_id": c,
            "bm25_norm": bm25_norm.get(c, 0.0),
            "dense_norm": dense_norm.get(c, 0.0),
            "is_value_dup_of_higher_ranked": is_dup[c],
            "cc_score": cc_score[c],
            "stage1_rank_norm": stage1_rank_norm[c],
            "ce_score_norm": ce_norm[c],
            "entity_match": 1.0 if id_to_entity.get(c) == entity else 0.0,
            "year_match": 1.0 if id_to_year.get(c) in years else 0.0,
            "is_table_row": 1.0 if id_to_type.get(c) == "table_row" else 0.0,
            "lexical_overlap": _lexical_overlap(question, id_to_text.get(c, "")),
            "label": 1 if c in gold_ids else 0,
            "hop_label": chunk_to_hop.get(c),
        })
    return pd.DataFrame(rows)


def build(dataset_path, out_path):
    id_to_entity, id_to_year, id_to_text, id_to_type = _load_metadata()
    reranker = _load_reranker()
    data = json.load(open(dataset_path))

    groups = []
    for i, q in enumerate(data, 1):
        df = extract_features_for_question(
            q["question"], set(q["gold_chunk_ids"]), q["hop_groups"],
            id_to_entity, id_to_year, id_to_text, id_to_type, reranker,
        )
        if df is None or df["label"].sum() == 0:
            continue  # skip questions where stage-1 found zero gold candidates - no ranking signal to learn from
        groups.append({
            "id": q["id"],
            "question": q["question"],
            "hop_groups": q["hop_groups"],
            "df": df,
        })
        if i % 20 == 0:
            print(f"{i}/{len(data)}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(groups, f)
    print(f"wrote {len(groups)} usable query groups (of {len(data)} total) to {out_path}")


if __name__ == "__main__":
    build(ROOT / "data" / "processed" / "rq2_dataset.json", ROOT / "data" / "processed" / "lambdamart_train_features.pkl")
    build(ROOT / "data" / "processed" / "rq2_val_dataset.json", ROOT / "data" / "processed" / "lambdamart_val_features.pkl")

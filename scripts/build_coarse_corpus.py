"""
Builds a coarser-granularity corpus: each TABLE becomes one retrievable
unit (all its rows concatenated back together), instead of one unit per
row. Narrative chunks are left as-is, since they're already paragraph-
level - comparable to the ~920-token whole-document granularity a
published benchmark (arxiv 2604.01733) used to reach Recall@5=0.816,
versus our current row-level split, which strips a table's surrounding
context from each individual fact and was the likely cause of the
"vocabulary mismatch" failures diagnosed earlier (a ratio question needs
to see the whole statement, not one isolated line item with no context).

Output: data/processed/coarse_chunks.parquet - one row per unit, with a
"member_chunk_ids" column listing which original row-level chunk_ids it
aggregates (needed to map gold labels up to this coarser level).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
OUT_PATH = ROOT / "data" / "processed" / "coarse_chunks.parquet"


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)

    table_rows = chunks[chunks["chunk_type"] == "table_row"]
    narrative = chunks[chunks["chunk_type"] == "narrative"]

    coarse_rows = []
    for parent_id, group in table_rows.groupby("parent_id"):
        combined_text = "\n".join(group.sort_values("chunk_id")["raw_text"])
        first = group.iloc[0]
        coarse_rows.append({
            "chunk_id": parent_id,
            "chunk_type": "table",
            "entity": first["entity"],
            "fiscal_period": first["fiscal_period"],
            "section_title": first["section_title"],
            "subsection_title": first["subsection_title"],
            "raw_text": combined_text,
            "member_chunk_ids": list(group["chunk_id"]),
        })

    for _, row in narrative.iterrows():
        coarse_rows.append({
            "chunk_id": row["chunk_id"],
            "chunk_type": "narrative",
            "entity": row["entity"],
            "fiscal_period": row["fiscal_period"],
            "section_title": row["section_title"],
            "subsection_title": row["subsection_title"],
            "raw_text": row["raw_text"],
            "member_chunk_ids": [row["chunk_id"]],
        })

    df = pd.DataFrame(coarse_rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"{len(df)} coarse units ({(df['chunk_type']=='table').sum()} tables, {(df['chunk_type']=='narrative').sum()} narrative)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

"""
Builds child_chunks.parquet from the Docling-converted markdown of the
84 FinanceBench source documents.

Design decisions, grounded in real evidence spans (see
financebench_merged.jsonl) rather than copying the FinReflectKG
pipeline's choices wholesale:

- Table rows are split individually (chunk_type="table_row") - answers
  are almost always pinned to ONE specific line item (e.g. "Purchases
  of property, plant and equipment"), same reasoning as before, and
  this corpus's tables are proper multi-line markdown (no boundary-
  artifact parsing needed this time).

- Narrative text is kept at PARAGRAPH granularity, not split into
  sentences. Checked directly: FinanceBench's text evidence spans are
  coherent multi-sentence blocks (e.g. a 4-sentence explanation of why
  SG&A changed), unlike FinReflectKG's KG-hop text which was reliably
  one atomic sentence. Splitting these would break the causal
  connection between sentences that makes the paragraph the real
  answer unit.

- section_title tracks the single most recent heading. Docling
  flattened all headings in this export to one level (no h1/h2/h3
  hierarchy to exploit), so subsection_title is left empty rather than
  faking a nesting that isn't actually there.

- entity/fiscal_period come directly from financebench_merged.jsonl's
  company/doc_period fields (joined on doc_name) - no inference needed,
  unlike FinReflectKG where we had to detect entities from chunk text.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_DIR = ROOT / "data" / "interim" / "docling_markdown"
OUT_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"

MIN_PARAGRAPH_LEN = 40  # drop short boilerplate fragments (page numbers, running headers)
JUNK_HEADING_MARKERS = ["table of contents"]


def _load_doc_metadata():
    rows = [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]
    meta = {}
    for r in rows:
        meta[r["doc_name"]] = {"company": r["company"], "fiscal_period": r["doc_period"]}
    return meta


def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-+:?", c.strip()) for c in cells if c.strip())


def _parse_table_block(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    header = rows[0]
    data_rows = [r for r in rows[1:] if not _is_separator_row(r)]
    return header, data_rows


def process_document(doc_name: str, text: str, company: str, fiscal_period: int) -> list[dict]:
    lines = text.split("\n")
    chunks = []
    current_heading = ""
    i = 0
    block_idx = 0
    paragraph_buffer = []

    def flush_paragraph():
        nonlocal block_idx
        if paragraph_buffer:
            para_text = " ".join(paragraph_buffer).strip()
            paragraph_buffer.clear()
            if len(para_text) >= MIN_PARAGRAPH_LEN:
                block_idx += 1
                chunk_id = f"{doc_name}__block{block_idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "parent_id": chunk_id,
                    "chunk_type": "narrative",
                    "entity": company,
                    "fiscal_period": fiscal_period,
                    "section_title": current_heading,
                    "subsection_title": None,
                    "raw_text": para_text,
                })

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        heading_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            heading_text = heading_match.group(1).strip()
            if not any(m in heading_text.lower() for m in JUNK_HEADING_MARKERS):
                current_heading = heading_text
            i += 1
            continue

        if _is_table_line(stripped):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and _is_table_line(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                header, data_rows = _parse_table_block(table_lines)
                block_idx += 1
                parent_id = f"{doc_name}__block{block_idx}"
                header_str = " | ".join(c for c in header if c)
                for row_idx, row in enumerate(data_rows):
                    row_text = " | ".join(c for c in row if c)
                    if not row_text.strip():
                        continue
                    chunks.append({
                        "chunk_id": f"{parent_id}__row{row_idx}",
                        "parent_id": parent_id,
                        "chunk_type": "table_row",
                        "entity": company,
                        "fiscal_period": fiscal_period,
                        "section_title": current_heading,
                        "subsection_title": header_str if header_str else None,
                        "raw_text": row_text,
                    })
            continue

        if not stripped:
            flush_paragraph()
        else:
            paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()
    return chunks


def main():
    doc_meta = _load_doc_metadata()
    all_chunks = []
    for md_path in sorted(MARKDOWN_DIR.glob("*.md")):
        doc_name = md_path.stem
        meta = doc_meta.get(doc_name)
        if meta is None:
            print(f"skipping {doc_name}: no metadata found")
            continue
        text = md_path.read_text()
        chunks = process_document(doc_name, text, meta["company"], meta["fiscal_period"])
        all_chunks.extend(chunks)
        print(f"{doc_name}: {len(chunks)} chunks", flush=True)

    df = pd.DataFrame(all_chunks)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n{len(df)} total chunks written to {OUT_PATH}")
    print(df["chunk_type"].value_counts())


if __name__ == "__main__":
    main()

"""
Computes a per-document page range to feed Docling's page_range
parameter, instead of converting entire 100-250 page 10-Ks - most of
that length is legal boilerplate/exhibits we don't need, and it's the
main reason full-document conversion took 296s vs 96s for a 51-page
range covering the same evidence in local testing.

Range = [min(evidence pages) - BUFFER, max(evidence pages) + BUFFER],
clipped to the document's actual page count. BUFFER=15 keeps real
surrounding non-relevant content (so retrieval stays a genuine search
task, not just "the only page in the corpus"), while still excluding
the bulk of front/back matter a 10-K doesn't need for financial QA.
"""
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
BUFFER = 15


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "raw" / "financebench_merged.jsonl")]
    pages_by_doc = {}
    for r in rows:
        doc = r["doc_name"]
        for e in r["evidence"]:
            pages_by_doc.setdefault(doc, []).append(e["evidence_page_num"])

    ranges = {}
    for doc, pages in pages_by_doc.items():
        pdf_path = ROOT / "data" / "raw" / "pdfs_needed" / f"{doc}.pdf"
        total_pages = len(PdfReader(str(pdf_path)).pages)
        # evidence_page_num is 0-indexed; Docling's page_range is 1-indexed inclusive
        start = max(1, min(pages) + 1 - BUFFER)
        end = min(total_pages, max(pages) + 1 + BUFFER)
        ranges[doc] = [start, end]

    out_path = ROOT / "data" / "interim" / "page_ranges.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(ranges, f, indent=2)

    total_pages_selected = sum(r[1] - r[0] + 1 for r in ranges.values())
    print(f"{len(ranges)} documents, {total_pages_selected} total pages selected (avg {total_pages_selected/len(ranges):.0f}/doc)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""
Maps each of the 150 FinanceBench questions' evidence_text (a full
extracted table/paragraph block) to the specific chunk_ids in our
corpus that fall within it.

Matching strategy: restrict to chunks from the SAME document (doc_name
is recoverable from chunk_id's prefix, since we control that format),
then for each chunk check if its content is actually present in the
evidence_text:
  - table_row: the row's line-item label AND at least one of its
    numeric values must both appear in the evidence_text (checking
    both, not just the label, avoids matching every row that happens
    to share a common label word).
  - narrative: a normalized (whitespace-collapsed) substring check,
    since paragraph chunks are exactly one of Docling's parsed blocks
    and evidence_text is usually a near-verbatim page/paragraph extract.

Tightening: for questions whose "justification" field explicitly names
the exact line item(s) actually used (e.g. "...named: TOTAL ASSETS"),
gold is restricted to just those named rows instead of every row that
happens to appear within the (often much broader, whole-statement)
evidence_text - this is what took the AES ROA example from 312 loosely-
matched rows down to the ~3 rows actually load-bearing for the answer.
Only ~32/150 questions have this explicit phrasing in their
justification; the rest fall back to the broader evidence_text match,
since there's no reliable way to identify the "actually necessary"
subset for those without it.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT.parent / "data" / "processed" / "child_chunks.parquet"
QUESTIONS_PATH = ROOT.parent / "data" / "raw" / "financebench_merged.jsonl"
OUT_PATH = ROOT.parent / "data" / "processed" / "gold_relevance.json"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _doc_name_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("__block")[0]


# labels that are bare years, bare dates, section headers, company/
# address identification rows, or too generic/common a single word to
# specifically identify a line item - found via real failure cases:
# a currency-exchange-rate table row labeled "Rate", a bare
# "2021 | 2020 | 2020" header row, "December 31, 2020" (a bare date),
# "Cash flows from financing activities:" (a section header, not a
# value), and "Corning Incorporated | New York" (a corporate address
# row) all got marked gold just because generic tokens they contain
# trivially appear inside broad evidence_text blocks. Same bug category
# as the original project's header-row-leaking-as-fake-segment issue,
# applied here to gold-matching instead of chunking.
_BARE_YEAR_PATTERN = re.compile(r"^\d{4}$")
_BARE_DATE_PATTERN = re.compile(r"^[a-z]+ \d{1,2},? \d{4}$")  # "December 31, 2020"
_COMPANY_SUFFIX_PATTERN = re.compile(r"\b(incorporated|corporation|company|inc\.?|corp\.?|llc)\b")
_GENERIC_LABELS = {
    "total", "rate", "assets", "amount", "value", "net", "gross", "other", "change",
    "interest", "period", "date", "years", "year", "balance", "days", "income", "expenses",
    "state", "foreign", "domestic", "federal", "current", "long-term", "short-term",
}


def _row_matches(row_text: str, evidence_norm: str, section_title: str = "") -> bool:
    parts = [p.strip() for p in row_text.split("|") if p.strip()]
    if not parts:
        return False
    label = _normalize(parts[0])
    if (
        len(label) < 5
        or label.endswith(":")
        or _BARE_YEAR_PATTERN.match(label)
        or _BARE_DATE_PATTERN.match(label)
        or _COMPANY_SUFFIX_PATTERN.search(label)
        or label in _GENERIC_LABELS
        # a row whose label duplicates its own section heading is a
        # parsed-table caption/title row, not real data - found via
        # "Consolidated Statements of Income | Consolidated Statements
        # of Income" getting marked gold for a Corning question
        or (section_title and label == _normalize(section_title))
    ):
        return False
    if label not in evidence_norm:
        return False
    numbers = re.findall(r"[\d,]{2,}", row_text)
    return any(n in evidence_norm for n in numbers) if numbers else True


_NAMED_LINE_ITEM_PATTERN = re.compile(r"named:\s*([^.\n]+)")


def _extract_named_line_items(justification: str | None) -> list[str]:
    if not justification:
        return []
    return [_normalize(name) for name in _NAMED_LINE_ITEM_PATTERN.findall(justification)]


def _matching_named_item(row_text: str, named_items: list[str]) -> str | None:
    """Returns the specific named item this row matches (for hop-group
    assignment), or None. Same matching rule as before - named item
    must be contained in the row's label, not the reverse direction."""
    label = _normalize(row_text.split("|")[0])
    for name in named_items:
        if len(name) >= 4 and (label == name or name in label):
            return name
    return None


_SINGLE_NUMBER_ANSWER_PATTERN = re.compile(r"^\$?-?\(?([\d,]+\.?\d*)\)?%?\$?$")


def _extract_answer_value(answer: str) -> float | None:
    """Returns the numeric value of `answer` if it's a single clean
    number (e.g. "$1577.00"), or None for narrative/multi-number
    answers where there's no one value to match against."""
    m = _SINGLE_NUMBER_ANSWER_PATTERN.match(answer.strip())
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    return float(num_str) if num_str else None


def _row_contains_value(row_text: str, target: float, tol: float = 0.5) -> bool:
    """Sign-agnostic match (accounting convention shows outflows as
    negative/parenthesized, e.g. "(1,577)", while an answer is usually
    stated as a positive magnitude, "$1577.00" - found via the 3M capex
    example initially failing to match until this was fixed). Skips
    |value| < 1 to avoid coincidentally matching stray small numbers
    (a percentage, a footnote index) that aren't the real answer."""
    numbers = re.findall(r"-?\(?[\d,]+\.?\d*\)?", row_text)
    for n in numbers:
        clean = n.replace(",", "").replace("(", "-").replace(")", "")
        try:
            val = float(clean)
        except ValueError:
            continue
        if abs(val) < 1:
            continue
        if abs(abs(val) - abs(target)) <= tol:
            return True
    return False


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    chunks["doc_name"] = chunks["chunk_id"].apply(_doc_name_from_chunk_id)
    chunks_by_doc = {doc: g for doc, g in chunks.groupby("doc_name")}

    questions = [json.loads(l) for l in open(QUESTIONS_PATH)]

    results = []
    unmatched = 0
    tightened = 0
    value_matched = 0
    for q in questions:
        named_items = _extract_named_line_items(q.get("justification"))
        answer_value = _extract_answer_value(str(q["answer"]))

        gold_ids = set()
        tightened_ids = set()
        value_matched_ids = set()
        hop_groups: dict[str, list[str]] = {}
        for ev in q["evidence"]:
            doc_name = ev["doc_name"]
            evidence_norm = _normalize(ev["evidence_text"])
            doc_chunks = chunks_by_doc.get(doc_name)
            if doc_chunks is None:
                continue
            for _, row in doc_chunks.iterrows():
                if row["chunk_type"] == "table_row":
                    matched = _row_matches(row["raw_text"], evidence_norm, row["section_title"])
                else:
                    matched = _normalize(row["raw_text"])[:200] in evidence_norm or evidence_norm[:200] in _normalize(row["raw_text"])
                if matched:
                    gold_ids.add(row["chunk_id"])
                    if named_items and row["chunk_type"] == "table_row":
                        matched_item = _matching_named_item(row["raw_text"], named_items)
                        if matched_item:
                            tightened_ids.add(row["chunk_id"])
                            hop_groups.setdefault(matched_item, []).append(row["chunk_id"])
                    # highest-confidence tier: row is BOTH inside the
                    # annotator's cited evidence_text AND contains the
                    # exact answer value - requiring both together is
                    # what eliminates coincidental number collisions
                    # elsewhere in the document (verified directly: a
                    # "Net sales" row and a "Total Company" row also
                    # happened to contain 1,577, but neither was inside
                    # the actual evidence_text for the 3M capex question)
                    if answer_value is not None and row["chunk_type"] == "table_row" and _row_contains_value(row["raw_text"], answer_value):
                        value_matched_ids.add(row["chunk_id"])

        # priority: value-matched (highest confidence) > justification-
        # tightened > broad evidence_text fallback
        if value_matched_ids:
            final_ids = value_matched_ids
            value_matched += 1
        elif tightened_ids:
            final_ids = tightened_ids
            tightened += 1
        else:
            final_ids = gold_ids
        if not final_ids:
            unmatched += 1
        results.append({
            "financebench_id": q["financebench_id"],
            "question": q["question"],
            "answer": q["answer"],
            "gold_chunk_ids": sorted(final_ids),
            "gold_tightened": bool(tightened_ids) or bool(value_matched_ids),
            "gold_value_matched": bool(value_matched_ids),
            # each value is the chunk_ids satisfying ONE distinct named
            # line item (a "hop") - only populated for tightened
            # questions, since that's the only case with real hop
            # boundaries rather than one flat gold blob
            "hop_groups": {k: sorted(v) for k, v in hop_groups.items()} if tightened_ids else {},
        })

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    n_gold = [len(r["gold_chunk_ids"]) for r in results]
    print(f"{len(results)} questions processed")
    print(f"unmatched (0 gold chunks found): {unmatched}")
    print(f"value-matched (highest confidence): {value_matched}")
    print(f"tightened via justification named line items: {tightened}")
    print(f"gold chunks per question: min={min(n_gold)} max={max(n_gold)} mean={sum(n_gold)/len(n_gold):.1f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

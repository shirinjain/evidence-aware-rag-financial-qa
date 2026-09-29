"""
Locates each company's Consolidated Statement of Income/Operations page
in its 10-K PDF and extracts clean "revenue" and "expense" line items -
building simple, atomic (company, period, metric, value) facts to
compose into synthetic multi-hop questions (e.g. "revenue and expense
from 2 companies").

Heading detection strips ALL whitespace before matching, since pypdf's
text extraction sometimes splits a word across a line break mid-token
(e.g. 3M's 10-K extracts "Income" as "Incom" + newline + "e" - a plain
substring/regex match on the raw text misses this; matching against a
whitespace-stripped copy is robust to it).

Revenue/expense line extraction works on the ORIGINAL (not stripped)
page text, since we need line boundaries to isolate one row - only
lines with an explicit "total" line item are trusted (e.g. "Total
revenues", "Total costs and expenses"), not summed from sub-line-items,
to avoid the same kind of fragile inference that caused problems in the
original FinReflectKG table-parsing work.
"""

from __future__ import annotations
import re
from pathlib import Path

from pypdf import PdfReader

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "pdfs"

REVENUE_LINE_PATTERNS = [
    r"total revenues?\b",
    r"total net (?:revenue|sales)s?\b",
    r"net sales\b",
    r"net revenues?\b",
]
EXPENSE_LINE_PATTERNS = [
    r"total costs? and expenses?\b",
    r"total operating expenses?\b",
    r"total expenses?\b",
]


def _extract_number(line: str) -> str | None:
    nums = re.findall(r"\(?\$?\s?[\d,]{3,}(?:\.\d+)?\)?", line)
    return nums[0].strip() if nums else None


def find_income_statement_page(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        stripped = re.sub(r"\s+", "", text.lower())
        if "consolidatedstatement" in stripped and ("ofincome" in stripped or "ofoperations" in stripped):
            if "revenue" in stripped or "netsales" in stripped:
                return i, text
    return None, None


def extract_revenue_expense(pdf_path: Path):
    idx, text = find_income_statement_page(pdf_path)
    if text is None:
        return None
    lines = text.split("\n")
    revenue_line = expense_line = None
    for line in lines:
        low = line.lower()
        if revenue_line is None and any(re.search(p, low) for p in REVENUE_LINE_PATTERNS):
            revenue_line = line.strip()
        if expense_line is None and any(re.search(p, low) for p in EXPENSE_LINE_PATTERNS):
            expense_line = line.strip()
    revenue_value = _extract_number(revenue_line) if revenue_line else None
    expense_value = _extract_number(expense_line) if expense_line else None
    return {
        "page": idx,
        "revenue_line": revenue_line,
        "revenue_value": revenue_value,
        "expense_line": expense_line,
        "expense_value": expense_value,
    }


if __name__ == "__main__":
    import sys

    targets = sys.argv[1:] or [f.name for f in PDF_DIR.glob("*10K.pdf")]
    for fname in targets:
        path = PDF_DIR / fname
        if not path.exists():
            print(f"{fname}: NOT FOUND")
            continue
        result = extract_revenue_expense(path)
        if result is None:
            print(f"{fname}: income statement page not found")
        else:
            print(f"{fname}: page={result['page']} revenue={result['revenue_value']!r} expense={result['expense_value']!r}")

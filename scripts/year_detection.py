"""
Detects which fiscal year a question refers to, so retrieval can be
filtered to that year's document on top of entity filtering - targets
the cross-year confusion found directly in this corpus (a 3M FY2018
question pulling a 3M_2023Q2_10Q chunk into dense's top 3).

Corpus spans fiscal years 2015-2024 (verified against child_chunks.parquet).
When a range is mentioned ("FY2019 - FY2021"), returns ALL years in
that range, not just the max, since these corpus documents are single-
year filings (unlike the original project's multi-year comparison
tables within one document) - a 3-year-average question genuinely
needs evidence from multiple distinct fiscal-year documents.
"""
import re

_VALID_YEARS = set(range(2015, 2025))
# no leading \b: "FY2018" has no boundary between "Y" and "2" (both are
# \w characters) so a leading \b would never match embedded years like
# this - found via direct testing showing 0/4 test cases matching
_YEAR_PATTERN = re.compile(r"(20\d{2})\b")


def detect_years(text: str) -> list[int]:
    """Returns all years mentioned in text that fall within this
    corpus's range, sorted. Empty list if none mentioned. When exactly
    2 years are mentioned (e.g. "FY2019 - FY2021"), fills in the years
    between them too - a "3 year average" question needs all 3 years'
    documents, not just the two endpoints."""
    years = {int(y) for y in _YEAR_PATTERN.findall(text)} & _VALID_YEARS
    if len(years) == 2:
        lo, hi = min(years), max(years)
        years = set(range(lo, hi + 1)) & _VALID_YEARS
    return sorted(years)


if __name__ == "__main__":
    tests = [
        "What is the FY2018 capital expenditure amount for 3M?",
        "What is the FY2019 - FY2021 3 year average operating margin for Corning?",
        "What is Coca Cola's FY2022 dividend payout ratio?",
        "Is Boeing's business subject to cyclicality?",
    ]
    for t in tests:
        print(f"{detect_years(t)!r:20} <- {t}")

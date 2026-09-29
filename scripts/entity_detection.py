"""
Detects which company a piece of query text refers to, so retrieval
can be filtered to that company's chunks - targets the exact cross-
company confusion found in this corpus (e.g. a Corning question
retrieving Coca-Cola chunks, a Coca-Cola question retrieving JPMorgan/
3M/MGM Resorts chunks) - generic financial line-item phrasing like
"Dividends paid to shareholders" is nearly identical across companies,
so BM25 has no way to prefer the right one without this.

Unlike the original project, the corpus's own entity values ARE full
company display names already (no ticker mapping needed) - matching
directly against those, plus a couple of common aliases (e.g. "Coke"
for "Coca-Cola") where the formal name might not appear verbatim in a
natural question.
"""

from __future__ import annotations
import re

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# a few common aliases where the formal entity name might not appear
# verbatim in a natural question
ALIASES = {
    "Coca-Cola": ["Coca-Cola", "Coca Cola", "Coke"],
    "Johnson & Johnson": ["Johnson & Johnson", "Johnson and Johnson", "JnJ", "J&J"],
    "AES Corporation": ["AES Corporation", "AES"],
    "CVS Health": ["CVS Health", "CVS"],
}


def _load_entities() -> list[str]:
    chunks = pd.read_parquet(ROOT / "data" / "processed" / "child_chunks.parquet")
    return sorted(chunks["entity"].unique())


_ENTITIES = _load_entities()
_VARIANTS = sorted(
    ((alias, entity) for entity in _ENTITIES for alias in ALIASES.get(entity, [entity])),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(alias) for alias, _ in _VARIANTS) + r")\b",
    re.IGNORECASE,
)
_ALIAS_TO_ENTITY = {alias.lower(): entity for alias, entity in _VARIANTS}


def detect_entity(text: str) -> str | None:
    """Returns a single entity name if exactly one company is
    unambiguously mentioned in text, else None."""
    hits = {_ALIAS_TO_ENTITY[m.group(1).lower()] for m in _PATTERN.finditer(text)}
    return next(iter(hits)) if len(hits) == 1 else None


if __name__ == "__main__":
    tests = [
        "Taking into account the information outlined in the income statement, what is the FY2019 - FY2021 3 year average unadjusted operating income % margin for Corning?",
        "What is Coca Cola's FY2022 dividend payout ratio?",
        "Compare Adobe and Microsoft revenue growth.",
        "What was the total revenue for the period?",
    ]
    for t in tests:
        print(f"{detect_entity(t)!r:20s} <- {t[:80]}")

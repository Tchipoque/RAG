import math
from collections import defaultdict
from typing import Any

from src.tokenizer import STOP_WORDS, tokenize_split

PART_WEIGHT = 0.5


def query_tokens(text: str) -> tuple[list[str], list[str]]:
    """Return (whole_words, parts) for a query; grammar words removed."""

    whole, parts = tokenize_split(text)
    kept = [t for t in whole if t not in STOP_WORDS]
    whole = kept or whole
    parts = [p for p in parts if p not in STOP_WORDS and p not in whole]
    return whole, parts


def bm25_search(
    query: str,
    index: dict[str, dict[int, int]],
    chunk_lenghts: list[int],
    avg_len: float,
    k: int = 5,
    K1: float = 1.5,
    B: float = 0.75
) -> list[tuple[int, Any]]:
    """Rank indexed chunks using the BM25 scoring algorithm."""
    n = len(chunk_lenghts)
    scores: defaultdict[int, float] = defaultdict(float)
    whole, parts = query_tokens(query)
    weighted = [(w, 1.0) for w in set(whole)] + [
        (p, PART_WEIGHT) for p in set(parts)
    ]

    for word, weight in weighted:
        postings = index.get(word)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for chunk_id, tf in postings.items():
            norm = tf + K1 * (
                1 - B + B * chunk_lenghts[int(chunk_id)] / avg_len
            )
            scores[int(chunk_id)] += weight * idf * tf * (K1 + 1) / norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]

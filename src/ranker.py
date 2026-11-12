import math
import re
from collections import defaultdict
from typing import Any


def query_tokens(text: str) -> list[str]:    
    """Tokenize a query and remove grammar words when possible."""
    GRAMMAR = {"what", "which", "who", "how", "why", "when", "where", "does",
           "do", "did", "is", "are", "was", "were", "be", "the", "a", "an",
           "of", "to", "in", "on", "for", "and", "or", "it", "its", "this",
           "that", "with", "by", "as", "at", "from", "can", "some"}
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    kept = [t for t in tokens if t not in GRAMMAR]
    return kept or tokens


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
    scores = defaultdict(float)
    tokens = query_tokens(query)

    for word in set(tokens):
        postings = index.get(word)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for chunk_id, tf in postings.items():
            norm = tf + K1 * (
                1 - B + B * chunk_lenghts[int(chunk_id)] / avg_len
            )
            scores[int(chunk_id)] += idf * tf * (K1 + 1) / norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]

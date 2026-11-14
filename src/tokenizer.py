import tokenize
from io import BytesIO
from pathlib import Path
import re
from typing import Any

INDENT_PART = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+"
)
STOP_WORDS = {
        "what", "which", "who", "how", "why", "when", "where", "does",
        "do", "did", "is", "are", "was", "were", "be", "the", "a", "an",
        "of", "to", "in", "on", "for", "and", "or", "it", "its", "this",
        "that", "with", "by", "as", "at", "from", "can", "some",
    }


def split_identifier(word: str) -> list[str]:
    """'ModelRunner' / 'max_num_seqs' -> ['model','runner'] /
    ['max','num','seqs'].
    Returns [] when the word has only one part."""
    parts = [
        p.lower()
        for piece in word.split("_")
        for p in INDENT_PART.findall(piece)
    ]
    return parts if len(parts) > 1 else []


def tokenize_split(text: str) -> tuple[list[str], list[str]]:
    """Return (whole_words, sub_parts). Case is read BEFORE lowercasing,
    so camelCase can be split."""
    whole: list[str] = []
    parts: list[str] = []
    for w in re.findall(r"[A-Za-z0-9_]+", text):
        whole.append(w.lower())
        parts.extend(split_identifier(w))

    return whole, parts


def tokenize_text(text: str) -> list[str]:
    """Words + identifier parts, minus grammar words (used for docs)."""
    whole, parts = tokenize_split(text)
    tokens = whole + parts
    kept = [x for x in tokens if x not in STOP_WORDS]

    return kept or tokens


def tokenize_py_text(code: str) -> list[str]:
    """Words + identifier parts from Python source (no stopword removal)."""
    whole, parts = tokenize_split(code)
    return whole + parts


def tokenize_py(code: str) -> Any:
    """Return the token stream for Python source."""
    return tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)


def indexer(
    text: str,
    chunk_id: int,
    index_dict: dict[str, dict[int, int]],
    file_path: str,
) -> int:
    """Add a chunk's token frequencies to an inverted index."""
    if Path(file_path).suffix == ".py":
        tokens = tokenize_py_text(text)
        n_tokens = 0
        for word in tokens:
            if not word.strip():
                continue
            n_tokens += 1
            inner = index_dict.setdefault(word, {})
            inner[chunk_id] = inner.get(chunk_id, 0) + 1

        return n_tokens

    tokens = tokenize_text(text)
    n_tokens = 0
    for word in tokens:
        if not word.strip():
            continue
        n_tokens += 1
        inner = index_dict.setdefault(word, {})
        inner[chunk_id] = inner.get(chunk_id, 0) + 1

    return n_tokens

from functools import lru_cache
from typing import Any, cast


@lru_cache(maxsize=256)
def read_file(path: str) -> str:
    """Read and cache a UTF-8 text file."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def get_text(source: dict[str, object]) -> str:
    """Extract the source span described by a chunk metadata mapping."""
    file_path = source["file_path"]
    first = source["first_character_index"]
    last = source["last_character_index"]
    if not (
        isinstance(file_path, str)
        and isinstance(first, int)
        and isinstance(last, int)
    ):
        raise TypeError("Invalid source metadata")
    content = read_file(file_path)
    return content[first:last]


def answer(query: str, sources: list[dict[str, object]], model: Any) -> str:
    """Generate an answer from source chunks using a language model."""
    prompt = """
You are a precise assistant answering questions about the vLLM
codebase. Answer only with what the provided context supports.
If the context does not contain the answer, say so plainly.
Answer in at most four sentences, with no preamble.

Context extracted from the codebase:
"""
    context = ""

    for source in sources[:5]:
        data = f"\n {get_text(source)}"
        context += data

    prompt += context

    prompt += f'\nQuestion: "{query}", Answer using only the context above.'
    prompt += '\n Answer: "'

    encoded_token = model.encode(prompt)[0].tolist()
    result = []

    print(
        f'\n\033[34mQuestion:\033[0m "{query}"\n'
        f'\033[34mAnswer:\033[0m "',
        end="",
    )

    for _ in range(100):
        logits = model.get_logits_from_input_ids(encoded_token)
        next_token = int(logits.index(max(logits)))

        encoded_token.append(next_token)
        result.append(next_token)

        # generated_text = model.decode(encoded_token)
        next_word = model.decode(next_token)
        print(next_word, end="", flush=True)
        if '"' in next_word:
            break

    output = model.decode(result)
    return cast(str, output)

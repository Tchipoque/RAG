import ast
import re
from src.models import MinimalSource


def build_line_offset(content: str) -> list[int]:
    """Return character offsets for the start of each line."""
    offsets = [0]
    for line in content.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def line_to_char(line_offsets: list[int], line_no: int) -> int:
    """Convert a one-based line number to a character offset."""
    return line_offsets[line_no - 1]


def parse_file(content: str) -> ast.Module:
    """Parse Python source code into an abstract syntax tree."""
    return ast.parse(content)


def get_top_level_nodes(tree: ast.Module) -> list[ast.stmt]:
    """Return the nodes directly contained in a syntax tree."""
    return list(ast.iter_child_nodes(tree))

def path_words(file_path: str) -> str:
    """Turn 'data/raw/vllm-0.10.1/vllm/worker/model_runner.py' into
    'vllm worker model_runner.py' (drops the data/raw/<repo>/ prefix)."""
    short = re.sub(r"^.*?data/raw/[^/]+", "", file_path)
    return short.replace("/", "")


def chunk_python(
    file_path: str, content: str, max_chunk_size: int
) -> list[dict[str, object]]:
    """Split Python source into chunks around top-level definitions."""
    line_offsets = build_line_offset(content)
    tree = ast.parse(content)
    chunks: list[MinimalSource] = []
    n_lines = len(content.splitlines())
    pending_start = 0

    for node in ast.iter_child_nodes(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
        ):
            start = line_to_char(line_offsets, node.lineno)

            if start > pending_start:
                chunks.extend(
                    make_sized_chunks(
                        file_path, content, pending_start, start, max_chunk_size,
                    )
                )

            end = (
                line_to_char(line_offsets, node.end_lineno + 1)
                if node.end_lineno < n_lines
                else len(content)
            )
            chunks.extend(
                make_sized_chunks(
                    file_path, content, start, end, max_chunk_size, label=node.name
                )
            )
            pending_start = end

    if pending_start < len(content):
        chunks.extend(
            make_sized_chunks(
                file_path, content, pending_start, len(content), max_chunk_size
            )
        )

    # print(f"\033[36mCreated {len(chunks)} chunks\033[0m")
    return chunks


def chunk_text(
    file_path: str, content: str, max_chunk_size: int
) -> list[dict[str, object]]:
    """Split text content into paragraph-sized chunks."""
    chunks: list[dict[str, object]] = []
    pos = 0
    para_start = 0
    in_fence = False

    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        pos += len(line)

        if not in_fence and not line.strip():
            para = content[para_start: pos].strip()
            if para.startswith("#") and "\n" not in para:
                continue
            chunks.extend(
                make_sized_chunks(
                    file_path, content, para_start, pos, max_chunk_size
                )
            )
            para_start = pos

    if para_start < len(content):
        chunks.extend(
            make_sized_chunks(
                file_path, content, para_start, len(content), max_chunk_size
            )
        )

    # print(f"\033[36mCreated {len(chunks)} chunks\033[0m")
    return chunks

def chunk_markdown(
    file_path: str, content: str, max_chunk_size: int
) -> list[dict[str, object]]:
    """Split Markdown content into paragraph-sized chunks."""
    HEADER = re.compile(r"^#{1,6}\s")
    spans, in_fence, pos, start = [], False, 0, 0

    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence

        if not in_fence and HEADER.match(line) and pos > start:
            spans.append((start, pos))
            start = pos
        pos += len(line)

    if pos > start:
        spans.append((start, pos))

    chunks = []
    for s, e in spans:
        chunks.extend(
            make_sized_chunks(
                file_path, content, s, e, max_chunk_size
            )
        )
    return chunks


def make_sized_chunks(
    file_path: str, content: str, start: int, end: int, max_chunk_size: int, label = ""
) -> list[dict[str, object]]:
    """Split a source span into chunks no larger than the requested size."""

    result = []
    overlap = int(max_chunk_size * 0.2)
    step = max_chunk_size - overlap
    prefix = f"{path_words(file_path)} {label}\n"
    pos = start
    while pos < end:
        piece_end = min(pos + max_chunk_size, end)
        text = content[pos:piece_end]
        if text.strip():
            result.append( {
                "source": MinimalSource(
                    file_path=file_path,
                    first_character_index=pos,
                    last_character_index=piece_end,
                ),
                "text": text,
                "index_text": prefix + text,
            })
        if piece_end == end:
            break
        pos += step

    return result

import ast
from models import MinimalSource

def build_line_offset(content: str) -> list[int]:
	offsets = [0]
	for line in content.splitlines(keepends=True):
		offsets.append(offsets[-1] + len(line))
	return offsets

def line_to_char(line_offsets: list[int], line_no: int) -> int:
	return line_offsets[line_no - 1]

def parse_file(content: str) -> ast.Module:
	return ast.parse(content)

def get_top_level_nodes(tree: ast.Module) -> list[ast.stmt]:
	return list(ast.iter_child_nodes(tree))

def chunk_python(file_path: str, content: str, max_chunk_size: int) -> list[dict[MinimalSource, str]]:
	line_offsets = build_line_offset(content)
	tree = ast.parse(content)
	chunks: list[MinimalSource] = []

	pending_start = 0

	for node in ast.iter_child_nodes(tree):
		if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
			start = line_to_char(line_offsets, node.lineno)

			if start > pending_start:
				chunks.extend(make_sized_chunks(file_path, content, pending_start, start, max_chunk_size))

			end = line_to_char(line_offsets, node.end_lineno + 1) if node.end_lineno < len(content.splitlines()) else len(content)
			chunks.extend(make_sized_chunks(file_path, content, start, end, max_chunk_size))
			pending_start = end

	if pending_start < len(content):
		chunks.extend(make_sized_chunks(file_path, content, pending_start, len(content), max_chunk_size))

	return chunks

def make_sized_chunks(file_path, content, start, end, max_chunk_size):
	span = end - start
	if span <= max_chunk_size:
		text  = content[start : end]
		if text.strip():
			data = {
					"source": MinimalSource(file_path=file_path, first_character_index=start, last_character_index=end),
					"text": text,
				}
			return [data]
		return []

	result = []
	pos = start
	while pos < end:
		piece_end = min(pos + max_chunk_size, end)
		text = content[pos : piece_end]
		if text.strip():
			data = {
				"source": MinimalSource(file_path=file_path, first_character_index=pos, last_character_index=piece_end),
				"text": text,
			}
			result.append(data)
			pos = piece_end

	return result




code = """
import os

def hello(name):
    print(name)

class Person:
    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(self.name)
"""

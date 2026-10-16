import tokenize
from io import BytesIO
from pathlib import Path
import re

def tokenize_text(text: str):
	return re.findall(r"[a-z0-9_]+", text.lower())

def tokenize_py(code: str):
	return tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)

def indexer(text, chunk_id, index_dict, file_path) -> int:
	py_f = False

	if Path(file_path).suffix == ".py":
		tokens = tokenize_py(text)
		py_f = True
	else:
		tokens = tokenize_text(text)

	n_tokens = 0
	for token in tokens:
		n_tokens += 1
		if py_f:
			word = token.string
		else:
			word = token
		if not word.strip():
			continue
		if py_f and token.type == tokenize.ENCODING:
			continue
		inner = index_dict.setdefault(word, {})
		inner[chunk_id] = inner.get(chunk_id, 0) + 1

	return n_tokens





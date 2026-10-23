import tokenize
from io import BytesIO
from pathlib import Path
import re
from tokenize import TokenError

def tokenize_text(text: str):
	return re.findall(r"[a-z0-9_]+", text.lower())

def tokenize_py(code: str):
	return tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)

def indexer(text, chunk_id, index_dict, file_path) -> int:
	if Path(file_path).suffix == ".py":
		n_tokens = 0
		try:
			for token in tokenize_py(text):
				n_tokens += 1
				word = token.string
				if token.type == tokenize.ENCODING or not word.strip():
					continue
				inner = index_dict.setdefault(word, {})
				inner[chunk_id] = inner.get(chunk_id, 0) + 1
			return n_tokens
		except (TokenError, IndentationError):
			tokens = tokenize_text(text)
			n_tokens = 0
			for word in tokens:
				n_tokens += 1
				if not word.strip():
					continue
				inner = index_dict.setdefault(word, {})
				inner[chunk_id] = inner.get(chunk_id, 0) + 1
			return n_tokens

	tokens = tokenize_text(text)
	n_tokens = 0
	for word in tokens:
		n_tokens += 1
		if not word.strip():
			continue
		inner = index_dict.setdefault(word, {})
		inner[chunk_id] = inner.get(chunk_id, 0) + 1

	return n_tokens





import tokenize
from io import BytesIO
from pathlib import Path
import re
from tokenize import TokenError

def tokenize_text(text: str):
	return re.findall(r"[a-z0-9_]+", text.lower())

def tokes(text):
	tokens = []
	for word in re.findall(r"[A-Za-z0-9_]+", text):
		tokens.append(word.lower())
		spaced = re.sub(r"([a-z0-9])(A-Z)", r"\1 \2", word)
		spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
		parts = [p.lower() for p in re.split(r"[_ ]", spaced) if p]
		if len(parts) > 1:
			tokens.extend(parts)
	return tokens

def tokenize_py(code: str):
	return tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)

def indexer(text, chunk_id, index_dict, file_path) -> int:
	# return indexer_test(text, chunk_id, index_dict, file_path)
	if Path(file_path).suffix == ".py":
		n_tokens = 0
		try:
			for token in tokenize_py(text):
				word = token.string
				if token.type == tokenize.ENCODING or not word.strip():
					continue
				n_tokens += 1
				inner = index_dict.setdefault(word, {})
				inner[chunk_id] = inner.get(chunk_id, 0) + 1
			return n_tokens
		except (TokenError, IndentationError):
			tokens = tokenize_text(text)
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

from collections import Counter


def indexer(text, chunk_id, index_dict, file_path) -> int:
    short_path = file_path.split("vllm-0.10.1/", 1)[-1]
    tokens = tokenize_text(text) + tokenize_text(short_path)
    for word, tf in Counter(tokens).items():
        index_dict.setdefault(word, {})[chunk_id] = tf
    return len(tokens)

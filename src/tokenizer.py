import tokenize
from io import BytesIO
from pathlib import Path
import re
from tokenize import TokenError

def tokenize_text(text: str):
	stop_words = {
    "what", "which", "who", "how", "why", "when", "where", "does", "do",
    "did", "is", "are", "was", "were", "be", "the", "a", "an", "of", "to",
    "in", "on", "for", "and", "or", "it", "its", "this", "that", "with",
    "by", "as", "at", "from", "can", "some", "used", "use", "using",
    "default", "value", "values", "parameter", "parameters", "method",
    "function", "class", "returns", "return",
	}
	tokens = re.findall(r"[a-z0-9_]+", text.lower())
	kept = [t for t in tokens if t not in stop_words]
	return kept or tokens


def tokenize_py_text(code: str):
	return re.findall(r"[a-z0-9_]+", code.lower())

def tokenize_py(code: str):
	return tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)

def indexer(text, chunk_id, index_dict, file_path) -> int:
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

	# 	try:
	# 		for token in tokenize_py(text):
	# 			word = token.string
	# 			if token.type == tokenize.ENCODING or not word.strip():
	# 				continue
	# 			n_tokens += 1
	# 			inner = index_dict.setdefault(word, {})
	# 			inner[chunk_id] = inner.get(chunk_id, 0) + 1
	# 		return n_tokens
	# 	except (TokenError, IndentationError):
	# 		tokens = tokenize_py_text(text)
	# 		n_tokens = 0
	# 		for word in tokens:
	# 			if not word.strip():
	# 				continue
	# 			n_tokens += 1
	# 			inner = index_dict.setdefault(word, {})
	# 			inner[chunk_id] = inner.get(chunk_id, 0) + 1
	# 		return n_tokens

	tokens = tokenize_text(text)
	n_tokens = 0
	for word in tokens:
		if not word.strip():
			continue
		n_tokens += 1
		inner = index_dict.setdefault(word, {})
		inner[chunk_id] = inner.get(chunk_id, 0) + 1

	return n_tokens

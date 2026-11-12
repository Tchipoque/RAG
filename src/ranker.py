import math
from collections import defaultdict
from src.tokenizer import tokenize_text
import re

GRAMMAR = {"what", "which", "who", "how", "why", "when", "where", "does",
           "do", "did", "is", "are", "was", "were", "be", "the", "a", "an",
           "of", "to", "in", "on", "for", "and", "or", "it", "its", "this",
           "that", "with", "by", "as", "at", "from", "can", "some"}
TEMPLATE = {"default", "value", "values", "parameter", "parameters",
            "method", "function", "class", "returns", "return", "used",
            "use", "using"}

def query_tokens(text: str, drop=GRAMMAR):
	tokens = re.findall(r"[a-z0-9_]+", text.lower())
	kept = [t for t in tokens if t not in drop]
	return kept or tokens

def bm25_search(query, index, chunk_lenghts, avg_len, k=5, K1=1.5, B=0.75, drop=GRAMMAR):
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
			norm = tf + K1 * (1 - B + B * chunk_lenghts[int(chunk_id)] / avg_len)
			scores[int(chunk_id)] += idf * tf * (K1 + 1) / norm

	ranked = sorted(scores.items(), key= lambda x:x[1], reverse=True)
	return ranked[:k]

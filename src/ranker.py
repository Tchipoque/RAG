import math
from collections import defaultdict
from src.tokenizer import tokenize_text


def bm25_search(query, index, chunk_lenghts, avg_len, k=5):
	K1, B = 1.5, 0.75
	n = len(chunk_lenghts)
	scores = defaultdict(float)
	tokens = tokenize_text(query)

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

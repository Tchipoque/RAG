from functools import lru_cache

@lru_cache(maxsize=256)
def read_file(path):
	with open(path, encoding="utf-8", newline="") as f:
		return f.read()

def get_text(source: dict):
	content = read_file(source["file_path"])
	return content[source["first_character_index"]: source["last_character_index"]]


def answer(query, sources, model):
	prompt ="""You are a code assistant that answers questions about a codebase.

Rules:
1. Answer ONLY with facts that are explicitly written in the Context below.
2. NEVER use your own knowledge, even if you know the answer.
3. If the Context does not explicitly contain the answer, reply with exactly this sentence and nothing else:
The provided context does not contain the answer.
4. If you can answer, use at most 4 sentences and do not repeat yourself.

Example:
Context: vLLM uses PagedAttention to manage the KV cache.
Question: What is the capital of France?
Answer: The provided context does not contain the answer.

Context:
"""
	context = ""

	for source in sources[:5]:
		data = f"\n {get_text(source)}"
		context += data

	prompt += context

	prompt += f'\nQuery: "{query}"'
	prompt += '\n Answer: "'

	encoded_token = model.encode(prompt)[0].tolist()
	result = []

	print(f'\n\033[32mQuestion:\033[0m "{query}"\n\033[32mAnswer:\033[0m "', end="")

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
	return output


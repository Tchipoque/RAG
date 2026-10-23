from functools import lru_cache

@lru_cache(maxsize=256)
def read_file(path):
	with open(path, encoding="utf-8", newline="") as f:
		return f.read()

def get_text(source: dict):
	content = read_file(source["file_path"])
	return content[source["first_character_index"]: source["last_character_index"]]


def answer(query, sources, model):
	prompt ="""
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

	print(f'\n\033[34mQuestion:\033[0m "{query}"\n\033[34mAnswer:\033[0m "', end="")

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


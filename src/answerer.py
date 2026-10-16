def answer(query, ranks, chunks):
	from llm_sdk.llm_sdk import Small_LLM_Model

	model = Small_LLM_Model()
	print("\033[1;32m\nANSWERING...\033[0m")


	prompt = """
You are a code assistant that answers questions about a codebase.

Instructions:
    - Use only the context provided by the user.\n
    - If the context does not contain the answer, say:
    \"The provided context does not contain the answer.\"\n
    - Answer in 2 to 4 sentences. Do not repeat yourself.\n

Context:"""
	context = ""

	for rank in ranks:
		data = f"""\n {chunks[rank[0]]["text"]}"""
		context += data

	prompt += context

	prompt += f'\nQuery: "{query}"'
	prompt += '\n Answer: "'

	encoded_token = model.encode(prompt)[0].tolist()

	print(prompt, end="")
	for _ in range(50):
		logits = model.get_logits_from_input_ids(encoded_token)
		next_token = int(logits.index(max(logits)))

		encoded_token.append(next_token)

		# generated_text = model.decode(encoded_token)
		next_word = model.decode(next_token)
		print(next_word, end="", flush=True)
		if '"' in next_word:
			break

	output = model.decode(encoded_token)
	print("\n\n", output)


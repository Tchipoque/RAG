from src.chunker import chunk_python, chunk_markdown
from src.tokenizer import indexer
import json
from src.ranker import bm25_search
from pathlib import Path
import fire
from pydantic import ValidationError


def index(max_chunk_size: int = 2000) -> None:
	chunks = []
	index_dict = {}
	chunk_lengths = []

	raw_dir = Path("data/test")

	for path in raw_dir.rglob("*"):
		if path.is_file():
			if path.suffix == ".py":
				with open(path, "r") as file:
					content = file.read()
				chunks.extend(chunk_python(str(path), content, max_chunk_size))
			elif path.suffix in [".md", ".txt"]:
				with open(path, "r") as file:
					content = file.read()
				chunks.extend(chunk_markdown(str(path), content, max_chunk_size))

	for id, chunk in enumerate(chunks):
		n_tokens = indexer(chunk["text"], id, index_dict, chunk["source"].file_path)
		chunk_lengths.append(n_tokens)

	avg_len = sum(chunk_lengths) / len(chunk_lengths)

	data = {
		"chunks": [{"source": c["source"].model_dump(), "text": c["text"]} for c in chunks],
		"index": index_dict,
		"chunk_lengths": chunk_lengths,
		"avg_len": avg_len,
	}

	with open("data/processed/index.json", "w") as f:
		json.dump(data, f)

	print("\033[1;32mINDEXING...\033[0m")


def search(query, k = 5) -> None:
	from src.answerer import answer

	with open("data/processed/index.json", "r") as f:
		data = json.load(f)

	print("\033[1;32mSEARCHING...\033[0m")

	ranks = bm25_search(query, data["index"], data["chunk_lengths"], data["avg_len"], k)

	for n, rank in enumerate(ranks):
		output = f"""
{n + 1}.	{data["chunks"][rank[0]]["source"]["file_path"]} - score: {rank[1]:.4f}
		{data["chunks"][rank[0]]["text"]}
"""
		print(output)

	answer(query, ranks, data["chunks"])



if __name__ == "__main__":
	fire.Fire({"index": index, "search": search})

	# try:
	# 	fire.Fire({"index": index, "search": search})
	# except ValidationError as e:
	# 	last_error = e.errors()[-1]
	# 	field_error =  " -> ".join(str(loc) for loc in last_error["loc"])
	# 	print(f"\n❌ [Pydantic Error] Field '{field_error}': {last_error['msg']} (Received value: {last_error['input']})")
	# except Exception as e:
	# 	print("🐍 Standard Python Error Caught!")
	# 	print(f"Error Type: {type(e).__name__}")
	# 	print(f"Msg: {e}")

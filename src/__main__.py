from src.chunker import chunk_python, chunk_markdown
from src.tokenizer import indexer
import json
from src.ranker import bm25_search
from pathlib import Path
import fire
from pydantic import ValidationError
from src.answerer import answer as ans
from src.models import *
# from src.models import RagDataset, MinimalSearchResults, StudentSearchResults, MinimalSource


def index(max_chunk_size: int = 2000) -> None:
	print(f"\033[1;33mINDEXING... \033[0m")
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
	file_path = Path("data/processed/index.json")
	file_path.parent.mkdir(parents=True, exist_ok=True)

	with open(file_path, "w") as f:
		json.dump(data, f, indent=4, ensure_ascii=False)

	print(f"\033[1;32mIngestion complete. Indexed {len(chunks)} chunks under data/processed \033[0m")


def search(query, k = 5) -> None:
	print("\033[1;36m\nSEARCHING...\n\033[0m")

	with open("data/processed/index.json", "r") as f:
		data = json.load(f)

	ranks = bm25_search(query, data["index"], data["chunk_lengths"], data["avg_len"], k)

	for cid, _ in ranks:
		source = data["chunks"][cid]["source"]
		output = f"{source["file_path"]} [{source["first_character_index"]} : {source["last_character_index"]}]"
		print(output)

def answer(query: str, k: int = 5):
	print("\033[1;35m\nANSWERING...\n\033[0m")

	from llm_sdk.llm_sdk import Small_LLM_Model

	model = Small_LLM_Model()

	with open("data/processed/index.json", "r") as f:
		data = json.load(f)

	ranks = bm25_search(query, data["index"], data["chunk_lengths"], data["avg_len"], k)
	sources = [data["chunks"][cid]["source"] for cid, _ in ranks]

	ans(query, sources, model)

def search_dataset(data_set, k, save_dir):
	with open("data/processed/index.json", "r") as f:
		index_data = json.load(f)

	with open(data_set, "r") as f:
		data = json.load(f)

	print("\033[1;33m\nSEARCHING...\n\033[0m")

	search_results = []
	dataset = RagDataset(rag_questions=data["rag_questions"])
	for q in dataset.rag_questions:
		ranks = bm25_search(q.question, index_data["index"], index_data["chunk_lengths"], index_data["avg_len"], k )
		sources = [index_data["chunks"][cid]["source"] for cid, _ in ranks]
		search_results.append(MinimalSearchResults(question_id=q.question_id, question=q.question, retrieved_sources=sources ))

	out = StudentSearchResults(search_results=search_results, k=k)
	Path(save_dir).mkdir(parents=True, exist_ok=True)
	out_path = Path(save_dir) / Path(data_set).name

	with out_path.open("w") as f:
		json.dump(out.model_dump(), f, indent=4, ensure_ascii=False)
	print(f"\033[1;32mSaved student_search_results to {out_path}\033[0m")

def answer_dataset(search_results_path: str, save_dir: str):
	print("\033[1;33m\nANSWERING...\n\033[0m")

	from llm_sdk.llm_sdk import Small_LLM_Model
	model = Small_LLM_Model()

	with open(search_results_path, "r") as f:
			results = json.load(f)

	search_results = []
	for r in results["search_results"]:
		answer = ans(r["question"], r["retrieved_sources"], model)
		search_results.append(MinimalAnswer(question_id=r["question_id"], question=r["question"], retrieved_sources= r["retrieved_sources"], answer=answer ))
		print(f"\033[1;35m\n{"_"*10}\033[0m")

	out = StudentSearchResultsAndAnswer(search_results=search_results , k=results["k"])
	Path(save_dir).mkdir(parents=True, exist_ok=True)
	out_path = Path(save_dir) / Path(search_results_path).name

	with out_path.open("w") as f:
		json.dump(out.model_dump(), f, indent=4, ensure_ascii=False)
	print(f"\033[1;32mSaved student_search_results_and_answers to {out_path}\033[0m")



if __name__ == "__main__":
	fire.Fire({
		"index": index,
		"search": search,
		"answer": answer,
		"search_dataset": search_dataset,
		"answer_dataset": answer_dataset,
		})

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

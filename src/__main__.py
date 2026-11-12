import json
import os
import time
from pathlib import Path

import fire
from src.answerer import answer as ans
from src.chunker import chunk_markdown, chunk_python
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.ranker import bm25_search
from src.tokenizer import indexer
from pydantic import ValidationError
from tqdm import tqdm


def index(max_chunk_size: int = 2000) -> None:
    """Index raw source files into the processed search index."""
    os.system("clear")
    chunks = []
    index_dict = {}
    chunk_lengths = []

    raw_dir = Path("data/raw")
    files = [f for f in raw_dir.rglob("*") if f.is_file() and f.suffix in [".py", ".md", ".txt"]]
    
    for path in tqdm(files, desc="Reading Files", unit="file"):
        with open(path, "r") as file:
            content = file.read()

        if path.suffix == ".py":
            chunks.extend(chunk_python(str(path), content, max_chunk_size))
        elif path.suffix in [".md", ".txt"]:
            chunks.extend(chunk_markdown(str(path), content, max_chunk_size))

    for id, chunk in enumerate(chunks):
        n_tokens = indexer(
            chunk["text"], id, index_dict, chunk["source"].file_path
        )
        chunk_lengths.append(n_tokens)

    avg_len = sum(chunk_lengths) / len(chunk_lengths)

    data = {
        "chunks": [
            {"source": c["source"].model_dump(), "text": c["text"]}
            for c in chunks
        ],
        "index": index_dict,
        "chunk_lengths": chunk_lengths,
        "avg_len": avg_len,
    }
    file_path = Path("data/processed/index.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(
        "\033[1;32m\nIngestion complete. Indexed "
        f"{len(chunks)} chunks under data/processed/index.json \033[0m"
    )


def search(query: str, k: int = 5) -> None:
    """Search the processed index and print matching source spans."""
    os.system("clear")
    print("\033[1;36m\nSEARCHING...\n\033[0m")

    with open("data/processed/index.json", "r") as f:
        data = json.load(f)

    ranks = bm25_search(
        query, data["index"], data["chunk_lengths"], data["avg_len"], k
    )

    for cid, _ in ranks:
        source = data["chunks"][cid]["source"]
        output = (
            f"{source['file_path']} [{source['first_character_index']} : "
            f"{source['last_character_index']}]"
        )
        print(output)


def answer(query: str, k: int = 5) -> None:
    """Answer a query using indexed sources and a language model."""
    from llm_sdk.llm_sdk import Small_LLM_Model
    model = Small_LLM_Model()

    os.system("clear")
    print("\033[1;35m\nANSWERING...\033[0m")

    with open("data/processed/index.json", "r") as f:
        data = json.load(f)

    ranks = bm25_search(
        query, data["index"], data["chunk_lengths"], data["avg_len"], k
    )
    sources = [data["chunks"][cid]["source"] for cid, _ in ranks]

    print(f'\033[1;36m{"_" * 40}\033[0m')
    ans(query, sources, model)


def search_dataset(
    dataset_path: str, k: int = 10, save_directory: str = "data/output/search_results/"
) -> None:
    """Search each question in a dataset and save retrieved sources."""
    os.system("clear")
    with open("data/processed/index.json", "r") as f:
        index_data = json.load(f)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    search_results = []
    dataset = RagDataset(rag_questions=data["rag_questions"])

    for q in tqdm(dataset.rag_questions, desc="Searching...", unit="questions"):
        ranks = bm25_search(
            q.question,
            index_data["index"],
            index_data["chunk_lengths"],
            index_data["avg_len"],
            k,
        )
        sources = [index_data["chunks"][cid]["source"] for cid, _ in ranks]
        search_results.append(
            MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources,
            )
        )

    out = StudentSearchResults(search_results=search_results, k=k)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_directory) / Path(dataset_path).name

    with out_path.open("w") as f:
        json.dump(out.model_dump(), f, indent=4, ensure_ascii=False)
    print(f"\033[1;32m\nSaved student_search_results to {out_path}\033[0m")


def answer_dataset(
    student_search_results_path: str,
    save_directory: str = "data/output/search_results_and_answer/",
) -> None:
    """Answer each question in saved search results and write the output."""
    # print("\033[1;33m\nANSWERING...\n\033[0m")

    from llm_sdk.llm_sdk import Small_LLM_Model
    model = Small_LLM_Model()
    os.system("clear")

    with open(student_search_results_path, "r") as f:
        results = json.load(f)

    search_results = []
    n_questions = 0
    for r in tqdm(results["search_results"], desc="Answering...", unit="answers"):
        n_questions += 1
        answer = ans(r["question"], r["retrieved_sources"], model)
        search_results.append(
            MinimalAnswer(
                question_id=r["question_id"],
                question=r["question"],
                retrieved_sources=r["retrieved_sources"],
                answer=answer,
            )
        )
        print(f'\033[1;35m\n{"_" * 40}\033[0m')

    out = StudentSearchResultsAndAnswer(
        search_results=search_results, k=results["k"]
    )
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_directory) / Path(student_search_results_path).name

    with out_path.open("w") as f:
        json.dump(out.model_dump(), f, indent=4, ensure_ascii=False)
    print(f"\033[36m Loaded {n_questions} questions ...\033[0m")
    print(
        "\033[1;32mSaved student_search_results_and_answers "
        f"to {out_path}\033[0m"
    )


if __name__ == "__main__":
    try:
        fire.Fire(
            {
                "index": index,
                "search": search,
                "answer": answer,
                "search_dataset": search_dataset,
                "answer_dataset": answer_dataset,
            }
        )
    except ValidationError as e:
        last_error = e.errors()[-1]
        field_error = " -> ".join(str(loc) for loc in last_error["loc"])
        print(
            f"\n❌ [Pydantic Error] Field '{field_error}': "
            f"{last_error['msg']} (Received value: {last_error['input']})"
        )
    except Exception as e:
        print("🐍 Standard Python Error Caught!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Msg: {e}")

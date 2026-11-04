import json
from src.ranker import bm25_search

idx = json.load(open("data/processed/index.json"))
ds = json.load(open("data/datasets/AnsweredQuestions/dataset_code_public.json"))
indexed_files = {c["source"]["file_path"] for c in idx["chunks"]}


def overlaps(a, b):
    if a["file_path"] != b["file_path"]:
        return False
    inter = min(a["last_character_index"], b["last_character_index"]) \
        - max(a["first_character_index"], b["first_character_index"])
    if inter <= 0:
        return False
    union = max(a["last_character_index"], b["last_character_index"]) \
        - min(a["first_character_index"], b["first_character_index"])
    return inter / union >= 0.05


stats = {"hit@5": 0, "hit@6-50": 0, "right file, wrong span": 0,
         "file not in top 50": 0, "file never indexed": 0}
examples = []

for q in ds["rag_questions"]:
    ranks = bm25_search(q["question"], idx["index"], idx["chunk_lengths"],
                        idx["avg_len"], 50)
    got = [idx["chunks"][cid]["source"] for cid, _ in ranks]
    for gold in q["sources"]:
        pos = next((i for i, s in enumerate(got) if overlaps(s, gold)), None)
        if pos is not None:
            stats["hit@5" if pos < 5 else "hit@6-50"] += 1
        elif gold["file_path"] not in indexed_files:
            stats["file never indexed"] += 1
            examples.append((q["question"], gold["file_path"], "NOT INDEXED"))
        elif any(s["file_path"] == gold["file_path"] for s in got):
            stats["right file, wrong span"] += 1
        else:
            stats["file not in top 50"] += 1
            examples.append((q["question"], gold["file_path"], got[0]["file_path"]))

print(stats)
for e in examples[:10]:
    print('\n', e)


def recall_at_k(idx, dataset, k=5, **params):
    total = 0.0
    questions = dataset["rag_questions"]
    for q in questions:
        ranks = bm25_search(q["question"], idx["index"], idx["chunk_lengths"],
                            idx["avg_len"], k, **params)
        got = [idx["chunks"][cid]["source"] for cid, _ in ranks]
        gold = q["sources"]
        found = sum(any(overlaps(g, s) for s in got) for g in gold)
        total += found / len(gold)
    return total / len(questions)

# idx = json.load(open("data/processed/index.json"))
# docs = json.load(open("data/datasets/AnsweredQuestions/dataset_docs_public.json"))
# code = json.load(open("data/datasets/AnsweredQuestions/dataset_code_public.json"))

# for k1 in (0.9, 1.2, 1.5, 2.0):
#     for b in (0.4, 0.6, 0.75, 0.9):
#         d = recall_at_k(idx, docs, 5, K1=k1, B=b)
#         c = recall_at_k(idx, code, 5, K1=k1, B=b)
#         print(f"k1={k1} b={b}  docs={d:.3f}  code={c:.3f}")
"""Small web server: a chat page + one endpoint that runs the RAG."""
import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.answerer import answer, get_text
from src.ranker import bm25_search

state: dict = {}          # holds the index and the model, loaded once
model_lock = threading.Lock()  # the model can only answer one question at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load the index and the model."""
    with open("data/processed/index.json") as f:
        state["data"] = json.load(f)
    from llm_sdk.llm_sdk import Small_LLM_Model
    state["model"] = Small_LLM_Model()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    question: str
    k: int = 5


@app.get("/")
def home() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    data = state["data"]
    ranks = bm25_search(
        question, data["index"], data["chunk_lengths"], data["avg_len"], req.k
    )
    sources = [data["chunks"][cid]["source"] for cid, _ in ranks]
    if not sources:
        return {"answer": "I could not find anything relevant.", "sources": []}

    with model_lock:
        text = answer(question, sources, state["model"])

    return {
        "answer": text.strip().strip('"'),
        "sources": [
            {
                "file_path": s["file_path"].split("data/raw/")[-1],
                "start": s["first_character_index"],
                "end": s["last_character_index"],
                "snippet": get_text(s)[:400],
            }
            for s in sources
        ],
    }

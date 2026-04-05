"""
FastAPI application entry point.
Loads all models at startup — not per request.
"""

import os
import pickle
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from src.hf_insecure_ssl import apply_if_configured

apply_if_configured()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.retrieval.biencoder import BiEncoder
from src.retrieval.colbert_reranker import ColBERTReranker
from src.personalization.user_model import UserInterestModel
from src.database.user_store import init_db
from api.state import get_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_state()
    print("Initializing models and indexes...")

    index_path  = os.getenv("INDEX_PATH",  "indexes/faiss_index.bin")
    bm25_path   = os.getenv("BM25_PATH",   "indexes/bm25_index.pkl")
    corpus_path = os.getenv("CORPUS_PATH", "indexes/corpus.pkl")
    db_path     = os.getenv("DB_PATH", "data/userdata.db")

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    biencoder = BiEncoder(model_name=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
    biencoder.load_index(index_path, corpus_path)

    s["biencoder"] = biencoder
    s["docs"] = biencoder.docs
    s["embeddings"] = biencoder.embeddings

    with open(bm25_path, "rb") as f:
        s["bm25"] = pickle.load(f)

    s["colbert"] = ColBERTReranker(
        model_name=os.getenv("COLBERT_MODEL", "bert-base-uncased")
    )

    s["user_model"] = UserInterestModel(
        decay=float(os.getenv("EMA_DECAY", 0.85))
    )

    await init_db(db_path)

    print(f"Ready. Corpus: {len(s['docs'])} docs")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Personalized Dense IR System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import search, feedback, analytics

app.include_router(search.router,    prefix="/api")
app.include_router(feedback.router,  prefix="/api")
app.include_router(analytics.router, prefix="/api")


@app.get("/api/health")
async def health():
    s = get_state()
    docs = s.get("docs")
    return {
        "status":      "ok",
        "corpus_size": len(docs) if docs else 0,
        "model":       os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"),
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

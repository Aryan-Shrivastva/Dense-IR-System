"""
Builds and saves FAISS index + BM25 index from corpus CSV.
Run once via: python scripts/build_index.py
"""

import os
import pickle
import numpy as np
import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.retrieval.hash_biencoder import encode_texts as hash_encode_texts


def build_indexes(
    corpus_path: str,
    index_path: str,
    bm25_path: str,
    corpus_pkl_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
):
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_csv(corpus_path).fillna("")

    # combine title + body for richer embedding
    df["text"] = df["title"].str.strip() + ". " + df["body"].str.strip()
    df = df[df["text"].str.len() > 60].reset_index(drop=True)
    docs = df.to_dict("records")
    print(f"Loaded {len(docs)} documents")

    texts = [d["text"] for d in docs]
    offline = os.getenv("OFFLINE_IR", "").strip().lower() in ("1", "true", "yes")

    # --- Dense embeddings ---
    if offline:
        print("OFFLINE_IR=1: using deterministic hash embeddings (no Hugging Face).")
        embeddings = hash_encode_texts(texts)
        bi_mode = "hash384"
    else:
        print(f"Encoding with {model_name}...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        bi_mode = "sentence_transformers"

    # --- FAISS index ---
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved -> {index_path} ({index.ntotal} vectors)")

    # --- BM25 index ---
    print("Building BM25 index...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved -> {bm25_path}")

    # --- Corpus pickle (docs + embeddings for personalization) ---
    with open(corpus_pkl_path, "wb") as f:
        pickle.dump(
            {"docs": docs, "embeddings": embeddings, "bi_encoder_mode": bi_mode},
            f,
        )
    print(f"Corpus pickle saved -> {corpus_pkl_path}")

    return docs, embeddings, index, bm25

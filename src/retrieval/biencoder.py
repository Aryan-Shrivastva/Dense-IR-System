"""
Stage 1: Bi-encoder retrieval using MiniLM + FAISS.
Embeds query and retrieves top-K candidates via inner product search.
"""

import numpy as np
import faiss
from typing import List, Dict, Any


class BiEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._hash_mode = False
        self.index = None
        self.docs = []
        self.embeddings = None

    def encode(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        if self._hash_mode:
            from src.retrieval.hash_biencoder import encode_texts
            return encode_texts(texts)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def load_index(self, index_path: str, corpus_path: str):
        import pickle
        from sentence_transformers import SentenceTransformer

        self.index = faiss.read_index(index_path)
        with open(corpus_path, "rb") as f:
            data = pickle.load(f)
        self.docs = data["docs"]
        self.embeddings = data["embeddings"]
        self._hash_mode = data.get("bi_encoder_mode") == "hash384"
        if self._hash_mode:
            print("Bi-encoder: hash384 mode (offline)")
        else:
            self.model = SentenceTransformer(self.model_name)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        q_emb = self.encode([query])
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.docs[idx].copy()
            doc["stage1_score"] = float(score)
            doc["local_idx"] = int(idx)
            results.append(doc)
        return results

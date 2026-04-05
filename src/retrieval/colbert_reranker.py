"""
Stage 2: ColBERT-style re-ranker using token-level MaxSim scoring.

Key idea: instead of comparing [CLS] embeddings (bi-encoder),
ColBERT compares EVERY query token against EVERY document token
and takes the MAX similarity for each query token, then sums.

This captures fine-grained semantic matches that bi-encoders miss.
e.g. query token "myocardial" matches doc token "cardiac" via context.
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple


def _env_offline() -> bool:
    return os.getenv("OFFLINE_IR", "").strip().lower() in ("1", "true", "yes")


class ColBERTReranker:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = None,
        offline: bool | None = None,
    ):
        self.offline = _env_offline() if offline is None else offline
        if self.offline:
            print("ColBERT reranker: OFFLINE stub (re-rank by stage1 score)")
            self.device = None
            self.tokenizer = None
            self.model = None
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ColBERT reranker using: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def _encode_tokens(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """
        Returns all token embeddings (not just [CLS]).
        Shape: (batch, seq_len, hidden_dim)
        """
        if self.offline:
            raise RuntimeError("ColBERT offline stub does not encode tokens")
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens)

        # all token embeddings
        embs = out.last_hidden_state  # (batch, seq_len, 768)

        # mask padding tokens before normalization
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
        embs = embs * attention_mask

        # L2 normalize each token vector
        norms = embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        embs = embs / norms

        return embs

    def maxsim(
        self, query_emb: torch.Tensor, doc_emb: torch.Tensor
    ) -> float:
        """
        ColBERT MaxSim scoring.
        For each query token: find the max cosine similarity over all doc tokens.
        Final score = sum of per-query-token max similarities.

        query_emb: (Lq, D)
        doc_emb:   (Ld, D)
        returns:   scalar score
        """
        # (Lq, Ld) similarity matrix
        sim = torch.matmul(query_emb, doc_emb.T)
        # max over doc tokens for each query token → (Lq,)
        per_token_max = sim.max(dim=-1).values
        # sum over query tokens
        return per_token_max.sum().item()

    def get_token_weights(
        self, query: str, top_doc_text: str
    ) -> List[Dict[str, float]]:
        """
        Returns per-query-token MaxSim weights — used for frontend visualization.
        Shows which query tokens contributed most to the score.
        """
        if self.offline:
            return []
        q_emb = self._encode_tokens([query])[0]       # (Lq, D)
        d_emb = self._encode_tokens([top_doc_text])[0] # (Ld, D)

        sim = torch.matmul(q_emb, d_emb.T)
        max_sims = sim.max(dim=-1).values.cpu().numpy()

        tokens = self.tokenizer.tokenize(query)
        # align — tokenizer may produce more tokens than words
        weights = []
        for tok, weight in zip(tokens[:len(max_sims)], max_sims):
            if not tok.startswith("##"):  # skip subword continuations
                weights.append({
                    "token": tok,
                    "weight": float(max(0, weight)),
                })
        return sorted(weights, key=lambda x: x["weight"], reverse=True)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Re-ranks candidates using ColBERT MaxSim.
        Returns (reranked_docs, colbert_scores)
        """
        if self.offline:
            top = sorted(
                candidates,
                key=lambda d: float(d.get("stage1_score", 0.0)),
                reverse=True,
            )[:top_k]
            scores = [float(d.get("stage1_score", 0.0)) for d in top]
            out = []
            for d, s in zip(top, scores):
                c = d.copy()
                c["colbert_score"] = s
                out.append(c)
            return out, scores

        q_emb = self._encode_tokens([query])[0]  # (Lq, D)

        scored = []
        for doc in candidates:
            text = (doc.get("title", "") + " " + doc.get("body", ""))[:512]
            d_emb = self._encode_tokens([text])[0]
            score = self.maxsim(q_emb, d_emb)
            doc_copy = doc.copy()
            doc_copy["colbert_score"] = score
            scored.append((score, doc_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        reranked_docs = [d for _, d in top]
        colbert_scores = [s for s, _ in top]

        return reranked_docs, colbert_scores

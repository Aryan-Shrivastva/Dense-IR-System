"""
POST /api/search — main search endpoint.

Pipeline:
1. Stage 1: bi-encoder → top-100 candidates (FAISS)
2. Stage 2: ColBERT MaxSim → re-rank to top-20
3. Load user profile from DB (if exists)
4. Compute dynamic alpha from ColBERT score variance
5. Hybrid rank: alpha*semantic + (1-alpha)*user
6. Return top-10 with full metadata for visualization
"""

import time
import os
from fastapi import APIRouter
import numpy as np

from api.schemas import SearchRequest, SearchResponse, SearchResult
from api.state import get_state
from src.personalization.hybrid_ranker import hybrid_rank
from src.database.user_store import load_user_profile, log_search

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0    = time.time()
    state = get_state()

    biencoder   = state["biencoder"]
    colbert     = state["colbert"]
    user_model  = state["user_model"]
    bm25        = state["bm25"]
    docs        = state["docs"]
    embeddings  = state["embeddings"]

    top_k_s1 = int(os.getenv("TOP_K_STAGE1", 100))
    top_k_s2 = int(os.getenv("TOP_K_STAGE2", 20))

    # ── Load user profile from DB into in-memory model ──────────────────────
    if not user_model.has_profile(req.user_id):
        profile = await load_user_profile(req.user_id)
        if profile:
            user_model.load_from_db(
                req.user_id,
                profile["interest_vector"],
                profile.get("click_count"),
            )

    # ── Stage 1: retrieve candidates ─────────────────────────────────────────
    if req.method == "bm25":
        tokens = req.query.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        top_idx = np.argsort(bm25_scores)[::-1][:top_k_s1]
        candidates = []
        for idx in top_idx:
            d = docs[idx].copy()
            d["stage1_score"] = float(bm25_scores[idx])
            d["local_idx"] = int(idx)
            candidates.append(d)
    else:
        candidates = biencoder.retrieve(req.query, top_k=top_k_s1)

    # ── Stage 2: ColBERT re-rank ─────────────────────────────────────────────
    if req.method in ("hybrid", "dense"):
        reranked, colbert_scores = colbert.rerank(
            req.query, candidates, top_k=top_k_s2
        )
    else:
        # BM25 mode: skip ColBERT, use BM25 scores directly
        reranked = candidates[:top_k_s2]
        colbert_scores = [d["stage1_score"] for d in reranked]

    # ── Stage 3: Hybrid adaptive ranking ─────────────────────────────────────
    doc_indices = [d.get("local_idx", -1) for d in reranked]

    ranked_docs, metadata = hybrid_rank(
        docs=reranked,
        colbert_scores=colbert_scores,
        user_model=user_model,
        user_id=req.user_id,
        doc_embeddings=embeddings,
        doc_indices=doc_indices,
        alpha_min=float(os.getenv("ALPHA_MIN", 0.3)),
        alpha_max=float(os.getenv("ALPHA_MAX", 0.9)),
        top_k=req.top_k,
    )

    # ── Token weights for visualization ──────────────────────────────────────
    token_weights = []
    if ranked_docs:
        top_text = ranked_docs[0].get("body", "")[:300]
        try:
            token_weights = colbert.get_token_weights(req.query, top_text)
        except Exception:
            pass

    # ── Format results ────────────────────────────────────────────────────────
    results = []
    for rank, doc in enumerate(ranked_docs, 1):
        results.append(SearchResult(
            rank           = rank,
            doc_id         = doc.get("docid", ""),
            doc_idx        = int(doc.get("local_idx", -1)),
            title          = doc.get("title", "")[:120],
            snippet        = doc.get("body", "")[:250] + "...",
            url            = doc.get("url", ""),
            colbert_score  = doc.get("colbert_score", 0.0),
            semantic_score = doc.get("semantic_score", 0.0),
            user_score     = doc.get("user_score", 0.0),
            final_score    = doc.get("final_score", 0.0),
            personalized   = doc.get("personalized", False),
            source         = doc.get("source", "msmarco"),
        ))

    latency_ms = round((time.time() - t0) * 1000, 1)

    await log_search(
        user_id=req.user_id, query=req.query,
        alpha=metadata["alpha"], method=req.method,
        result_count=len(results), latency_ms=latency_ms,
    )

    return SearchResponse(
        query            = req.query,
        user_id          = req.user_id,
        method           = req.method,
        results          = results,
        alpha            = metadata["alpha"],
        alpha_label      = metadata["alpha_label"],
        colbert_variance = metadata["colbert_variance"],
        has_user_profile = metadata["has_user_profile"],
        user_click_count = metadata["user_click_count"],
        latency_ms       = latency_ms,
        stage1_count     = len(candidates),
        stage2_count     = len(reranked),
        token_weights    = token_weights,
    )

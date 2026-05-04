"""
POST /api/search — two-stage retrieval pipeline.
Stage 1: BM25, Dense, or Hybrid (RRF fusion of both)
Stage 2: ColBERT MaxSim reranker over Stage 1 shortlist
Stage 3: Dynamic alpha hybrid with user profile
"""

import time
import os
import numpy as np
from fastapi import APIRouter

from api.schemas import SearchRequest, SearchResponse, SearchResult
from api.state import get_state
from src.personalization.hybrid_ranker import hybrid_rank
from src.database.user_store import load_user_profile, log_search

router = APIRouter()


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Combine ranked lists using RRF. ranked_lists = [corpus_indices_ordered_by_score, ...]"""
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.time()
    state = get_state()

    biencoder = state["biencoder"]
    colbert = state["colbert"]
    user_model = state["user_model"]
    bm25 = state["bm25"]
    docs = state["docs"]
    embeddings = state["embeddings"]

    top_k_s1 = int(os.getenv("TOP_K_STAGE1", 100))
    top_k_s2 = int(os.getenv("TOP_K_STAGE2", 20))

    if not user_model.has_profile(req.user_id):
        profile = await load_user_profile(req.user_id)
        if profile:
            user_model.load_from_db(req.user_id, profile["interest_vector"], profile.get("click_count"))

    # ---- Stage 1: Candidate retrieval ----
    if req.method == "bm25":
        tokens = req.query.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        top_idx = np.argsort(bm25_scores)[::-1][:top_k_s1]
        candidates = [
            {**docs[i], "stage1_score": float(bm25_scores[i]), "local_idx": int(i)}
            for i in top_idx
        ]
        doc_indices = [d["local_idx"] for d in candidates]

    elif req.method == "dense":
        candidates = biencoder.retrieve(req.query, top_k=top_k_s1)
        doc_indices = [d.get("local_idx", -1) for d in candidates]

    else:  # hybrid: RRF of BM25 + Dense
        tokens = req.query.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_order = np.argsort(bm25_scores)[::-1][:top_k_s1]

        dense_cands = biencoder.retrieve(req.query, top_k=top_k_s1)
        dense_order = [d["local_idx"] for d in dense_cands]

        # RRF over corpus indices
        top_rrf = reciprocal_rank_fusion([bm25_order, dense_order])[:top_k_s1]
        candidates = []
        for idx in top_rrf:
            c = docs[idx].copy()
            c["stage1_score"] = float(bm25_scores[idx])
            c["local_idx"] = int(idx)
            candidates.append(c)
        doc_indices = [d["local_idx"] for d in candidates]

    # ---- Stage 2: ColBERT reranking ----
    # Collect all stage-1 candidate scores for the score distribution chart
    stage1_all_scores = [d.get("stage1_score", 0.0) for d in candidates]

    if req.method in ("hybrid", "dense"):
        reranked, colbert_scores = colbert.rerank(req.query, candidates, top_k=top_k_s2)
        doc_indices = [d.get("local_idx", -1) for d in reranked]
    else:  # BM25: skip ColBERT, use BM25 scores
        reranked = candidates[:top_k_s2]
        colbert_scores = [d["stage1_score"] for d in reranked]

    # ---- Stage 3: Hybrid ranking with dynamic alpha ----
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

    # Token weights for visualization
    token_weights = []
    if ranked_docs:
        try:
            token_weights = colbert.get_token_weights(
                req.query, ranked_docs[0].get("body", "")[:300]
            )
        except Exception:
            pass

    results = [
        SearchResult(
            rank=rank,
            doc_id=d.get("docid", ""),
            doc_idx=int(d.get("local_idx", -1)),
            title=(d.get("title") or "")[:120],
            snippet=(d.get("body") or "")[:250] + "...",
            url=d.get("url", ""),
            colbert_score=d.get("colbert_score", 0.0),
            semantic_score=d.get("semantic_score", 0.0),
            user_score=d.get("user_score", 0.0),
            final_score=d.get("final_score", 0.0),
            personalized=d.get("personalized", False),
            source=d.get("source", "msmarco"),
        )
        for rank, d in enumerate(ranked_docs, 1)
    ]

    latency_ms = round((time.time() - t0) * 1000, 1)
    await log_search(req.user_id, req.query, metadata["alpha"], req.method, len(results), latency_ms)

    # Build all stage-1 ColBERT scores (top-100 candidates before reranking)
    all_colbert_scores = []
    if req.method in ("hybrid", "dense"):
        # After ColBERT reranking, scores are only for top-20 reranked docs
        # We need to show the full distribution — pad with 0 for non-reranked candidates
        reranked_idx = set(d.get("local_idx", -1) for d in reranked)
        all_colbert_scores = []
        for c in candidates:
            lidx = c.get("local_idx", -1)
            # Find this candidate's ColBERT score if it was reranked
            matched = [cs for cs, rd in zip(colbert_scores, reranked) if rd.get("local_idx") == lidx]
            all_colbert_scores.append(matched[0] if matched else 0.0)
    else:
        all_colbert_scores = stage1_all_scores

    return SearchResponse(
        query=req.query,
        user_id=req.user_id,
        method=req.method,
        results=results,
        alpha=metadata["alpha"],
        alpha_label=metadata["alpha_label"],
        colbert_variance=metadata["colbert_variance"],
        has_user_profile=metadata["has_user_profile"],
        user_click_count=metadata["user_click_count"],
        latency_ms=latency_ms,
        stage1_count=len(candidates),
        stage2_count=len(reranked),
        token_weights=token_weights,
    ).model_copy(update={"stage1_all_scores": all_colbert_scores})
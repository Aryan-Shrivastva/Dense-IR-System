"""
Hybrid ranker — combines ColBERT scores with user interest scores.

Final score = alpha * colbert_score_normalized + (1-alpha) * user_score

Where alpha is computed dynamically per query by the confidence estimator.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .user_model import UserInterestModel
from .confidence import compute_dynamic_alpha, score_variance_category


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def hybrid_rank(
    docs: List[Dict[str, Any]],
    colbert_scores: List[float],
    user_model: UserInterestModel,
    user_id: str,
    doc_embeddings: np.ndarray,
    doc_indices: List[int],
    alpha_min: float = 0.3,
    alpha_max: float = 0.9,
    top_k: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Ranks documents using a dynamic combination of ColBERT + user interest.

    Returns:
        ranked_docs: list of docs with scores attached
        metadata:    alpha, variance info for frontend visualization
    """
    alpha = compute_dynamic_alpha(colbert_scores, alpha_min, alpha_max)
    alpha_label = score_variance_category(alpha)

    norm_colbert = normalize_scores(colbert_scores)
    has_user_profile = user_model.has_profile(user_id)

    ranked = []
    for i, (doc, col_score, norm_col) in enumerate(
        zip(docs, colbert_scores, norm_colbert)
    ):
        user_score = 0.0
        if has_user_profile and i < len(doc_indices):
            idx = doc_indices[i]
            if 0 <= idx < len(doc_embeddings):
                user_score = user_model.score(user_id, doc_embeddings[idx])
                # normalize user score from [-1,1] to [0,1]
                user_score = (user_score + 1) / 2

        final_score = alpha * norm_col + (1 - alpha) * user_score
        personalized = has_user_profile and user_score > 0.5

        doc_out = doc.copy()
        doc_out["colbert_score"]   = round(col_score, 4)
        doc_out["semantic_score"]  = round(norm_col, 4)
        doc_out["user_score"]      = round(user_score, 4)
        doc_out["final_score"]     = round(final_score, 4)
        doc_out["personalized"]    = personalized
        ranked.append(doc_out)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    metadata = {
        "alpha": round(alpha, 3),
        "alpha_label": alpha_label,
        "colbert_variance": round(float(np.var(colbert_scores)), 6),
        "has_user_profile": has_user_profile,
        "user_click_count": user_model.get_click_count(user_id),
    }

    return ranked[:top_k], metadata

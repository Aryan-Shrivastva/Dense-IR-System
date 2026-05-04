"""
Hybrid ranker — combines ColBERT semantic scores with user interest scores.
final = alpha * semantic_normalized + (1-alpha) * user_normalized
alpha is computed dynamically from ColBERT score variance.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .user_model import UserInterestModel
from .confidence import compute_dynamic_alpha, score_variance_category
from .normalize import normalize


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
    alpha = compute_dynamic_alpha(colbert_scores, alpha_min, alpha_max)
    print(f"[ALPHA] alpha={alpha:.3f} var={np.var(colbert_scores):.6f} label='{score_variance_category(alpha)}'")

    has_profile = user_model.has_profile(user_id)
    norm_sem = normalize(colbert_scores)

    ranked = []
    for i, doc in enumerate(docs):
        user_score = 0.0
        if has_profile and i < len(doc_indices):
            idx = doc_indices[i]
            if 0 <= idx < len(doc_embeddings):
                us = user_model.score(user_id, doc_embeddings[idx])
                user_score = (us + 1) / 2

        final = alpha * norm_sem[i] + (1 - alpha) * user_score

        d = doc.copy()
        d["colbert_score"]  = round(colbert_scores[i], 4)
        d["semantic_score"] = round(norm_sem[i], 4)
        d["user_score"]     = round(user_score, 4)
        d["final_score"]    = round(final, 4)
        d["personalized"]  = has_profile and user_score > 0.5
        ranked.append(d)

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    return ranked[:top_k], {
        "alpha": round(alpha, 3),
        "alpha_label": score_variance_category(alpha),
        "colbert_variance": round(float(np.var(colbert_scores)), 6),
        "has_user_profile": has_profile,
        "user_click_count": user_model.get_click_count(user_id),
    }
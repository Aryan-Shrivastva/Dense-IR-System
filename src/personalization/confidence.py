"""
Confidence estimator — computes dynamic alpha per query.

Key idea:
- When ColBERT scores have HIGH variance → top result is clearly better
  → trust the semantic ranker more (alpha closer to 0.9)
- When ColBERT scores have LOW variance → results are similarly relevant
  → let user preferences break the tie (alpha closer to 0.3)

This makes personalization only kick in meaningfully when it actually helps.
"""

import numpy as np
from typing import List


def compute_dynamic_alpha(
    colbert_scores: List[float],
    alpha_min: float = 0.3,
    alpha_max: float = 0.9,
    sensitivity: float = 80.0,
) -> float:
    """
    Maps ColBERT score variance → alpha in [alpha_min, alpha_max].

    sensitivity: higher = alpha responds more aggressively to variance changes
    """
    if not colbert_scores or len(colbert_scores) < 2:
        return (alpha_min + alpha_max) / 2

    variance = float(np.var(colbert_scores))

    # sigmoid-style mapping: high variance → high alpha
    alpha = alpha_min + (alpha_max - alpha_min) * (
        1 - np.exp(-sensitivity * variance)
    )
    return float(np.clip(alpha, alpha_min, alpha_max))


def score_variance_category(alpha: float) -> str:
    """Human-readable label for the UI."""
    if alpha >= 0.75:
        return "high semantic confidence — trusting ranker"
    elif alpha >= 0.5:
        return "moderate confidence — balanced blend"
    else:
        return "low confidence — personalizing for you"

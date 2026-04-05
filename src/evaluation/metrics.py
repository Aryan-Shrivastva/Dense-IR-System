import numpy as np
from typing import List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / k if k > 0 else 0.0


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for rank, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    dcg, idcg = 0.0, 0.0
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant:
            dcg += 1 / np.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    for i in range(1, ideal_hits + 1):
        idcg += 1 / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0

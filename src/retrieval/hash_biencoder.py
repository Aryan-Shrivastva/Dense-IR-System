"""Deterministic pseudo-embeddings for offline / no-HF environments (not for research quality)."""

import hashlib
from typing import List

import numpy as np


def encode_texts(texts: List[str], dim: int = 384) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = int.from_bytes(
            hashlib.sha256(t.encode("utf-8", errors="ignore")).digest()[:8],
            "little",
        )
        rng = np.random.RandomState(h % (2**32))
        v = rng.standard_normal(dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            v /= n
        out[i] = v
    return out

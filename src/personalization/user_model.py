"""
Adaptive user interest model using Exponential Moving Average (EMA).

Key idea:
- Each user has an "interest vector" in the same embedding space as documents
- After every click, we update: interest = decay * old + (1-decay) * clicked_doc_emb
- Recent clicks matter more than old ones (controlled by decay parameter)
- At query time, we compute cosine similarity between interest vector and each candidate

This is online learning — no retraining, updates in milliseconds.
"""

import numpy as np
from typing import Dict, Optional


class UserInterestModel:
    def __init__(self, decay: float = 0.85):
        """
        decay: weight given to historical interest vs new click
               0.85 = new click contributes 15%, history 85%
               higher = slower adaptation, lower = faster but noisier
        """
        self.decay = decay
        self.profiles: Dict[str, np.ndarray] = {}
        self.click_counts: Dict[str, int] = {}

    def update(self, user_id: str, doc_embedding: np.ndarray):
        """Update interest vector with a new clicked document embedding."""
        # normalize the incoming embedding
        emb = doc_embedding.astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        if user_id not in self.profiles:
            # first click — initialize directly
            self.profiles[user_id] = emb.copy()
            self.click_counts[user_id] = 1
        else:
            old = self.profiles[user_id]
            # EMA update
            updated = self.decay * old + (1 - self.decay) * emb
            # re-normalize so it stays a unit vector
            updated_norm = np.linalg.norm(updated)
            if updated_norm > 0:
                updated = updated / updated_norm
            self.profiles[user_id] = updated
            self.click_counts[user_id] = self.click_counts.get(user_id, 0) + 1

    def score(self, user_id: str, doc_embedding: np.ndarray) -> float:
        """Returns cosine similarity between user interest and a document."""
        if user_id not in self.profiles:
            return 0.0
        interest = self.profiles[user_id]
        emb = doc_embedding.astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return float(np.dot(interest, emb))

    def get_profile(self, user_id: str) -> Optional[np.ndarray]:
        return self.profiles.get(user_id)

    def has_profile(self, user_id: str) -> bool:
        return user_id in self.profiles

    def get_click_count(self, user_id: str) -> int:
        return self.click_counts.get(user_id, 0)

    def load_from_db(
        self, user_id: str, interest_vector: list, click_count: Optional[int] = None
    ):
        """Load a previously persisted interest vector from the database."""
        if interest_vector:
            vec = np.array(interest_vector, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.profiles[user_id] = vec
        if click_count is not None:
            self.click_counts[user_id] = int(click_count)

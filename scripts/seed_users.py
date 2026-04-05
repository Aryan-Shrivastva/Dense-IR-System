"""
Seeds 3 demo users with realistic click histories so personalization
is visible immediately during the live demo.
Run AFTER the API has been started at least once (so the DB exists).

Usage: python scripts/seed_users.py
"""

import asyncio
import pickle
import os
import numpy as np
from dotenv import load_dotenv

from src.personalization.user_model import UserInterestModel
from src.database.user_store import save_user_profile, init_db, log_click

load_dotenv()

CORPUS_PATH = "indexes/corpus.pkl"
DB_PATH     = os.getenv("DB_PATH", "data/userdata.db")

USERS = {
    "tech_user": [
        "machine learning", "neural network", "deep learning",
        "transformer architecture", "GPU computing",
    ],
    "science_user": [
        "DNA replication", "quantum mechanics", "climate change",
        "black hole", "protein folding",
    ],
    "medical_user": [
        "cardiovascular disease", "cancer treatment", "vaccine development",
        "antibiotic resistance", "neuroscience",
    ],
}


async def seed():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    await init_db(DB_PATH)

    with open(CORPUS_PATH, "rb") as f:
        data = pickle.load(f)
    docs       = data["docs"]
    embeddings = data["embeddings"]

    for user_id, interest_queries in USERS.items():
        model = UserInterestModel(decay=0.85)
        print(f"Seeding {user_id}...")

        for query in interest_queries:
            if data.get("bi_encoder_mode") == "hash384":
                from src.retrieval.hash_biencoder import encode_texts
                q_emb = encode_texts([query])
            else:
                from sentence_transformers import SentenceTransformer
                st = SentenceTransformer(os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
                q_emb = st.encode(
                    [query], normalize_embeddings=True, convert_to_numpy=True
                ).astype("float32")
            # find top-3 docs for this interest query
            sims  = embeddings @ q_emb.T
            top_i = np.argsort(sims.flatten())[::-1][:3]

            for idx in top_i:
                model.update(user_id, embeddings[idx])
                await log_click(
                    user_id=user_id,
                    doc_id=docs[idx]["docid"],
                    doc_idx=int(idx),
                    query=query,
                    db_path=DB_PATH,
                )

        profile = model.get_profile(user_id)
        if profile is not None:
            await save_user_profile(
                user_id=user_id,
                interest_vector=profile.tolist(),
                click_count=model.get_click_count(user_id),
                db_path=DB_PATH,
            )
        print(f"  {user_id}: {model.get_click_count(user_id)} simulated clicks")

    print("Seeding complete.")


if __name__ == "__main__":
    asyncio.run(seed())

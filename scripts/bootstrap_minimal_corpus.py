"""
Offline fallback: tiny corpus + eval stubs when Hugging Face is unreachable (e.g. SSL proxy).
Run: python scripts/bootstrap_minimal_corpus.py
"""

import os
import pandas as pd

DATA = "data"
os.makedirs(DATA, exist_ok=True)

TOPICS = [
    ("machine learning", "Supervised and unsupervised learning train models from data."),
    ("neural networks", "Deep networks stack layers of nonlinear transformations for representation learning."),
    ("climate change", "Rising greenhouse gases shift temperatures and ocean circulation patterns."),
    ("quantum mechanics", "Wavefunctions evolve under the Schrödinger equation with probabilistic outcomes."),
    ("cardiovascular health", "Blood pressure and cholesterol are key modifiable risk markers."),
    ("vaccine immunity", "Adaptive immune memory reduces severity after antigen exposure."),
    ("protein folding", "Amino acid sequences determine three-dimensional folded structures."),
    ("GPU computing", "Parallel SIMD units accelerate dense linear algebra for training."),
    ("DNA replication", "Polymerases copy complementary strands with proofreading machinery."),
    ("black holes", "Event horizons trap light; mergers emit gravitational waves."),
]

extras = []
for i in range(120):
    t, b = TOPICS[i % len(TOPICS)]
    extras.append({
        "docid": f"bootstrap_{i:05d}",
        "title": f"{t} — note {i}",
        "body": (b + " ") * 8 + f" Additional context paragraph {i} with more text for BM25 and dense retrieval.",
        "url": "",
    })

df = pd.DataFrame(extras)
df.to_csv(os.path.join(DATA, "corpus.csv"), index=False)
print(f"Wrote {len(df)} rows -> data/corpus.csv")

queries = [
    {"qid": 1, "query": "machine learning training"},
    {"qid": 2, "query": "climate ocean temperature"},
    {"qid": 3, "query": "quantum wavefunction"},
]
pd.DataFrame(queries).to_csv(os.path.join(DATA, "dev_queries.tsv"), sep="\t", index=False)

qrels = [
    {"qid": 1, "docid": "bootstrap_00000", "rel": 1},
    {"qid": 2, "docid": "bootstrap_00002", "rel": 1},
    {"qid": 3, "docid": "bootstrap_00003", "rel": 1},
]
pd.DataFrame(qrels).to_csv(os.path.join(DATA, "dev_qrels.tsv"), sep="\t", index=False)
print("Wrote data/dev_queries.tsv and data/dev_qrels.tsv")

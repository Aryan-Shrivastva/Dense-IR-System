# Complete Agent Instructions — Personalized Dense Information Retrieval System

You are building a **fully working, end-to-end information retrieval research project**.
This is not a demo. Every component must be real, connected, and functional.
The codebase must be clean, well-structured, reproducible, and GitHub-ready.

---

## Project overview

Build a two-stage retrieval system with adaptive personalization:

1. **Stage 1 — Bi-encoder retrieval**: MiniLM + FAISS fetches top-100 candidates fast
2. **Stage 2 — ColBERT re-ranking**: Token-level MaxSim scoring re-ranks top-100 → top-20
3. **Confidence estimator**: Score variance → dynamic alpha (how much to trust semantic vs user signal)
4. **Adaptive user model**: EMA (exponential moving average) over clicked document embeddings — updates live after every click
5. **Hybrid ranker**: `alpha(query) × colbert_score + (1-alpha) × user_score`
6. **Live visualization dashboard**: Real charts connected to the real backend — not mocked

The novelty is: (a) ColBERT-style late interaction re-ranking at inference time without training, and (b) a confidence-aware dynamic alpha that decides per-query how much personalization to apply.

---

## Repository structure to create

Create this exact folder structure before writing any code:

```
ir-project/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── README.md                  ← instructions for downloading datasets
│   ├── download_msmarco.py        ← script to download + sample MS MARCO
│   ├── download_wikipedia.py      ← script to build Wikipedia corpus
│   └── .gitkeep
│
├── indexes/
│   └── .gitkeep                   ← FAISS index saved here (gitignored)
│
├── models/
│   └── .gitkeep                   ← model cache (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── biencoder.py           ← MiniLM + FAISS Stage 1
│   │   ├── colbert_reranker.py    ← ColBERT MaxSim Stage 2
│   │   └── index_builder.py       ← builds + saves FAISS index
│   │
│   ├── personalization/
│   │   ├── __init__.py
│   │   ├── user_model.py          ← EMA user interest model
│   │   ├── confidence.py          ← score variance → dynamic alpha
│   │   └── hybrid_ranker.py       ← combines colbert + user scores
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             ← Precision@K, MRR, NDCG
│   │   └── evaluate.py            ← full evaluation script
│   │
│   └── database/
│       ├── __init__.py
│       └── user_store.py          ← SQLite user profile + click history
│
├── api/
│   ├── __init__.py
│   ├── main.py                    ← FastAPI app
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── search.py              ← POST /search
│   │   ├── feedback.py            ← POST /click, POST /feedback
│   │   └── analytics.py          ← GET /analytics/* (feeds frontend charts)
│   └── schemas.py                 ← Pydantic request/response models
│
├── frontend/
│   ├── index.html                 ← main UI
│   ├── dashboard.html             ← live visualization dashboard
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── search.js              ← search + results logic
│       ├── charts.js              ← Chart.js visualizations (live data)
│       └── api.js                 ← all fetch calls to FastAPI
│
├── scripts/
│   ├── setup.sh                   ← one-command environment setup
│   ├── build_index.py             ← run once: builds FAISS + BM25 indexes
│   └── seed_users.py              ← seeds 3 demo users with click history
│
└── notebooks/
    └── exploration.ipynb          ← optional: Colab-ready exploration notebook
```

---

## Step 1 — Environment setup

### Create `requirements.txt` with exactly these packages:

```
# Core ML
sentence-transformers==2.7.0
transformers==4.40.0
torch==2.3.0
faiss-cpu==1.8.0

# Search
rank-bm25==0.2.2

# Data
datasets==2.19.0
wikipedia-api==0.6.0
pandas==2.2.0
numpy==1.26.4

# API
fastapi==0.111.0
uvicorn[standard]==0.30.0
pydantic==2.7.0
python-multipart==0.0.9
python-dotenv==1.0.1

# Database
aiosqlite==0.20.0

# Evaluation
ir-measures==0.3.4

# Utils
tqdm==4.66.4
httpx==0.27.0
```

### Create `.env.example`:

```
# Copy this to .env and fill in values
CORPUS_SIZE=50000
INDEX_PATH=indexes/faiss_index.bin
BM25_PATH=indexes/bm25_index.pkl
CORPUS_PATH=indexes/corpus.pkl
DB_PATH=data/userdata.db
MODEL_NAME=all-MiniLM-L6-v2
COLBERT_MODEL=bert-base-uncased
TOP_K_STAGE1=100
TOP_K_STAGE2=20
TOP_K_FINAL=10
EMA_DECAY=0.85
ALPHA_MIN=0.3
ALPHA_MAX=0.9
```

### Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/

# Data and indexes (too large for git)
indexes/*.bin
indexes/*.pkl
data/*.csv
data/*.tsv
data/*.json
data/userdata.db
models/

# Environment
.env

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
```

### Create `scripts/setup.sh`:

```bash
#!/bin/bash
set -e
echo "Setting up IR project environment..."

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
mkdir -p indexes data models

echo ""
echo "Setup complete. Next steps:"
echo "  1. Run: python data/download_msmarco.py"
echo "  2. Run: python scripts/build_index.py"
echo "  3. Run: uvicorn api.main:app --reload"
echo "  4. Open frontend/index.html in your browser"
```

---

## Step 2 — Dataset download scripts

### Create `data/download_msmarco.py`:

This script downloads MS MARCO passage dataset via HuggingFace — no manual file downloads needed. It saves a sampled corpus CSV and the dev queries + qrels for evaluation.

```python
"""
Downloads MS MARCO passage dataset via HuggingFace datasets library.
No manual download required. Run once — takes ~5 minutes.

Output files:
  data/corpus.csv          — sampled passages for indexing
  data/dev_queries.tsv     — evaluation queries
  data/dev_qrels.tsv       — relevance labels for evaluation
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

CORPUS_SIZE = int(os.getenv("CORPUS_SIZE", 50000))
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_corpus():
    print(f"Downloading MS MARCO passages (sampling {CORPUS_SIZE} docs)...")
    # This streams directly — no 22GB download needed
    dataset = load_dataset(
        "ms_marco", "v2.1",
        split=f"train[:{CORPUS_SIZE}]",
        trust_remote_code=True
    )

    docs = []
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        for j, passage in enumerate(item["passages"]["passage_text"]):
            docs.append({
                "docid": f"msmarco_{i:06d}_{j}",
                "title": item.get("query", "")[:100],
                "body":  passage,
                "url":   "",
            })
            if len(docs) >= CORPUS_SIZE:
                break
        if len(docs) >= CORPUS_SIZE:
            break

    df = pd.DataFrame(docs[:CORPUS_SIZE])
    df = df[df["body"].str.len() > 50].reset_index(drop=True)
    out = os.path.join(DATA_DIR, "corpus.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} passages → {out}")
    return df


def download_eval_data():
    print("Downloading MS MARCO dev queries and qrels...")
    dev = load_dataset("ms_marco", "v2.1", split="validation[:5000]", trust_remote_code=True)

    queries, qrels = [], []
    for item in tqdm(dev, desc="Processing eval"):
        qid = item["query_id"]
        queries.append({"qid": qid, "query": item["query"]})
        for did, rel in zip(
            item["passages"]["passage_id"],
            item["passages"]["is_selected"]
        ):
            if rel > 0:
                qrels.append({"qid": qid, "docid": str(did), "rel": int(rel)})

    pd.DataFrame(queries).to_csv(
        os.path.join(DATA_DIR, "dev_queries.tsv"), sep="\t", index=False
    )
    pd.DataFrame(qrels).to_csv(
        os.path.join(DATA_DIR, "dev_qrels.tsv"), sep="\t", index=False
    )
    print(f"Saved {len(queries)} queries and {len(qrels)} qrels")


if __name__ == "__main__":
    download_corpus()
    download_eval_data()
    print("\nDataset download complete.")
```

### Create `data/download_wikipedia.py`:

This is an OPTIONAL additional corpus for the live demo. Run it if you also want Wikipedia articles (recommended for demo — more topically diverse).

```python
"""
Builds a Wikipedia corpus using HuggingFace datasets.
More topically diverse than MS MARCO — better for live demos.
Run this AFTER download_msmarco.py if you want a combined corpus.

Output: data/wiki_corpus.csv
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

WIKI_SIZE = 20000
DATA_DIR = "data"


def download_wiki():
    print(f"Downloading Wikipedia articles (sampling {WIKI_SIZE})...")
    # Uses the 2022-03-01 snapshot — topically current for most academic topics
    wiki = load_dataset(
        "wikipedia", "20220301.en",
        split=f"train[:{WIKI_SIZE}]",
        trust_remote_code=True
    )

    docs = []
    for i, article in enumerate(tqdm(wiki, desc="Processing")):
        text = article["text"]
        # Split each article into ~300-word chunks
        words = text.split()
        chunks = [" ".join(words[j:j+300]) for j in range(0, len(words), 300)]
        for k, chunk in enumerate(chunks[:5]):  # max 5 chunks per article
            if len(chunk) > 100:
                docs.append({
                    "docid": f"wiki_{i:06d}_{k}",
                    "title": article["title"],
                    "body":  chunk,
                    "url":   f"https://en.wikipedia.org/wiki/{article['title'].replace(' ','_')}",
                })

    df = pd.DataFrame(docs)
    out = os.path.join(DATA_DIR, "wiki_corpus.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} Wikipedia chunks → {out}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_wiki()
```

### Create `data/README.md`:

```markdown
# Data Directory

## Setup (run in order)

### Step 1 — MS MARCO (required for evaluation)
```bash
python data/download_msmarco.py
```
Downloads via HuggingFace. No manual download needed.
Produces: `corpus.csv`, `dev_queries.tsv`, `dev_qrels.tsv`

### Step 2 — Wikipedia (recommended for live demo)
```bash
python data/download_wikipedia.py
```
Produces: `wiki_corpus.csv`

## Files (generated — not committed to git)

| File | Description |
|---|---|
| `corpus.csv` | MS MARCO passages (50k) — used for indexing + evaluation |
| `wiki_corpus.csv` | Wikipedia chunks — used for live demo queries |
| `dev_queries.tsv` | MS MARCO dev queries for evaluation |
| `dev_qrels.tsv` | Relevance labels for evaluation |
| `userdata.db` | SQLite database for user profiles and clicks |
```

---

## Step 3 — Core retrieval modules

### Create `src/retrieval/biencoder.py`:

```python
"""
Stage 1: Bi-encoder retrieval using MiniLM + FAISS.
Embeds query and retrieves top-K candidates via inner product search.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os


class BiEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs = []

    def encode(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def load_index(self, index_path: str, corpus_path: str):
        import pickle
        self.index = faiss.read_index(index_path)
        with open(corpus_path, "rb") as f:
            data = pickle.load(f)
        self.docs = data["docs"]
        self.embeddings = data["embeddings"]
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        q_emb = self.encode([query])
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.docs[idx].copy()
            doc["stage1_score"] = float(score)
            doc["local_idx"] = int(idx)
            results.append(doc)
        return results
```

### Create `src/retrieval/colbert_reranker.py`:

This is the most important file — implement ColBERT MaxSim properly.

```python
"""
Stage 2: ColBERT-style re-ranker using token-level MaxSim scoring.

Key idea: instead of comparing [CLS] embeddings (bi-encoder),
ColBERT compares EVERY query token against EVERY document token
and takes the MAX similarity for each query token, then sums.

This captures fine-grained semantic matches that bi-encoders miss.
e.g. query token "myocardial" matches doc token "cardiac" via context.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple


class ColBERTReranker:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ColBERT reranker using: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def _encode_tokens(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """
        Returns all token embeddings (not just [CLS]).
        Shape: (batch, seq_len, hidden_dim)
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens)

        # all token embeddings
        embs = out.last_hidden_state  # (batch, seq_len, 768)

        # mask padding tokens before normalization
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
        embs = embs * attention_mask

        # L2 normalize each token vector
        norms = embs.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        embs = embs / norms

        return embs

    def maxsim(
        self, query_emb: torch.Tensor, doc_emb: torch.Tensor
    ) -> float:
        """
        ColBERT MaxSim scoring.
        For each query token: find the max cosine similarity over all doc tokens.
        Final score = sum of per-query-token max similarities.

        query_emb: (Lq, D)
        doc_emb:   (Ld, D)
        returns:   scalar score
        """
        # (Lq, Ld) similarity matrix
        sim = torch.matmul(query_emb, doc_emb.T)
        # max over doc tokens for each query token → (Lq,)
        per_token_max = sim.max(dim=-1).values
        # sum over query tokens
        return per_token_max.sum().item()

    def get_token_weights(
        self, query: str, top_doc_text: str
    ) -> List[Dict[str, float]]:
        """
        Returns per-query-token MaxSim weights — used for frontend visualization.
        Shows which query tokens contributed most to the score.
        """
        q_emb = self._encode_tokens([query])[0]       # (Lq, D)
        d_emb = self._encode_tokens([top_doc_text])[0] # (Ld, D)

        sim = torch.matmul(q_emb, d_emb.T)
        max_sims = sim.max(dim=-1).values.cpu().numpy()

        tokens = self.tokenizer.tokenize(query)
        # align — tokenizer may produce more tokens than words
        weights = []
        for tok, weight in zip(tokens[:len(max_sims)], max_sims):
            if not tok.startswith("##"):  # skip subword continuations
                weights.append({
                    "token": tok,
                    "weight": float(max(0, weight)),
                })
        return sorted(weights, key=lambda x: x["weight"], reverse=True)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Re-ranks candidates using ColBERT MaxSim.
        Returns (reranked_docs, colbert_scores)
        """
        q_emb = self._encode_tokens([query])[0]  # (Lq, D)

        scored = []
        for doc in candidates:
            text = (doc.get("title", "") + " " + doc.get("body", ""))[:512]
            d_emb = self._encode_tokens([text])[0]
            score = self.maxsim(q_emb, d_emb)
            doc_copy = doc.copy()
            doc_copy["colbert_score"] = score
            scored.append((score, doc_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        reranked_docs = [d for _, d in top]
        colbert_scores = [s for s, _ in top]

        return reranked_docs, colbert_scores
```

### Create `src/retrieval/index_builder.py`:

```python
"""
Builds and saves FAISS index + BM25 index from corpus CSV.
Run once via: python scripts/build_index.py
"""

import os
import pickle
import numpy as np
import faiss
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def build_indexes(
    corpus_path: str,
    index_path: str,
    bm25_path: str,
    corpus_pkl_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
):
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_csv(corpus_path).fillna("")

    # combine title + body for richer embedding
    df["text"] = df["title"].str.strip() + ". " + df["body"].str.strip()
    df = df[df["text"].str.len() > 60].reset_index(drop=True)
    docs = df.to_dict("records")
    print(f"Loaded {len(docs)} documents")

    # --- Dense embeddings ---
    print(f"Encoding with {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # --- FAISS index ---
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved → {index_path} ({index.ntotal} vectors)")

    # --- BM25 index ---
    print("Building BM25 index...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index saved → {bm25_path}")

    # --- Corpus pickle (docs + embeddings for personalization) ---
    with open(corpus_pkl_path, "wb") as f:
        pickle.dump({"docs": docs, "embeddings": embeddings}, f)
    print(f"Corpus pickle saved → {corpus_pkl_path}")

    return docs, embeddings, index, bm25
```

---

## Step 4 — Personalization modules

### Create `src/personalization/user_model.py`:

```python
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

    def load_from_db(self, user_id: str, interest_vector: list):
        """Load a previously persisted interest vector from the database."""
        if interest_vector:
            vec = np.array(interest_vector, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.profiles[user_id] = vec
```

### Create `src/personalization/confidence.py`:

```python
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
```

### Create `src/personalization/hybrid_ranker.py`:

```python
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
```

---

## Step 5 — Database

### Create `src/database/user_store.py`:

```python
"""
SQLite-based persistence for user profiles and click history.
Uses aiosqlite for async FastAPI compatibility.
"""

import json
import time
import aiosqlite
from typing import List, Dict, Optional


DB_PATH = "data/userdata.db"


async def init_db(db_path: str = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS clicks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                doc_idx     INTEGER DEFAULT -1,
                query       TEXT,
                timestamp   REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id          TEXT PRIMARY KEY,
                interest_vector  TEXT,
                click_count      INTEGER DEFAULT 0,
                updated_at       REAL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT,
                query       TEXT,
                alpha       REAL,
                method      TEXT,
                result_count INTEGER,
                latency_ms  REAL,
                timestamp   REAL
            )
        """)
        await db.commit()


async def log_click(
    user_id: str, doc_id: str, doc_idx: int, query: str,
    db_path: str = DB_PATH
):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO clicks (user_id, doc_id, doc_idx, query, timestamp)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, doc_id, doc_idx, query, time.time())
        )
        await db.commit()


async def save_user_profile(
    user_id: str, interest_vector: List[float], click_count: int,
    db_path: str = DB_PATH
):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            INSERT INTO user_profiles (user_id, interest_vector, click_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                interest_vector=excluded.interest_vector,
                click_count=excluded.click_count,
                updated_at=excluded.updated_at
        """, (user_id, json.dumps(interest_vector), click_count, time.time()))
        await db.commit()


async def load_user_profile(
    user_id: str, db_path: str = DB_PATH
) -> Optional[Dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT interest_vector, click_count FROM user_profiles WHERE user_id=?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["interest_vector"]:
                return {
                    "interest_vector": json.loads(row["interest_vector"]),
                    "click_count": row["click_count"],
                }
    return None


async def get_click_history(
    user_id: str, limit: int = 20, db_path: str = DB_PATH
) -> List[Dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT doc_id, query, timestamp FROM clicks"
            " WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def log_search(
    user_id: str, query: str, alpha: float, method: str,
    result_count: int, latency_ms: float, db_path: str = DB_PATH
):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO search_logs"
            " (user_id, query, alpha, method, result_count, latency_ms, timestamp)"
            " VALUES (?,?,?,?,?,?,?)",
            (user_id, query, alpha, method, result_count, latency_ms, time.time())
        )
        await db.commit()


async def get_analytics(db_path: str = DB_PATH) -> Dict:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT COUNT(*) as n FROM search_logs") as c:
            total_searches = (await c.fetchone())["n"]

        async with db.execute("SELECT COUNT(*) as n FROM clicks") as c:
            total_clicks = (await c.fetchone())["n"]

        async with db.execute(
            "SELECT AVG(alpha) as avg_alpha FROM search_logs WHERE alpha IS NOT NULL"
        ) as c:
            avg_alpha = (await c.fetchone())["avg_alpha"] or 0

        async with db.execute(
            "SELECT AVG(latency_ms) as avg_lat FROM search_logs"
        ) as c:
            avg_latency = (await c.fetchone())["avg_lat"] or 0

        async with db.execute(
            "SELECT query, COUNT(*) as cnt FROM search_logs"
            " GROUP BY query ORDER BY cnt DESC LIMIT 10"
        ) as c:
            top_queries = [dict(r) for r in await c.fetchall()]

        return {
            "total_searches": total_searches,
            "total_clicks": total_clicks,
            "avg_alpha": round(avg_alpha, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "top_queries": top_queries,
        }
```

---

## Step 6 — FastAPI application

### Create `api/schemas.py`:

```python
from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query:   str             = Field(..., min_length=1, max_length=500)
    user_id: str             = Field(default="anonymous")
    method:  str             = Field(default="hybrid")  # hybrid | dense | bm25
    top_k:   int             = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    rank:           int
    doc_id:         str
    title:          str
    snippet:        str
    url:            str
    colbert_score:  float
    semantic_score: float
    user_score:     float
    final_score:    float
    personalized:   bool
    source:         str


class SearchResponse(BaseModel):
    query:          str
    user_id:        str
    method:         str
    results:        List[SearchResult]
    alpha:          float
    alpha_label:    str
    colbert_variance: float
    has_user_profile: bool
    user_click_count: int
    latency_ms:     float
    stage1_count:   int
    stage2_count:   int
    token_weights:  List[dict]


class ClickRequest(BaseModel):
    user_id: str
    doc_id:  str
    doc_idx: int = -1
    query:   str = ""


class ClickResponse(BaseModel):
    status:      str
    user_id:     str
    click_count: int


class AnalyticsResponse(BaseModel):
    total_searches:  int
    total_clicks:    int
    avg_alpha:       float
    avg_latency_ms:  float
    top_queries:     List[dict]
```

### Create `api/main.py`:

```python
"""
FastAPI application entry point.
Loads all models at startup — not per request.
"""

import os
import pickle
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.retrieval.biencoder import BiEncoder
from src.retrieval.colbert_reranker import ColBERTReranker
from src.personalization.user_model import UserInterestModel
from src.database.user_store import init_db

load_dotenv()

# ── Global state ─────────────────────────────────────────────────────────────
biencoder   = None
colbert     = None
user_model  = None
bm25        = None
docs        = None
embeddings  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global biencoder, colbert, user_model, bm25, docs, embeddings

    print("Initializing models and indexes...")

    index_path  = os.getenv("INDEX_PATH",  "indexes/faiss_index.bin")
    bm25_path   = os.getenv("BM25_PATH",   "indexes/bm25_index.pkl")
    corpus_path = os.getenv("CORPUS_PATH", "indexes/corpus.pkl")

    biencoder = BiEncoder(model_name=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
    biencoder.load_index(index_path, corpus_path)

    docs       = biencoder.docs
    embeddings = biencoder.embeddings

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    colbert = ColBERTReranker(
        model_name=os.getenv("COLBERT_MODEL", "bert-base-uncased")
    )

    user_model = UserInterestModel(
        decay=float(os.getenv("EMA_DECAY", 0.85))
    )

    await init_db(os.getenv("DB_PATH", "data/userdata.db"))

    print(f"Ready. Corpus: {len(docs)} docs")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Personalized Dense IR System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────
from api.routes import search, feedback, analytics

app.include_router(search.router,    prefix="/api")
app.include_router(feedback.router,  prefix="/api")
app.include_router(analytics.router, prefix="/api")

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {
        "status":      "ok",
        "corpus_size": len(docs) if docs else 0,
        "model":       os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"),
    }


# ── Expose shared state to routes ─────────────────────────────────────────────
def get_state():
    return {
        "biencoder":  biencoder,
        "colbert":    colbert,
        "user_model": user_model,
        "bm25":       bm25,
        "docs":       docs,
        "embeddings": embeddings,
    }
```

### Create `api/routes/search.py`:

```python
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
from rank_bm25 import BM25Okapi
import numpy as np

from api.schemas import SearchRequest, SearchResponse, SearchResult
from api.main import get_state
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
                req.user_id, profile["interest_vector"]
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
```

### Create `api/routes/feedback.py`:

```python
"""
POST /api/click — logs a click and updates the user interest model live.
"""

from fastapi import APIRouter
from api.schemas import ClickRequest, ClickResponse
from api.main import get_state
from src.database.user_store import log_click, save_user_profile

router = APIRouter()


@router.post("/click", response_model=ClickResponse)
async def record_click(req: ClickRequest):
    state      = get_state()
    user_model = state["user_model"]
    embeddings = state["embeddings"]

    # Update in-memory EMA model
    idx = req.doc_idx
    if 0 <= idx < len(embeddings):
        user_model.update(req.user_id, embeddings[idx])

    # Persist click to DB
    await log_click(
        user_id=req.user_id,
        doc_id=req.doc_id,
        doc_idx=req.doc_idx,
        query=req.query,
    )

    # Persist updated interest vector
    profile = user_model.get_profile(req.user_id)
    if profile is not None:
        await save_user_profile(
            user_id=req.user_id,
            interest_vector=profile.tolist(),
            click_count=user_model.get_click_count(req.user_id),
        )

    return ClickResponse(
        status="ok",
        user_id=req.user_id,
        click_count=user_model.get_click_count(req.user_id),
    )
```

### Create `api/routes/analytics.py`:

```python
"""
GET /api/analytics — system-wide analytics for the dashboard charts.
GET /api/user/{user_id}/profile — per-user profile info.
GET /api/user/{user_id}/history — click history.
"""

from fastapi import APIRouter
from api.main import get_state
from src.database.user_store import get_analytics, get_click_history, load_user_profile

router = APIRouter()


@router.get("/analytics")
async def analytics():
    return await get_analytics()


@router.get("/user/{user_id}/profile")
async def user_profile(user_id: str):
    state      = get_state()
    user_model = state["user_model"]

    return {
        "user_id":     user_id,
        "has_profile": user_model.has_profile(user_id),
        "click_count": user_model.get_click_count(user_id),
    }


@router.get("/user/{user_id}/history")
async def user_history(user_id: str, limit: int = 20):
    history = await get_click_history(user_id, limit)
    return {"user_id": user_id, "history": history}
```

---

## Step 7 — Evaluation script

### Create `src/evaluation/metrics.py`:

```python
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
```

### Create `src/evaluation/evaluate.py`:

```python
"""
Full evaluation script.
Compares: BM25 | Dense only | Dense+ColBERT | Dense+ColBERT+Personalization

Run: python -m src.evaluation.evaluate
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.biencoder import BiEncoder
from src.retrieval.colbert_reranker import ColBERTReranker
from src.personalization.user_model import UserInterestModel
from src.personalization.hybrid_ranker import hybrid_rank
from src.evaluation.metrics import precision_at_k, mrr, ndcg_at_k

N_EVAL = 50
K = 5


def main():
    print("Loading indexes...")
    biencoder = BiEncoder()
    biencoder.load_index(
        os.getenv("INDEX_PATH", "indexes/faiss_index.bin"),
        os.getenv("CORPUS_PATH", "indexes/corpus.pkl"),
    )
    docs       = biencoder.docs
    embeddings = biencoder.embeddings

    with open(os.getenv("BM25_PATH", "indexes/bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)

    colbert    = ColBERTReranker()
    user_model = UserInterestModel()

    print("Loading eval queries and qrels...")
    queries_df = pd.read_csv("data/dev_queries.tsv", sep="\t")
    qrels_df   = pd.read_csv("data/dev_qrels.tsv", sep="\t")
    qrels      = qrels_df.groupby("qid")["docid"].apply(set).to_dict()

    eval_qids = [q for q in list(qrels.keys())[:N_EVAL] if q in set(queries_df["qid"])]

    # warm user: simulate clicks on first 5 retrieved docs before eval
    warm_user = "warm_eval_user"
    warmup_q  = queries_df[queries_df["qid"] == eval_qids[0]].iloc[0]["query"]
    warmup    = biencoder.retrieve(warmup_q, top_k=5)
    for doc in warmup:
        idx = doc.get("local_idx", -1)
        if 0 <= idx < len(embeddings):
            user_model.update(warm_user, embeddings[idx])

    results = {
        "BM25":                      {"P@5": [], "MRR": [], "NDCG@5": []},
        "Dense (bi-encoder)":        {"P@5": [], "MRR": [], "NDCG@5": []},
        "Dense + ColBERT":           {"P@5": [], "MRR": [], "NDCG@5": []},
        "Dense + ColBERT + User":    {"P@5": [], "MRR": [], "NDCG@5": []},
    }

    print(f"Evaluating {len(eval_qids)} queries...")
    for qid in tqdm(eval_qids):
        row   = queries_df[queries_df["qid"] == qid]
        if row.empty: continue
        query = row.iloc[0]["query"]
        rel   = qrels.get(qid, set())
        rel   = {str(r) for r in rel}
        if not rel: continue

        # BM25
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_top    = np.argsort(bm25_scores)[::-1][:K]
        bm25_ids    = [str(docs[i]["docid"]) for i in bm25_top]
        results["BM25"]["P@5"].append(precision_at_k(bm25_ids, rel, K))
        results["BM25"]["MRR"].append(mrr(bm25_ids, rel))
        results["BM25"]["NDCG@5"].append(ndcg_at_k(bm25_ids, rel, K))

        # Dense only
        dense_cands = biencoder.retrieve(query, top_k=100)
        dense_ids   = [str(d["docid"]) for d in dense_cands[:K]]
        results["Dense (bi-encoder)"]["P@5"].append(precision_at_k(dense_ids, rel, K))
        results["Dense (bi-encoder)"]["MRR"].append(mrr(dense_ids, rel))
        results["Dense (bi-encoder)"]["NDCG@5"].append(ndcg_at_k(dense_ids, rel, K))

        # Dense + ColBERT
        reranked, col_scores = colbert.rerank(query, dense_cands, top_k=20)
        col_ids = [str(d["docid"]) for d in reranked[:K]]
        results["Dense + ColBERT"]["P@5"].append(precision_at_k(col_ids, rel, K))
        results["Dense + ColBERT"]["MRR"].append(mrr(col_ids, rel))
        results["Dense + ColBERT"]["NDCG@5"].append(ndcg_at_k(col_ids, rel, K))

        # Dense + ColBERT + User
        doc_indices = [d.get("local_idx", -1) for d in reranked]
        ranked, _   = hybrid_rank(
            docs=reranked, colbert_scores=col_scores,
            user_model=user_model, user_id=warm_user,
            doc_embeddings=embeddings, doc_indices=doc_indices,
            top_k=K,
        )
        user_ids = [str(d["docid"]) for d in ranked]
        results["Dense + ColBERT + User"]["P@5"].append(precision_at_k(user_ids, rel, K))
        results["Dense + ColBERT + User"]["MRR"].append(mrr(user_ids, rel))
        results["Dense + ColBERT + User"]["NDCG@5"].append(ndcg_at_k(user_ids, rel, K))

    print("\n" + "="*65)
    print(f"{'Method':<30} {'P@5':>8} {'MRR':>8} {'NDCG@5':>8}")
    print("="*65)
    for method, m in results.items():
        p5    = np.mean(m["P@5"])
        mrr_s = np.mean(m["MRR"])
        ndcg  = np.mean(m["NDCG@5"])
        print(f"{method:<30} {p5:>8.4f} {mrr_s:>8.4f} {ndcg:>8.4f}")
    print("="*65)
    print("\nCopy this table into your presentation.")


if __name__ == "__main__":
    main()
```

---

## Step 8 — Index builder script

### Create `scripts/build_index.py`:

```python
"""
One-time script to build FAISS + BM25 indexes from the corpus.
Run AFTER data/download_msmarco.py.

Usage: python scripts/build_index.py
       python scripts/build_index.py --corpus data/wiki_corpus.csv  (Wikipedia)
"""

import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.index_builder import build_indexes

parser = argparse.ArgumentParser()
parser.add_argument(
    "--corpus", default=os.getenv("CORPUS_PATH_CSV", "data/corpus.csv")
)
args = parser.parse_args()

build_indexes(
    corpus_path   = args.corpus,
    index_path    = os.getenv("INDEX_PATH",  "indexes/faiss_index.bin"),
    bm25_path     = os.getenv("BM25_PATH",   "indexes/bm25_index.pkl"),
    corpus_pkl_path = "indexes/corpus.pkl",
    model_name    = os.getenv("MODEL_NAME",  "all-MiniLM-L6-v2"),
)
```

### Create `scripts/seed_users.py`:

```python
"""
Seeds 3 demo users with realistic click histories so personalization
is visible immediately during the live demo.
Run AFTER the API has been started at least once (so the DB exists).

Usage: python scripts/seed_users.py
"""

import asyncio
import pickle
import numpy as np
from src.personalization.user_model import UserInterestModel
from src.database.user_store import save_user_profile, init_db, log_click
from dotenv import load_dotenv
import os

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
    await init_db(DB_PATH)

    with open(CORPUS_PATH, "rb") as f:
        data = pickle.load(f)
    docs       = data["docs"]
    embeddings = data["embeddings"]

    from src.retrieval.biencoder import BiEncoder
    encoder = BiEncoder()

    for user_id, interest_queries in USERS.items():
        model = UserInterestModel(decay=0.85)
        print(f"Seeding {user_id}...")

        for query in interest_queries:
            q_emb = encoder.model.encode(
                [query], normalize_embeddings=True
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
                )

        profile = model.get_profile(user_id)
        if profile is not None:
            await save_user_profile(
                user_id=user_id,
                interest_vector=profile.tolist(),
                click_count=model.get_click_count(user_id),
            )
        print(f"  {user_id}: {model.get_click_count(user_id)} simulated clicks")

    print("Seeding complete.")


if __name__ == "__main__":
    asyncio.run(seed())
```

---

## Step 9 — Frontend

### Create `frontend/index.html`:

Full working UI — connected to the real API. No mocked data anywhere.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personalized Dense IR</title>
<link rel="stylesheet" href="css/style.css">
</head>
<body>
<header>
  <h1>Personalized Dense IR System</h1>
  <p>Two-stage retrieval · ColBERT re-ranking · Adaptive user model</p>
</header>

<div class="layout">
  <div class="sidebar">
    <div class="card">
      <label class="field-label">User</label>
      <select id="user-select">
        <option value="anonymous">New user (no history)</option>
        <option value="tech_user">Tech user</option>
        <option value="science_user">Science user</option>
        <option value="medical_user">Medical user</option>
      </select>

      <label class="field-label" style="margin-top:16px">Method</label>
      <div class="method-btns">
        <button class="method-btn" data-method="bm25">BM25</button>
        <button class="method-btn" data-method="dense">Dense</button>
        <button class="method-btn active" data-method="hybrid">Hybrid</button>
      </div>

      <label class="field-label" style="margin-top:16px">Results</label>
      <select id="topk-select">
        <option value="5">5</option>
        <option value="10" selected>10</option>
        <option value="20">20</option>
      </select>
    </div>

    <div class="card" id="alpha-card" style="display:none">
      <div class="field-label">Alpha (α)</div>
      <div class="alpha-display" id="alpha-val">—</div>
      <div class="alpha-bar-bg"><div class="alpha-bar" id="alpha-bar"></div></div>
      <div class="alpha-label" id="alpha-label"></div>
    </div>

    <div class="card" id="pipeline-card" style="display:none">
      <div class="field-label">Pipeline</div>
      <div class="pipeline-row">
        <span class="stage-label">Stage 1 (FAISS)</span>
        <span class="stage-val" id="s1-count">—</span>
      </div>
      <div class="pipeline-row">
        <span class="stage-label">Stage 2 (ColBERT)</span>
        <span class="stage-val" id="s2-count">—</span>
      </div>
      <div class="pipeline-row">
        <span class="stage-label">Latency</span>
        <span class="stage-val" id="latency-val">—</span>
      </div>
    </div>

    <div class="card" id="profile-card">
      <div class="field-label">User profile</div>
      <div id="profile-info" style="font-size:13px;color:var(--text-muted)">No clicks yet</div>
    </div>
  </div>

  <div class="main">
    <div class="search-bar">
      <input id="query-input" type="text"
             placeholder="Enter a query — e.g. 'effects of climate change on ocean biodiversity'"/>
      <button id="search-btn" onclick="doSearch()">Search</button>
    </div>

    <div id="status" class="status"></div>
    <div id="results"></div>
  </div>
</div>

<script src="js/api.js"></script>
<script src="js/search.js"></script>
</body>
</html>
```

### Create `frontend/dashboard.html`:

Live visualization dashboard — all charts fed from real API data.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IR Dashboard</title>
<link rel="stylesheet" href="css/style.css">
</head>
<body>
<header>
  <h1>Live Analytics Dashboard</h1>
  <p>All charts update in real time from the backend</p>
  <nav><a href="index.html">← Back to search</a></nav>
</header>

<div class="dashboard-grid">
  <div class="metric-card"><div class="metric-label">Total searches</div><div class="metric-val" id="d-searches">—</div></div>
  <div class="metric-card"><div class="metric-label">Total clicks</div><div class="metric-val" id="d-clicks">—</div></div>
  <div class="metric-card"><div class="metric-label">Avg alpha</div><div class="metric-val" id="d-alpha">—</div></div>
  <div class="metric-card"><div class="metric-label">Avg latency</div><div class="metric-val" id="d-latency">—</div></div>
</div>

<div class="charts-grid">
  <div class="chart-card" id="score-card">
    <h3>ColBERT score distribution</h3>
    <p>MaxSim scores for last query</p>
    <div style="position:relative;height:200px"><canvas id="scoreChart"></canvas></div>
  </div>

  <div class="chart-card" id="blend-card">
    <h3>Semantic vs user blend</h3>
    <p>Dynamic alpha per query</p>
    <div style="position:relative;height:200px"><canvas id="blendChart"></canvas></div>
  </div>

  <div class="chart-card full" id="rank-card">
    <h3>Score comparison — top 10 results</h3>
    <p>Semantic score vs final hybrid score</p>
    <div style="position:relative;height:220px"><canvas id="rankChart"></canvas></div>
  </div>

  <div class="chart-card" id="token-card">
    <h3>Query token weights</h3>
    <p>ColBERT MaxSim per token</p>
    <div id="token-grid" class="token-grid"></div>
  </div>

  <div class="chart-card" id="queries-card">
    <h3>Top queries</h3>
    <p>Most frequent searches</p>
    <div id="top-queries"></div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="js/api.js"></script>
<script src="js/charts.js"></script>
</body>
</html>
```

### Create `frontend/css/style.css`:

Complete stylesheet — clean, professional, works in both light and dark environments.

```css
:root {
  --bg:        #ffffff;
  --bg2:       #f6f6f4;
  --border:    rgba(0,0,0,0.1);
  --text:      #1a1a1a;
  --text-muted:#666;
  --purple:    #534AB7;
  --purple-l:  #EEEDFE;
  --teal:      #1D9E75;
  --teal-l:    #E1F5EE;
  --amber:     #BA7517;
  --coral:     #D85A30;
  --radius:    10px;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#1a1a1a; --bg2:#242424; --border:rgba(255,255,255,0.1); --text:#e0e0e0; --text-muted:#999; --purple-l:#26215C; --teal-l:#04342C; }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg2); color: var(--text); min-height: 100vh; }
header { background: var(--bg); border-bottom: 1px solid var(--border); padding: 16px 32px; }
header h1 { font-size: 18px; font-weight: 600; }
header p  { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
header nav { margin-top: 8px; }
header nav a { font-size: 13px; color: var(--purple); text-decoration: none; }
.layout { display: grid; grid-template-columns: 240px 1fr; gap: 20px; padding: 24px 32px; max-width: 1100px; margin: 0 auto; }
.card { background: var(--bg); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 16px; margin-bottom: 14px; }
.field-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; display: block; }
select, input[type=text] { width: 100%; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg2); color: var(--text); font-size: 14px; }
.method-btns { display: flex; gap: 6px; }
.method-btn { flex: 1; padding: 7px 0; border: 1px solid var(--border); border-radius: 6px; background: var(--bg2); color: var(--text-muted); font-size: 12px; cursor: pointer; }
.method-btn.active { border-color: var(--purple); background: var(--purple-l); color: var(--purple); font-weight: 500; }
.alpha-display { font-size: 28px; font-weight: 500; color: var(--text); margin: 4px 0; }
.alpha-bar-bg { height: 6px; background: var(--bg2); border-radius: 3px; overflow: hidden; margin: 6px 0; }
.alpha-bar { height: 6px; background: var(--purple); border-radius: 3px; transition: width .5s ease; }
.alpha-label { font-size: 11px; color: var(--text-muted); line-height: 1.4; }
.pipeline-row { display: flex; justify-content: space-between; font-size: 13px; padding: 4px 0; border-bottom: 0.5px solid var(--border); }
.pipeline-row:last-child { border-bottom: none; }
.stage-label { color: var(--text-muted); }
.stage-val { font-weight: 500; }
.search-bar { display: flex; gap: 10px; margin-bottom: 16px; }
.search-bar input { flex: 1; padding: 11px 14px; border: 1.5px solid var(--border); border-radius: 8px; font-size: 15px; background: var(--bg); color: var(--text); }
.search-bar input:focus { outline: none; border-color: var(--purple); }
#search-btn { padding: 11px 22px; background: var(--purple); color: #fff; border: none; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; }
#search-btn:hover { background: #3C3489; }
.status { font-size: 13px; color: var(--text-muted); margin-bottom: 12px; min-height: 18px; }
.result-card { background: var(--bg); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 14px 18px; margin-bottom: 10px; cursor: pointer; transition: border-color .15s; }
.result-card:hover { border-color: var(--purple); }
.result-card.clicked { border-color: var(--teal); background: var(--teal-l); }
.result-top { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px; }
.result-rank { font-size: 12px; color: var(--text-muted); font-weight: 600; min-width: 22px; }
.result-title { font-size: 15px; font-weight: 500; flex: 1; }
.result-snippet { font-size: 13px; color: var(--text-muted); line-height: 1.6; }
.result-meta { font-size: 11px; color: var(--text-muted); margin-top: 6px; display: flex; gap: 12px; }
.badges { display: flex; gap: 5px; flex-wrap: wrap; flex-shrink: 0; }
.badge { font-size: 10px; padding: 2px 8px; border-radius: 12px; font-weight: 600; letter-spacing: .2px; }
.badge-colbert { background: var(--purple-l); color: var(--purple); }
.badge-bm25 { background: #FAEEDA; color: #633806; }
.badge-personalized { background: var(--teal-l); color: #085041; }
.score-bars { display: flex; gap: 6px; align-items: center; margin-top: 6px; }
.score-bar-label { font-size: 10px; color: var(--text-muted); min-width: 50px; }
.score-bar-bg { flex: 1; height: 4px; background: var(--bg2); border-radius: 2px; overflow: hidden; }
.score-bar-fill { height: 4px; border-radius: 2px; }
.dashboard-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; padding: 24px 32px 0; max-width: 1100px; margin: 0 auto; }
.metric-card { background: var(--bg); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 14px; }
.metric-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
.metric-val { font-size: 26px; font-weight: 500; }
.charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 16px 32px 32px; max-width: 1100px; margin: 0 auto; }
.chart-card { background: var(--bg); border: 0.5px solid var(--border); border-radius: var(--radius); padding: 16px; }
.chart-card.full { grid-column: 1 / -1; }
.chart-card h3 { font-size: 14px; font-weight: 500; margin-bottom: 2px; }
.chart-card p { font-size: 12px; color: var(--text-muted); margin-bottom: 12px; }
.token-grid { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.token { padding: 3px 10px; border-radius: 12px; font-size: 13px; font-weight: 500; }
#top-queries { font-size: 13px; }
.query-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 0.5px solid var(--border); color: var(--text-muted); }
.query-row:last-child { border-bottom: none; }
```

### Create `frontend/js/api.js`:

```javascript
const API_BASE = "";  // same origin — FastAPI serves the frontend

async function apiSearch(query, userId, method, topK) {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query, user_id: userId, method, top_k: topK
    }),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.status}`);
  return res.json();
}

async function apiClick(userId, docId, docIdx, query) {
  const res = await fetch(`${API_BASE}/api/click`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId, doc_id: docId, doc_idx: docIdx, query
    }),
  });
  if (!res.ok) throw new Error(`Click log failed: ${res.status}`);
  return res.json();
}

async function apiUserProfile(userId) {
  const res = await fetch(`${API_BASE}/api/user/${userId}/profile`);
  return res.json();
}

async function apiUserHistory(userId) {
  const res = await fetch(`${API_BASE}/api/user/${userId}/history`);
  return res.json();
}

async function apiAnalytics() {
  const res = await fetch(`${API_BASE}/api/analytics`);
  return res.json();
}
```

### Create `frontend/js/search.js`:

```javascript
let currentMethod = "hybrid";
let lastQuery = "";
let lastResponse = null;
const clickedDocs = new Set();

// Method toggle
document.querySelectorAll(".method-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".method-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentMethod = btn.dataset.method;
  });
});

// User change — refresh profile
document.getElementById("user-select").addEventListener("change", async function () {
  clickedDocs.clear();
  await updateProfileCard(this.value);
});

async function doSearch() {
  const query  = document.getElementById("query-input").value.trim();
  const userId = document.getElementById("user-select").value;
  const topK   = parseInt(document.getElementById("topk-select").value);
  if (!query) return;

  lastQuery = query;
  document.getElementById("status").textContent = "Searching...";
  document.getElementById("results").innerHTML = "";

  try {
    const data = await apiSearch(query, userId, currentMethod, topK);
    lastResponse = data;
    renderResults(data, userId);
    updateSidebar(data);
    // dispatch event for dashboard if open
    window.dispatchEvent(new CustomEvent("searchComplete", { detail: data }));
  } catch (e) {
    document.getElementById("status").textContent =
      "Error: make sure the API is running. " + e.message;
  }
}

function renderResults(data, userId) {
  const { results, latency_ms, method, alpha } = data;
  document.getElementById("status").textContent =
    `${results.length} results · ${method} · ${latency_ms}ms · α=${alpha.toFixed(2)}`;

  const container = document.getElementById("results");
  container.innerHTML = "";

  results.forEach(r => {
    const wasClicked = clickedDocs.has(r.doc_id);
    const card = document.createElement("div");
    card.className = "result-card" + (wasClicked ? " clicked" : "");

    const colBadge = `<span class="badge badge-colbert">${method === "bm25" ? "BM25" : "ColBERT"}</span>`;
    const persBadge = r.personalized
      ? `<span class="badge badge-personalized">personalized</span>` : "";

    card.innerHTML = `
      <div class="result-top">
        <span class="result-rank">#${r.rank}</span>
        <div style="flex:1">
          <div class="result-title">${r.title || "(no title)"}</div>
        </div>
        <div class="badges">${colBadge}${persBadge}</div>
      </div>
      <div class="result-snippet">${r.snippet}</div>
      <div class="score-bars">
        <span class="score-bar-label">Semantic</span>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:${(r.semantic_score*100).toFixed(0)}%;background:#7F77DD"></div></div>
        <span class="score-bar-label" style="min-width:30px;text-align:right">${r.semantic_score.toFixed(2)}</span>
      </div>
      <div class="score-bars">
        <span class="score-bar-label">Hybrid</span>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:${(r.final_score*100).toFixed(0)}%;background:#1D9E75"></div></div>
        <span class="score-bar-label" style="min-width:30px;text-align:right">${r.final_score.toFixed(2)}</span>
      </div>
      <div class="result-meta">
        <span>user score: ${r.user_score.toFixed(3)}</span>
        ${r.url ? `<a href="${r.url}" target="_blank" style="color:#534AB7">${new URL(r.url).hostname}</a>` : ""}
      </div>`;

    card.addEventListener("click", () => handleClick(r, userId, card));
    container.appendChild(card);
  });
}

async function handleClick(result, userId, card) {
  if (clickedDocs.has(result.doc_id)) return;
  clickedDocs.add(result.doc_id);
  card.classList.add("clicked");

  try {
    const resp = await apiClick(userId, result.doc_id, result.rank - 1, lastQuery);
    await updateProfileCard(userId);
    document.getElementById("profile-info").textContent =
      `${resp.click_count} click${resp.click_count !== 1 ? "s" : ""} logged`;
  } catch (e) {
    console.error("Click error:", e);
  }
}

function updateSidebar(data) {
  document.getElementById("alpha-card").style.display = "block";
  document.getElementById("pipeline-card").style.display = "block";

  const alphaPct = ((data.alpha - 0.3) / 0.6 * 100).toFixed(0);
  document.getElementById("alpha-val").textContent = data.alpha.toFixed(2);
  document.getElementById("alpha-bar").style.width = alphaPct + "%";
  document.getElementById("alpha-label").textContent = data.alpha_label;

  document.getElementById("s1-count").textContent = data.stage1_count + " docs";
  document.getElementById("s2-count").textContent = data.stage2_count + " docs";
  document.getElementById("latency-val").textContent = data.latency_ms + "ms";
}

async function updateProfileCard(userId) {
  try {
    const p = await apiUserProfile(userId);
    document.getElementById("profile-info").textContent =
      p.has_profile
        ? `${p.click_count} click${p.click_count !== 1 ? "s" : ""} · profile active`
        : "No clicks yet";
  } catch (e) {}
}

// Enter key
document.getElementById("query-input").addEventListener("keydown", e => {
  if (e.key === "Enter") doSearch();
});

// Init
updateProfileCard(document.getElementById("user-select").value);
```

### Create `frontend/js/charts.js`:

```javascript
// Dashboard charts — all data from real API
let scoreChart, blendChart, rankChart;

async function loadAnalytics() {
  try {
    const data = await apiAnalytics();
    document.getElementById("d-searches").textContent = data.total_searches;
    document.getElementById("d-clicks").textContent   = data.total_clicks;
    document.getElementById("d-alpha").textContent    = data.avg_alpha.toFixed(2);
    document.getElementById("d-latency").textContent  = data.avg_latency_ms + "ms";

    const qList = document.getElementById("top-queries");
    qList.innerHTML = (data.top_queries || []).map(q =>
      `<div class="query-row"><span>${q.query}</span><span>${q.cnt}×</span></div>`
    ).join("");
  } catch (e) {
    console.error("Analytics error:", e);
  }
}

function renderScoreChart(colbertScores) {
  if (scoreChart) scoreChart.destroy();
  const ctx = document.getElementById("scoreChart").getContext("2d");
  scoreChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: colbertScores.map((_, i) => i + 1),
      datasets: [{
        label: "ColBERT MaxSim",
        data: colbertScores.map(s => parseFloat(s.toFixed(4))),
        borderColor: "#534AB7",
        backgroundColor: "#EEEDFE",
        borderWidth: 2, pointRadius: 2, fill: true, tension: 0.3,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "rank", font: { size: 10 } } },
        y: { min: 0, ticks: { font: { size: 10 } } }
      }
    }
  });
}

function renderBlendChart(alpha) {
  if (blendChart) blendChart.destroy();
  const ctx = document.getElementById("blendChart").getContext("2d");
  blendChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Semantic (ColBERT)", "User interest"],
      datasets: [{
        data: [
          parseFloat((alpha * 100).toFixed(1)),
          parseFloat(((1 - alpha) * 100).toFixed(1))
        ],
        backgroundColor: ["#534AB7", "#1D9E75"],
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "60%",
      plugins: { legend: { display: true, position: "bottom",
        labels: { font: { size: 11 }, boxWidth: 12 } } }
    }
  });
}

function renderRankChart(results) {
  if (rankChart) rankChart.destroy();
  const ctx = document.getElementById("rankChart").getContext("2d");
  rankChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: results.map(r => `#${r.rank} ${(r.title || "").split(" ").slice(0, 3).join(" ")}`),
      datasets: [
        {
          label: "Semantic",
          data: results.map(r => parseFloat(r.semantic_score.toFixed(4))),
          backgroundColor: "#534AB799",
          borderColor: "#534AB7", borderWidth: 1,
        },
        {
          label: "Hybrid final",
          data: results.map(r => parseFloat(r.final_score.toFixed(4))),
          backgroundColor: "#1D9E7599",
          borderColor: "#1D9E75", borderWidth: 1,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { font: { size: 11 }, boxWidth: 12 } } },
      scales: {
        x: { ticks: { font: { size: 9 }, maxRotation: 35, autoSkip: false } },
        y: { min: 0, max: 1, ticks: { font: { size: 10 }, stepSize: 0.2 } }
      }
    }
  });
}

function renderTokenGrid(tokenWeights) {
  const grid = document.getElementById("token-grid");
  if (!tokenWeights || tokenWeights.length === 0) {
    grid.innerHTML = "<span style='color:#888;font-size:13px'>Run a search first</span>";
    return;
  }
  const max = tokenWeights[0].weight;
  grid.innerHTML = tokenWeights.map(t => {
    const norm = t.weight / (max || 1);
    const bg   = norm > 0.7 ? "#EEEDFE" : norm > 0.4 ? "#E1F5EE" : "#F1EFE8";
    const col  = norm > 0.7 ? "#3C3489" : norm > 0.4 ? "#085041" : "#5F5E5A";
    const size = Math.round(11 + norm * 4);
    return `<span class="token" style="background:${bg};color:${col};font-size:${size}px">
      ${t.token} <span style="opacity:.6;font-size:10px">${t.weight.toFixed(2)}</span>
    </span>`;
  }).join("");
}

// Listen for searches from the main page (if both pages are open)
window.addEventListener("searchComplete", e => {
  const data = e.detail;
  renderScoreChart(data.results.map(r => r.colbert_score));
  renderBlendChart(data.alpha);
  renderRankChart(data.results);
  renderTokenGrid(data.token_weights || []);
  loadAnalytics();
});

// Auto-refresh analytics every 10 seconds
loadAnalytics();
setInterval(loadAnalytics, 10000);
```

---

## Step 10 — README

### Create `README.md`:

```markdown
# Personalized Dense Information Retrieval System

A two-stage retrieval system with ColBERT re-ranking and adaptive user personalization.

## Architecture

1. **Stage 1**: MiniLM bi-encoder + FAISS → top-100 candidates
2. **Stage 2**: ColBERT MaxSim re-ranker → top-20
3. **Confidence estimator**: score variance → dynamic alpha per query
4. **User model**: EMA over click embeddings → live interest profile
5. **Hybrid ranker**: `α × semantic + (1-α) × user_score`

## Setup

### Requirements
- Python 3.10+
- GPU recommended (Colab Pro / Kaggle)

### Install
```bash
bash scripts/setup.sh
source venv/bin/activate
```

### Download data (run in order)
```bash
python data/download_msmarco.py       # MS MARCO corpus + eval data
python data/download_wikipedia.py     # Wikipedia corpus (optional, better for demos)
```

### Build indexes
```bash
python scripts/build_index.py                          # MS MARCO
python scripts/build_index.py --corpus data/wiki_corpus.csv  # Wikipedia
```

### Start the system
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Seed demo users (optional but recommended)
```bash
python scripts/seed_users.py
```

### Open the UI
Open `http://localhost:8000` in your browser.
Open `http://localhost:8000/dashboard.html` for the live analytics dashboard.

### Run evaluation
```bash
python -m src.evaluation.evaluate
```

## Reproducing on Colab / Kaggle

```python
# 1. Clone repo
!git clone https://github.com/YOUR_REPO/ir-project.git
%cd ir-project

# 2. Install
!pip install -r requirements.txt -q

# 3. Download data
!python data/download_msmarco.py

# 4. Build index
!python scripts/build_index.py

# 5. Start API in background
import subprocess
proc = subprocess.Popen(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

# 6. Expose with ngrok (Colab) or use Kaggle's port forwarding
```

## Project structure
See directory tree in AGENT_INSTRUCTIONS.md.

## Team
- Akash Choudhary (230797)
- Aryan (230836)
- Vaibhaav Tiwari (230599)
```

---

## Final agent checklist

After writing all files, verify the following before stopping:

1. Every `__init__.py` exists in every package folder
2. `api/main.py` imports correctly resolve — check all relative imports
3. `get_state()` in `api/main.py` is importable from all route files without circular imports — use a module-level dict if needed
4. `frontend/index.html` loads `js/api.js` before `js/search.js`
5. `frontend/dashboard.html` loads Chart.js CDN before `js/charts.js`
6. All file paths in `.env.example` are relative to project root
7. `scripts/setup.sh` is executable (`chmod +x scripts/setup.sh`)
8. `.gitignore` excludes all large files (`indexes/`, `data/*.csv`, `data/*.db`, `models/`)
9. `data/README.md` clearly explains how to download data with exact commands
10. The API serves the frontend from `/` — verify `StaticFiles` mount is last in `main.py`

---

## Commands the user needs to run (in order)

```bash
# 1. Setup
bash scripts/setup.sh
source venv/bin/activate

# 2. Download data
python data/download_msmarco.py

# 3. Build indexes  (~5 min on GPU, ~15 min on CPU)
python scripts/build_index.py

# 4. Start API
uvicorn api.main:app --reload

# 5. Seed demo users (in a second terminal)
python scripts/seed_users.py

# 6. Run evaluation
python -m src.evaluation.evaluate
```

That is the complete list. No other terminal commands are needed.

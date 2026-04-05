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

On Windows (PowerShell), create a venv and install manually:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
mkdir -Force indexes, data, models
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
!git clone https://github.com/YOUR_REPO/Dense-IR-System.git
%cd Dense-IR-System

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

See `AGENT_INSTRUCTIONS.md` for the full directory tree and implementation notes.

## Team

- Akash Choudhary (230797)
- Aryan (230836)
- Vaibhaav Tiwari (230599)

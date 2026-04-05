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

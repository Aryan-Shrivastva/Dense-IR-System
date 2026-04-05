# Data Directory

## Setup (run in order)

### Step 1 — MS MARCO (required for evaluation)
```bash
python data/download_msmarco.py
```
Downloads via HuggingFace. No manual download needed.
Produces: `corpus.csv`, `dev_queries.tsv`, `dev_qrels.tsv`

### Step 2 — Wikipedia (recommended for live demo)

**Option A — Download fresh from HuggingFace (default):**
```bash
python data/download_wikipedia.py
```

**Option B — Load from a pre-downloaded disk copy (GPU server):**

If you already have Wikipedia saved on a GPU/remote server via `save_to_disk()` (a folder containing `state.json` and `.arrow` files), run the script with `--disk-path`:
```bash
# Run on the server inside the project directory
python data/download_wikipedia.py --disk-path ~/wiki_project/wikipedia_full

# Or set the env var in .env instead:
# WIKI_DISK_PATH=/home/user/wiki_project/wikipedia_full
# Then just run:
python data/download_wikipedia.py
```

Control how many articles are processed (default 20 000):
```bash
python data/download_wikipedia.py --disk-path ~/wiki_project/wikipedia_full --size 50000
```

Both options produce: `wiki_corpus.csv`

---

## Files (generated — not committed to git)

| File | Description |
|---|---|
| `corpus.csv` | MS MARCO passages (50k) — used for indexing + evaluation |
| `wiki_corpus.csv` | Wikipedia chunks — used for live demo queries |
| `dev_queries.tsv` | MS MARCO dev queries for evaluation |
| `dev_qrels.tsv` | Relevance labels for evaluation |
| `userdata.db` | SQLite database for user profiles and clicks |

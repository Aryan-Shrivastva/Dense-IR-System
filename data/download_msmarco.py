"""
Downloads MS MARCO passage dataset via HuggingFace datasets library.
No manual download required. Run once — takes ~5 minutes.

Output files:
  data/corpus.csv          — sampled passages for indexing
  data/dev_queries.tsv     — evaluation queries
  data/dev_qrels.tsv       — relevance labels for evaluation
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()
from src.hf_insecure_ssl import apply_if_configured

apply_if_configured()

CORPUS_SIZE = int(os.getenv("CORPUS_SIZE", 50000))
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_corpus():
    print(f"Downloading MS MARCO passages (sampling {CORPUS_SIZE} docs)...")
    # This streams directly — no 22GB download needed
    dataset = load_dataset(
        "ms_marco", "v2.1",
        split=f"train[:{CORPUS_SIZE}]",
    )

    docs = []
    for i, item in enumerate(tqdm(dataset, desc="Processing")):
        texts = item["passages"]["passage_text"]
        pids = item["passages"].get("passage_id") or []
        for j, passage in enumerate(texts):
            pid = str(pids[j]) if j < len(pids) else f"msmarco_{i:06d}_{j}"
            docs.append({
                "docid": pid,
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
    dev = load_dataset("ms_marco", "v2.1", split="validation[:5000]")

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

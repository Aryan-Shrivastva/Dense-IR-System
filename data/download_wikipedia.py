"""
Builds a Wikipedia corpus using HuggingFace datasets.
More topically diverse than MS MARCO — better for live demos.
Run this AFTER download_msmarco.py if you want a combined corpus.

Output: data/wiki_corpus.csv

Usage:
  # Download fresh from HuggingFace
  python data/download_wikipedia.py

  # Use a pre-downloaded dataset on disk (wikipedia_full/ folder from save_to_disk)
  python data/download_wikipedia.py --disk-path /path/to/wikipedia_full

  # Or set env var and run without flag:
  # WIKI_DISK_PATH=/path/to/wikipedia_full python data/download_wikipedia.py
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    from src.hf_insecure_ssl import apply_if_configured
    apply_if_configured()
except ImportError:
    pass

WIKI_SIZE = int(os.getenv("WIKI_SIZE", 20000))
DATA_DIR = "data"


def _load_wiki_from_disk(disk_path: str):
    from datasets import load_from_disk
    print(f"Loading Wikipedia from disk: {disk_path}")
    ds = load_from_disk(disk_path)
    # load_from_disk may return a Dataset or a DatasetDict
    if hasattr(ds, "keys"):
        split = "train" if "train" in ds else list(ds.keys())[0]
        ds = ds[split]
    print(f"Loaded {len(ds):,} articles from disk.")
    return ds


def _load_wiki_from_hub():
    from datasets import load_dataset
    print(f"Downloading Wikipedia articles from HuggingFace (sampling {WIKI_SIZE})...")
    return load_dataset(
        "wikipedia", "20220301.en",
        split=f"train[:{WIKI_SIZE}]",
    )


def process_wiki(wiki, limit: int = None):
    docs = []
    for i, article in enumerate(tqdm(wiki, desc="Processing")):
        if limit and i >= limit:
            break
        text = article["text"]
        words = text.split()
        chunks = [" ".join(words[j:j+300]) for j in range(0, len(words), 300)]
        for k, chunk in enumerate(chunks[:5]):  # max 5 chunks per article
            if len(chunk) > 100:
                docs.append({
                    "docid": f"wiki_{i:06d}_{k}",
                    "title": article["title"],
                    "body":  chunk,
                    "url":   f"https://en.wikipedia.org/wiki/{article['title'].replace(' ', '_')}",
                })
    return docs


def download_wiki(disk_path: str = None):
    disk_path = disk_path or os.getenv("WIKI_DISK_PATH", "").strip()

    if disk_path:
        wiki = _load_wiki_from_disk(disk_path)
        limit = WIKI_SIZE  # still cap to WIKI_SIZE articles even from full dump
    else:
        wiki = _load_wiki_from_hub()
        limit = None  # hub split already limits via train[:{WIKI_SIZE}]

    docs = process_wiki(wiki, limit=limit)

    df = pd.DataFrame(docs)
    out = os.path.join(DATA_DIR, "wiki_corpus.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} Wikipedia chunks → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disk-path",
        default=None,
        help="Path to a Wikipedia dataset saved with save_to_disk() "
             "(e.g. ~/wiki_project/wikipedia_full). "
             "Overrides WIKI_DISK_PATH env var and skips HuggingFace download.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Max number of Wikipedia articles to process (default: WIKI_SIZE env / 20000).",
    )
    args = parser.parse_args()

    if args.size:
        WIKI_SIZE = args.size

    os.makedirs(DATA_DIR, exist_ok=True)
    download_wiki(disk_path=args.disk_path)

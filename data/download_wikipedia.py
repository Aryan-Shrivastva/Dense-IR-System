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

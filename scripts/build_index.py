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

from src.hf_insecure_ssl import apply_if_configured

apply_if_configured()

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
    corpus_pkl_path = os.getenv("CORPUS_PATH", "indexes/corpus.pkl"),
    model_name    = os.getenv("MODEL_NAME",  "all-MiniLM-L6-v2"),
)

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
        if row.empty:
            continue
        query = row.iloc[0]["query"]
        rel   = qrels.get(qid, set())
        rel   = {str(r) for r in rel}
        if not rel:
            continue

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

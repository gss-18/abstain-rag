"""CP1: retrieval baseline.

1. Parse the FAR HTML into section-level passages -> data/far/corpus.jsonl
2. Index SciFact, FiQA, NFCorpus and FAR with bge-small-en-v1.5 (embeddings cached in data/index/)
3. Score the three BEIR test sets and compare with the scores published on the model card

Usage: uv run python scripts/cp1_retrieval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from abstain_rag.beir import doc_text, load_corpus, load_qrels, load_queries
from abstain_rag.far import chunk, parse_folder
from abstain_rag.metrics import evaluate
from abstain_rag.retrieval import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "runs" / "cp1"

# From the bge-small-en-v1.5 model card (MTEB results), in percent.
PUBLISHED = {
    "scifact": {"ndcg@10": 71.275, "recall@100": 94.167},
    "fiqa": {"ndcg@10": 40.343, "recall@100": 69.552},
    "nfcorpus": {"ndcg@10": 34.305, "recall@100": 31.135},
}


def build_far_corpus() -> Path:
    sections = parse_folder(DATA / "raw" / "far")
    passages = chunk(sections)
    folder = DATA / "far"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "corpus.jsonl").open("w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")
    print(f"[far] {len(sections):,} sections -> {len(passages):,} passages")
    return folder


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    far_folder = build_far_corpus()
    retriever = DenseRetriever()
    results = {}
    for name in ["scifact", "fiqa", "nfcorpus"]:
        folder = DATA / "beir" / name
        corpus = load_corpus(folder)
        ids = list(corpus)
        t = time.perf_counter()
        retriever.build(ids, [doc_text(corpus[i]) for i in ids], cache_dir=DATA / "index" / name)
        index_secs = time.perf_counter() - t
        queries, qrels = load_queries(folder), load_qrels(folder, "test")
        qids = [q for q in qrels if q in queries]
        t = time.perf_counter()
        _, ranked = retriever.search([queries[q] for q in qids], k=100)
        ms_per_query = (time.perf_counter() - t) / len(qids) * 1000
        scores = evaluate(dict(zip(qids, ranked, strict=True)), qrels)
        ours = {k: round(scores[k] * 100, 2) for k in ("ndcg@10", "recall@100")}
        results[name] = {**ours, "published": PUBLISHED[name], "queries": scores["queries"],
                         "docs": len(ids), "index_secs": round(index_secs, 1),
                         "ms_per_query": round(ms_per_query, 2)}
        print(f"[{name}] nDCG@10 {ours['ndcg@10']} (published {PUBLISHED[name]['ndcg@10']}), "
              f"recall@100 {ours['recall@100']} (published {PUBLISHED[name]['recall@100']}), "
              f"{ms_per_query:.1f} ms/query")

    far = load_corpus(far_folder)
    ids = list(far)
    retriever.build(ids, [doc_text(far[i]) for i in ids], cache_dir=DATA / "index" / "far")
    probes = ["When can the government reject nonconforming commercial items?",
              "What is the micro-purchase threshold?",
              "How does the contracting officer do proposal analysis?"]
    scores, ranked = retriever.search(probes, k=3)
    for q, s, r in zip(probes, scores, ranked, strict=True):
        print(f"[far] {q}\n      -> {r} scores {[round(float(x), 3) for x in s]}")
    results["far"] = {"passages": len(ids)}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {OUT / 'results.json'}")


if __name__ == "__main__":
    main()

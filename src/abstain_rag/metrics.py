"""Retrieval metrics, matching trec_eval / pytrec_eval definitions (linear gain, log2 discount)."""

from __future__ import annotations

import math

Run = dict[str, list[str]]  # query_id -> ranked doc ids (best first)


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int = 100) -> float:
    if not rels:
        return 0.0
    return len(set(ranked[:k]) & set(rels)) / len(rels)


def evaluate(run: Run, qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Mean nDCG@10 and recall@100 over queries that have at least one relevant doc."""
    qids = [q for q in qrels if qrels[q]]
    n = len(qids)
    return {
        "ndcg@10": sum(ndcg_at_k(run.get(q, []), qrels[q], 10) for q in qids) / n,
        "recall@100": sum(recall_at_k(run.get(q, []), qrels[q], 100) for q in qids) / n,
        "queries": n,
    }

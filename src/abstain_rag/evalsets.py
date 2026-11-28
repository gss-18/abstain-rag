"""Evaluation query sets (answerable + unanswerable) for the four corpora, shared by CP3 and CP4."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .beir import load_qrels, load_queries
from .gates import section_scores
from .retrieval import DenseRetriever

BEIR = ["scifact", "fiqa", "nfcorpus"]
CORPORA = BEIR + ["far"]
DEV_SPLIT = {"scifact": "train", "fiqa": "dev", "nfcorpus": "dev"}  # SciFact has no dev split
# Unanswerable queries for a corpus come from an unrelated dataset (finance vs biomedical).
UNRELATED = {"scifact": ["fiqa"], "nfcorpus": ["fiqa"], "fiqa": ["scifact", "nfcorpus"]}


def far_items(data: Path) -> list[dict]:
    items = [json.loads(line) for line in (data / "far" / "testset.jsonl").open()]
    return [{"id": it["id"], "question": it["question"], "answerable": it["answerable"], "split": it["split"],
             "type": it["type"], "relevant": {it["section"]} if it["answerable"] else set()} for it in items]


def beir_items(data: Path, name: str, ood_llm: list[dict], max_dev: int = 300, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    folder = data / "beir" / name
    queries = load_queries(folder)
    items = []
    for split, qsplit in [("dev", DEV_SPLIT[name]), ("test", "test")]:
        qrels = load_qrels(folder, qsplit)
        qids = sorted(q for q in qrels if q in queries)
        if split == "dev" and len(qids) > max_dev:
            qids = rng.sample(qids, max_dev)
        items += [{"id": f"{name}:{q}", "question": queries[q], "answerable": True, "split": split,
                   "type": "answerable", "relevant": set(qrels[q])} for q in qids]
    n_ans = len(items)
    pool = [t for other in UNRELATED[name] for t in load_queries(data / "beir" / other).values()]
    cross = rng.sample(pool, n_ans - len(ood_llm))
    n_dev = round(len(cross) * 0.3)
    items += [{"id": f"{name}:unrel{k}", "question": q, "answerable": False, "split": "dev" if k < n_dev else "test",
               "type": "unrelated_beir", "relevant": set()} for k, q in enumerate(cross)]
    items += [{**it, "relevant": set()} for it in ood_llm]
    return items


def all_items(data: Path) -> dict[str, list[dict]]:
    far = far_items(data)
    ood_llm = [it for it in far if it["type"] == "ood_llm"]
    out = {name: beir_items(data, name, ood_llm) for name in BEIR}
    out["far"] = far
    return out


def retrieve_units(retriever: DenseRetriever, questions: list[str], is_far: bool,
                   k: int = 50) -> tuple[list[np.ndarray], list[list[str]], list[list[str]]]:
    """Retrieve and return (scores, ranked units, ranked passage ids).

    A "unit" is a document for BEIR and a section for FAR (best passage per section).
    """
    scores, ranked = retriever.search(questions, k=k)
    if not is_far:
        return list(scores), ranked, ranked
    out_s, out_r = [], []
    for s, r in zip(scores, ranked, strict=True):
        out_s.append(section_scores(s, r))
        out_r.append(list(dict.fromkeys(pid.split("#")[0] for pid in r)))
    return out_s, out_r, ranked

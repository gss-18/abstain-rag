"""Load datasets stored in the BEIR layout (corpus.jsonl, queries.jsonl, qrels/<split>.tsv)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

Corpus = dict[str, dict[str, str]]   # doc_id -> {"title", "text"}
Queries = dict[str, str]             # query_id -> text
Qrels = dict[str, dict[str, int]]    # query_id -> {doc_id: relevance}


def load_corpus(folder: Path) -> Corpus:
    corpus: Corpus = {}
    with (folder / "corpus.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    return corpus


def load_queries(folder: Path) -> Queries:
    with (folder / "queries.jsonl").open() as f:
        return {d["_id"]: d["text"] for d in map(json.loads, f)}


def load_qrels(folder: Path, split: str) -> Qrels:
    """Read qrels, keeping only judgments with relevance > 0."""
    qrels: Qrels = {}
    with (folder / "qrels" / f"{split}.tsv").open() as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def doc_text(doc: dict[str, str]) -> str:
    """Text the encoder sees for a document: title and body, as in the BEIR/MTEB evaluation."""
    return f"{doc['title']} {doc['text']}".strip()

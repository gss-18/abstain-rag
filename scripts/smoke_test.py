"""CP0 smoke test: embeddings, FAISS search and both local LLMs work end to end.

Usage: uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import httpx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OLLAMA = "http://localhost:11434"
LLMS = ["llama3.1:8b", "qwen2.5:7b"]


def check_embeddings() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)
    docs = []
    with (ROOT / "data/beir/scifact/corpus.jsonl").open() as f:
        for _, line in zip(range(500), f, strict=False):
            d = json.loads(line)
            docs.append(f"{d['title']}. {d['text']}")
    t = time.perf_counter()
    emb = model.encode(docs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    secs = time.perf_counter() - t
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))
    q = model.encode([QUERY_PREFIX + "diffusion tensor imaging of newborn white matter"], normalize_embeddings=True)
    scores, ids = index.search(q.astype(np.float32), 3)
    print(f"[embed] {EMBED_MODEL} on {device}: 500 docs in {secs:.1f}s ({500 / secs:.0f} docs/s), dim {emb.shape[1]}")
    print(f"[faiss] top-3 scores {np.round(scores[0], 3).tolist()} -> {docs[ids[0][0]][:80]}...")


def check_llms() -> None:
    with httpx.Client(timeout=300) as client:
        have = {m["name"] for m in client.get(f"{OLLAMA}/api/tags").json()["models"]}
        for name in LLMS:
            if name not in have:
                print(f"[llm] {name}: NOT PULLED")
                continue
            t = time.perf_counter()
            r = client.post(
                f"{OLLAMA}/api/generate",
                json={"model": name, "prompt": "In one sentence: what is the FAR?", "stream": False,
                      "options": {"temperature": 0, "num_predict": 60}},
            ).json()
            secs = time.perf_counter() - t
            tps = r["eval_count"] / (r["eval_duration"] / 1e9)
            print(f"[llm] {name}: {secs:.1f}s total, {tps:.0f} tok/s -> {r['response'].strip()[:100]}")


if __name__ == "__main__":
    check_embeddings()
    check_llms()

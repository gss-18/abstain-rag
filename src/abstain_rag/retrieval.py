"""Dense retrieval with bge-small and an exact FAISS inner-product index."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL, query_prefix: str = BGE_QUERY_PREFIX,
                 device: str | None = None) -> None:
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        self.index: faiss.IndexFlatIP | None = None
        self.doc_ids: list[str] = []

    def encode(self, texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=progress)
        return emb.astype(np.float32)

    def build(self, doc_ids: list[str], texts: list[str], cache_dir: Path | None = None) -> None:
        """Embed documents (or load cached embeddings) and build the index."""
        emb_path = cache_dir / "embeddings.npy" if cache_dir else None
        ids_path = cache_dir / "doc_ids.json" if cache_dir else None
        if emb_path and emb_path.exists() and json.loads(ids_path.read_text()) == doc_ids:
            emb = np.load(emb_path)
        else:
            # Encode longest texts first: batches of similar length waste less padding.
            order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
            sorted_emb = self.encode([texts[i] for i in order], progress=True)
            emb = np.empty_like(sorted_emb)
            emb[order] = sorted_emb
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(emb_path, emb)
                ids_path.write_text(json.dumps(doc_ids))
        self.doc_ids = list(doc_ids)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, queries: list[str], k: int = 100) -> tuple[np.ndarray, list[list[str]]]:
        """Return (scores [n, k], ranked doc ids) for each query."""
        if self.index is None:
            raise RuntimeError("call build() first")
        q = self.encode([self.query_prefix + t for t in queries])
        scores, idx = self.index.search(q, k)
        return scores, [[self.doc_ids[j] for j in row if j >= 0] for row in idx]

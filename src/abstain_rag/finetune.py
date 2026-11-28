"""CP5: fine-tune the bi-encoder on FAR with synthetic questions (contrastive loss, in-batch negatives).

Kept small and explicit on purpose: a plain PyTorch loop instead of a trainer framework, so every step is visible.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def training_passages(passages: list[dict], held_out_sections: set[str], per_section: int = 1,
                      min_words: int = 60, seed: int = 42) -> list[dict]:
    """Passages to write training questions for: never from a dev/test section (no leakage), at most
    `per_section` per section so large parts of the FAR do not dominate, each long enough to ask about."""
    by_section: dict[str, list[dict]] = {}
    for p in passages:
        if p["section"] not in held_out_sections and len(p["text"].split()) >= min_words:
            by_section.setdefault(p["section"], []).append(p)
    rng = random.Random(seed)
    out = []
    for s in sorted(by_section):
        out += rng.sample(by_section[s], min(per_section, len(by_section[s])))
    return out


def batches(pairs: list[tuple[str, str, str]], batch_size: int, seed: int) -> list[list[tuple[str, str, str]]]:
    """Shuffle (query, passage, section) pairs into batches with no section repeated inside a batch.

    In-batch negatives assume every other passage in the batch is wrong for this query; two passages from the
    same section would be false negatives, so a repeated section moves to a later batch."""
    rng = random.Random(seed)
    pending = pairs[:]
    rng.shuffle(pending)
    out = []
    while pending:
        batch, seen, rest = [], set(), []
        for p in pending:
            if len(batch) < batch_size and p[2] not in seen:
                batch.append(p)
                seen.add(p[2])
            else:
                rest.append(p)
        out.append(batch)
        pending = rest
    return out


def contrastive_loss(q: torch.Tensor, p: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    """Multiple-negatives ranking loss: each query should score its own passage above every other passage in
    the batch. q, p: (batch, dim), L2-normalised. Cross-entropy over the scaled cosine-similarity matrix."""
    logits = scale * q @ p.T
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))


def embed(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    features = model.tokenize(texts)
    features = {k: v.to(model.device) for k, v in features.items()}
    return F.normalize(model(features)["sentence_embedding"], dim=-1)


def train_epoch(model: SentenceTransformer, pairs: list[tuple[str, str, str]], optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler, query_prefix: str, batch_size: int = 32,
                seed: int = 0) -> float:
    model.train()
    losses = []
    for batch in batches(pairs, batch_size, seed):
        if len(batch) < 2:
            continue
        q = embed(model, [query_prefix + b[0] for b in batch])
        p = embed(model, [b[1] for b in batch])
        loss = contrastive_loss(q, p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    model.eval()
    return float(np.mean(losses))

"""Abstention gates: decide from retrieval results whether to answer or refuse.

Every gate works on cheap signals computed from the ranked retrieval scores (no LLM call):
  top1      - similarity of the best hit
  margin    - how much the best hit stands out: top1 minus the mean of hits 2..10
  vocab     - share of the query's content words that occur in the corpus

Gates (numbering follows the project plan; #6, LLM self-refusal, lives in CP4):
  1 no_gate          always answer
  2 fixed_threshold  top1 >= t, one global t for every corpus
  3 calibrated_floor top1 >= floor, floor set per corpus from nonsense probes (no labeled refusals needed)
  4 floor_margin     floor AND margin >= m
  5 floor_vocab      floor AND vocab >= v
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN = re.compile(r"[a-z][a-z0-9\-]+")


def content_words(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if len(w) >= 3 and w not in ENGLISH_STOP_WORDS]


class Vocabulary:
    """Content words that occur in at least min_df documents of a corpus."""

    def __init__(self, texts: list[str], min_df: int = 2) -> None:
        df: Counter = Counter()
        for t in texts:
            df.update(set(content_words(t)))
        self.words = sorted(w for w, c in df.items() if c >= min_df)
        self._set = set(self.words)

    def coverage(self, query: str) -> float:
        words = content_words(query)
        return sum(w in self._set for w in words) / len(words) if words else 0.0


def section_scores(scores: np.ndarray, ids: list[str]) -> np.ndarray:
    """Collapse passage hits ("52.212-4#3") to one score per section (its best passage), in rank order."""
    seen: dict[str, float] = {}
    for s, pid in zip(scores, ids, strict=False):
        seen.setdefault(pid.split("#")[0], float(s))
    return np.array(list(seen.values()), dtype=np.float32)


def signals(scores: np.ndarray, query: str, vocab: Vocabulary) -> dict[str, float]:
    """Signals for one query from its ranked scores (best first, at least 2 hits)."""
    top = scores[:10]
    return {"top1": float(top[0]), "margin": float(top[0] - top[1:].mean()), "vocab": vocab.coverage(query)}


def word_salad_probes(vocab: Vocabulary, n: int = 200, length: int = 6, seed: int = 42) -> list[str]:
    """Nonsense queries made of random in-domain words.

    These are a harder (and better) probe than random characters: in-domain words make the encoder
    produce corpus-like vectors, so the floor ends up above where real off-topic questions land.
    """
    rng = random.Random(seed)
    return [" ".join(rng.sample(vocab.words, length)) for _ in range(n)]


def calibrate_floor(probe_top1: np.ndarray, known_good_top1: np.ndarray | None = None,
                    probe_quantile: float = 0.95, good_quantile: float = 0.10) -> float:
    """Floor = 95th percentile of nonsense-probe scores, capped so it never refuses more than
    10% of known-good queries. Needs no labeled unanswerable questions."""
    floor = float(np.quantile(probe_top1, probe_quantile))
    if known_good_top1 is not None and len(known_good_top1):
        floor = min(floor, float(np.quantile(known_good_top1, good_quantile)))
    return floor


def balanced_accuracy(answer: np.ndarray, answerable: np.ndarray) -> float:
    """Mean of: share of answerable that were answered, share of unanswerable that were refused."""
    tpr = answer[answerable].mean() if answerable.any() else 0.0
    tnr = (~answer[~answerable]).mean() if (~answerable).any() else 0.0
    return float((tpr + tnr) / 2)


def tune_threshold(values: np.ndarray, answerable: np.ndarray, base: np.ndarray | None = None) -> float:
    """Pick t maximizing balanced accuracy of (base AND values >= t) on dev data."""
    base = np.ones_like(answerable, dtype=bool) if base is None else base
    candidates = np.unique(np.concatenate([values, [values.min() - 1e-6]]))
    best_t, best = float(candidates[0]), -1.0
    for t in candidates:
        acc = balanced_accuracy(base & (values >= t), answerable)
        if acc > best:
            best_t, best = float(t), acc
    return best_t


@dataclass(frozen=True)
class GateParams:
    fixed_t: float
    floor: float
    margin_m: float
    vocab_v: float


def decide(sig: dict[str, np.ndarray], p: GateParams) -> dict[str, np.ndarray]:
    """Answer (True) / refuse (False) per query for gates 1-5."""
    floor = sig["top1"] >= p.floor
    return {
        "1_no_gate": np.ones_like(floor, dtype=bool),
        "2_fixed_threshold": sig["top1"] >= p.fixed_t,
        "3_calibrated_floor": floor,
        "4_floor_margin": floor & (sig["margin"] >= p.margin_m),
        "5_floor_vocab": floor & (sig["vocab"] >= p.vocab_v),
    }

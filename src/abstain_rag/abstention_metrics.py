"""Metrics for answer-or-refuse decisions.

Two kinds of ground truth:
  answerable - should the system answer at all? (used for AUROC and refusal precision/recall)
  correct    - would an answer be grounded? Here: answerable AND the right doc is in the top-5 retrieved.
               (used for risk; CP4 replaces it with the LLM judge's verdict on the actual answer)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(confidence: np.ndarray, answerable: np.ndarray) -> float:
    """Probability that a random answerable query gets higher confidence than a random unanswerable one."""
    return float(roc_auc_score(answerable, confidence))


def decision_metrics(answer: np.ndarray, answerable: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    refuse = ~answer
    n_ref = refuse.sum()
    return {
        "coverage": float(answer.mean()),
        "refusal_precision": float((refuse & ~answerable).sum() / n_ref) if n_ref else float("nan"),
        "refusal_recall": float((refuse & ~answerable).sum() / (~answerable).sum()),
        "false_refusal_rate": float((refuse & answerable).sum() / answerable.sum()),
        "risk": float((answer & ~correct).sum() / answer.sum()) if answer.any() else 0.0,
    }


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer the most confident queries first. For each coverage level, the error rate among answered ones."""
    order = np.argsort(-confidence, kind="stable")
    wrong = (~correct[order]).astype(float)
    n = np.arange(1, len(order) + 1)
    return n / len(order), np.cumsum(wrong) / n


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve: average risk over all coverage levels (lower is better)."""
    _, risk = risk_coverage(confidence, correct)
    return float(risk.mean())

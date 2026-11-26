import math

import pytest

from abstain_rag.metrics import evaluate, ndcg_at_k, recall_at_k


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # relevant doc at rank 2 only: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert ndcg_at_k(["x", "a"], {"a": 1}) == pytest.approx(1 / math.log2(3))


def test_ndcg_ignores_ranks_past_k():
    assert ndcg_at_k(["x"] * 10 + ["a"], {"a": 1}, k=10) == 0.0


def test_recall():
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1, "c": 1}, k=2) == pytest.approx(1 / 3)


def test_evaluate_averages_over_judged_queries():
    run = {"q1": ["a"], "q2": ["x"]}
    out = evaluate(run, {"q1": {"a": 1}, "q2": {"b": 1}})
    assert out["ndcg@10"] == pytest.approx(0.5)
    assert out["queries"] == 2

import numpy as np
import pytest

from abstain_rag.abstention_metrics import aurc, auroc, decision_metrics, risk_coverage
from abstain_rag.gates import (
    GateParams,
    Vocabulary,
    calibrate_floor,
    decide,
    section_scores,
    signals,
    tune_threshold,
)


def test_vocabulary_coverage_needs_min_df():
    v = Vocabulary(["contract award price", "contract price clause"], min_df=2)
    assert v.words == ["contract", "price"]
    assert v.coverage("contract award") == 0.5
    assert v.coverage("the of and") == 0.0  # no content words


def test_section_scores_keeps_best_passage_per_section():
    s = section_scores(np.array([0.9, 0.8, 0.7]), ["1.1#0", "1.1#2", "2.2#0"])
    assert s.tolist() == pytest.approx([0.9, 0.7])


def test_signals_margin():
    v = Vocabulary(["a b"], min_df=1)
    out = signals(np.array([0.9, 0.5, 0.3]), "query", v)
    assert out["margin"] == pytest.approx(0.5)


def test_floor_is_capped_by_known_good():
    probes = np.array([0.1, 0.2, 0.9])
    assert calibrate_floor(probes) == pytest.approx(np.quantile(probes, 0.95))
    assert calibrate_floor(probes, known_good_top1=np.array([0.5, 0.6])) == pytest.approx(0.51)


def test_tune_threshold_separates_clean_data():
    vals = np.array([0.9, 0.8, 0.3, 0.2])
    ans = np.array([True, True, False, False])
    t = tune_threshold(vals, ans)
    assert 0.3 < t <= 0.8


def test_decide_combines_floor_and_margin():
    sig = {"top1": np.array([0.9, 0.9, 0.1]), "margin": np.array([0.2, 0.01, 0.5]), "vocab": np.array([1, 1, 1.0])}
    out = decide(sig, GateParams(fixed_t=0.5, floor=0.5, margin_m=0.1, vocab_v=0.5))
    assert out["4_floor_margin"].tolist() == [True, False, False]
    assert out["1_no_gate"].all()


def test_decision_metrics():
    answer = np.array([True, True, False, False])
    answerable = np.array([True, False, True, False])
    correct = np.array([True, False, True, False])
    m = decision_metrics(answer, answerable, correct)
    assert m["coverage"] == 0.5
    assert m["refusal_precision"] == 0.5
    assert m["refusal_recall"] == 0.5
    assert m["risk"] == 0.5


def test_risk_coverage_and_aurc():
    conf = np.array([0.9, 0.8, 0.1])
    correct = np.array([True, False, True])
    cov, risk = risk_coverage(conf, correct)
    assert cov.tolist() == pytest.approx([1 / 3, 2 / 3, 1])
    assert risk.tolist() == pytest.approx([0, 0.5, 1 / 3])
    assert aurc(conf, correct) == pytest.approx((0 + 0.5 + 1 / 3) / 3)
    assert auroc(conf, correct) == pytest.approx(0.5)

import numpy as np

from abstain_rag.testset import (
    longest_shared_span,
    near_duplicates,
    reject_reason,
    sample_passages,
    split_dev_test,
)


def test_longest_shared_span():
    # longest copied runs are "the agency" and "pay early"
    assert longest_shared_span("when may the agency pay early?", "The agency may pay early in some cases") == 2


def test_reject_reasons():
    src = "The contracting officer shall evaluate the reasonableness of the offered prices in every case"
    assert reject_reason("Price?") == "too_short"
    assert reject_reason("What does FAR 15.404-1 say about price analysis rules?") == "cites_source"
    assert reject_reason("What does this section require of contractors here?") == "cites_source"
    assert reject_reason("Tell me about price analysis for contracts please") == "not_a_question"
    assert reject_reason("Must the contracting officer shall evaluate the reasonableness of the offered prices?",
                         src) == "copies_source"
    assert reject_reason("Who checks whether a vendor's quoted price is fair?", src) is None


def test_sample_passages_one_per_section_and_min_words():
    long = " ".join(["w"] * 80)
    ps = [{"section": "1.1", "text": long}, {"section": "1.1", "text": long},
          {"section": "2.2", "text": long}, {"section": "3.3", "text": "short"}]
    out = sample_passages(ps, n=10)
    assert sorted(p["section"] for p in out) == ["1.1", "2.2"]


def test_near_duplicates_keeps_first():
    e = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert near_duplicates(e) == {1}


def test_split_is_stratified_and_deterministic():
    items = [{"type": "a"} for _ in range(10)] + [{"type": "b"} for _ in range(10)]
    out = split_dev_test(items, dev_fraction=0.3)
    for t in "ab":
        assert sum(1 for i in out if i["type"] == t and i["split"] == "dev") == 3

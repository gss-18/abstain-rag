from abstain_rag.rag import REFUSAL, build_sources, far_source_text, is_refusal, truncate_words


def test_is_refusal():
    assert is_refusal(REFUSAL)
    assert is_refusal("The sources do not contain information about per diem rates.")
    assert is_refusal("There is no information in the provided sources about Demat brokers.")
    assert is_refusal("I'm not able to answer the question as it is not based on the provided sources.")
    assert is_refusal("Unfortunately, the sources provided do not address 0% financing. However, [4] says...")
    assert is_refusal("The sources do not mention DFARS 252.204-7012, but they do mention FAR 52.204-25 [3].")
    assert not is_refusal("The micro-purchase threshold is $10,000 [1].")
    # a hedge after a real first sentence is still an answer
    assert not is_refusal("Rebalancing keeps a target mix [1]. The sources do not specify how often.")


def test_truncate_words():
    assert truncate_words("a b c d", 2) == "a b ..."
    assert truncate_words("a b", 5) == "a b"


def test_far_source_text_appends_next_window_without_overlap():
    corpus = {"1.1#0": {"text": "x " * 250}, "1.1#1": {"text": " ".join(["o"] * 50 + ["new", "words"])}}
    assert far_source_text("1.1#0", corpus).endswith("new words")
    assert far_source_text("1.1#1", corpus).endswith("new words")  # last window: nothing appended


def test_build_sources_one_passage_per_far_section():
    corpus = {p: {"title": p, "text": "t"} for p in ["1.1#0", "1.1#1", "2.2#0"]}
    text, used = build_sources(["1.1#0", "1.1#1", "2.2#0"], corpus, is_far=True)
    assert used == ["1.1#0", "2.2#0"]
    assert text.startswith("[1] 1.1#0")

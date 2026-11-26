import pytest

from abstain_rag.far import Section, chunk, parse_part

HTML = """
<html><body>
<article class="topic" id="FAR_15_404">
  <h3><span class="ph autonumber">15.404</span> Proposal analysis.</h3>
  <div class="body"><p class="ListL2"><a class="xref fm:TOC" href="#x">15.404-1 Techniques</a></p></div>
  <article class="topic" id="FAR_15_404_1">
    <h4><span class="ph autonumber">15.404-1</span> Proposal analysis
    techniques.</h4>
    <div class="body"><p>(a) The objective is a fair and reasonable price.</p></div>
  </article>
</article>
</body></html>
"""


def test_parse_part_keeps_own_text_and_drops_toc():
    sections = parse_part(HTML, part=15)
    assert sections == [
        Section("15.404-1", "Proposal analysis techniques.", 15, "(a) The objective is a fair and reasonable price.")
    ]


def test_chunk_short_section_is_one_passage():
    out = chunk([Section("1.101", "Purpose.", 1, "word " * 10)])
    assert [p["_id"] for p in out] == ["1.101#0"]
    assert out[0]["section"] == "1.101"


def test_chunk_long_section_overlaps():
    words = " ".join(str(i) for i in range(500))
    out = chunk([Section("52.212-4", "Terms.", 52, words)], max_words=250, overlap=50)
    assert [p["_id"] for p in out] == ["52.212-4#0", "52.212-4#1", "52.212-4#2"]
    assert out[1]["text"].split()[0] == "200"  # starts 50 words before the end of chunk 0
    assert all(len(p["text"].split()) <= 250 for p in out)


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk([], max_words=10, overlap=10)

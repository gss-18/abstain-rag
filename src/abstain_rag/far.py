"""Parse the FAR (Federal Acquisition Regulation) HTML into section-level passages.

Each FAR section (e.g. 52.212-4) is an <article> whose own text lives in its direct
child <div class="body">. Nested articles are child sections, so they are parsed separately.
Long sections are split into overlapping word windows that fit the encoder's input limit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Section:
    number: str   # "52.212-4"
    title: str    # "Contract Terms and Conditions—Commercial Products and Commercial Services."
    part: int
    text: str


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip()


def parse_part(html: str, part: int) -> list[Section]:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", class_="fm:TOC"):  # table-of-contents links are not content
        a.parent.decompose()
    sections: list[Section] = []
    for art in soup.find_all("article"):
        heading = art.find(re.compile(r"^h\d$"), recursive=False)
        body = art.find("div", class_="body", recursive=False)
        if not isinstance(heading, Tag) or not isinstance(body, Tag):
            continue
        num_tag = heading.find("span", class_="autonumber")
        number = _clean(num_tag.get_text()) if num_tag else ""
        if num_tag:
            num_tag.extract()
        text = _clean(body.get_text(" "))
        if number and text:
            sections.append(Section(number, _clean(heading.get_text(" ")), part, text))
    return sections


def parse_folder(folder: Path) -> list[Section]:
    out: list[Section] = []
    for path in sorted(folder.glob("part_*.html"), key=lambda p: int(p.stem.split("_")[1])):
        out.extend(parse_part(path.read_text(encoding="utf-8", errors="replace"), int(path.stem.split("_")[1])))
    return out


def chunk(sections: list[Section], max_words: int = 250, overlap: int = 50) -> list[dict[str, str]]:
    """Split sections into passages of at most max_words words, in BEIR corpus format.

    Passage ids are "<section>#<n>", so every passage maps back to its FAR section.
    """
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    passages = []
    for s in sections:
        words = s.text.split()
        step = max_words - overlap
        starts = range(0, max(len(words) - overlap, 1), step)
        for n, i in enumerate(starts):
            passages.append({
                "_id": f"{s.number}#{n}",
                "title": f"FAR {s.number} {s.title}",
                "text": " ".join(words[i:i + max_words]),
                "section": s.number,
            })
    return passages

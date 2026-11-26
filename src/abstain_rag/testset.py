"""Building blocks for the FAR test set: sampling, quality filters and splitting.

All functions are pure (no model calls), so they are easy to test.
"""

from __future__ import annotations

import random
import re

_WORD = re.compile(r"[a-z0-9]+")
# A FAR citation like "52.212-4", "15.404-1" or "13.003", or a phrase that points at the source text.
_SECTION_REF = re.compile(r"\b\d{1,2}\.\d{3}(-\d+)?\b|\b(this|the above|the following) (section|clause|passage|text)\b",
                          re.IGNORECASE)


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def sample_passages(passages: list[dict], n: int, min_words: int = 60, seed: int = 42) -> list[dict]:
    """Pick n passages from n different sections (uniform over sections), each with >= min_words words."""
    by_section: dict[str, list[dict]] = {}
    for p in passages:
        if len(p["text"].split()) >= min_words:
            by_section.setdefault(p["section"], []).append(p)
    rng = random.Random(seed)
    sections = sorted(by_section)
    chosen = rng.sample(sections, min(n, len(sections)))
    return [rng.choice(by_section[s]) for s in chosen]


def longest_shared_span(question: str, source: str) -> int:
    """Length in words of the longest run of consecutive words the question copies from the source."""
    q, s = words(question), words(source)
    grams = {}
    best = 0
    for n in range(1, len(q) + 1):
        grams = {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
        if any(tuple(q[i:i + n]) in grams for i in range(len(q) - n + 1)):
            best = n
        else:
            break
    return best


def reject_reason(question: str, source: str | None = None, max_copied_words: int = 6,
                  min_words: int = 5, max_words: int = 40) -> str | None:
    """Return why a generated question should be dropped, or None if it passes."""
    n = len(words(question))
    if n < min_words:
        return "too_short"
    if n > max_words:
        return "too_long"
    if not question.strip().endswith("?"):
        return "not_a_question"
    if _SECTION_REF.search(question):
        return "cites_source"
    if source is not None and longest_shared_span(question, source) > max_copied_words:
        return "copies_source"
    return None


def near_duplicates(embeddings, threshold: float = 0.95) -> set[int]:
    """Indices to drop so no two kept items have cosine similarity above threshold (keeps the first)."""
    sims = embeddings @ embeddings.T
    drop: set[int] = set()
    for i in range(len(embeddings)):
        if i in drop:
            continue
        for j in range(i + 1, len(embeddings)):
            if j not in drop and sims[i, j] > threshold:
                drop.add(j)
    return drop


def split_dev_test(items: list[dict], dev_fraction: float = 0.3, seed: int = 42) -> list[dict]:
    """Add "split": "dev" or "test" to each item, stratified by item["type"].

    dev is for tuning gate thresholds (CP3); test is only for final numbers.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for it in items:
        by_type.setdefault(it["type"], []).append(it)
    for group in by_type.values():
        rng.shuffle(group)
        n_dev = round(len(group) * dev_fraction)
        for k, it in enumerate(group):
            it["split"] = "dev" if k < n_dev else "test"
    return items

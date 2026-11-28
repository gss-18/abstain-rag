"""Answer generation over retrieved sources, LLM self-refusal (gate 6), and the groundedness judge."""

from __future__ import annotations

import re

MAX_WORDS_PER_SOURCE = 220
N_SOURCES = 5

REFUSAL = "I don't know based on the provided sources."
_REFUSAL_RE = re.compile(
    r"\bi(?: do not| don't| cannot| can't| am not able to|'m not able to| am unable to|'m unable to) (know|answer|find)"
    r"|\b(do|does) not (contain|include|mention|provide|address|discuss|cover|specify)"
    r"|\bno (direct |specific |relevant )?(information|mention|answer)\b"
    r"|\bnot (mentioned|addressed|covered|provided) in the (provided )?sources",
    re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


SYSTEM = "You answer questions using only the numbered sources you are given. Be concise and factual."

ANSWER_ALWAYS = """Sources:
{sources}

Question: {question}

Answer in 2-4 sentences using the sources. Cite the sources you use like [1] or [2][3]."""

ANSWER_OR_REFUSE = """Sources:
{sources}

Question: {question}

Answer in 2-4 sentences using only the sources. Cite the sources you use like [1] or [2][3].
If the sources do not contain the answer, reply with exactly this sentence and nothing else:
""" + REFUSAL

JUDGE_SYSTEM = "You are a strict fact-checker. You compare an answer with the sources it was supposed to use."

JUDGE_PROMPT = """Sources:
{sources}

Question: {question}

Answer to check:
{answer}

Decide two things:
1. grounded: is EVERY factual claim in the answer stated in, or directly implied by, the sources?
   Claims from general knowledge that the sources do not state make it NOT grounded.
2. answers_question: does the answer actually address what the question asks (not a related but different topic)?
Give a one-sentence reason."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "grounded": {"type": "boolean"},
                   "answers_question": {"type": "boolean"}},
    "required": ["reason", "grounded", "answers_question"],
}


def truncate_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def far_source_text(pid: str, corpus: dict[str, dict], extra_words: int = 120) -> str:
    """A FAR passage plus the start of the next passage in the same section.

    Passages are fixed word windows, so they can stop mid-clause (found in the CP2 human review).
    Adding the beginning of the following window gives the generator the rest of the sentence.
    """
    section, n = pid.split("#")
    text = corpus[pid]["text"]
    nxt = corpus.get(f"{section}#{int(n) + 1}")
    if nxt:
        text += " " + " ".join(nxt["text"].split()[50:50 + extra_words])  # skip the 50-word overlap
    return text


def build_sources(passage_ids: list[str], corpus: dict[str, dict], is_far: bool) -> tuple[str, list[str]]:
    """Numbered source block from the top passages (one per FAR section). Returns (text, ids used)."""
    used, seen_sections = [], set()
    for pid in passage_ids:
        key = pid.split("#")[0] if is_far else pid
        if key in seen_sections:
            continue
        seen_sections.add(key)
        used.append(pid)
        if len(used) == N_SOURCES:
            break
    blocks = []
    for i, pid in enumerate(used, 1):
        doc = corpus[pid]
        body = far_source_text(pid, corpus) if is_far else doc["text"]
        title = doc.get("title", "")
        blocks.append(f"[{i}] {title}\n{truncate_words(body, MAX_WORDS_PER_SOURCE)}".strip())
    return "\n\n".join(blocks), used


def is_refusal(answer: str) -> bool:
    """True if the first sentence declines to answer. Later hedges inside a real answer do not count."""
    first = _SENTENCE_END.split(answer.strip(), maxsplit=1)[0]
    return bool(_REFUSAL_RE.search(first))

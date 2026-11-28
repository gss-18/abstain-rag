"""CP4: can a local LLM judge replace the human groundedness check?

Re-judges the 40 human-labeled answers with a stricter claim-by-claim prompt and reports agreement with the human
labels. Measured (Nov 2026): qwen2.5:7b 50% grounded / 72% answers; qwen2.5:14b 60% / 57%. The original judge
(qwen2.5:7b, short prompt) scored 45% / 75%. None is close to the 85% target, so CP4 reports human-labeled rates.

Usage: uv run python scripts/cp4_judge_check.py qwen2.5:14b   (2 parallel requests; ~25 min for 14b on a 16 GB Mac)
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from abstain_rag.llm import Ollama
from abstain_rag.rag import JUDGE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels" / "cp4_judge_review_40.csv"
ITEMS = ROOT / "runs" / "cp4" / "items.jsonl"

CLAIM_PROMPT = """Sources:
{sources}

Question: {question}

Answer to check:
{answer}

Check the answer strictly against the sources.
Step 1. List each factual claim in the answer (numbers, findings, rules, recommendations, paragraph references).
Step 2. For each claim, find the sentence in the sources that states it. A claim is unsupported if no source states it,
if it is stronger or more general than the source (e.g. "essential" vs "enhances", "always" vs "in one study"),
if it cites the wrong source or paragraph, or if it comes from general knowledge.
Step 3. grounded = true only if every claim is supported. If the answer makes no factual claim
(it refuses, says the sources do not cover the question, or asks for clarification), grounded = true.
answers_question = true if the answer addresses what was asked; a refusal or an answer to a different question is false.
Put your claim-by-claim check in "reason", then give the two booleans."""


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    rows = {(r["corpus"], r["id"]): r for r in map(json.loads, ITEMS.open())}
    labels = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))
    llm = Ollama(model, cache_path=ROOT / "data" / "cache" / "judge_check.jsonl")

    def judge(lab: dict) -> dict:
        r = rows[(lab["corpus"], lab["id"])]
        prompt = CLAIM_PROMPT.format(sources=r["sources"], question=r["question"], answer=r["answer_always"])
        return llm.generate(prompt, schema=JUDGE_SCHEMA, max_tokens=600).json()

    with ThreadPoolExecutor(2) as pool:
        verdicts = list(pool.map(judge, labels))
    result = {"model": model, "labeled": len(labels)}
    for field, col in [("grounded", "label_grounded"), ("answers_question", "label_answers")]:
        key = next(k for k in labels[0] if k.startswith(col))
        agree = [lab[key].strip().upper().startswith("Y") == v[field] for lab, v in zip(labels, verdicts, strict=True)]
        result[f"{field}_agreement"] = round(sum(agree) / len(agree), 3)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

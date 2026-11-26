"""CP2: build the FAR test set (answerable + unanswerable questions) and a 100-item review sheet.

Question types:
  answerable  - llama3.1 writes a question from one FAR passage; that passage's section is the answer key
  near_miss   - government-contracting questions on topics the FAR does not cover (should abstain)
  ood_llm     - everyday off-topic questions (should abstain)
  ood_beir    - real science/medical queries from SciFact and NFCorpus (should abstain)

Outputs (data/far/):
  testset.jsonl   one question per line: id, type, question, answerable, section, passage, split
  qrels/{dev,test}.tsv   section-level answer key for answerable questions
  review.csv      100 questions for human labeling

Usage: uv run python scripts/cp2_build_testset.py [--limit N]   (--limit for a quick prompt check)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from abstain_rag.beir import doc_text, load_corpus, load_qrels, load_queries
from abstain_rag.llm import Ollama
from abstain_rag.retrieval import DenseRetriever
from abstain_rag.testset import near_duplicates, reject_reason, sample_passages, split_dev_test

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FAR = DATA / "far"
GENERATOR = "llama3.1:8b"

SYSTEM = "You write realistic test questions for a search system over the US Federal Acquisition Regulation (FAR)."

ANSWERABLE_PROMPT = """Passage from {title}:
\"\"\"{text}\"\"\"

Write ONE question that a contracting officer or a government contractor might type into a search box,
which this passage answers.
Rules:
- Use your own everyday words. Do not copy phrases from the passage.
- Do not mention section or clause numbers, and do not say "this passage" or "this clause".
- The question must make sense on its own, without seeing the passage.
- 8 to 25 words, ending with a question mark.
Also give the short answer, based only on the passage."""

LIST_PROMPT = """Write exactly {n} different questions that a person might type into a search box.
Topic: {topic}
Rules:
- Each question is specific and realistic, 8 to 25 words, ending with a question mark.
- Vary the wording and the sub-topic. No numbering."""

# Contracting-flavoured topics that the FAR itself does not answer (agency supplements, grants, state law,
# portals, live data, other federal rules). Near-misses test whether the system abstains on plausible questions.
NEAR_MISS_TOPICS = [
    "specific clause requirements in the Defense FAR Supplement (DFARS), e.g. DFARS 252.204-7012 cyber reporting",
    "federal grants and cooperative agreements under 2 CFR 200 (the Uniform Guidance), not contracts",
    "Georgia state government purchasing rules and the Georgia Procurement Registry",
    "technical help with SAM.gov or login.gov accounts, passwords and website errors",
    "current GSA per diem lodging and meal rates for specific US cities",
    "who won a specific named federal contract and its exact dollar value",
    "federal employee hiring, GS pay grades, and USAJOBS applications",
    "IRS tax filing deadlines and deductions for small businesses",
    "ITAR export registration fees and State Department licensing steps",
    "OSHA workplace safety inspection procedures for private employers",
    "Medicare and Medicaid billing codes and reimbursement rates",
    "city and county construction permits and local building codes",
    "how much a specific federal agency's annual budget is this year",
    "Veterans Affairs disability benefits and claims process for individuals",
    "European Union public procurement directives and thresholds",
]

OOD_TOPICS = ["cooking and recipes", "professional sports results and rules", "travel and tourism tips",
              "python and javascript programming errors", "personal fitness and nutrition", "world history",
              "movies and television shows", "home gardening and plants"]

QUESTION_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                   "required": ["question", "answer"]}


def list_schema(n: int) -> dict:
    # minItems/maxItems make Ollama's constrained decoding return exactly n questions.
    return {"type": "object", "required": ["questions"],
            "properties": {"questions": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}}}


def gen_answerable(llm: Ollama, passages: list[dict], n: int) -> tuple[list[dict], Counter]:
    rejected: Counter = Counter()
    items = []
    for k, p in enumerate(sample_passages(passages, n)):
        out = llm.generate(ANSWERABLE_PROMPT.format(title=p["title"], text=p["text"]), system=SYSTEM,
                           schema=QUESTION_SCHEMA, temperature=0.7, seed=k, max_tokens=200).json()
        q = out["question"].strip()
        reason = reject_reason(q, p["text"])
        if reason:
            rejected[reason] += 1
            continue
        items.append({"type": "answerable", "question": q, "answerable": True, "section": p["section"],
                      "passage": p["_id"], "generated_answer": out["answer"].strip()})
    return items, rejected


def gen_list(llm: Ollama, topics: list[str], kind: str, per_topic: int) -> tuple[list[dict], Counter]:
    rejected: Counter = Counter()
    items = []
    for k, topic in enumerate(topics):
        out = llm.generate(LIST_PROMPT.format(n=per_topic, topic=topic), schema=list_schema(per_topic),
                           temperature=0.8, seed=k, max_tokens=800).json()
        for q in out["questions"][:per_topic]:
            q = q.strip()
            reason = reject_reason(q)
            if reason:
                rejected[reason] += 1
                continue
            items.append({"type": kind, "question": q, "answerable": False, "topic": topic})
    return items, rejected


def beir_ood(n_each: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for name in ["scifact", "nfcorpus"]:
        folder = DATA / "beir" / name
        queries, qrels = load_queries(folder), load_qrels(folder, "test")
        qids = sorted(q for q in qrels if q in queries)
        for q in rng.sample(qids, n_each):
            items.append({"type": "ood_beir", "question": queries[q], "answerable": False, "source": f"{name}:{q}"})
    return items


def write_review(items: list[dict], far: dict, retriever: DenseRetriever, path: Path, seed: int = 42) -> None:
    """100 questions for a human: 50 answerable, 30 near-miss, 20 off-topic."""
    rng = random.Random(seed)
    want = {"answerable": 50, "near_miss": 30, "ood_llm": 20}
    sample = []
    for kind, n in want.items():
        pool = [it for it in items if it["type"] == kind]
        sample += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(sample)
    _, ranked = retriever.search([it["question"] for it in sample], k=3)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "source_section", "source_text", "generated_answer",
                    "top3_far_hits", "label_answerable_by_far (Y/N)", "label_question_ok (Y/N)", "notes"])
        for it, hits in zip(sample, ranked, strict=True):
            src = far.get(it.get("passage", ""), {})
            w.writerow([it["id"], it["type"], it["question"], it.get("section", ""), src.get("text", ""),
                        it.get("generated_answer", ""), " | ".join(far[h]["title"] for h in hits), "", "", ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="quick check: only this many answerable questions, 1 topic each")
    args = ap.parse_args()
    llm = Ollama(GENERATOR, cache_path=FAR / "llm_cache.jsonl")
    passages = [json.loads(line) for line in (FAR / "corpus.jsonl").open()]

    if args.limit:
        ans, _ = gen_answerable(llm, passages, args.limit)
        nm, _ = gen_list(llm, NEAR_MISS_TOPICS[:2], "near_miss", 5)
        for it in ans + nm:
            print(f"[{it['type']}] {it['question']}  ({it.get('section', it.get('topic', ''))[:60]})")
        return

    ans, rej_a = gen_answerable(llm, passages, 400)
    nm, rej_n = gen_list(llm, NEAR_MISS_TOPICS, "near_miss", 12)
    ood, rej_o = gen_list(llm, OOD_TOPICS, "ood_llm", 12)
    items = ans + nm + ood + beir_ood(40)
    print(f"generated: {Counter(it['type'] for it in items)}")
    print(f"rejected: answerable {dict(rej_a)}, near_miss {dict(rej_n)}, ood {dict(rej_o)}")

    retriever = DenseRetriever()
    drop = near_duplicates(retriever.encode([it["question"] for it in items]), threshold=0.95)
    items = [it for k, it in enumerate(items) if k not in drop]
    print(f"dropped {len(drop)} near-duplicates")
    for k, it in enumerate(items):
        it["id"] = f"far-q{k:04d}"
    split_dev_test(items)

    with (FAR / "testset.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    (FAR / "qrels").mkdir(exist_ok=True)
    for split in ["dev", "test"]:
        with (FAR / "qrels" / f"{split}.tsv").open("w") as f:
            f.write("query-id\tsection\tscore\n")
            for it in items:
                if it["answerable"] and it["split"] == split:
                    f.write(f"{it['id']}\t{it['section']}\t1\n")
    print(f"final: {Counter((it['type'], it['split']) for it in items)}")

    far = load_corpus(FAR)
    far_ids = list(far)
    retriever.build(far_ids, [doc_text(far[i]) for i in far_ids], cache_dir=DATA / "index" / "far")

    # Sanity check: how often does retrieval find the source section for answerable questions?
    answerable = [it for it in items if it["answerable"]]
    _, ranked = retriever.search([it["question"] for it in answerable], k=50)
    hit1 = hit10 = 0
    for it, r in zip(answerable, ranked, strict=True):
        secs = list(dict.fromkeys(pid.split("#")[0] for pid in r))  # passages -> unique sections, in rank order
        hit1 += secs[:1] == [it["section"]]
        hit10 += it["section"] in secs[:10]
    print(f"section hit@1 {hit1 / len(answerable):.2%}, hit@10 {hit10 / len(answerable):.2%}")

    write_review(items, far, retriever, FAR / "review.csv")
    print(f"saved {FAR / 'testset.jsonl'} and {FAR / 'review.csv'}")


if __name__ == "__main__":
    main()

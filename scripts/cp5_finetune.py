"""CP5: fine-tune bge-small on the FAR with synthetic questions, then re-measure retrieval and abstention.

Stages (run in order; LLM calls are cached, so `questions` can be stopped and resumed):
  questions  llama3.1:8b writes one search question per passage, one passage per FAR section, skipping every
             section used by the dev/test questions (2,084 sections -> training pairs)
  train      contrastive fine-tuning (in-batch negatives); up to 3 epochs, the epoch with the best dev nDCG@10
             is kept -> models/bge-small-far
  eval       base vs fine-tuned on the FAR test set: nDCG@10, recall@100, right section in top 5, AUROC of top1,
             and the calibrated-floor gate re-calibrated for each model -> runs/cp5/results.json

Usage: uv run python scripts/cp5_finetune.py --stage questions [--workers 3] | train | eval
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# faiss must load before torch: the other order segfaults on macOS (two OpenMP runtimes).
import faiss  # noqa: F401
import numpy as np
import torch

from abstain_rag.abstention_metrics import auroc, decision_metrics
from abstain_rag.beir import doc_text, load_corpus
from abstain_rag.evalsets import far_items, retrieve_units
from abstain_rag.finetune import train_epoch, training_passages
from abstain_rag.gates import Vocabulary, calibrate_floor, signals, word_salad_probes
from abstain_rag.llm import Ollama
from abstain_rag.metrics import evaluate
from abstain_rag.retrieval import BGE_QUERY_PREFIX, DEFAULT_MODEL, DenseRetriever
from abstain_rag.testset import reject_reason

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FAR = DATA / "far"
OUT = ROOT / "runs" / "cp5"
MODEL_DIR = ROOT / "models" / "bge-small-far"
QUESTIONS = DATA / "cp5_train_questions.jsonl"
GENERATOR = "llama3.1:8b"

# Same prompt as CP2 (scripts/cp2_build_testset.py), so training questions look like real search questions.
SYSTEM = "You write realistic test questions for a search system over the US Federal Acquisition Regulation (FAR)."
PROMPT = """Passage from {title}:
\"\"\"{text}\"\"\"

Write ONE question that a contracting officer or a government contractor might type into a search box,
which this passage answers.
Rules:
- Use your own everyday words. Do not copy phrases from the passage.
- Do not mention section or clause numbers, and do not say "this passage" or "this clause".
- The question must make sense on its own, without seeing the passage.
- 8 to 25 words, ending with a question mark.
Also give the short answer, based only on the passage."""
SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
          "required": ["question", "answer"]}


def held_out_sections() -> set[str]:
    return {json.loads(line)["section"] for line in (FAR / "testset.jsonl").open() if json.loads(line).get("section")}


def stage_questions(workers: int) -> None:
    corpus = load_corpus(FAR)
    passages = [{"_id": i, "section": i.split("#")[0], **corpus[i]} for i in corpus]
    chosen = training_passages(passages, held_out_sections())
    llm = Ollama(GENERATOR, cache_path=DATA / "cache" / "cp5_questions.jsonl")

    def work(k: int, p: dict) -> dict | None:
        out = llm.generate(PROMPT.format(title=p["title"], text=p["text"]), system=SYSTEM, schema=SCHEMA,
                           temperature=0.7, seed=100_000 + k, max_tokens=200).json()
        q = out["question"].strip()
        reason = reject_reason(q, p["text"])
        return None if reason else {"question": q, "passage": p["_id"], "section": p["section"]}

    t0, kept = time.time(), []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(work, k, p) for k, p in enumerate(chosen)]
        for n, fut in enumerate(as_completed(futures), 1):
            if (row := fut.result()) is not None:
                kept.append(row)
            if n % 100 == 0:
                left = (time.time() - t0) / n * (len(chosen) - n) / 60
                print(f"[questions] {n}/{len(chosen)}  kept {len(kept)}  ~{left:.0f} min left", flush=True)
    kept.sort(key=lambda r: r["passage"])
    QUESTIONS.write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"[questions] {len(kept)} of {len(chosen)} kept -> {QUESTIONS}")


def far_eval(retriever: DenseRetriever, corpus: dict, items: list[dict], cache: Path) -> dict:
    """Retrieval and abstention metrics for one model on the FAR dev/test questions."""
    ids = list(corpus)
    retriever.build(ids, [doc_text(corpus[i]) for i in ids], cache_dir=cache)
    scores, ranked, _ = retrieve_units(retriever, [it["question"] for it in items], is_far=True, k=100)
    vocab = Vocabulary([doc_text(corpus[i]) for i in ids])
    top1 = np.array([signals(s, it["question"], vocab)["top1"] for s, it in zip(scores, items, strict=True)])
    ans = np.array([it["answerable"] for it in items])
    dev = np.array([it["split"] == "dev" for it in items])
    correct = np.array([it["answerable"] and bool(it["relevant"] & set(r[:5]))
                        for it, r in zip(items, ranked, strict=True)])
    probe_scores, _, _ = retrieve_units(retriever, word_salad_probes(vocab), is_far=True)
    floor = calibrate_floor(np.array([s[0] for s in probe_scores]), top1[dev & ans])

    def split_metrics(mask: np.ndarray) -> dict:
        run = {it["id"]: r for it, r, m in zip(items, ranked, mask & ans, strict=True) if m}
        qrels = {it["id"]: dict.fromkeys(it["relevant"], 1) for it, m in zip(items, mask & ans, strict=True) if m}
        gate = decision_metrics(top1[mask] >= floor, ans[mask], correct[mask])
        return {**{k: round(v, 4) for k, v in evaluate(run, qrels).items()},
                "right_section_in_top5": round(float(correct[mask & ans].mean()), 4),
                "auroc_top1": round(auroc(top1[mask], ans[mask]), 4),
                "floor_gate": {k: round(v, 4) for k, v in gate.items()}}

    return {"floor": round(float(floor), 4), "dev": split_metrics(dev), "test": split_metrics(~dev)}


def stage_train(epochs: int = 3, lr: float = 2e-5, batch_size: int = 32) -> None:
    rows = [json.loads(line) for line in QUESTIONS.open()]
    corpus = load_corpus(FAR)
    pairs = [(r["question"], doc_text(corpus[r["passage"]]), r["section"]) for r in rows]
    items = far_items(DATA)
    retriever = DenseRetriever()
    model = retriever.model
    steps = epochs * math.ceil(len(pairs) / batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))
    torch.manual_seed(0)
    random.seed(0)
    best, log = -1.0, []
    base = far_eval(retriever, corpus, items, cache=None)["dev"]["ndcg@10"]
    log.append({"epoch": 0, "dev_ndcg@10": base})
    print(f"[train] {len(pairs)} pairs; epoch 0 (base) dev nDCG@10 {base:.4f}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, pairs, optimizer, scheduler, BGE_QUERY_PREFIX, batch_size, seed=epoch)
        dev_ndcg = far_eval(retriever, corpus, items, cache=None)["dev"]["ndcg@10"]
        log.append({"epoch": epoch, "loss": round(loss, 4), "dev_ndcg@10": dev_ndcg})
        print(f"[train] epoch {epoch}: loss {loss:.4f}  dev nDCG@10 {dev_ndcg:.4f}  ({time.time() - t0:.0f}s)")
        if dev_ndcg > best:
            best = dev_ndcg
            model.save(str(MODEL_DIR))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"[train] kept epoch with dev nDCG@10 {best:.4f} -> {MODEL_DIR}")


def stage_eval() -> None:
    corpus = load_corpus(FAR)
    items = far_items(DATA)
    shutil.rmtree(DATA / "index" / "far-ft", ignore_errors=True)  # embeddings from an older fine-tune would be stale
    results = {}
    for name, path, cache in [("base", DEFAULT_MODEL, DATA / "index" / "far"),
                              ("fine_tuned", str(MODEL_DIR), DATA / "index" / "far-ft")]:
        results[name] = far_eval(DenseRetriever(path), corpus, items, cache)
        t = results[name]["test"]
        print(f"[{name}] test nDCG@10 {t['ndcg@10']}  recall@100 {t['recall@100']}  "
              f"top5 {t['right_section_in_top5']}  AUROC {t['auroc_top1']}  floor-gate {t['floor_gate']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["questions", "train", "eval"])
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    {"questions": lambda: stage_questions(args.workers), "train": stage_train, "eval": stage_eval}[args.stage]()


if __name__ == "__main__":
    main()

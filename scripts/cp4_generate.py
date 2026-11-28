"""CP4: generate answers, LLM self-refusal (gate 6), groundedness judge, human check, end-to-end report.

Stages (run in order; every LLM call is cached, so any stage can be stopped and restarted):
  generate  llama3.1:8b writes two answers per question: "always answer" (used by gates 1-5)
            and "answer or say I don't know" (gate 6)
  judge     qwen2.5:7b checks every non-refusal answer: grounded in the sources? answers the question?
  review    writes 100 judged answers to data/cp4_review.csv for a human
  report    metrics per corpus and gate -> runs/cp4/results.json, risk_coverage.png

Usage: uv run python scripts/cp4_generate.py --stage generate|judge|review|report|llm
       (llm = generate then judge, for the overnight run)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from abstain_rag.abstention_metrics import risk_coverage  # noqa: E402
from abstain_rag.beir import doc_text, load_corpus  # noqa: E402
from abstain_rag.evalsets import CORPORA, all_items, retrieve_units  # noqa: E402
from abstain_rag.gates import GateParams, Vocabulary, decide, signals  # noqa: E402
from abstain_rag.llm import Ollama  # noqa: E402
from abstain_rag.rag import (  # noqa: E402
    ANSWER_ALWAYS,
    ANSWER_OR_REFUSE,
    JUDGE_PROMPT,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    SYSTEM,
    build_sources,
    is_refusal,
)
from abstain_rag.retrieval import DenseRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "runs" / "cp4"
ITEMS = OUT / "items.jsonl"
REVIEW = DATA / "cp4_review_40.csv"
LABELS = ROOT / "labels" / "cp4_judge_review_40.csv"
PER_CORPUS = 10  # 40 rows in total: small enough for one person to label by hand
GENERATOR, JUDGE = "llama3.1:8b", "qwen2.5:7b"
OPTIONS = {"temperature": 0.0, "seed": 42}
WORKERS = 3  # parallel requests; Ollama must run with OLLAMA_NUM_PARALLEL >= WORKERS
BEIR_PER_CLASS = 100  # BEIR: 100 answerable + 100 unanswerable test queries per corpus; FAR: full test set


def select(items: list[dict], name: str, seed: int = 42) -> list[dict]:
    test = [it for it in items if it["split"] == "test"]
    if name == "far":
        return test
    rng = random.Random(seed)
    ans = [it for it in test if it["answerable"]]
    unans = [it for it in test if not it["answerable"]]
    return rng.sample(ans, BEIR_PER_CLASS) + rng.sample(unans, BEIR_PER_CLASS)


def prepare() -> list[dict]:
    """Pick questions, retrieve, compute signals and CP3 gate decisions, build the source block."""
    cp3 = json.loads((ROOT / "runs" / "cp3" / "results.json").read_text())
    retriever = DenseRetriever()
    sets = all_items(DATA)
    rows = []
    for name in CORPORA:
        is_far = name == "far"
        corpus = load_corpus(DATA / ("far" if is_far else f"beir/{name}"))
        ids = list(corpus)
        retriever.build(ids, [doc_text(corpus[i]) for i in ids], cache_dir=DATA / "index" / name)
        vocab = Vocabulary([doc_text(corpus[i]) for i in ids])
        chosen = select(sets[name], name)
        t = time.perf_counter()
        scores, units, passages = retrieve_units(retriever, [it["question"] for it in chosen], is_far)
        retrieve_ms = (time.perf_counter() - t) / len(chosen) * 1000
        params = GateParams(**cp3[name]["params"])
        for it, s, u, p in zip(chosen, scores, units, passages, strict=True):
            sig = signals(s, it["question"], vocab)
            gates = {g: bool(v[0]) for g, v in decide({k: np.array([v]) for k, v in sig.items()}, params).items()}
            sources, used = build_sources(p, corpus, is_far)
            rows.append({"id": it["id"], "corpus": name, "type": it["type"], "question": it["question"],
                         "answerable": it["answerable"], "correct_at_5": bool(it["relevant"] & set(u[:5])),
                         "signals": sig, "retrieval_gates": gates, "sources": sources, "source_ids": used,
                         "retrieve_ms": retrieve_ms})
        print(f"[prepare] {name}: {len(chosen)} questions")
    return rows


def load_rows() -> list[dict]:
    return [json.loads(line) for line in ITEMS.open()] if ITEMS.exists() else []


def save_rows(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = ITEMS.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(ITEMS)


def run_parallel(rows: list[dict], work, label: str) -> None:
    """Apply work(row) to every row with WORKERS threads; save progress every 25 rows (atomic rewrite)."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(work, r) for r in rows]
        for k, fut in enumerate(as_completed(futures), 1):
            fut.result()  # re-raise any error from the worker
            if k % 25 == 0:
                save_rows(rows)
                rate = (time.time() - t0) / k
                print(f"[{label}] {k}/{len(rows)}  ~{rate * (len(rows) - k) / 60:.0f} min left", flush=True)
    save_rows(rows)


def stage_generate(rows: list[dict]) -> None:
    llm = Ollama(GENERATOR, cache_path=DATA / "cache" / "generate.jsonl")

    def work(r: dict) -> None:
        for mode, template in [("always", ANSWER_ALWAYS), ("refuse", ANSWER_OR_REFUSE)]:
            g = llm.generate(template.format(sources=r["sources"], question=r["question"]), system=SYSTEM,
                             max_tokens=220, **OPTIONS)
            r[f"answer_{mode}"] = g.text.strip()
            r[f"gen_{mode}"] = {"seconds": round(g.seconds, 2), "prompt_tokens": g.prompt_tokens,
                                "output_tokens": g.output_tokens}

    run_parallel(rows, work, "generate")


def stage_judge(rows: list[dict]) -> None:
    llm = Ollama(JUDGE, cache_path=DATA / "cache" / "judge.jsonl")

    def work(r: dict) -> None:
        for mode in ["always", "refuse"]:
            answer = r[f"answer_{mode}"]
            if is_refusal(answer):
                r[f"judge_{mode}"] = None
                continue
            g = llm.generate(JUDGE_PROMPT.format(sources=r["sources"], question=r["question"], answer=answer),
                             system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA, max_tokens=150, **OPTIONS)
            r[f"judge_{mode}"] = {**g.json(), "seconds": round(g.seconds, 2)}

    run_parallel(rows, work, "judge")


def cited_sources(sources: str, answer: str) -> str:
    """Only the sources the answer cites (all of them if it cites none), so the sheet is quick to read."""
    blocks = re.split(r"\n(?=\[\d+\] )", sources)
    cited = set(re.findall(r"\[(\d+)\]", answer))
    keep = [b for b in blocks if re.match(r"\[(\d+)\]", b) and re.match(r"\[(\d+)\]", b).group(1) in cited]
    return "\n\n".join(keep or blocks)


def stage_review(rows: list[dict], seed: int = 7) -> None:
    """PER_CORPUS judged 'always' answers per corpus, half the judge called grounded, half not (where possible)."""
    rng = random.Random(seed)
    sample = []
    for name in CORPORA:
        judged = [r for r in rows if r["corpus"] == name and r.get("judge_always")]
        yes = [r for r in judged if r["judge_always"]["grounded"]]
        no = [r for r in judged if not r["judge_always"]["grounded"]]
        n_no = min(len(no), PER_CORPUS // 2)
        sample += rng.sample(no, n_no) + rng.sample(yes, PER_CORPUS - n_no)
    rng.shuffle(sample)
    with REVIEW.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "corpus", "question", "answer", "cited_sources", "label_grounded (Y/N)",
                    "label_answers_question (Y/N)", "notes"])
        for r in sample:
            w.writerow([r["id"], r["corpus"], r["question"], r["answer_always"],
                        cited_sources(r["sources"], r["answer_always"]), "", "", ""])
    print(f"[review] wrote {len(sample)} rows to {REVIEW} (judge verdicts hidden so they do not bias you)")


def good(judge: dict | None) -> bool:
    return bool(judge and judge["grounded"] and judge["answers_question"])


def outcomes(rows: list[dict]) -> dict[str, list[tuple[bool, bool]]]:
    """Per gate, per question: (answered?, good answer?)."""
    out: dict[str, list[tuple[bool, bool]]] = {}
    for r in rows:
        always_ok = not is_refusal(r["answer_always"])
        refuse_ok = not is_refusal(r["answer_refuse"])
        for g, passed in r["retrieval_gates"].items():
            out.setdefault(g, []).append((passed and always_ok, good(r["judge_always"])))
        out.setdefault("6_llm_self_refusal", []).append((refuse_ok, good(r["judge_refuse"])))
        floor_and_self = r["retrieval_gates"]["3_calibrated_floor"] and refuse_ok
        out.setdefault("3+6_floor_then_self_refusal", []).append((floor_and_self, good(r["judge_refuse"])))
    return out


def llm_ran(row: dict, gate: str) -> bool:
    """Generation only runs when the retrieval gate lets the question through (gate 6 alone always generates)."""
    if gate == "6_llm_self_refusal":
        return True
    if gate.startswith("3+"):
        return row["retrieval_gates"]["3_calibrated_floor"]
    return row["retrieval_gates"][gate]


def gate_metrics(rows: list[dict], decisions: list[tuple[bool, bool]], gate: str) -> dict:
    answered = np.array([a for a, _ in decisions])
    good_ans = np.array([g for _, g in decisions])
    answerable = np.array([r["answerable"] for r in rows])
    mode = "refuse" if "6" in gate else "always"
    grounded = np.array([bool(r[f"judge_{mode}"] and r[f"judge_{mode}"]["grounded"]) for r in rows])
    n_ans = answered.sum()
    refused = ~answered
    gen_s = np.array([r[f"gen_{mode}"]["seconds"] for r in rows])
    tokens = np.array([r[f"gen_{mode}"]["prompt_tokens"] + r[f"gen_{mode}"]["output_tokens"] for r in rows])
    runs_llm = np.array([llm_ran(r, gate) for r in rows])
    return {
        "coverage": round(float(answered.mean()), 4),
        "hallucination_rate": round(float((answered & ~grounded).sum() / n_ans), 4) if n_ans else 0.0,
        "risk": round(float((answered & ~good_ans).sum() / n_ans), 4) if n_ans else 0.0,
        "good_answers_share_of_answerable": round(
            float((answered & good_ans & answerable).sum() / answerable.sum()), 4),
        "refusal_precision": round(float((refused & ~answerable).sum() / refused.sum()), 4) if refused.any() else None,
        "refusal_recall": round(float((refused & ~answerable).sum() / (~answerable).sum()), 4),
        "avg_llm_seconds_per_query": round(float((gen_s * runs_llm).mean()), 2),
        "avg_llm_tokens_per_query": round(float((tokens * runs_llm).mean()), 1),
    }


def load_labels(rows: list[dict]) -> list[tuple[dict, dict]] | None:
    """(item row, human label) pairs. Labels hold Y/N strings; returned as booleans."""
    if not LABELS.exists():
        return None
    # Off-topic questions are reused in every corpus under the same id, so (corpus, id) is the key.
    by_key = {(r["corpus"], r["id"]): r for r in rows}
    labels = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))
    g_col = next(k for k in labels[0] if k.startswith("label_grounded"))
    a_col = next(k for k in labels[0] if k.startswith("label_answers"))
    yes = lambda v: v.strip().upper().startswith("Y")  # noqa: E731
    return [(by_key[(lab["corpus"], lab["id"])], {"grounded": yes(lab[g_col]), "answers_question": yes(lab[a_col])})
            for lab in labels if lab[g_col].strip()]


def judge_agreement(pairs: list[tuple[dict, dict]]) -> dict:
    out: dict = {"labeled": len(pairs)}
    for field in ("grounded", "answers_question"):
        h = np.array([lab[field] for _, lab in pairs])
        j = np.array([r["judge_always"][field] for r, _ in pairs])
        out[field] = {"agreement": round(float((h == j).mean()), 3),
                      "human_no_judge_yes": int((~h & j).sum()), "human_yes_judge_no": int((h & ~j).sum())}
    return out


def human_rate(rows: list[dict], pairs: list[tuple[dict, dict]], keep, reps: int = 2000, seed: int = 0) -> dict:
    """Share of answers a human calls NOT grounded, among judged 'always' answers that pass `keep`.

    The review sample was drawn per corpus and per judge verdict (half yes, half no), so each labeled answer
    stands for N/n answers of its stratum. Weighted ratio estimate; 95% interval from a stratified bootstrap."""
    strata: dict[tuple[str, bool], list[dict]] = {}
    for r in rows:
        if r.get("judge_always"):
            strata.setdefault((r["corpus"], r["judge_always"]["grounded"]), []).append(r)
    sample: dict[tuple[str, bool], list[tuple[bool, bool]]] = {}
    for r, lab in pairs:
        sample.setdefault((r["corpus"], r["judge_always"]["grounded"]), []).append((keep(r), not lab["grounded"]))

    def estimate(draw: dict) -> float:
        num = den = 0.0
        for s, obs in draw.items():
            w = len(strata[s]) / len(obs)
            num += w * sum(k and bad for k, bad in obs)
            den += w * sum(k for k, _ in obs)
        return num / den if den else float("nan")

    rng = np.random.default_rng(seed)
    boots = [estimate({s: [obs[i] for i in rng.integers(0, len(obs), len(obs))] for s, obs in sample.items()})
             for _ in range(reps)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    n_kept = sum(keep(r) for r, _ in pairs)
    return {"rate": round(estimate(sample), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "labeled_answers_used": n_kept}


def stage_report(rows: list[dict]) -> None:
    results: dict = {}
    pairs = load_labels(rows)
    if pairs:
        results["judge_vs_human"] = judge_agreement(pairs)
        # The judge is not reliable enough on grounding (see LEARN.md CP4), so the headline hallucination rates
        # come from the human labels. Only the 'always' answers were labeled, so this covers gates 1-5.
        answered = lambda r: not is_refusal(r["answer_always"])  # noqa: E731
        results["human_hallucination_rate"] = {
            g: human_rate(rows, pairs, lambda r, g=g: answered(r) and r["retrieval_gates"][g])
            for g in ("1_no_gate", "2_fixed_threshold", "3_calibrated_floor")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    for ax, name in zip(axes.flat, CORPORA, strict=True):
        sub = [r for r in rows if r["corpus"] == name]
        res = {g: gate_metrics(sub, d, g) for g, d in outcomes(sub).items()}
        results[name] = {"questions": len(sub), "answerable": sum(r["answerable"] for r in sub),
                         "retrieve_ms": round(sub[0]["retrieve_ms"], 2), "gates": res}
        conf = np.array([r["signals"]["top1"] for r in sub])
        ok = np.array([good(r["judge_always"]) and not is_refusal(r["answer_always"]) for r in sub])
        cov, risk = risk_coverage(conf, ok)
        ax.plot(cov, risk, lw=1.6, label=f"rank by top1 (AURC {risk.mean():.3f})")
        for g, m in res.items():
            ax.scatter(m["coverage"], m["risk"], s=55, zorder=5, edgecolor="black", label=g)
        ax.set(title=name, ylim=(0, 1), xlabel="coverage", ylabel="risk (answers judged bad)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("End-to-end risk-coverage with generated answers (judge: qwen2.5:7b)")
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=150)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results.get(k) for k in ("judge_vs_human", "human_hallucination_rate")}, indent=1))
    for name in CORPORA:
        head = f"{'gate':30s} {'cover':>6s} {'halluc':>7s} {'risk':>6s} {'refP':>6s} {'refR':>6s} {'llm_s':>6s}"
        print(f"\n[{name}]  {head}")
        for g, m in results[name]["gates"].items():
            rp = m["refusal_precision"]
            print(f"   {g:36s} {m['coverage']:6.3f} {m['hallucination_rate']:7.3f} {m['risk']:6.3f} "
                  f"{(rp if rp is not None else float('nan')):6.3f} {m['refusal_recall']:6.3f} "
                  f"{m['avg_llm_seconds_per_query']:6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["generate", "judge", "llm", "review", "report"])
    args = ap.parse_args()
    rows = load_rows()
    if not rows:
        rows = prepare()
        save_rows(rows)
    if args.stage in ("generate", "llm"):
        stage_generate(rows)
    if args.stage in ("judge", "llm"):
        stage_judge(rows)
    if args.stage == "review":
        stage_review(rows)
    if args.stage == "report":
        stage_report(rows)


if __name__ == "__main__":
    main()

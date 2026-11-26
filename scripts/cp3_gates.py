"""CP3: abstention gates 1-5 on all four corpora.

For each corpus: build answerable + unanswerable query sets (dev and test), retrieve, compute signals,
tune gate thresholds on dev only, and report on test:
  AUROC per signal, coverage, refusal precision/recall, false-refusal rate, risk, AURC, risk-coverage curves.

Unanswerable queries for the BEIR corpora are real queries from an unrelated dataset (finance vs biomedical)
plus the FAR off-topic questions. For FAR they come from the CP2 test set (near-miss + off-topic).

Usage: uv run python scripts/cp3_gates.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from abstain_rag.abstention_metrics import aurc, auroc, decision_metrics, risk_coverage  # noqa: E402
from abstain_rag.beir import doc_text, load_corpus, load_qrels, load_queries  # noqa: E402
from abstain_rag.gates import (  # noqa: E402
    GateParams,
    Vocabulary,
    calibrate_floor,
    decide,
    section_scores,
    signals,
    tune_threshold,
    word_salad_probes,
)
from abstain_rag.retrieval import DenseRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "runs" / "cp3"
BEIR = ["scifact", "fiqa", "nfcorpus"]
DEV_SPLIT = {"scifact": "train", "fiqa": "dev", "nfcorpus": "dev"}  # SciFact has no dev split
UNRELATED = {"scifact": ["fiqa"], "nfcorpus": ["fiqa"], "fiqa": ["scifact", "nfcorpus"]}
SIGNALS = ["top1", "margin", "vocab"]
TOP_K_CORRECT = 5


def far_items() -> list[dict]:
    items = [json.loads(line) for line in (DATA / "far" / "testset.jsonl").open()]
    return [{"question": it["question"], "answerable": it["answerable"], "split": it["split"], "type": it["type"],
             "relevant": {it["section"]} if it["answerable"] else set()} for it in items]


def beir_items(name: str, ood_llm: list[dict], max_dev: int = 300, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    folder = DATA / "beir" / name
    queries = load_queries(folder)
    items = []
    for split, qsplit in [("dev", DEV_SPLIT[name]), ("test", "test")]:
        qrels = load_qrels(folder, qsplit)
        qids = sorted(q for q in qrels if q in queries)
        if split == "dev" and len(qids) > max_dev:
            qids = rng.sample(qids, max_dev)
        items += [{"question": queries[q], "answerable": True, "split": split, "type": "answerable",
                   "relevant": set(qrels[q])} for q in qids]
    n_ans = len(items)
    pool = [t for other in UNRELATED[name] for t in load_queries(DATA / "beir" / other).values()]
    cross = rng.sample(pool, n_ans - len(ood_llm))
    n_dev = round(len(cross) * 0.3)
    items += [{"question": q, "answerable": False, "split": "dev" if k < n_dev else "test", "type": "unrelated_beir",
               "relevant": set()} for k, q in enumerate(cross)]
    items += [{**it, "relevant": set()} for it in ood_llm]
    return items


def run_corpus(name: str, items: list[dict], retriever: DenseRetriever, corpus: dict) -> dict:
    is_far = name == "far"
    texts = [doc_text(corpus[i]) for i in corpus]
    vocab = Vocabulary(texts)

    def retrieve(questions: list[str]) -> tuple[list[np.ndarray], list[list[str]]]:
        scores, ranked = retriever.search(questions, k=50)
        if not is_far:
            return list(scores), ranked
        out_s, out_r = [], []
        for s, r in zip(scores, ranked, strict=True):  # passages -> sections, best passage per section
            out_s.append(section_scores(s, r))
            out_r.append(list(dict.fromkeys(pid.split("#")[0] for pid in r)))
        return out_s, out_r

    t = time.perf_counter()
    scores, ranked = retrieve([it["question"] for it in items])
    retrieve_ms = (time.perf_counter() - t) / len(items) * 1000
    t = time.perf_counter()
    sig_rows = [signals(s, it["question"], vocab) for s, it in zip(scores, items, strict=True)]
    signal_ms = (time.perf_counter() - t) / len(items) * 1000

    sig = {k: np.array([r[k] for r in sig_rows]) for k in SIGNALS}
    answerable = np.array([it["answerable"] for it in items])
    correct = np.array([it["answerable"] and bool(it["relevant"] & set(r[:TOP_K_CORRECT]))
                        for it, r in zip(items, ranked, strict=True)])
    dev = np.array([it["split"] == "dev" for it in items])
    probe_scores, _ = retrieve(word_salad_probes(vocab))
    probe_top1 = np.array([s[0] for s in probe_scores])
    return {"sig": sig, "answerable": answerable, "correct": correct, "dev": dev, "probe_top1": probe_top1,
            "retrieve_ms": retrieve_ms, "signal_ms": signal_ms, "types": [it["type"] for it in items]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    retriever = DenseRetriever()
    far = far_items()
    ood_llm = [it for it in far if it["type"] == "ood_llm"]
    runs = {}
    for name in BEIR + ["far"]:
        folder = DATA / ("far" if name == "far" else f"beir/{name}")
        corpus = load_corpus(folder)
        ids = list(corpus)
        retriever.build(ids, [doc_text(corpus[i]) for i in ids], cache_dir=DATA / "index" / name)
        items = far if name == "far" else beir_items(name, ood_llm)
        runs[name] = run_corpus(name, items, retriever, corpus)
        print(f"[{name}] {len(items)} queries ({int(runs[name]['answerable'].sum())} answerable)")

    # Gate 2: one global threshold, tuned on all dev data pooled.
    pooled_top1 = np.concatenate([r["sig"]["top1"][r["dev"]] for r in runs.values()])
    pooled_ans = np.concatenate([r["answerable"][r["dev"]] for r in runs.values()])
    fixed_t = tune_threshold(pooled_top1, pooled_ans)

    results, fig_data = {"fixed_threshold_global": round(fixed_t, 4)}, {}
    for name, r in runs.items():
        sig, ans, cor, dev = r["sig"], r["answerable"], r["correct"], r["dev"]
        good_dev = sig["top1"][dev & ans]
        floor = calibrate_floor(r["probe_top1"], good_dev)
        floor_dev = sig["top1"][dev] >= floor
        dev_sig = {k: v[dev] for k, v in sig.items()}
        params = GateParams(
            fixed_t=fixed_t, floor=floor,
            margin_m=tune_threshold(dev_sig["margin"], ans[dev], base=floor_dev),
            vocab_v=tune_threshold(dev_sig["vocab"], ans[dev], base=floor_dev),
        )
        test = ~dev
        t_sig = {k: v[test] for k, v in sig.items()}
        decisions = decide(t_sig, params)
        res = {
            "test_queries": int(test.sum()), "test_answerable": int(ans[test].sum()),
            "retrieval_correct_at_5": round(float(cor[test & ans].mean()), 4),
            "params": {k: round(v, 4) for k, v in params.__dict__.items()},
            "probe_top1_p95": round(float(np.quantile(r["probe_top1"], 0.95)), 4),
            "auroc": {k: round(auroc(t_sig[k], ans[test]), 4) for k in SIGNALS},
            "aurc": {k: round(aurc(t_sig[k], cor[test]), 4) for k in SIGNALS},
            "gates": {g: {k: round(v, 4) for k, v in decision_metrics(d, ans[test], cor[test]).items()}
                      for g, d in decisions.items()},
            "ms_per_query": {"retrieval": round(r["retrieve_ms"], 2), "signals_and_gates": round(r["signal_ms"], 3)},
        }
        if name == "far":
            types = np.array(r["types"])[test]
            res["refused_by_type"] = {g: {t: round(float((~d[types == t]).mean()), 3) for t in sorted(set(types))}
                                      for g, d in decisions.items()}
        results[name] = res
        fig_data[name] = (t_sig, cor[test], res["gates"])

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    plot(fig_data)
    print_table(results)


def plot(fig_data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    markers = {"1_no_gate": "o", "2_fixed_threshold": "s", "3_calibrated_floor": "^", "4_floor_margin": "D",
               "5_floor_vocab": "v"}
    for ax, (name, (sig, cor, gates)) in zip(axes.flat, fig_data.items(), strict=True):
        for k in SIGNALS:
            cov, risk = risk_coverage(sig[k], cor)
            ax.plot(cov, risk, label=f"rank by {k} (AURC {risk.mean():.3f})", lw=1.6)
        for g, m in gates.items():
            ax.scatter(m["coverage"], m["risk"], marker=markers[g], s=55, zorder=5, edgecolor="black",
                       label=g.split("_", 1)[1].replace("_", " "))
        ax.set_title(name)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_xlabel("coverage (share of questions answered)")
        ax.set_ylabel("risk (share of answers without support)")
    for ax in axes.flat:
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Risk-coverage on test sets (lower curve = better; markers = gates)")
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=150)


def print_table(results: dict) -> None:
    print(f"\nglobal fixed threshold: {results['fixed_threshold_global']}")
    for name in BEIR + ["far"]:
        r = results[name]
        print(f"\n[{name}] AUROC {r['auroc']}  floor {r['params']['floor']}  correct@5 {r['retrieval_correct_at_5']}")
        print(f"  {'gate':20s} {'cover':>6s} {'ref_P':>6s} {'ref_R':>6s} {'falseR':>6s} {'risk':>6s}")
        for g, m in r["gates"].items():
            print(f"  {g:20s} {m['coverage']:6.3f} {m['refusal_precision']:6.3f} {m['refusal_recall']:6.3f} "
                  f"{m['false_refusal_rate']:6.3f} {m['risk']:6.3f}")
    print(f"\n[far] refused share by question type:\n{json.dumps(results['far']['refused_by_type'], indent=1)}")


if __name__ == "__main__":
    main()

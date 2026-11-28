"""README figure: FAR gate results (CP4) and the judge-vs-human check -> docs/img/far_gates.png"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GATES = {"1_no_gate": "No gate", "3_calibrated_floor": "Calibrated\nfloor", "6_llm_self_refusal": "LLM\nself-refusal",
         "3+6_floor_then_self_refusal": "Floor,\nthen LLM"}
# Judge agreement with the 40 human labels (scripts/cp4_judge_check.py, LEARN.md CP4).
JUDGES = {"qwen2.5:7b": 0.45, "qwen2.5:7b\nclaim-by-claim": 0.50, "qwen2.5:14b\nclaim-by-claim": 0.60,
          "Claude\n(blind)": 0.725}


def main() -> None:
    far = json.loads((ROOT / "runs" / "cp4" / "results.json").read_text())["far"]["gates"]
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    names = [GATES[g] for g in GATES]
    recall = [far[g]["refusal_recall"] * 100 for g in GATES]
    secs = [far[g]["avg_llm_seconds_per_query"] for g in GATES]
    bars = a.bar(names, recall, color=["#afb8c1", "#0969da", "#8250df", "#1a7f37"], width=0.6)
    for bar, r, s in zip(bars, recall, secs, strict=True):
        a.text(bar.get_x() + bar.get_width() / 2, r + 1.5, f"{r:.0f}%\n{s:.1f} s LLM", ha="center", fontsize=9)
    a.set(ylim=(0, 118), ylabel="Unanswerable questions refused (%)",
          title="FAR test set: refusals and LLM time per question")
    a.set_yticks(range(0, 101, 25))

    jn = list(JUDGES)
    jv = [v * 100 for v in JUDGES.values()]
    b.bar(jn, jv, color=["#cf222e", "#cf222e", "#bc4c00", "#57606a"], width=0.6)
    for x, v in enumerate(jv):
        b.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    b.axhline(85, color="#1f2328", ls="--", lw=1)
    b.text(3.45, 86.5, "target 85%", ha="right", fontsize=8.5)
    b.set(ylim=(0, 100), ylabel="Agreement with human on 'grounded' (%)",
          title="Can an LLM judge replace the human check?")
    fig.tight_layout()
    out = ROOT / "docs" / "img" / "far_gates.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

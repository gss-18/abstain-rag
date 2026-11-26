"""CP2: apply the human review to the FAR test set.

Reads labels/far_review_100.csv (100 hand-labeled questions), measures how often the generated labels were
wrong, removes questions the reviewer marked as not OK, and rewrites data/far/testset.jsonl and qrels.

Usage: uv run python scripts/cp2_apply_review.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAR = ROOT / "data" / "far"
LABELS = ROOT / "labels" / "far_review_100.csv"
OUT = ROOT / "runs" / "cp2"


def yes(v: str) -> bool:
    return v.strip().upper().startswith("Y")


def main() -> None:
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8-sig")))
    a_col = next(k for k in rows[0] if k.startswith("label_answerable"))
    q_col = next(k for k in rows[0] if k.startswith("label_question"))
    n = len(rows)
    disagree = [r["id"] for r in rows if yes(r[a_col]) != (r["type"] == "answerable")]
    not_ok = {r["id"] for r in rows if not yes(r[q_col])}
    # "Rule of three": with 0 errors in n samples, the 95% upper bound on the error rate is about 3/n.
    upper = 3 / n if not disagree else None
    summary = {
        "reviewed": n,
        "answerable_label_agreement": round(1 - len(disagree) / n, 3),
        "answerable_label_disagreements": disagree,
        "answerable_label_error_95pct_upper": round(upper, 3) if upper is not None else None,
        "question_ok_rate": round(1 - len(not_ok) / n, 3),
        "removed_not_ok": sorted(not_ok),
    }

    items = [json.loads(line) for line in (FAR / "testset.jsonl").open()]
    reviewed = {r["id"]: r for r in rows}
    kept = []
    for it in items:
        if it["id"] in not_ok:
            continue
        if it["id"] in reviewed:
            it["human_reviewed"] = True
            # Any label disagreement is resolved in favour of the human.
            it["answerable"] = yes(reviewed[it["id"]][a_col])
        kept.append(it)
    with (FAR / "testset.jsonl").open("w") as f:
        for it in kept:
            f.write(json.dumps(it) + "\n")
    for split in ["dev", "test"]:
        with (FAR / "qrels" / f"{split}.tsv").open("w") as f:
            f.write("query-id\tsection\tscore\n")
            for it in kept:
                if it["answerable"] and it["split"] == split:
                    f.write(f"{it['id']}\t{it['section']}\t1\n")
    summary["final_questions"] = len(kept)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert len(items) - len(kept) == len(not_ok)


if __name__ == "__main__":
    main()

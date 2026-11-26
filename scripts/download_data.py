"""Download the public datasets used by the benchmark.

BEIR (SciFact, FiQA, NFCorpus) comes from the BeIR mirrors on Hugging Face and is
written to data/beir/<name>/ in the standard BEIR layout:
    corpus.jsonl   {"_id", "title", "text"}
    queries.jsonl  {"_id", "text"}
    qrels/<split>.tsv  query-id \t corpus-id \t score  (with header)

The FAR (Federal Acquisition Regulation) is downloaded as raw HTML, one file per
part, to data/raw/far/. It is parsed into passages by a later step.

Usage: uv run python scripts/download_data.py [--only beir|far]
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HF = "https://huggingface.co/datasets/BeIR"
BEIR_SETS = {
    "scifact": ["train", "test"],
    "fiqa": ["train", "dev", "test"],
    "nfcorpus": ["train", "dev", "test"],
}
FAR_BASE = "https://www.acquisition.gov/sites/default/files/current/far/compiled_html"
FAR_INDEX = "https://www.acquisition.gov/browse/index/far"
HEADERS = {"User-Agent": "abstain-rag-benchmark/0.1"}


def fetch(client: httpx.Client, url: str) -> bytes:
    for attempt in range(4):
        try:
            r = client.get(url, follow_redirects=True, timeout=120)
            r.raise_for_status()
            return r.content
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def write_jsonl(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    with path.open("w") as f:
        for row in df[cols].itertuples(index=False):
            f.write(json.dumps(dict(zip(cols, (str(v) for v in row), strict=True))) + "\n")


def download_beir(client: httpx.Client) -> None:
    for name, splits in BEIR_SETS.items():
        out = DATA / "beir" / name
        if (out / "corpus.jsonl").exists():
            print(f"[beir] {name}: already present, skipping")
            continue
        (out / "qrels").mkdir(parents=True, exist_ok=True)
        base = f"{HF}/{name}/resolve/main"
        corpus = pd.read_parquet(io.BytesIO(fetch(client, f"{base}/corpus/corpus-00000-of-00001.parquet")))
        queries = pd.read_parquet(io.BytesIO(fetch(client, f"{base}/queries/queries-00000-of-00001.parquet")))
        write_jsonl(corpus, ["_id", "title", "text"], out / "corpus.jsonl")
        write_jsonl(queries, ["_id", "text"], out / "queries.jsonl")
        for split in splits:
            tsv = fetch(client, f"{HF}/{name}-qrels/resolve/main/{split}.tsv")
            (out / "qrels" / f"{split}.tsv").write_bytes(tsv)
        print(f"[beir] {name}: {len(corpus):,} docs, {len(queries):,} queries, splits {splits}")


def download_far(client: httpx.Client) -> None:
    import re

    out = DATA / "raw" / "far"
    out.mkdir(parents=True, exist_ok=True)
    index = fetch(client, FAR_INDEX).decode("utf-8", "replace")
    parts = sorted({int(n) for n in re.findall(r"compiled_html/part_(\d+)\.html", index)})
    if not parts:
        raise RuntimeError("No FAR parts found on the index page; the site layout may have changed.")
    for n in parts:
        path = out / f"part_{n}.html"
        if path.exists() and path.stat().st_size > 0:
            continue
        path.write_bytes(fetch(client, f"{FAR_BASE}/part_{n}.html"))
        time.sleep(0.5)  # be polite to acquisition.gov
    total = sum(p.stat().st_size for p in out.glob("part_*.html"))
    print(f"[far] {len(parts)} parts, {total / 1e6:.1f} MB in {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["beir", "far"])
    args = ap.parse_args()
    with httpx.Client(headers=HEADERS) as client:
        if args.only in (None, "beir"):
            download_beir(client)
        if args.only in (None, "far"):
            download_far(client)


if __name__ == "__main__":
    main()

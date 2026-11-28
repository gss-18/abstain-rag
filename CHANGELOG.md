# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-28

First release: the full benchmark, CP0 to CP5.

### Added

- **CP1 retrieval:** bge-small-en-v1.5 + exact FAISS search, checked against published BEIR scores (SciFact,
  FiQA, NFCorpus, all within 0.1 nDCG point); FAR parsed into 6,003 passages over 3,117 sections.
- **CP2 FAR test set:** 719 LLM-written questions (answerable, near-miss, off-topic), filtered for copying and
  near-duplicates, 100 reviewed by hand; dev/test split.
- **CP3 gates:** fixed threshold, calibrated floor (from nonsense probes, no labeled refusals), floor + margin,
  floor + vocabulary; AUROC, refusal precision/recall, risk-coverage curves on four corpora.
- **CP4 generation:** llama3.1:8b answers, LLM self-refusal (gate 6) and floor-then-LLM stacking; qwen2.5 judge
  checked against 40 human labels (45-60% agreement for local judges); human-weighted hallucination rates with
  stratified bootstrap intervals; refusal and cost results.
- **CP5 fine-tuning:** contrastive fine-tuning on 2,009 synthetic FAR questions from sections the test never uses;
  nDCG@10 0.712 → 0.757; shows the floor must be recalibrated after fine-tuning.
- Disk cache for every LLM call, so long runs resume where they stopped.
- `docs/LEARN.md` walkthrough with check-yourself questions, 32 tests, ruff, CI.

[1.0.0]: https://github.com/gss-18/abstain-rag/releases/tag/v1.0.0

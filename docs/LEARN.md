# Learn abstain-rag: build it yourself

One section per checkpoint: **what** each step does, **why**, and the **exact commands**.
At the end of each section there are self-check questions (likely interview questions).

---

## CP0: Setup, data, smoke test

### 1. Tools

| Tool | What it is | Why we need it |
|---|---|---|
| **Homebrew** | Mac package installer | Installs Ollama in one command |
| **Ollama** | Runs open LLMs locally, with a web API on `localhost:11434` | Free answer generation and judging, no API key |
| **uv** | Fast Python project manager (replaces pip + venv + poetry) | One tool for the virtual env, dependencies and lockfile |

```bash
brew install ollama
brew services start ollama      # runs Ollama in the background, also after reboot
ollama pull llama3.1:8b         # generator (~4.9 GB)
ollama pull qwen2.5:7b          # judge (~4.7 GB): a different family, so it does not grade its own work
ollama list                     # confirm both are there
```

**Why 7-8B models?** On 16 GB RAM, a 4-bit 8B model uses about 5 GB and runs at ~25 tokens/s.
Bigger models are too slow or do not fit.

### 2. Project setup

```bash
mkdir ~/Documents/abstain-rag && cd ~/Documents/abstain-rag
git init
uv init --lib --name abstain-rag --python 3.12 .
```

- `--lib` creates the `src/abstain_rag/` layout, so the project can be pip-installed later.
- `pyproject.toml` is the project's ID card: name, version, dependencies.

```bash
uv add numpy faiss-cpu sentence-transformers scikit-learn pandas httpx tqdm pyarrow beautifulsoup4 lxml
uv add --dev pytest ruff
```

- `uv add` installs a package **and** records it in `pyproject.toml` and `uv.lock`.
- `uv.lock` pins exact versions, so anyone who clones the repo gets the same setup (reproducibility).
- `--dev` packages are for building only (tests, lint), not for users.

| Package | Job |
|---|---|
| sentence-transformers | Loads the embedding model (bge-small) |
| faiss-cpu | Fast vector search |
| numpy / pandas / pyarrow | Arrays, tables, reading parquet files |
| httpx | Downloads and calls to Ollama |
| beautifulsoup4 + lxml | Parsing FAR HTML |
| scikit-learn | Metrics later (AUROC) |
| pytest / ruff | Tests / lint |

`.gitignore` excludes `data/` and `runs/`. The data is large and public, and the download script rebuilds it.
Commit code, not data.

### 3. The data

**BEIR format** (every BEIR dataset looks like this):

| File | Contains | Example |
|---|---|---|
| `corpus.jsonl` | Documents to search | `{"_id": "4983", "title": "...", "text": "..."}` |
| `queries.jsonl` | Questions | `{"_id": "1", "text": "..."}` |
| `qrels/test.tsv` | **Answer key**: which doc is right for which query | `1  31715818  1` = query 1 -> doc 31715818 is relevant |

- **qrels** = "query relevance judgments", labeled by humans. They make scoring possible.
- **Splits:** `train` (fine-tuning, CP5), `dev` (tuning), `test` (**final scores only**; tuning on it is cheating).

**Source:** the official BEIR server (TU Darmstadt) timed out, so we use the Hugging Face mirror
(`huggingface.co/datasets/BeIR/...`): same data, stored as parquet.
Doc counts match the BEIR paper: SciFact 5,183, FiQA 57,638, NFCorpus 3,633.

**FAR:** acquisition.gov publishes one HTML file per part (53 parts). The script reads the index page
to find all part numbers, so it adapts if parts are added.

**Download script design** (`scripts/download_data.py`):

- **Retries** up to 4 times, because networks fail.
- **Skips files it already has**, so re-running is safe ("idempotent").
- **Waits 0.5 s between FAR requests**, to be polite to a government server.
- **Generic user-agent** (`abstain-rag-benchmark/0.1`), so no personal details are sent.

```bash
uv run python scripts/download_data.py          # everything
uv run python scripts/download_data.py --only far
```

`uv run` runs the command inside the project's virtual env. No `source activate` needed.

### 4. Smoke test (`scripts/smoke_test.py`)

**Embeddings:** the model turns text into 384 numbers. Similar meaning -> nearby vectors.

- `normalize_embeddings=True` scales each vector to length 1, so **inner product = cosine similarity**.
  That is why the index is `IndexFlatIP` (IP = inner product).
- **Query prefix:** bge expects queries to start with
  `"Represent this sentence for searching relevant passages: "`. Documents get no prefix.
  Queries and documents are different kinds of text ("asymmetric retrieval"). Forget it and scores drop.
- **MPS** = Apple's GPU, ~120 docs/s here.

**FAISS:** `IndexFlatIP` is **exact** search (compares the query with every doc). Fine at our size
(58K docs max). Approximate indexes like HNSW are only needed at millions of docs.

**LLMs:** POST to Ollama `/api/generate` with:

- `temperature: 0` -> same answer every time (needed for a reproducible benchmark)
- `num_predict: 60` -> max 60 tokens
- **tokens/s** = `eval_count / eval_duration`, used for the latency and cost metric

Result: both models work (~24-25 tok/s). qwen2.5 answered "What is the FAR?" with
"False Acceptance Rate": a live example of an LLM answering confidently without grounding.
That is the problem this project measures.

### 5. Commit

```bash
uv run ruff check scripts     # lint: catch mistakes and style issues
git add -A && git commit -m "CP0: project setup, data download and smoke test"
```

### Check yourself

1. Why does normalizing vectors let us use inner product?
   *For unit vectors, cosine = dot product.*
2. Why do bge queries get a prefix but documents do not?
   *Asymmetric retrieval: short questions vs long passages.*
3. Why must you never tune on the `test` split?
   *The score would overstate real performance (leakage).*
4. Why exact search (Flat) and not HNSW here?
   *58K docs is small: exact is fast and has no approximation error.*

---

## CP1: Retrieval baseline

**Goal:** prove the search pipeline is correct by reproducing the published bge-small scores,
then build the FAR index.

```
FAR HTML ──► far.py ──► passages ─┐
BEIR files ─► beir.py ───────────┼─► retrieval.py (embed + FAISS) ─► ranked docs ─► metrics.py ─► scores
queries + answer key ─────────────┘
```

### 1. `beir.py`: load the datasets

| Function | Returns |
|---|---|
| `load_corpus` | `{doc_id: {"title", "text"}}` |
| `load_queries` | `{query_id: text}` |
| `load_qrels(split)` | `{query_id: {doc_id: relevance}}`, score-0 rows dropped (0 = judged *not* relevant) |

`doc_text` = `title + " " + text`, exactly as BEIR/MTEB do. Any other join makes scores not comparable.

### 2. `far.py`: FAR HTML -> passages

Each section is `<article>` with a heading (`<span class="autonumber">52.212-4</span>`) and a
direct child `<div class="body">`. Child sections are nested `<article>`s.

- Delete table-of-contents links (`fm:TOC`): titles, not content.
- Read only the **direct** heading and body (`recursive=False`), so a parent does not swallow its children's text.
- **`chunk`**: bge-small truncates after 512 tokens (~380 words). Long sections are cut into
  250-word windows with 50 words of overlap. IDs are `52.212-4#0`, `#1`, ... so every passage maps back
  to its section (the CP2 answer key is per section).

### 3. `retrieval.py`: `DenseRetriever`

- Loads bge-small on MPS, `max_seq_length = 512`.
- `encode`: normalized 384-dim vectors.
- `build`: caches `embeddings.npy` + `doc_ids.json` in `data/index/<name>/` (checks IDs match before reuse);
  sorts texts by length before encoding (less padding = faster), then restores the order.
- `search`: adds the bge query prefix, returns **scores and IDs**. The scores feed the abstention gates in CP3.

### 4. `metrics.py`

- **recall@100**: share of the right docs found in the top 100. Retrieval misses cannot be fixed later.
- **nDCG@10**: right docs near the top score more (gain / log2(rank + 1)), divided by the best possible.
  Graded relevance (2 vs 1) counts.
- Same formulas as trec_eval (used by BEIR and MTEB). Averaged over queries with at least one right doc.
- Source: the BEIR paper (Thakur et al., 2021) uses nDCG@10 as its main metric; the bge-small model card
  publishes both numbers for our datasets.

### 5. Run

```bash
uv run pytest -q
uv run ruff check .
uv run python scripts/cp1_retrieval.py     # ~9 min first time (FiQA), seconds after (cache)
```

### Results (`runs/cp1/results.json`)

| Dataset | nDCG@10 ours / published | recall@100 ours / published | ms/query |
|---|---|---|---|
| SciFact | 71.27 / 71.28 | 94.17 / 94.17 | 1.9 |
| FiQA | 40.35 / 40.34 | 69.63 / 69.55 | 0.9 |
| NFCorpus | 34.30 / 34.31 | 31.10 / 31.14 | 1.0 |

All within 0.1 point: the pipeline is correct. FAR: 3,117 sections -> 6,003 passages.
Spot checks hit the right sections (nonconforming items -> 46.407, micro-purchase -> 13.003/13.201,
proposal analysis -> 15.404-1).

**Finding for CP3:** FAR top-3 scores are tightly packed (e.g. 0.742 / 0.736 / 0.729). A single fixed score
cutoff will struggle; this motivates the margin and calibrated-floor gates.

### Check yourself

1. Why `recursive=False`? *Otherwise parent sections duplicate their children's text.*
2. Why 250-word chunks? *bge truncates past ~380 words; the rest would be invisible.*
3. Why does `search` return scores? *The abstention gates decide from them.*
4. Why reproduce published numbers before anything else? *If the baseline is wrong, every later result is wrong too.*

---

## CP2: FAR test set

**Problem:** BEIR ships questions and an answer key; the FAR does not. We build both, plus questions the
system *should refuse*.

### Question types

| Type | How it is made | Should the system answer? |
|---|---|---|
| `answerable` | llama3.1 reads one FAR passage and writes a question it answers; that passage's **section** is the answer key | Yes |
| `near_miss` | Contracting-flavoured topics the FAR does not cover (DFARS, grants under 2 CFR 200, Georgia state rules, SAM.gov login help, per diem rates, ...) | No |
| `ood_llm` | Everyday off-topic questions (cooking, sports, travel, ...) | No |
| `ood_beir` | Real science/medical queries from SciFact and NFCorpus | No |

Near-misses are the hard, honest case: they *sound* like FAR questions, so retrieval returns
confident-looking hits.

### Code

- **`llm.py` (`Ollama`)**: POST to `/api/generate` with
  - `format` = a **JSON schema** (constrained decoding: the reply must parse). `minItems`/`maxItems` force list
    length; without them llama3.1 returned 1 question when asked for 5.
  - a fixed `seed` per item -> reproducible; `temperature` 0.7-0.8 for variety.
  - a **disk cache** (`data/far/llm_cache.jsonl`) keyed by a hash of the request: reruns are free.
- **`testset.py`** (pure functions, all tested):
  - `sample_passages`: one passage per section, uniform over sections, passages >= 60 words.
  - `reject_reason`: drops questions that are too short/long, not a question, cite a section number or
    say "this section", or **copy > 6 consecutive words** from the passage (copied wording makes retrieval
    artificially easy).
  - `near_duplicates`: drop questions with cosine > 0.95 to an earlier one.
  - `split_dev_test`: 30% dev / 70% test, stratified by type. **dev** tunes gate thresholds (CP3), **test** is
    for final numbers only.
- **`scripts/cp2_build_testset.py`**: generates, filters, splits, writes `testset.jsonl`, section-level
  `qrels/{dev,test}.tsv`, a retrieval sanity check, and `review.csv`.

```bash
uv run python scripts/cp2_build_testset.py --limit 6   # quick prompt check first
uv run python scripts/cp2_build_testset.py             # full run, ~25 min
```

### Results

- Generated 384 answerable, 178 near-miss, 83 + 80 off-topic. Rejected: 16 answerable copied the passage,
  13 off-topic were not questions. 3 near-duplicates dropped. **722 questions total.**
- Retrieval on answerable questions: **source section at rank 1: 53.9%, in top 10: 86.5%.**
  Harder than SciFact because FAR has many near-identical sections (e.g. a policy section and its clause).

### Human review (`data/far/review.csv`)

100 questions (50 answerable, 30 near-miss, 20 off-topic), labeled by hand:
`label_answerable_by_far (Y/N)` and `label_question_ok (Y/N)`. This measures how often the generated labels
are wrong. Questions found wrong are fixed or removed before CP3.

### Check yourself

1. Why reject questions that copy the passage? *Word overlap makes retrieval look better than it is.*
2. Why near-miss questions at all? *Off-topic questions are easy to refuse; the real risk is plausible ones.*
3. Why a section-level answer key, not passage-level? *A long section is several passages; any of them answers.*
4. Why a human check of 100? *Generated labels are not ground truth until someone measures their error rate.*

### Review results (`scripts/cp2_apply_review.py`, `runs/cp2/review_summary.json`)

- **Answerable labels: 100/100 agree** with the human. With 0 errors in 100, the "rule of three" gives a
  95% upper bound of ~3% on the label error rate.
- **Question quality: 97/100 OK.** Removed: 2 questions with a misleading premise (e.g. "minimum value for the
  FAR to apply", where $20,000 is really a labor-clause threshold) and 1 vague off-topic question.
- **Final set: 719 questions.** Human labels live in `labels/far_review_100.csv` (committed; `data/` is not).

**Findings from the reviewer's notes (used later):**

1. **Passages start mid-sentence.** 250-word windows cut clauses. The section-level answer key is right, but in
   CP4 the generator should see neighbouring passages of the same section, not one window.
2. **False-premise questions exist** even in generated "answerable" sets. A good system answers by correcting
   the premise. Worth a separate category later.
3. **Near-misses share vocabulary with the FAR** (per diem, SAM registration, cost allowability), so they pull
   real FAR hits. This is exactly where a score threshold alone fails.

**Rule of three:** if you see 0 events in n trials, the 95% upper bound on the true rate is about 3/n.

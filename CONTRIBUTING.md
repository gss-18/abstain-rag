# Contributing

Issues and pull requests are welcome.

## Set up

```bash
uv sync
uv run ruff check .
uv run pytest -q        # no datasets, index or LLM needed
```

The full benchmark needs Ollama with `llama3.1:8b` and `qwen2.5:7b`; see "Run it" in the README.

## Ground rules

- **Tune on dev, report on test.** Never choose a threshold, epoch or prompt by looking at test numbers.
- **Every LLM call goes through the cached client** (`abstain_rag.llm.Ollama`), so runs can be resumed and
  reproduced.
- **Do not replace human labels with model labels.** Judge-agreement checks compare against people.
- If a change moves a reported number, update the README and `docs/LEARN.md` in the same pull request.
- Write commit messages in plain English: what changed and why.

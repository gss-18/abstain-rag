"""Minimal client for a local Ollama server, with a disk cache so reruns cost nothing."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

OLLAMA_URL = "http://localhost:11434"


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    cached: bool = False

    def json(self) -> Any:
        return json.loads(self.text)


class Ollama:
    def __init__(self, model: str, url: str = OLLAMA_URL, cache_path: Path | None = None,
                 client: httpx.Client | None = None, timeout: float = 300) -> None:
        self.model = model
        self.url = url
        self.client = client or httpx.Client(timeout=timeout)
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            with cache_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    self._cache[row["key"]] = row

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def generate(self, prompt: str, *, system: str | None = None, schema: dict | None = None,
                 temperature: float = 0.0, seed: int = 42, max_tokens: int = 512) -> Generation:
        payload: dict[str, Any] = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if schema:
            payload["format"] = schema  # Ollama structured output: the reply must match this JSON schema
        key = self._key(payload)
        if key in self._cache:
            c = self._cache[key]
            return Generation(c["text"], c["prompt_tokens"], c["output_tokens"], c["seconds"], cached=True)
        t = time.perf_counter()
        r = self.client.post(f"{self.url}/api/generate", json=payload)
        r.raise_for_status()
        body = r.json()
        gen = Generation(body["response"], body.get("prompt_eval_count", 0), body.get("eval_count", 0),
                         time.perf_counter() - t)
        if self.cache_path:
            row = {"key": key, "text": gen.text, "prompt_tokens": gen.prompt_tokens,
                   "output_tokens": gen.output_tokens, "seconds": gen.seconds}
            self._cache[key] = row
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return gen

import json

import httpx

from abstain_rag.llm import Ollama


def _client(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"response": '{"q": "hi?"}', "prompt_eval_count": 5, "eval_count": 3})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_generate_sends_schema_and_seed(tmp_path):
    calls = []
    llm = Ollama("m", client=_client(calls), cache_path=tmp_path / "c.jsonl")
    gen = llm.generate("p", schema={"type": "object"}, seed=7)
    assert gen.json() == {"q": "hi?"}
    assert calls[0]["format"] == {"type": "object"}
    assert calls[0]["options"]["seed"] == 7


def test_cache_avoids_second_call_and_survives_restart(tmp_path):
    calls = []
    cache = tmp_path / "c.jsonl"
    Ollama("m", client=_client(calls), cache_path=cache).generate("p")
    again = Ollama("m", client=_client(calls), cache_path=cache).generate("p")
    assert again.cached and len(calls) == 1


def test_parallel_generate_writes_every_row_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    calls = []
    cache = tmp_path / "c.jsonl"
    llm = Ollama("m", client=_client(calls), cache_path=cache)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: llm.generate(f"p{i}"), range(30)))
    lines = cache.read_text().splitlines()
    assert len(lines) == 30 and len({json.loads(x)["key"] for x in lines}) == 30

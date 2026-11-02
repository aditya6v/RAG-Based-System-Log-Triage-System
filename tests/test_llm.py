"""Provider chain: retries, fallback, recording, redaction. Hosted providers
are faked with httpx.MockTransport and Ollama with a stub client, so nothing
leaves the machine and no model runs."""
import json

import httpx
import pytest

import config
from agents import llm

LINE = ("1124721545 2005.08.22 R73-M1-N4-I:J18-U01 2005-08-22-07.39.05.715470 R73-M1-N4-I:J18-U01 RAS APP "
        "FATAL ciod: Error reading message prefix on CioStream socket to 172.16.96.116:53370, Link has been severed")


def ok(text, model="m", prompt_tokens=11, completion_tokens=3):
    return httpx.Response(200, json={"model": model, "choices": [{"message": {"content": text}}],
                                     "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})


class Hosted:
    """Fake Groq/OpenRouter. `script` maps provider -> list of responses (or
    exceptions) returned in order; every request body is kept."""
    def __init__(self, script):
        self.script, self.seen = {k: list(v) for k, v in script.items()}, []

    def __call__(self, request: httpx.Request):
        name = "groq" if "groq" in request.url.host else "openrouter"
        self.seen.append((name, json.loads(request.content)))
        nxt = self.script[name].pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class StubOllama:
    def __init__(self, reply="ollama says hi", fail=None):
        self.reply, self.fail, self.seen = reply, fail, []

    def chat(self, model, messages, stream=False, **kw):
        self.seen.append(messages)
        if self.fail:
            raise self.fail
        if stream:
            return iter([{"message": {"content": "ollama "}, "done": False},
                         {"message": {"content": "streamed"}, "done": True, "prompt_eval_count": 7, "eval_count": 2}])
        return {"message": {"content": self.reply}, "prompt_eval_count": 7, "eval_count": 2}


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER_CHAIN", "groq,openrouter,ollama")
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or_test")
    monkeypatch.setattr(config, "REDACT_HOSTED", True)
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    sleeps = []

    def install(script, ollama=None):
        fake = Hosted(script)
        monkeypatch.setattr(llm, "_http", httpx.Client(transport=httpx.MockTransport(fake)))
        stub = ollama or StubOllama()
        monkeypatch.setattr(llm, "_get_ollama_client", lambda: stub)
        return fake, stub, sleeps
    return install


def test_chain_skips_hosted_providers_without_keys(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER_CHAIN", "groq, openrouter ,ollama")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "x")
    assert llm.provider_chain() == ["openrouter", "ollama"]


def test_first_provider_answers_and_is_recorded(setup):
    fake, _, _ = setup({"groq": [ok("from groq", model="llama-3.3-70b")]})
    calls = llm.start_recording()
    assert llm.chat_text("hello") == "from groq"
    assert calls == [{"provider": "groq", "model": "llama-3.3-70b", "ms": calls[0]["ms"], "attempts": 1,
                      "failed_before": 0, "prompt_tokens": 11, "completion_tokens": 3, "redacted": True}]


def test_429_is_retried_with_retry_after_then_succeeds(setup):
    fake, _, sleeps = setup({"groq": [httpx.Response(429, headers={"retry-after": "2"}), httpx.Response(429), ok("third time")]})
    calls = llm.start_recording()
    assert llm.chat_text("q") == "third time"
    assert sleeps == [2.0, 3.0]            # Retry-After honoured, then normal backoff
    assert calls[0]["attempts"] == 3 and calls[0]["failed_before"] == 2


def test_retry_after_is_capped(setup):
    _, _, sleeps = setup({"groq": [httpx.Response(429, headers={"retry-after": "120"}), ok("ok")]})
    llm.chat_text("q")
    assert sleeps == [llm.MAX_RETRY_AFTER_S]


def test_5xx_exhausts_retries_then_falls_back_to_next_provider(setup):
    fake, _, _ = setup({"groq": [httpx.Response(503)] * 3, "openrouter": [ok("from openrouter")]})
    calls = llm.start_recording()
    assert llm.chat_text("q") == "from openrouter"
    assert [n for n, _ in fake.seen] == ["groq", "groq", "groq", "openrouter"]
    assert calls[0]["provider"] == "openrouter" and calls[0]["failed_before"] == 3


def test_timeout_is_retried(setup):
    fake, _, _ = setup({"groq": [httpx.ReadTimeout("slow"), ok("after timeout")]})
    assert llm.chat_text("q") == "after timeout"
    assert len(fake.seen) == 2


def test_bad_key_is_not_retried(setup):
    fake, _, sleeps = setup({"groq": [httpx.Response(401, json={"error": "invalid key"})], "openrouter": [ok("or")]})
    assert llm.chat_text("q") == "or"
    assert [n for n, _ in fake.seen] == ["groq", "openrouter"] and sleeps == []


def test_falls_back_to_local_ollama_last(setup):
    fake, stub, _ = setup({"groq": [httpx.Response(401)], "openrouter": [httpx.Response(404)]})
    calls = llm.start_recording()
    assert llm.chat_text("q") == "ollama says hi"
    assert calls[0]["provider"] == "ollama" and calls[0]["prompt_tokens"] == 7 and calls[0]["redacted"] is False


def test_all_providers_failing_lists_every_reason(setup):
    setup({"groq": [httpx.Response(401)], "openrouter": [httpx.Response(400)]},
          ollama=StubOllama(fail=ConnectionError("refused")))
    with pytest.raises(RuntimeError) as e:
        llm.chat_text("q")
    msg = str(e.value)
    assert "groq (attempt 1): HTTP 401" in msg and "openrouter (attempt 1): HTTP 400" in msg and "ollama" in msg


def test_no_provider_configured(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER_CHAIN", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(RuntimeError, match="no LLM provider configured"):
        llm.chat_text("q")


def test_hosted_providers_get_redacted_text_ollama_does_not(setup):
    fake, stub, _ = setup({"groq": [httpx.Response(401)], "openrouter": [ok("x")]})
    llm.chat_text(LINE)
    sent = fake.seen[-1][1]["messages"][0]["content"]
    assert "172.16.96.116" not in sent and "R73-M1-N4" not in sent and "<ip>" in sent and "<node>" in sent
    assert "Link has been severed" in sent                         # the message itself survives
    setup({"groq": [httpx.Response(401)], "openrouter": [httpx.Response(401)]}, ollama=stub)
    llm.chat_text(LINE)
    assert stub.seen[-1][0]["content"] == LINE                      # local model sees the real line


def test_redaction_can_be_turned_off(setup, monkeypatch):
    fake, _, _ = setup({"groq": [ok("x")]})
    monkeypatch.setattr(config, "REDACT_HOSTED", False)
    llm.chat_text(LINE)
    assert fake.seen[0][1]["messages"][0]["content"] == LINE


def test_fastembed_embedding_option(monkeypatch):
    class FakeModel:
        def embed(self, texts):
            return iter([[0.25] * 768 for _ in texts])
    monkeypatch.setattr(config, "EMBED_PROVIDER", "fastembed")
    monkeypatch.setattr(llm, "_fastembed", FakeModel())
    v = llm.embed_text("KERNEL FATAL data TLB error interrupt")
    assert len(v) == 768 and all(isinstance(x, float) for x in v)


def sse(*chunks, status=200):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})


def test_stream_from_hosted_provider(setup):
    setup({"groq": [sse({"model": "g", "choices": [{"delta": {"content": "Hel"}}]},
                        {"choices": [{"delta": {"content": "lo"}}]},
                        {"choices": [], "x_groq": {"usage": {"prompt_tokens": 20, "completion_tokens": 2}}})]})
    calls = llm.start_recording()
    assert "".join(llm.chat_stream([{"role": "user", "content": "hi"}])) == "Hello"
    assert calls[0]["provider"] == "groq" and calls[0]["completion_tokens"] == 2


def test_stream_falls_back_before_the_first_chunk(setup):
    setup({"groq": [httpx.Response(429)] * 3, "openrouter": [httpx.Response(500)] * 3})
    calls = llm.start_recording()
    assert "".join(llm.chat_stream([{"role": "user", "content": "hi"}])) == "ollama streamed"
    assert calls[0]["provider"] == "ollama" and calls[0]["failed_before"] == 6


def test_stream_failure_after_first_chunk_is_not_hidden(setup, monkeypatch):
    def broken(name, messages, temperature):
        yield "text", "partial "
        raise llm.ProviderError(name, "connection dropped", retryable=True)
    setup({})
    monkeypatch.setattr(llm, "_stream_with", broken)
    out = []
    with pytest.raises(RuntimeError, match="part-way"):
        for chunk in llm.chat_stream([{"role": "user", "content": "hi"}]):
            out.append(chunk)
    assert out == ["partial "]

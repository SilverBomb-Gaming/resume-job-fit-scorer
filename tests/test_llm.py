"""HTTP clients without a live model."""

import json

import httpx
import pytest

from fit_score.errors import LLMError
from fit_score.llm import OllamaClient, OpenAICompatibleClient, build_client


def test_ollama_requests_json_chat() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"ok": true}'}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(
            base_url="http://127.0.0.1:11434",
            model="llama3.2",
            timeout=5,
            num_ctx=8192,
            http=http,
        )
        assert client.complete(system="sys", user="usr") == '{"ok": true}'

    body = seen["body"]
    assert seen["path"] == "/api/chat"
    assert isinstance(body, dict)
    assert body["model"] == "llama3.2"
    assert body["stream"] is False
    assert body["format"] == "json"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["options"]["temperature"] == 0.1


def test_ollama_404_mentions_pull() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'llama3.2' not found"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(
            base_url="http://127.0.0.1:11434",
            model="llama3.2",
            timeout=5,
            num_ctx=8192,
            http=http,
        )
        with pytest.raises(LLMError, match="ollama pull llama3.2"):
            client.complete(system="sys", user="usr")


def test_openai_compatible_request_shape() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatibleClient(
            base_url="http://localhost:8080/v1",
            api_key="test-key",
            model="local-model",
            timeout=5,
            http=http,
        )
        assert client.complete(system="sys", user="Return JSON") == '{"ok": true}'

    body = seen["body"]
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert isinstance(body, dict)
    assert body["model"] == "local-model"
    assert body["response_format"] == {"type": "json_object"}


def test_default_provider_is_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIT_SCORE_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = build_client()
    try:
        assert isinstance(client, OllamaClient)
        assert client.model == "llama3.2"
        assert client.base_url == "http://127.0.0.1:11434"
    finally:
        client.close()


def test_openai_hosted_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        build_client(provider="openai")


def test_local_openai_compatible_allows_an_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("OPENAI_MODEL", "local-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_client(provider="openai")
    try:
        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "local-model"
        assert client.api_key == ""
    finally:
        client.close()

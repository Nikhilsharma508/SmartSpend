"""Unit tests for the copilot provider clients (no network — HTTP is mocked)."""

from __future__ import annotations

import json
import os

import pytest
import requests

from packages.copilot.providers import (
    DECISIONS_PATH,
    DecisionsClient,
    NeurometricWriter,
    OllamaWriter,
    ProviderError,
    TemplateWriter,
    build_prompt,
)
from packages.copilot.schemas import (
    CampaignScenario,
    DraftRequest,
    Intent,
    ToolRun,
)

# --- helpers ---------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: dict, status_error: Exception | None = None) -> None:
        self.payload = payload
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict:
        return self.payload


def _draft(intent: Intent = Intent.PLAN) -> DraftRequest:
    scenario = CampaignScenario(
        intent=intent,
        question="I have $500 at $0.10 each, how deep?",
        budget_usd=500.0,
        cost_per_contact_usd=0.10,
        assumptions_used=["cost per contact defaulted to $0.10"],
    )
    runs = [
        ToolRun(name="profit_point", ok=True, output={"profit_usd": 756.32, "n_targeted": 5000}),
        ToolRun(name="optimize_budget", ok=False, error="ValueError: boom"),
    ]
    return DraftRequest(
        intent=intent, question=scenario.question, scenario=scenario, tool_runs=runs
    )


# --- DecisionsClient -------------------------------------------------------


def test_decisions_payload_shape_and_headers(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse({"answers": {}})

    monkeypatch.setattr("packages.copilot.providers.requests.post", fake_post)
    client = DecisionsClient("https://host.example", "secret-key", "my-model", timeout=9)
    state = {"user_question": "q"}
    questions = {"intent": {"type": "choice", "name": "intent", "criteria": {"plan": "..."}}}
    client.decide(state, questions)

    assert captured["url"] == f"https://host.example{DECISIONS_PATH}"
    assert captured["headers"]["api-key"] == "secret-key"
    assert captured["timeout"] == 9
    assert captured["payload"] == {"model": "my-model", "state": state, "questions": questions}


def test_decisions_trailing_slash_is_normalised(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return FakeResponse({"answers": {}})

    monkeypatch.setattr("packages.copilot.providers.requests.post", fake_post)
    DecisionsClient("https://host.example/", "k", "m").decide({}, {})
    assert captured["url"] == f"https://host.example{DECISIONS_PATH}"


def test_decisions_http_error_raises_provider_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse({}, status_error=requests.HTTPError("500 Server Error"))

    monkeypatch.setattr("packages.copilot.providers.requests.post", fake_post)
    with pytest.raises(ProviderError, match=r"\[decisions\]"):
        DecisionsClient("https://h", "k", "m").decide({}, {})


def test_decisions_timeout_raises_provider_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.Timeout("took too long")

    monkeypatch.setattr("packages.copilot.providers.requests.post", fake_post)
    with pytest.raises(ProviderError, match=r"\[decisions\]"):
        DecisionsClient("https://h", "k", "m").decide({}, {})


def test_decisions_non_json_raises_provider_error(monkeypatch):
    class _BadJSON(FakeResponse):
        def json(self):
            raise json.JSONDecodeError("boom", "doc", 0)

    def fake_post(url, headers=None, json=None, timeout=None):
        return _BadJSON({})

    monkeypatch.setattr("packages.copilot.providers.requests.post", fake_post)
    with pytest.raises(ProviderError, match="not JSON"):
        DecisionsClient("https://h", "k", "m").decide({}, {})


def test_from_env_missing_lists_names_only(monkeypatch):
    for var in ("AZURE_ENDPOINT", "AZURE_API_KEY", "AZURE_DEPLOYMENT_NAME"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ProviderError) as err:
        DecisionsClient.from_env()
    message = str(err.value)
    assert "AZURE_ENDPOINT" in message and "AZURE_API_KEY" in message
    assert ".env.example" in message


def test_from_env_builds_client(monkeypatch):
    monkeypatch.setenv("AZURE_ENDPOINT", "https://e")
    monkeypatch.setenv("AZURE_API_KEY", "k")
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "m")
    client = DecisionsClient.from_env()
    assert (client.endpoint, client.model) == ("https://e", "m")


# --- build_prompt ----------------------------------------------------------


def test_build_prompt_contains_contract_and_data():
    prompt = build_prompt(_draft())
    assert "VERBATIM" in prompt.system
    assert "Recommendation" in prompt.system  # per-intent sections, titled
    assert "QUESTION:" in prompt.user and "INTENT: plan" in prompt.user
    assert "DATA (JSON" in prompt.user
    assert "756.32" in prompt.user  # tool outputs ride verbatim
    assert "UNAVAILABLE" in prompt.user  # failed tools surface as stubs
    assert "FAILED these validation checks" not in prompt.user  # no retry yet


def test_build_prompt_retry_carries_validator_errors():
    request = _draft().model_copy(update={"retry_errors": ["V1_number_trace: no source for 9"]})
    prompt = build_prompt(request)
    assert "FAILED these validation checks" in prompt.user
    assert "V1_number_trace: no source for 9" in prompt.user


def test_build_prompt_sections_differ_per_intent():
    plan = build_prompt(_draft(Intent.PLAN)).system
    explain = build_prompt(_draft(Intent.EXPLAIN)).system
    assert "Next Test" in plan
    assert "Segments" in explain and "Next Test" not in explain


# --- TemplateWriter --------------------------------------------------------


def test_template_writer_is_deterministic_and_local():
    writer = TemplateWriter()
    request = _draft()
    first, second = writer.write(request), writer.write(request)
    assert first.text == second.text
    assert first.provider == "template"
    assert "## Recommendation" in first.text


# --- chat writers (ollama / neurometric) -----------------------------------


def _chat_post(monkeypatch, payload: dict, captured: dict):
    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeResponse(payload)

    monkeypatch.setattr("packages.copilot.providers.requests.post", fake_post)


def test_ollama_payload_shape_and_parse(monkeypatch):
    captured: dict = {}
    _chat_post(
        monkeypatch,
        {
            "choices": [{"message": {"content": "**Brief**"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        captured,
    )
    writer = OllamaWriter(model="gpt-oss:120b-cloud", base_url="http://localhost:11434/")
    result = writer.write(_draft())
    assert result.text == "**Brief**"
    assert result.provider == "ollama"
    assert (result.input_tokens, result.output_tokens) == (10, 5)
    assert captured["url"] == "http://localhost:11434/chat/completions"
    assert captured["payload"]["model"] == "gpt-oss:120b-cloud"
    roles = [m["role"] for m in captured["payload"]["messages"]]
    assert roles == ["system", "user"]
    # no auth header for local ollama, and nothing forbidden in the body
    assert "Authorization" not in captured["headers"]
    assert set(captured["payload"]) == {"model", "messages"}


def test_neurometric_sends_bearer_token(monkeypatch):
    captured: dict = {}
    _chat_post(monkeypatch, {"choices": [{"message": {"content": "ok"}}]}, captured)
    writer = NeurometricWriter(
        model="neurometric/goal-setter", base_url="https://gw.example/v1", api_key="tok"
    )
    writer.write(_draft(Intent.EXPLAIN))
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["payload"]["model"] == "neurometric/goal-setter"


def test_chat_writer_maps_http_error_to_provider_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse({}, status_error=requests.HTTPError("404"))

    monkeypatch.setattr("packages.copilot.providers.requests.post", fake_post)
    writer = OllamaWriter(model="m", base_url="http://h")
    with pytest.raises(ProviderError, match=r"\[ollama\]"):
        writer.write(_draft())


def test_chat_writer_maps_bad_shape_to_provider_error(monkeypatch):
    _chat_post(monkeypatch, {"unexpected": True}, {})
    writer = OllamaWriter(model="m", base_url="http://h")
    with pytest.raises(ProviderError, match="shape"):
        writer.write(_draft())


def test_neurometric_from_env_requires_vars(monkeypatch):
    monkeypatch.delenv("NEUROMETRIC_BASE_URL", raising=False)
    monkeypatch.delenv("NEUROMETRIC_API", raising=False)
    with pytest.raises(ProviderError, match=r"\[neurometric\]"):
        NeurometricWriter.from_env("neurometric/goal-setter")


# --- live smoke (skipped without env; opt-in locally, never CI) ------------


@pytest.mark.skipif(
    not all(
        os.getenv(v) for v in ("AZURE_ENDPOINT", "AZURE_API_KEY", "AZURE_DEPLOYMENT_NAME")
    ),
    reason="Decisions API env not configured",
)
def test_live_decisions_tiny_question():
    body = DecisionsClient.from_env(timeout=30).decide(
        {"user_question": "hello"},
        {"ping": {"type": "noul", "name": "ping", "instructions": "Is this a greeting?",
                  "criteria": {"true": "It greets.", "false": "It does not."}}},
    )
    assert "answers" in body and "ping" in body["answers"]

"""Provider ordering, retry budget, and Bedrock request contract."""
import sys
import types

import pytest

import llm_providers


class _Provider:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def available(self):
        return True


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    monkeypatch.setattr(llm_providers.time, "sleep", lambda *_: None)
    llm_providers._PROVIDER_DISABLED_UNTIL.clear()


def test_default_chain_is_bedrock_then_gemini(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_ORDER", raising=False)
    assert llm_providers.get_fallback_chain() == ["bedrock", "gemini"]
    assert llm_providers.MAX_ATTEMPTS_PER_PROVIDER == 3


def test_bedrock_api_key_marks_provider_available(monkeypatch):
    monkeypatch.delenv("BEDROCK_ENABLED", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "development-key")

    assert llm_providers.BedrockNovaProvider().available() is True


def test_three_transient_attempts_then_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_ORDER", "bedrock,gemini")
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.setenv("LLM_GEMINI_MODEL", "")
    monkeypatch.setitem(llm_providers.PROVIDERS, "bedrock", _Provider("bedrock"))
    monkeypatch.setitem(llm_providers.PROVIDERS, "gemini", _Provider("gemini"))
    calls = []

    def invoke(provider, model):
        calls.append((provider.name(), model))
        if provider.name() == "bedrock":
            raise RuntimeError("503 service unavailable")
        return {"windows": []}, {"model": model}

    parsed, cost, used = llm_providers.call_with_fallback(
        invoke, "gemini-custom", force_available=["bedrock", "gemini"])

    assert [name for name, _ in calls] == ["bedrock"] * 3 + ["gemini"]
    assert {model for name, model in calls if name == "bedrock"} == {
        "amazon.nova-lite-v1:0"}
    assert calls[-1] == ("gemini", "gemini-custom")
    assert parsed == {"windows": []}
    assert cost["model"] == "gemini-custom"
    assert used == "gemini"


def test_fatal_provider_error_skips_retries_but_still_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_ORDER", "bedrock,gemini")
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.setenv("LLM_GEMINI_MODEL", "")
    monkeypatch.setitem(llm_providers.PROVIDERS, "bedrock", _Provider("bedrock"))
    monkeypatch.setitem(llm_providers.PROVIDERS, "gemini", _Provider("gemini"))
    calls = []

    def invoke(provider, _model):
        calls.append(provider.name())
        if provider.name() == "bedrock":
            raise RuntimeError("AccessDeniedException: forbidden")
        return {"shorts": []}, {}

    _, _, used = llm_providers.call_with_fallback(
        invoke, "", force_available=["bedrock", "gemini"])
    assert calls == ["bedrock", "gemini"]
    assert used == "gemini"


def test_fatal_provider_error_opens_circuit_for_later_batches(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_ORDER", "bedrock,gemini")
    monkeypatch.setitem(llm_providers.PROVIDERS, "bedrock", _Provider("bedrock"))
    monkeypatch.setitem(llm_providers.PROVIDERS, "gemini", _Provider("gemini"))
    calls = []

    def invoke(provider, _model):
        calls.append(provider.name())
        if provider.name() == "bedrock":
            raise RuntimeError("AccessDeniedException: Authentication failed")
        return {"windows": []}, {}

    llm_providers.call_with_fallback(invoke, "", stage_label="score batch 1")
    llm_providers.call_with_fallback(invoke, "", stage_label="score batch 2")

    assert calls == ["bedrock", "gemini", "gemini"]


def test_provider_specific_model_envs_do_not_cross(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    assert llm_providers.get_model_for_provider("bedrock", "wrong-gemini") == \
        "amazon.nova-pro-v1:0"
    assert llm_providers.get_model_for_provider("gemini", "caller-model") == \
        "gemini-test"


def test_bedrock_uses_converse_and_reports_usage(monkeypatch):
    seen = {}

    class _Runtime:
        def converse(self, **kwargs):
            seen.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": '{"windows": []}'}]}},
                "usage": {"inputTokens": 12, "outputTokens": 4},
            }

    fake_boto3 = types.SimpleNamespace(
        client=lambda service, region_name: (
            seen.update(service=service, region=region_name) or _Runtime()))
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setenv("BEDROCK_REGION", "us-west-2")

    parsed, cost = llm_providers.BedrockNovaProvider().score(
        "score this", "amazon.nova-lite-v1:0")

    assert seen["service"] == "bedrock-runtime"
    assert seen["region"] == "us-west-2"
    assert seen["modelId"] == "amazon.nova-lite-v1:0"
    assert seen["messages"][0]["content"] == [{"text": "score this"}]
    assert parsed == {"windows": []}
    assert cost["input_tokens"] == 12
    assert cost["output_tokens"] == 4

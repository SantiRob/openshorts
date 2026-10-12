"""
Multi-provider LLM adapter with configurable fallback chain.

Supports Gemini, Claude (Anthropic API + Bedrock), OpenAI, Amazon Bedrock (Nova),
Grok, and DeepSeek. Each provider implements the same schema (ScoreResponse,
DetailResponse) so the rest of the pipeline sees a uniform interface.

Configuration:
  - LLM_FALLBACK_ORDER: comma-separated provider names (default: bedrock,gemini)
  - Provider-specific API keys via env vars: GEMINI_API_KEY, ANTHROPIC_API_KEY,
    OPENAI_API_KEY, DEEPSEEK_API_KEY, XAI_API_KEY, AWS_REGION (for Bedrock)
  - BEDROCK_MODEL_ID and GEMINI_MODEL select each provider's model independently
  - Log which provider ultimately handled each job via logging.info()
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# One job can split scoring into several independent batches. If a provider
# rejects credentials or configuration, retrying it for every batch only adds
# latency and repeats the same warning. Transient failures still get their full
# three attempts; deterministic failures open this short process-local circuit.
_PROVIDER_DISABLED_UNTIL: Dict[str, float] = {}
_FATAL_PROVIDER_COOLDOWN_SECONDS = 300


class ProviderError(Exception):
    """Transient or fatal error from a provider."""
    pass


class BaseProvider(ABC):
    """Common interface for clip-selection providers."""

    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'gemini', 'claude', 'openai')."""
        pass

    @abstractmethod
    def available(self) -> bool:
        """True if this provider is ready to use (API key, network, etc.)."""
        pass

    @abstractmethod
    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Score windows using this provider.
        Returns (parsed_response, cost_analysis).
        parsed_response should have a 'windows' key with scored windows.
        cost_analysis should have 'input_tokens', 'output_tokens', 'total_cost', 'model'.
        """
        pass

    @abstractmethod
    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract detailed clips using this provider.
        Returns (parsed_response, cost_analysis).
        parsed_response should have a 'shorts' key with clip details.
        cost_analysis should have 'input_tokens', 'output_tokens', 'total_cost', 'model'.
        """
        pass


class GeminiProvider(BaseProvider):
    """Google Gemini via google-ai Python SDK."""

    def name(self) -> str:
        return "gemini"

    def available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Gemini."""
        try:
            from google import genai
            from google.genai import types as genai_types
            import gemini_worker
        except ImportError:
            raise ProviderError("Gemini SDK not available; install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.ScoreResponse,
        )

        response = client.models.generate_content(
            model=model or "gemini-3.1-flash-lite",
            contents=prompt,
            config=config
        )
        gemini_worker.raise_if_blocked(response)

        parsed_obj = getattr(response, "parsed", None)
        if parsed_obj is not None:
            parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        else:
            parsed = gemini_worker._parse_json_response_text(
                gemini_worker._get_response_text(response)
            )
        cost = gemini_worker._calculate_cost_analysis(response, model or "gemini-3.1-flash-lite")
        return parsed, cost or {}

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Gemini."""
        try:
            from google import genai
            from google.genai import types as genai_types
            import gemini_worker
        except ImportError:
            raise ProviderError("Gemini SDK not available")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.DetailResponse,
        )

        response = client.models.generate_content(
            model=model or "gemini-3.1-flash-lite",
            contents=prompt,
            config=config
        )
        gemini_worker.raise_if_blocked(response)

        parsed_obj = getattr(response, "parsed", None)
        if parsed_obj is not None:
            parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        else:
            parsed = gemini_worker._parse_json_response_text(
                gemini_worker._get_response_text(response)
            )
        cost = gemini_worker._calculate_cost_analysis(response, model or "gemini-3.1-flash-lite")
        return parsed, cost or {}


class ClaudeProvider(BaseProvider):
    """Claude via Anthropic API."""

    def name(self) -> str:
        return "claude"

    def available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Claude."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ProviderError("Anthropic SDK not available; install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY not set")

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or "claude-3-5-haiku-latest",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response text as JSON
        text = response.content[0].text if response.content else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("Claude returned invalid JSON")
            else:
                raise ProviderError("Claude returned invalid JSON")

        cost = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_cost": 0.0,  # Anthropic billing is per-token, calculate offline if needed
            "model": model or "claude-3-5-haiku-latest"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Claude."""
        return self.score(prompt, model)  # Same endpoint for both stages


class OpenAIProvider(BaseProvider):
    """OpenAI API (GPT-4o mini for cost efficiency)."""

    def name(self) -> str:
        return "openai"

    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using OpenAI."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK not available; install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("OpenAI returned invalid JSON")
            else:
                raise ProviderError("OpenAI returned invalid JSON")

        cost = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_cost": 0.0,
            "model": model or "gpt-4o-mini"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using OpenAI."""
        return self.score(prompt, model)


class DeepSeekProvider(BaseProvider):
    """DeepSeek API (very cheap option)."""

    def name(self) -> str:
        return "deepseek"

    def available(self) -> bool:
        return bool(os.getenv("DEEPSEEK_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using DeepSeek."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK (used for DeepSeek) not available")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderError("DEEPSEEK_API_KEY not set")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("DeepSeek returned invalid JSON")
            else:
                raise ProviderError("DeepSeek returned invalid JSON")

        cost = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_cost": 0.0,
            "model": model or "deepseek-chat"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using DeepSeek."""
        return self.score(prompt, model)


class GrokProvider(BaseProvider):
    """Grok (xAI) API."""

    def name(self) -> str:
        return "grok"

    def available(self) -> bool:
        return bool(os.getenv("XAI_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Grok."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK (used for Grok) not available")

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ProviderError("XAI_API_KEY not set")

        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model=model or "grok-2-1212",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("Grok returned invalid JSON")
            else:
                raise ProviderError("Grok returned invalid JSON")

        cost = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_cost": 0.0,
            "model": model or "grok-2-1212"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Grok."""
        return self.score(prompt, model)


class BedrockNovaProvider(BaseProvider):
    """Amazon Bedrock - Amazon Nova Lite (via AWS credentials)."""

    def name(self) -> str:
        return "bedrock"

    def available(self) -> bool:
        # Avoid probing EC2 metadata merely to render configuration or choose a
        # chain. Container/web-identity credentials are discoverable from env;
        # instance-profile users can opt in explicitly with BEDROCK_ENABLED=1.
        try:
            import boto3
        except ImportError:
            return False
        enabled = (os.getenv("BEDROCK_ENABLED") or "").strip().lower()
        if enabled in {"1", "true", "yes", "on"}:
            return True
        if enabled in {"0", "false", "no", "off"}:
            return False
        credential_hints = (
            "AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        )
        return any((os.getenv(name) or "").strip() for name in credential_hints)

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Amazon Nova on Bedrock."""
        try:
            import boto3
            import json as json_module
        except ImportError:
            raise ProviderError("boto3 not available for Bedrock")

        region = (os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION")
                  or os.getenv("AWS_DEFAULT_REGION") or "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        model_id = model or "amazon.nova-lite-v1:0"
        # Converse is Bedrock's provider-neutral API.  Its content-block shape
        # is also the native shape expected by Nova; the previous invoke_model
        # payload used OpenAI-style strings and was rejected by Nova.
        try:
            response = client.converse(
                modelId=model_id,
                system=[{"text": "Return one JSON object only, with no markdown."}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 4096, "temperature": 0.2},
            )
        except Exception as exc:
            message = str(exc)
            if "AccessDeniedException" in message and "Authentication failed" in message:
                raise ProviderError(
                    "AccessDeniedException: Bedrock rejected the API key in region "
                    f"{region}. Short-term keys are region-scoped and inherit IAM "
                    "permissions; generate it in this region (or change "
                    "BEDROCK_REGION) and allow bedrock:CallWithBearerToken."
                ) from exc
            raise

        blocks = (((response.get("output") or {}).get("message") or {})
                  .get("content") or [])
        text = "".join(block.get("text", "") for block in blocks
                       if isinstance(block, dict))
        if not text.strip():
            raise ProviderError("Bedrock returned an empty response body")

        try:
            parsed = json_module.loads(text)
        except json_module.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json_module.loads(text[start:end])
                else:
                    raise ProviderError("Bedrock returned invalid JSON")
            else:
                raise ProviderError("Bedrock returned invalid JSON")

        usage = response.get("usage") or {}
        cost = {
            "input_tokens": int(usage.get("inputTokens") or 0),
            "output_tokens": int(usage.get("outputTokens") or 0),
            "total_cost": 0.0,
            "model": model_id
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Amazon Nova on Bedrock."""
        return self.score(prompt, model)


# Provider registry
PROVIDERS = {
    "gemini": GeminiProvider(),
    "claude": ClaudeProvider(),
    "openai": OpenAIProvider(),
    "deepseek": DeepSeekProvider(),
    "grok": GrokProvider(),
    "bedrock": BedrockNovaProvider(),
}
# Backward-compatible name used by the first multi-provider implementation.
PROVIDERS["bedrock-nova"] = PROVIDERS["bedrock"]

# Bedrock is the primary provider; Gemini is the intentionally small fallback
# surface. Extra providers remain opt-in through LLM_FALLBACK_ORDER so existing
# self-hosted deployments do not lose compatibility.
DEFAULT_FALLBACK_ORDER = "bedrock,gemini"
MAX_ATTEMPTS_PER_PROVIDER = 3


def get_fallback_chain() -> List[str]:
    """Parse LLM_FALLBACK_ORDER env var or use default."""
    order = os.getenv("LLM_FALLBACK_ORDER", DEFAULT_FALLBACK_ORDER)
    return [p.strip() for p in order.split(",") if p.strip()]


def get_model_for_provider(provider_name: str, requested_model: str = "") -> str:
    """Resolve a model without leaking one provider's model into another.

    ``requested_model`` is retained for the historical Gemini caller. Every
    other provider uses its own environment variable or safe default.
    """
    canonical = "bedrock" if provider_name == "bedrock-nova" else provider_name
    env_names = {
        "bedrock": ("BEDROCK_MODEL_ID", "BEDROCK_MODEL", "LLM_BEDROCK_MODEL"),
        "gemini": ("GEMINI_MODEL", "LLM_GEMINI_MODEL"),
        "claude": ("ANTHROPIC_MODEL", "LLM_CLAUDE_MODEL"),
        "openai": ("OPENAI_MODEL", "LLM_OPENAI_MODEL"),
        "deepseek": ("DEEPSEEK_MODEL", "LLM_DEEPSEEK_MODEL"),
        "grok": ("XAI_MODEL", "LLM_GROK_MODEL"),
    }
    for env_name in env_names.get(canonical, ()):
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value
    if canonical == "gemini" and requested_model:
        return requested_model
    defaults = {
        "gemini": "gemini-3.1-flash-lite",
        "claude": "claude-3-5-haiku-latest",
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "grok": "grok-2-1212",
        "bedrock": "amazon.nova-lite-v1:0",
    }
    return defaults.get(canonical, "")


def describe_configuration() -> Dict[str, Any]:
    """Non-secret provider state suitable for ``/api/config`` and a UI."""
    chain = get_fallback_chain()
    return {
        "fallbackOrder": chain,
        "maxAttemptsPerProvider": MAX_ATTEMPTS_PER_PROVIDER,
        "providers": [
            {
                "name": name,
                "model": get_model_for_provider(name),
                "available": bool(PROVIDERS.get(name) and PROVIDERS[name].available()),
            }
            for name in chain
        ],
    }


def _retryable(exc: Exception) -> bool:
    """Whether another call can plausibly succeed without configuration changes."""
    # Gemini policy blocks and ordinary 4xx errors are deterministic. Invalid
    # or truncated model JSON is retried because it commonly changes per call.
    try:
        import gemini_worker
        if isinstance(exc, gemini_worker.GeminiBlockedError):
            return False
    except ImportError:
        pass
    msg = str(exc).lower()
    fatal_tokens = ("invalid_argument", "accessdenied", "unauthorized", "forbidden",
                    "no credentials", "unrecognizedclient", "validationexception")
    if any(token in msg for token in fatal_tokens):
        return False
    retryable_tokens = (
        "429", "500", "502", "503", "504", "unavailable", "throttl",
        "resource_exhausted", "overloaded", "deadline", "timeout", "connection",
        "empty response", "did not contain a json object", "failed to parse",
        "invalid json", "validation error", "internal",
    )
    return any(token in msg for token in retryable_tokens)


def call_with_fallback(
    callback,
    model_name: str,
    stage_label: str = "stage",
    force_available: Optional[List[str]] = None,
    raise_last: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """
    Call callback(provider, model_name) across fallback chain until success.
    Returns (parsed_response, cost_analysis, provider_name) or (None, None, "")
    on failure. With ``raise_last=True``, preserves the final provider exception
    (used by the pipeline for policy-block and diagnostic compatibility).

    callback: function(provider: BaseProvider, model: str) -> (parsed, cost)
    model_name: the model to request from each provider
    stage_label: for logging (e.g., "score" or "detail")
    """
    chain = get_fallback_chain()
    errors = []
    last_exception = None
    forced = set(force_available or [])

    for provider_name in chain:
        if provider_name not in PROVIDERS:
            logger.warning(f"Unknown provider in fallback chain: {provider_name}")
            continue

        provider = PROVIDERS[provider_name]
        disabled_until = _PROVIDER_DISABLED_UNTIL.get(provider_name, 0)
        if provider_name not in forced and disabled_until > time.monotonic():
            logger.info(
                "Skipping %s for %s after a fatal configuration/authentication error",
                provider_name, stage_label)
            continue
        if provider_name not in forced and not provider.available():
            logger.debug(f"Provider {provider_name} not available (missing API key?), skipping")
            continue

        provider_model = get_model_for_provider(provider_name, model_name)
        for attempt in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
            try:
                logger.info("Trying %s for %s (attempt %s/%s, model=%s)",
                            provider_name, stage_label, attempt,
                            MAX_ATTEMPTS_PER_PROVIDER, provider_model)
                parsed, cost = callback(provider, provider_model)
                logger.info("✅ %s succeeded for %s", provider_name, stage_label)
                return parsed, cost, provider_name
            except Exception as e:
                last_exception = e
                error_msg = str(e)[:200]
                errors.append(f"{provider_name}[{attempt}]: {error_msg}")
                logger.warning("❌ %s failed for %s (%s/%s): %s",
                               provider_name, stage_label, attempt,
                               MAX_ATTEMPTS_PER_PROVIDER, error_msg)
                retryable = _retryable(e)
                if not retryable:
                    _PROVIDER_DISABLED_UNTIL[provider_name] = (
                        time.monotonic() + _FATAL_PROVIDER_COOLDOWN_SECONDS
                    )
                if attempt == MAX_ATTEMPTS_PER_PROVIDER or not retryable:
                    break
                # Keep tests fast via monkeypatch and production gentle on
                # throttled APIs. Two waits means exactly three total attempts.
                time.sleep(5 * (2 ** (attempt - 1)))

    logger.error(f"All providers exhausted for {stage_label}. Errors: {'; '.join(errors)}")
    if raise_last and last_exception is not None:
        raise last_exception
    return None, None, ""

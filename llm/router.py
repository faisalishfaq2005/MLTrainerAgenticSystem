"""
router.py
---------
The single LLM abstraction layer for the entire agent system.

Every agent calls router.complete(system, user) and gets back a string.
To switch from Claude to GPT-4o to Gemini, the user changes ONE value —
the model string — and nothing else in the codebase changes.

HOW IT WORKS:
  LiteLLM is a unified wrapper that speaks to 100+ LLM providers using
  the same OpenAI-style messages API. You pass a model string like:
    "claude-sonnet-4-6"          → routes to Anthropic
    "gpt-4o"                     → routes to OpenAI
    "gemini/gemini-2.0-flash"    → routes to Google
    "ollama/llama3"              → routes to local Ollama
  LiteLLM handles authentication, request formatting, and response
  parsing for each provider automatically.

USAGE:
  # Create once in the orchestrator, pass to every agent
  from llm.router import LLMRouter

  router = LLMRouter(
      provider="anthropic",
      model="claude-sonnet-4-6",
      api_key="sk-ant-...",
  )
  response = router.complete(
      system_prompt="You are a config agent...",
      user_message="Generate training config for this task...",
  )
  # response is a plain string — agent doesn't know or care which LLM it is

  # Switching the LLM = change these 3 lines, NOTHING else changes:
  router = LLMRouter(provider="openai",  model="gpt-4o",            api_key="sk-...")
  router = LLMRouter(provider="google",  model="gemini-2.0-flash",  api_key="AIza...")
  router = LLMRouter(provider="ollama",  model="llama3",            api_key=None)
"""

import os
import logging
from typing import Optional, Any,Union

import litellm
from litellm import completion
import json

logger = logging.getLogger(__name__)

# Silence litellm's own verbose logging — we handle our own
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# Provider → litellm model string mapping
# ---------------------------------------------------------------------------
# LiteLLM uses specific model string formats per provider.
# This registry maps (provider, model_name) → the exact string litellm needs.
#
# To add a new model:  add one line to the right provider block below.
# To add a new provider: add a new key block + entry in DEFAULT_MODEL
#                        + API_KEY_ENV_VAR. Nothing else needs changing.

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "anthropic": {
        # user-facing name          litellm model string
        "claude-sonnet-4-6":        "claude-sonnet-4-6",
        "claude-opus-4-6":          "claude-opus-4-6",
        "claude-haiku-4-5-20251001":"claude-haiku-4-5-20251001",
    },
    "openai": {
        "gpt-4o":       "gpt-4o",
        "gpt-4o-mini":  "gpt-4o-mini",
        "gpt-4-turbo":  "gpt-4-turbo",
    },
    "google": {
        # Google models need the "gemini/" prefix in litellm
        "gemini-2.0-flash": "gemini/gemini-2.0-flash",
        "gemini-1.5-pro":   "gemini/gemini-1.5-pro",
    },
    "groq": {
        # Groq models are routed via the groq provider prefix
        "qwen/qwen3-32b": "groq/qwen/qwen3-32b",
        "llama-3.3-70b-versatile": "groq/llama-3.3-70b-versatile",
    },
    "ollama": {
        # Ollama models need the "ollama/" prefix in litellm
        "qwen2.5-coder:14b": "ollama/qwen2.5-coder:14b",
        "qwen2.5-coder:7b":  "ollama/qwen2.5-coder:7b",
        "llama3":            "ollama/llama3",
        "mistral":           "ollama/mistral",
        "phi3":              "ollama/phi3",
    },
}

# Default model used when user picks a provider but not a specific model
DEFAULT_MODEL: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "qwen/qwen3-32b",
    "ollama":    "qwen2.5-coder:14b",
    
}

# Free fallback used when user does not provide provider/model, or provides a
# paid provider without a key. Ordered strongest-to-lightest for coding tasks.
FREE_MODEL_FALLBACKS: list[tuple[str, str]] = [
    ("groq", "llama-3.3-70b-versatile"),
    ("ollama", "qwen2.5-coder:14b"),
    ("ollama", "qwen2.5-coder:7b"),
    ("ollama", "llama3"),
]

# Environment variable each provider reads its API key from.
# LiteLLM checks these automatically, but we set them explicitly
# so the key from collected_info always takes effect.
API_KEY_ENV_VAR: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "google":    "GEMINI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "ollama":    "",   # local — no key needed
}


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    The single LLM abstraction all agents use.

    Create one instance in the orchestrator and pass it to every agent.
    Agents call router.complete(system, user) and receive a string back.
    They never import litellm or any provider SDK directly.

    Args:
        provider:    "anthropic" | "openai" | "google" | "groq" | "ollama"
        model:       Model name from MODEL_REGISTRY (e.g. "claude-sonnet-4-6").
                     Defaults to DEFAULT_MODEL[provider] if not given.
        api_key:     Provider API key. Falls back to environment variable if None.
        max_tokens:  Max tokens in the response. Default 2048.
        temperature: Sampling temperature. Use 0.0 for agents that output JSON
                     or code (deterministic). Default 0.0.
        timeout:     Request timeout in seconds. Default 60.
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        provider = provider.lower().strip()

        if provider not in MODEL_REGISTRY:
            raise LLMRouterError(
                f"Unknown provider '{provider}'. "
                f"Supported: {sorted(MODEL_REGISTRY.keys())}"
            )

        self.provider    = provider
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout

        # Resolve which model to use
        model_name = model or DEFAULT_MODEL[provider]

        if model_name not in MODEL_REGISTRY[provider]:
            raise LLMRouterError(
                f"Unknown model '{model_name}' for provider '{provider}'. "
                f"Available: {sorted(MODEL_REGISTRY[provider].keys())}"
            )

        # _litellm_model is the exact string passed to litellm.completion()
        self._litellm_model = MODEL_REGISTRY[provider][model_name]
        self._display_name  = f"{provider}/{model_name}"
        self._api_key       = api_key

        # If no explicit key is passed, attempt provider env var.
        if not api_key and provider != "ollama":
            env_var = API_KEY_ENV_VAR.get(provider, "")
            if env_var:
                api_key = self._clean_opt(os.getenv(env_var))

        # Inject the API key into the environment so litellm can find it.
        # This is necessary because litellm reads from env vars internally.
        if api_key and provider != "ollama":
            env_var = API_KEY_ENV_VAR.get(provider, "")
            if env_var:
                os.environ[env_var] = api_key

        if provider != "ollama" and not api_key:
            raise LLMRouterError(
                f"No API key available for provider '{provider}'. "
                f"Set {API_KEY_ENV_VAR.get(provider, '<PROVIDER>_API_KEY')} or pass api_key explicitly."
            )

        logger.info(f"LLMRouter ready: {self._display_name}")

    # -----------------------------------------------------------------------
    # The one method every agent calls
    # -----------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_message: Optional[str],
        message_history: Optional[list[dict[str, str]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Union[str,dict]:
        """
        Send a (system, user) message pair and return the response text.

        This is the ONLY method agents call. The agent's code looks identical
        regardless of whether the underlying model is Claude, GPT-4o, or Gemini.

        Args:
            system_prompt:  Role + rules for the LLM (e.g. "You are a config
                            agent. Rules: 1. Return only JSON...").
            user_message:   The actual content for this call (the prompt,
                            data, template, etc.).
            message_history:
                Optional prior turns in OpenAI/LiteLLM format:
                [{"role": "user"|"assistant", "content": "..."}, ...]
                These turns are inserted between system_prompt and user_message.
            tools:
                Optional LiteLLM/OpenAI-compatible tools list for tool calling.
            tool_choice:
                Optional tool selection policy, e.g. "auto", "none", or a
                provider-specific explicit tool choice object.

        Returns:
            The model's reply as a plain stripped string.

        Raises:
            LLMRouterError: For any API, auth, rate-limit, or timeout error.
                            Agents catch this single exception type — they
                            never deal with provider-specific exceptions.
        """
        messages = [{"role": "system", "content": system_prompt}]

        if message_history:
            messages.extend(message_history)

        if user_message is not None:
            messages.append({"role": "user", "content": user_message})

        user_len = len(user_message) if isinstance(user_message, str) else 0

        logger.debug(
            f"→ {self._display_name} | "
            f"system={len(system_prompt)}c user={user_len}c"
        )

        try:
            request_kwargs = {
                "model": self._litellm_model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timeout": self.timeout,
                # Pass api_key directly for providers that need it per-request
                # Ollama is local — no key
                "api_key": self._api_key if self.provider != "ollama" else None,
            }

            if tools is not None:
                request_kwargs["tools"] = tools

            if tool_choice is not None:
                request_kwargs["tool_choice"] = tool_choice

            response = completion(
                **request_kwargs,
            )
            
            message=response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return {
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments)
                        }
                        for tc in message.tool_calls
                    ]
                }
           

            if message.content is None:
                raise LLMRouterError(
                    f"{self._display_name} returned an empty content field."
                )

            

            logger.debug(
                f"← {self._display_name} | "
                f"{len(message.content)}c | "
                f"tokens={self._token_summary(response)}"
            )

            return message.content

        # Re-raise our own error type unchanged
        except LLMRouterError:
            raise

        # Map litellm exceptions → LLMRouterError with helpful messages
        except litellm.AuthenticationError as e:
            raise LLMRouterError(
                f"Authentication failed for {self._display_name}. "
                f"Your API key is invalid or expired. Details: {e}"
            ) from e

        except litellm.RateLimitError as e:
            raise LLMRouterError(
                f"Rate limit hit on {self._display_name}. "
                f"Wait a few seconds and retry. Details: {e}"
            ) from e

        except litellm.ContextWindowExceededError as e:
            raise LLMRouterError(
                f"Prompt too long for {self._display_name}. "
                f"Reduce the input size. Details: {e}"
            ) from e

        except litellm.Timeout as e:
            raise LLMRouterError(
                f"Request to {self._display_name} timed out after {self.timeout}s. "
                f"Increase timeout= or choose a faster model."
            ) from e

        except litellm.APIError as e:
            raise LLMRouterError(
                f"API error from {self._display_name}: {e}"
            ) from e

        except Exception as e:
            raise LLMRouterError(
                f"Unexpected error calling {self._display_name}: "
                f"{type(e).__name__}: {e}"
            ) from e

    # -----------------------------------------------------------------------
    # Convenience factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_collected_info(
        cls,
        collected_info: dict,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> "LLMRouter":
        """
        Build a router from context.collected_info (written by IntakeManagerAgent).

        This is the standard way the orchestrator creates the router:

            router = LLMRouter.from_collected_info(context.collected_info)
            # now pass router into every agent's constructor

        collected_info keys used:
            llm_provider  — "anthropic" | "openai" | "google" | "groq" | "ollama" | None
            llm_model     — model name e.g. "claude-sonnet-4-6"
            llm_api_key   — validated API key (None for ollama / free fallback)

        Fallback behavior:
            1) If provider is missing, infer it from key prefix when possible.
            2) If still missing, choose FREE_MODEL_FALLBACKS[0].
            3) If provider is paid but key is missing, switch to free fallback.
            4) If model is missing, use DEFAULT_MODEL for the resolved provider.

        Per-agent overrides:
            max_tokens, temperature, timeout can be set directly here so
            different agents can use different token/latency budgets.
        kwargs:
            Any additional LLMRouter constructor overrides.
        """
        provider = cls._clean_opt(collected_info.get("llm_provider"))
        model = cls._clean_opt(collected_info.get("llm_model"))
        api_key = cls._clean_opt(collected_info.get("llm_api_key"))

        if not provider and api_key:
            provider = cls._infer_provider_from_api_key(api_key)

        if not provider:
            provider, model = cls._pick_free_fallback()
            logger.warning(
                "No llm_provider provided in collected_info. "
                "Using free fallback model %s/%s.",
                provider,
                model,
            )

        # If a paid provider is selected without a key, degrade gracefully to a
        # free fallback model so the pipeline can still run.
        if provider in {"anthropic", "openai", "google", "groq"} and not api_key:
            fallback_provider, fallback_model = cls._pick_free_fallback()
            logger.warning(
                "No API key provided for paid provider '%s'. "
                "Switching to free fallback %s/%s.",
                provider,
                fallback_provider,
                fallback_model,
            )
            provider, model = fallback_provider, fallback_model
            env_var = API_KEY_ENV_VAR.get(provider, "")
            api_key = cls._clean_opt(os.getenv(env_var)) if env_var else None

        if not model:
            model = DEFAULT_MODEL.get(provider)

        init_kwargs = dict(kwargs)
        if max_tokens is not None:
            init_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            init_kwargs["temperature"] = temperature
        if timeout is not None:
            init_kwargs["timeout"] = timeout

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            **init_kwargs,
        )

    @staticmethod
    def _clean_opt(value: Optional[str]) -> Optional[str]:
        """Normalize optional strings by trimming; empty -> None."""
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _infer_provider_from_api_key(api_key: str) -> Optional[str]:
        """Infer provider from common API key prefixes."""
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        if api_key.startswith("gsk_"):
            return "groq"
        if api_key.startswith("sk-"):
            return "openai"
        if api_key.startswith("AIza"):
            return "google"
        return None

    @staticmethod
    def _pick_free_fallback() -> tuple[str, str]:
        """
        Return the first free fallback model that exists in MODEL_REGISTRY.
        Raises LLMRouterError only if fallback config is invalid.
        """
        for provider, model in FREE_MODEL_FALLBACKS:
            if provider in MODEL_REGISTRY and model in MODEL_REGISTRY[provider]:
                return provider, model
        raise LLMRouterError(
            "No valid free fallback model configured in FREE_MODEL_FALLBACKS."
        )

    # -----------------------------------------------------------------------
    # Read-only properties
    # -----------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Human-readable name e.g. 'anthropic/claude-sonnet-4-6'."""
        return self._display_name

    @property
    def litellm_model(self) -> str:
        """Exact litellm model string e.g. 'claude-sonnet-4-6'."""
        return self._litellm_model

    def __repr__(self) -> str:
        return (
            f"LLMRouter(model={self._display_name!r}, "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature})"
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _token_summary(response) -> str:
        """Extract token usage for log line. Returns '-' if unavailable."""
        try:
            u = response.usage
            return f"in={u.prompt_tokens} out={u.completion_tokens}"
        except Exception:
            return "-"


# ---------------------------------------------------------------------------
# Custom exception — the only exception type agents ever see
# ---------------------------------------------------------------------------

class LLMRouterError(Exception):
    """
    Raised by LLMRouter.complete() for any error condition.
    Wraps all provider-specific exceptions (litellm.AuthenticationError,
    litellm.RateLimitError, etc.) into a single type.

    Agents catch this and raise AgentError:
        try:
            response = self.llm.complete(system, user)
        except LLMRouterError as e:
            raise AgentError(agent_name=self.name, stage="...", reason=str(e))
    """
    pass


# ---------------------------------------------------------------------------
# Reference function — used by IntakeManagerAgent for the model menu
# ---------------------------------------------------------------------------

def get_available_models() -> dict[str, list[str]]:
    """
    Returns the full list of supported models per provider.
    IntakeManagerAgent calls this to build the model selection menu.

    Returns:
        {"anthropic": ["claude-sonnet-4-6", ...], "openai": [...], ...}
    """
    return {
        provider: list(models.keys())
        for provider, models in MODEL_REGISTRY.items()
    }
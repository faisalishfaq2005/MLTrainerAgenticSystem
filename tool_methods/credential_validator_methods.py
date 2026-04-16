"""
credential_validator.py
------------------------
Validates all credentials BEFORE the pipeline starts.
Called during the conversational intake phase.

Each method returns (is_valid: bool, error_message: str | None).
Fast — each check is a lightweight API call, not a full operation.
"""

import httpx
import logging


logger = logging.getLogger(__name__)


class CredentialValidator:

    # ── HuggingFace ────────────────────────────────────────────────────

    @staticmethod
    def validate_hf_token(token: str) -> tuple[bool, str | None]:
        """
        Calls HF /api/whoami to confirm the token is valid.
        Returns (True, None) on success, (False, error_msg) on failure.
        """
        if not token or not token.startswith("hf_"):
            return False, "HuggingFace tokens must start with 'hf_'. Please check your token."

        try:
            resp = httpx.get(
                "https://huggingface.co/api/whoami",
                headers={"Authorization": f"Bearer {token}"},
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                username = data.get("name", "unknown")
                logger.info(f"HF token valid for user: {username}")
                return True, None
            elif resp.status_code == 401:
                return False, "HuggingFace token is invalid or expired. Generate a new one at https://huggingface.co/settings/tokens"
            else:
                return False, f"HuggingFace API returned unexpected status {resp.status_code}. Please try again."
        except httpx.TimeoutException:
            return False, "HuggingFace API timed out. Check your internet connection."
        except Exception as e:
            return False, f"Could not reach HuggingFace API: {e}"

    @staticmethod
    def get_hf_username(token: str) -> str | None:
        """Returns the HF username for a valid token, or None."""
        try:
            resp = httpx.get(
                "https://huggingface.co/api/whoami",
                headers={"Authorization": f"Bearer {token}"},
                timeout=8,
            )
            if resp.status_code == 200:
                return resp.json().get("name")
        except Exception:
            pass
        return None

    # ── Kaggle ───────────────────────────────────────────────────────────────

    @staticmethod
    def validate_kaggle_credentials(username: str, key: str) -> tuple[bool, str | None]:
        """
        Calls Kaggle API to verify username + API key combination.
        Uses the competitions list endpoint as a lightweight auth check.
        """
        if not username or not key:
            return False, "Both Kaggle username and API key are required."

        try:
            resp = httpx.get(
                "https://www.kaggle.com/api/v1/competitions/list",
                auth=(username, key),
                timeout=10,
            )
            if resp.status_code == 200:
                return True, None
            elif resp.status_code == 401:
                return False, (
                    "Kaggle credentials are invalid. "
                    "Get your API key at https://www.kaggle.com/settings → API → Create New Token"
                )
            else:
                return False, f"Kaggle API returned status {resp.status_code}."
        except httpx.TimeoutException:
            return False, "Kaggle API timed out. Check your internet connection."
        except Exception as e:
            return False, f"Could not reach Kaggle API: {e}"

    # ── LLM API keys ─────────────────────────────────────────────────────────

    @staticmethod
    def validate_llm_api_key(provider: str, api_key: str) -> tuple[bool, str | None]:
        """
        Lightweight check for LLM provider API keys.
        Sends a minimal request (1 token) to confirm the key works.
        """
        provider = provider.lower()

        if provider == "anthropic":
            return CredentialValidator._check_anthropic(api_key)
        elif provider == "openai":
            return CredentialValidator._check_openai(api_key)
        elif provider == "google":
            return CredentialValidator._check_google(api_key)
        elif provider == "ollama":
            return True, None  # local, no key needed
        else:
            return False, f"Unknown LLM provider '{provider}'. Supported: anthropic, openai, google, ollama"

    @staticmethod
    def _check_anthropic(api_key: str) -> tuple[bool, str | None]:
        if not api_key.startswith("sk-ant-"):
            return False, "Anthropic API keys start with 'sk-ant-'. Please check your key."
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=10,
            )
            if resp.status_code in (200, 201):
                return True, None
            elif resp.status_code == 401:
                return False, "Anthropic API key is invalid. Check https://console.anthropic.com/settings/keys"
            elif resp.status_code == 429:
                return True, None  # key valid, just rate limited
            else:
                return False, f"Anthropic API returned status {resp.status_code}."
        except Exception as e:
            return False, f"Could not reach Anthropic API: {e}"

    @staticmethod
    def _check_openai(api_key: str) -> tuple[bool, str | None]:
        if not api_key.startswith("sk-"):
            return False, "OpenAI API keys start with 'sk-'. Please check your key."
        try:
            resp = httpx.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, None
            elif resp.status_code == 401:
                return False, "OpenAI API key is invalid. Check https://platform.openai.com/api-keys"
            else:
                return False, f"OpenAI API returned status {resp.status_code}."
        except Exception as e:
            return False, f"Could not reach OpenAI API: {e}"

    @staticmethod
    def _check_google(api_key: str) -> tuple[bool, str | None]:
        try:
            resp = httpx.get(
                f"https://generativelanguage.googleapis.com/v1/models?key={api_key}",
                timeout=10,
            )
            if resp.status_code == 200:
                return True, None
            elif resp.status_code == 400 or resp.status_code == 403:
                return False, "Google API key is invalid. Check https://aistudio.google.com/apikey"
            else:
                return False, f"Google API returned status {resp.status_code}."
        except Exception as e:
            return False, f"Could not reach Google API: {e}"
        

 
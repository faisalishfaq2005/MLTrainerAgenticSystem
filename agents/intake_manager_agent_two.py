"""
intake_manager_agent.py
------------------------
Stage 1 of the pipeline — the conversational gatekeeper.

This IS a proper BaseAgent subclass. Unlike all other agents it is
interactive: _execute() drives a back-and-forth conversation with the
user until all required information is collected and validated.

Sole responsibility:
  - Converse with user to collect all required info
  - Validate credentials live (HF token, Kaggle creds, LLM API key)
  - Write THREE fields into JobContext:
      context.raw_prompt            (str)
      context.conversation_history  (list)
      context.collected_info        (dict)  <- all validated fields

It does NOT parse intent, does NOT decide architecture, does NOT touch
any other context field. Those are the next agents' jobs.

The orchestrator calls this agent differently from others: instead of
a single run(context) call, it feeds user messages one at a time via
run_turn(context, user_message) until run_turn returns ready=True,
then calls finalise(context) to write the result into the context.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentError
from agent_support.credentials_validator import CredentialValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state — lives only inside this agent, never exposed to context
# ---------------------------------------------------------------------------


@dataclass
class _IntakeState:
    raw_prompt: Optional[str] = None

    # Dataset
    dataset_url: Optional[str] = None
    dataset_source: Optional[str] = None   # "huggingface"|"kaggle"|"url"|"upload"

    # Runtime
    runtime: Optional[str] = None          # "kaggle"|"modal"

    # HuggingFace
    hf_token: Optional[str] = None
    hf_username: Optional[str] = None      # fetched from HF API
    hf_repo_name: Optional[str] = None
    hf_org: Optional[str] = None

    # Kaggle (conditional)
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None

    # LLM for the agent system
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None

    # Validation flags
    hf_token_valid: bool = False
    kaggle_creds_valid: bool = False
    llm_key_valid: bool = False

    # Error messages to surface to user on retry
    _hf_token_error: Optional[str] = None
    _kaggle_error: Optional[str] = None
    _llm_key_error: Optional[str] = None

    # Full conversation for context.conversation_history
    messages: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class IntakeManagerAgent(BaseAgent):
    """
    Conversational intake agent — Stage 1.

    The orchestrator loop:
        agent = IntakeManagerAgent(bypass_validation=False)
        print(agent.get_opening_message())

        while True:
            user_msg = input("User: ")
            result = agent.run_turn(user_msg)
            print("Agent:", result["message"])
            if result["ready"]:
                agent.finalise(context)   # writes to context
                break
        # Now context.raw_prompt, context.conversation_history,
        # and context.collected_info are populated.
        # Hand off to IntentParserAgent.
    """

    LLM_MODELS = {
        "anthropic": ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
        "openai":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "google":    ["gemini-2.0-flash", "gemini-1.5-pro"],
        "ollama":    ["llama3", "mistral", "phi3"],
    }

    RUNTIME_DESCRIPTIONS = {
        "kaggle": "Free — Kaggle Notebooks (T4/P100 GPU, 30 hr/week limit)",
        "modal":  "Paid — Modal.com (any GPU, pay-per-second, best for large models)",
    }

    def __init__(self, bypass_validation: bool = False):
        # No llm_router needed — intake is rule-based, not LLM-driven
        super().__init__(name="intake_manager", llm_router=None)
        self._state = _IntakeState()
        self._validator = CredentialValidator()
        self._bypass = bypass_validation

    # -----------------------------------------------------------------------
    # BaseAgent contract
    # -----------------------------------------------------------------------

    def _execute(self, context) -> dict:
        """
        Not used directly — intake is interactive.
        The orchestrator uses run_turn() + finalise() instead.
        This method exists to satisfy BaseAgent's abstract contract and is
        called only if someone mistakenly calls agent.run(context) directly.
        """
        raise AgentError(
            agent_name=self.name,
            stage="intake",
            reason=(
                "IntakeManagerAgent is interactive and cannot be run via run(context). "
                "Use run_turn(user_message) in a loop, then finalise(context)."
            ),
        )

    # -----------------------------------------------------------------------
    # Public interactive API
    # -----------------------------------------------------------------------

    def get_opening_message(self) -> str:
        return (
            "👋 Welcome to the ML Training Agent!\n\n"
            "I'll help you train and deploy a model to HuggingFace — automatically.\n\n"
            "Tell me what you want to build. For example:\n"
            "  • \"Train a spam email classifier on this dataset: [link]\"\n"
            "  • \"Fine-tune Mistral-7B on my instruction dataset with QLoRA\"\n"
            "  • \"Image classifier for 5 dog breeds, EfficientNet backbone\"\n\n"
            "You can give as much or as little detail as you like — "
            "I'll ask for anything I need."
        )

    def run_turn(self, user_message: str) -> dict:
        """
        Process one user message. Call this in a loop until ready=True.

        Returns:
            {
                "message": str,    reply to show the user
                "ready":   bool,   True = all info collected, call finalise()
                "stage":   str,    current intake stage label (for UI progress)
            }
        """
        self._state.messages.append({"role": "user", "content": user_message})
        self._absorb(user_message)

        reply, stage = self._next_question()

        if stage == "ready":
            return {"message": self._summary(), "ready": True, "stage": "complete"}

        self._state.messages.append({"role": "assistant", "content": reply})
        return {"message": reply, "ready": False, "stage": stage}

    def finalise(self, context) -> None:
        """
        Write the three intake outputs into JobContext.
        Call this once run_turn() returns ready=True.

        Writes:
            context.raw_prompt           str
            context.conversation_history list
            context.collected_info       dict
        """
        s = self._state
        context.raw_prompt = s.raw_prompt
        context.conversation_history = list(s.messages)
        context.collected_info = {
            "dataset_url":      s.dataset_url,
            "dataset_source":   s.dataset_source,
            "runtime":          s.runtime,
            "hf_token":         s.hf_token,
            "hf_username":      s.hf_username,
            "hf_repo_name":     s.hf_repo_name,
            "hf_org":           s.hf_org,
            "kaggle_username":  s.kaggle_username,
            "kaggle_key":       s.kaggle_key,
            "llm_provider":     s.llm_provider,
            "llm_model":        s.llm_model,
            "llm_api_key":      s.llm_api_key,
        }

    # -----------------------------------------------------------------------
    # Absorb info from a user message (order-independent)
    # -----------------------------------------------------------------------

    def _absorb(self, msg: str) -> None:
        """
        Extract any useful info from msg and update internal state.
        Called on every turn regardless of which stage we're in.
        """
        s = self._state
        lower = msg.lower().strip()

        # Raw prompt — first substantive message
        if s.raw_prompt is None and len(msg.strip()) > 10:
            s.raw_prompt = msg.strip()

        # Dataset URL
        if s.dataset_url is None:
            url = self._extract_url(msg)
            if url:
                s.dataset_url = url
                if "huggingface.co/datasets" in url:
                    s.dataset_source = "huggingface"
                elif "kaggle.com" in url:
                    s.dataset_source = "kaggle"
                else:
                    s.dataset_source = "url"

        # Runtime
        if s.runtime is None:
            if re.search(r'\bkaggle\b', lower):
                s.runtime = "kaggle"
            elif re.search(r'\bmodal\b', lower) or re.search(r'\bpaid\b', lower):
                s.runtime = "modal"

        # HuggingFace token
        hf_match = re.search(r'\bhf_[A-Za-z0-9_]{8,}\b', msg)
        if hf_match and not s.hf_token_valid:
            token = hf_match.group(0)
            if self._bypass:
                s.hf_token, s.hf_token_valid, s.hf_username = token, True, "demo-user"
            else:
                ok, err = self._validator.validate_hf_token(token)
                if ok:
                    s.hf_token = token
                    s.hf_token_valid = True
                    s.hf_username = self._validator.get_hf_username(token)
                else:
                    s._hf_token_error = err

        # HF repo name
        if s.hf_repo_name is None:
            # Try keyword-led extraction first
            repo_match = re.search(
                r'(?:repo|repository|model|push|deploy|call\s+it|named?)'
                r'[^\w]*([a-z0-9][a-z0-9\-_]{1,50})',
                lower
            )
            if repo_match:
                s.hf_repo_name = repo_match.group(1)
            elif (s.hf_token_valid
                  and re.fullmatch(r'[a-z0-9][a-z0-9\-_]{1,50}', lower.strip())):
                # Standalone slug — treat as repo name when we're in hf_repo stage
                s.hf_repo_name = lower.strip()

        # Kaggle credentials (key is a 32-char hex string)
        kaggle_key_match = re.search(r'\b([a-f0-9]{32})\b', msg)
        if kaggle_key_match and not s.kaggle_creds_valid:
            key = kaggle_key_match.group(1)
            username = None
            for line in msg.splitlines():
                if re.search(r'\buser(name)?\b', line, re.I):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        username = parts[1].strip()
            if username:
                if self._bypass:
                    s.kaggle_username, s.kaggle_key, s.kaggle_creds_valid = username, key, True
                else:
                    ok, err = self._validator.validate_kaggle_credentials(username, key)
                    if ok:
                        s.kaggle_username, s.kaggle_key, s.kaggle_creds_valid = username, key, True
                    else:
                        s._kaggle_error = err

        # LLM provider
        if s.llm_provider is None:
            for p in ["anthropic", "openai", "ollama"]:
                if re.search(rf'\b{p}\b', lower):
                    s.llm_provider = p
                    if p == "ollama":
                        s.llm_key_valid = True
                    break
            if s.llm_provider is None:
                if re.search(r'\bgoogle\b', lower) and "drive.google" not in lower:
                    s.llm_provider = "google"
            # Aliases
            if s.llm_provider is None:
                if re.search(r'\bclaude\b', lower):   s.llm_provider = "anthropic"
                elif re.search(r'\bgpt\b', lower):    s.llm_provider = "openai"
                elif re.search(r'\bgemini\b', lower): s.llm_provider = "google"

        # LLM model
        if s.llm_model is None and s.llm_provider:
            all_models = self.LLM_MODELS.get(s.llm_provider, [])
            for m in all_models:
                if m.lower() in lower:
                    s.llm_model = m
                    break
            # Exact standalone match
            if s.llm_model is None and msg.strip() in all_models:
                s.llm_model = msg.strip()

        # LLM API key
        key_patterns = [
            r'\bsk-ant-[A-Za-z0-9\-_]{10,}\b',
            r'\bsk-[A-Za-z0-9\-_]{10,}\b',
            r'\bAIza[A-Za-z0-9\-_]{20,}\b',
        ]
        if not s.llm_key_valid and s.llm_provider not in (None, "ollama"):
            for pat in key_patterns:
                m = re.search(pat, msg)
                if m:
                    key = m.group(0)
                    provider = s.llm_provider or self._infer_provider(key)
                    if provider:
                        if self._bypass:
                            s.llm_api_key, s.llm_provider, s.llm_key_valid = key, provider, True
                        else:
                            ok, err = self._validator.validate_llm_api_key(provider, key)
                            if ok:
                                s.llm_api_key, s.llm_provider, s.llm_key_valid = key, provider, True
                            else:
                                s._llm_key_error = err
                    break

    # -----------------------------------------------------------------------
    # Determine the next question to ask
    # -----------------------------------------------------------------------

    def _next_question(self) -> tuple[str, str]:
        """
        Returns (message_to_user, stage_label).
        Asks for exactly ONE missing thing in priority order.
        Returns ("", "ready") when everything is collected.
        """
        s = self._state

        if s.raw_prompt is None:
            return (
                "What would you like to train? Describe your task — "
                "even a sentence like 'classify customer reviews as positive or negative' "
                "is enough to start.",
                "task"
            )

        if s.dataset_url is None:
            return (
                "📦 **Dataset** — where is your data?\n\n"
                "Accepted formats:\n"
                "  • HuggingFace: `https://huggingface.co/datasets/your-dataset`\n"
                "  • Kaggle: `https://www.kaggle.com/datasets/...`\n"
                "  • Direct URL to a CSV, ZIP, or JSONL file\n"
                "  • Type `upload` if you want to upload a local file",
                "dataset"
            )

        if s.runtime is None:
            return (
                "💻 **Training runtime** — where should the model be trained?\n\n"
                "  • `kaggle` — Free (T4/P100 GPU, 30 hr/week). Good for models up to ~3B.\n"
                "  • `modal`  — Paid (any GPU, pay-per-second). Best for large models.\n\n"
                "Type `kaggle` or `modal`.",
                "runtime"
            )

        if not s.hf_token_valid:
            note = f"\n\n⚠️ {s._hf_token_error}" if s._hf_token_error else ""
            return (
                f"🤗 **HuggingFace token** — needed to push your model after training.{note}\n\n"
                "Get yours at https://huggingface.co/settings/tokens → New token → **Write** access.\n"
                "Paste it here (it starts with `hf_`).",
                "hf_token"
            )

        if s.hf_repo_name is None:
            username = s.hf_username or "your-username"
            return (
                f"📛 **Model repo name** — what should the HuggingFace repo be called?\n\n"
                f"It will be published as: `{username}/YOUR-REPO-NAME`\n"
                f"Example: `spam-detector`, `dog-breed-classifier`, `mistral-finetuned`",
                "hf_repo"
            )

        needs_kaggle = (s.runtime == "kaggle" or s.dataset_source == "kaggle")
        if needs_kaggle and not s.kaggle_creds_valid:
            note = f"\n\n⚠️ {s._kaggle_error}" if s._kaggle_error else ""
            return (
                f"🏁 **Kaggle credentials** — needed to run training on Kaggle.{note}\n\n"
                "Get your API key at https://www.kaggle.com/settings → API → Create New Token\n"
                "Paste the values like this:\n\n"
                "```\nusername: YOUR_KAGGLE_USERNAME\nkey: YOUR_32_CHAR_KEY\n```",
                "kaggle_creds"
            )

        if s.llm_provider is None:
            return (
                "🧠 **LLM for the agents** — which AI should power the agent system?\n\n"
                "  • `anthropic` — Claude (recommended)\n"
                "  • `openai`    — GPT-4o\n"
                "  • `google`    — Gemini\n"
                "  • `ollama`    — Local model (free, no API key needed)\n\n"
                "Type the provider name.",
                "llm_provider"
            )

        if s.llm_model is None:
            models = self.LLM_MODELS.get(s.llm_provider, [])
            model_list = "\n".join(f"  • `{m}`" for m in models)
            default = models[0] if models else "default"
            return (
                f"Which {s.llm_provider} model?\n\n{model_list}\n\n"
                f"Or press Enter for the default: `{default}`",
                "llm_model"
            )

        if s.llm_provider != "ollama" and not s.llm_key_valid:
            note = f"\n\n⚠️ {s._llm_key_error}" if s._llm_key_error else ""
            key_url = {
                "anthropic": "https://console.anthropic.com/settings/keys",
                "openai":    "https://platform.openai.com/api-keys",
                "google":    "https://aistudio.google.com/apikey",
            }.get(s.llm_provider, "your provider's dashboard")
            return (
                f"🔑 **{s.llm_provider.title()} API key** — so the agents can think.{note}\n\n"
                f"Get it at: {key_url}\nPaste it here.",
                "llm_key"
            )

        # Apply default model if user skipped that step
        if s.llm_model is None:
            models = self.LLM_MODELS.get(s.llm_provider, [])
            s.llm_model = models[0] if models else None

        return ("", "ready")

    # -----------------------------------------------------------------------
    # Summary shown to user when all info is collected
    # -----------------------------------------------------------------------

    def _summary(self) -> str:
        s = self._state
        username = s.hf_username or "unknown"
        repo_full = f"{s.hf_org or username}/{s.hf_repo_name}"
        return (
            f"✅ **All set! Here's what I have:**\n\n"
            f"  📋 Task: {s.raw_prompt[:120]}{'...' if len(s.raw_prompt) > 120 else ''}\n"
            f"  📦 Dataset: `{s.dataset_url}` ({s.dataset_source})\n"
            f"  💻 Runtime: `{s.runtime}` — {self.RUNTIME_DESCRIPTIONS[s.runtime]}\n"
            f"  🤗 HuggingFace repo: `{repo_full}` (logged in as `{username}`)\n"
            f"  🧠 LLM: `{s.llm_provider}/{s.llm_model}`\n\n"
            f"🚀 **Handing off to the agent pipeline...**\n"
            f"The agents are now planning your training job. "
            f"I'll send you updates as each stage completes."
        )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_url(text: str) -> Optional[str]:
        m = re.search(r'https?://[^\s\]\)>]+', text)
        return m.group(0).rstrip('.,;') if m else None

    @staticmethod
    def _infer_provider(key: str) -> Optional[str]:
        if key.startswith("sk-ant-"): return "anthropic"
        if key.startswith("sk-"):     return "openai"
        if key.startswith("AIza"):    return "google"
        return None
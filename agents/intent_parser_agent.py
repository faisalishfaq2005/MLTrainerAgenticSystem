"""
intent_parser_agent.py
-----------------------
Stage 2 of the pipeline — produces the universal parsed_intent JSON.

Reads from JobContext:
    context.raw_prompt           — user's original free-text description
    context.conversation_history — full intake conversation (for extra context)
    context.collected_info       — all validated fields from intake agent
                                   (hf_token, dataset_url, runtime, etc.)

Writes to JobContext:
    context.parsed_intent        — the complete IntentSchema JSON dict

Design:
  - LLM receives (1) raw_prompt, (2) conversation_history summary,
    (3) collected_info key-value pairs, (4) the JSON template to fill.
  - LLM extracts task intent, architecture, hyperparameters from the prompt.
  - collected_info values are embedded directly into the JSON by _patch_verified()
    AFTER the LLM responds — so validated values always win over LLM extraction.
  - Fields the user did not specify remain null for downstream agents to fill.
"""

import json
import re
import logging

from agents.base_agent import BaseAgent, AgentError
from agent_schemas.intent_parser_schema import INTENT_JSON_TEMPLATE
from llm.prompts.intent_parser_agent_prompt import INTENT_PARSER_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class IntentParserAgent(BaseAgent):
    """
    Stage 2: converts raw_prompt + collected_info → parsed_intent JSON.

    Reads:  context.raw_prompt, context.conversation_history, context.collected_info
    Writes: context.parsed_intent
    """

    def __init__(self, llm_router):
        super().__init__(name="intent_parser", llm_router=llm_router)

    def _execute(self, context) -> dict:
        self._require_context_keys(context, "raw_prompt", "collected_info")

        user_message = self._build_llm_input(context)

        self.logger.info("Calling LLM to parse intent...")
        raw_response = self.llm_router.complete(
            system_prompt=INTENT_PARSER_AGENT_SYSTEM_PROMPT,
            user_message=user_message,
        )

        parsed = self._parse_response(raw_response)
        self._patch_verified(parsed, context.collected_info)
        self._ensure_structure(parsed)
        self._validate(parsed)

        self.logger.info(
            f"parsed_intent ready: task={parsed.get('task_type')}, "
            f"expertise={parsed.get('user_expertise_level')}"
        )

        # Orchestrator merges this return dict into context:
        #   context.parsed_intent = parsed
        return {"parsed_intent": parsed}

    # -----------------------------------------------------------------------
    # Build the LLM input
    # -----------------------------------------------------------------------

    def _build_llm_input(self, context) -> str:
        """
        Builds the user-turn message sent to the LLM.

        Section 1 — raw_prompt: what the user said they want.
        Section 2 — conversation summary: last few exchanges (extra signal).
        Section 3 — collected_info: the verified values intake gathered.
                    Listed explicitly so the LLM embeds them directly.
        Section 4 — the JSON template to fill.
        """
        parts = []

        # 1. Raw prompt
        parts.append(f"USER'S TRAINING REQUEST:\n{context.raw_prompt}")

        # 2. Last 6 exchanges from conversation (gives the LLM extra context
        #    for things the user said in follow-up messages, e.g. repo name)
        history = context.conversation_history or []
        if len(history) > 2:
            recent = history[-6:]  # last 3 user + 3 assistant turns
            lines = [f"  [{m['role']}]: {m['content'][:200]}" for m in recent]
            parts.append("\nRECENT CONVERSATION (last few turns):\n" + "\n".join(lines))

        # 3. Collected info — all validated by intake agent.
        #    Include every non-None value; skip Nones so LLM isn't confused.
        info = context.collected_info or {}
        verified_lines = []
        field_order = [
            "dataset_url", "dataset_source", "runtime",
            "hf_token", "hf_username", "hf_repo_name", "hf_org",
            "kaggle_username", "kaggle_key",
            "llm_provider", "llm_model", "llm_api_key",
        ]
        for key in field_order:
            val = info.get(key)
            if val is not None:
                verified_lines.append(f"  {key}: {val}")

        if verified_lines:
            parts.append(
                "\nVERIFIED COLLECTED INFO (validated by intake — "
                "embed these directly into the JSON without modification):\n"
                + "\n".join(verified_lines)
            )

        # 4. Template + instruction
        parts.append(f"\nJSON TEMPLATE TO FILL:\n{INTENT_JSON_TEMPLATE}")
        parts.append(
            "\nFill the template. "
            "Use null for fields the user did NOT explicitly specify. "
            "Return ONLY the JSON."
        )

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # Parse LLM response
    # -----------------------------------------------------------------------

    def _parse_response(self, raw: str) -> dict:
        """Strip markdown fences and parse JSON. Raise AgentError if invalid."""
        cleaned = raw.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: find first {...} block
            m = re.search(r'\{[\s\S]+\}', cleaned)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
            raise AgentError(
                agent_name=self.name,
                stage="intent_parsing",
                reason=f"LLM returned invalid JSON. First 300 chars: {raw[:300]}"
            )

    # -----------------------------------------------------------------------
    # Post-LLM processing
    # -----------------------------------------------------------------------

    def _patch_verified(self, parsed: dict, collected_info: dict) -> None:
        """
        Overwrite parsed_intent fields with the validated values from collected_info.

        Why: The LLM may extract credentials or URLs from the raw prompt text,
        but intake-validated values are more reliable (they passed live API checks).
        Rule: collected_info always wins over LLM-extracted values for these fields.

        For hf_username / hf_repo_name / hf_org: use setdefault() because the LLM
        may have correctly parsed them from the prompt (e.g. "push to my-org/my-model"),
        and we only want to fill in what the LLM left as null.
        """
        if not collected_info:
            return

        creds  = parsed.setdefault("credentials", {})
        deploy = parsed.setdefault("deploy", {})
        ds     = parsed.setdefault("dataset", {})

        # Credentials — always overwrite (intake-validated is authoritative)
        for key in ("hf_token", "kaggle_username", "kaggle_key",
                    "llm_provider", "llm_model", "llm_api_key"):
            val = collected_info.get(key)
            if val is not None:
                creds[key] = val

        # Deploy — hf_token always overwrite; username/repo/org use setdefault
        if collected_info.get("hf_token"):
            deploy["hf_token"] = collected_info["hf_token"]
        for key in ("hf_username", "hf_repo_name", "hf_org"):
            val = collected_info.get(key)
            if val is not None:
                deploy.setdefault(key, val)

        # Dataset — use setdefault (LLM may have found more detail in the prompt)
        if collected_info.get("dataset_url"):
            ds.setdefault("url", collected_info["dataset_url"])
        if collected_info.get("dataset_source"):
            ds.setdefault("source_type", collected_info["dataset_source"])

        # Top-level runtime — always from collected_info
        if collected_info.get("runtime"):
            parsed["runtime"] = collected_info["runtime"]

    def _ensure_structure(self, parsed: dict) -> None:
        """Guarantee all expected sub-dicts exist (even if LLM omitted one)."""
        for key in ("dataset", "hyperparameters", "peft", "precision",
                    "architecture", "deploy", "credentials"):
            parsed.setdefault(key, {})

    def _validate(self, parsed: dict) -> None:
        """Hard validation: task_type must be a recognised value."""
        valid_tasks = {
            "tabular_classification", "tabular_regression",
            "image_classification",   "image_regression",
            "text_classification",    "text_generation",
            "llm_finetuning",         "token_classification",
            "summarization",          "translation",
            "object_detection",       "time_series_forecasting",
            "clustering",
        }
        task = parsed.get("task_type")
        if task not in valid_tasks:
            raise AgentError(
                agent_name=self.name,
                stage="intent_parsing",
                reason=(
                    f"LLM returned unrecognised task_type='{task}'. "
                    f"Must be one of: {', '.join(sorted(valid_tasks))}"
                ),
            )
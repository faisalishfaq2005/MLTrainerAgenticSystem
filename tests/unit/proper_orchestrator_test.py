"""
proper_orchestrator_test.py
---------------------------
Interactive CLI test that runs the intake + intent_parser pipeline
through the Orchestrator class.

Run from the project root:
    python -m tests.unit.proper_orchestrator_test

Two LLM routers are used:
  - intake_router  : free fallback (Groq/Ollama) — used during the intake
                     conversation before the user has provided any credentials.
  - pipeline_router: built from context.collected_info after intake completes,
                     so it uses whichever provider/model/key the user specified.

Console shows ONLY the agent's messages and your input.
Stages without a registered agent are skipped (expected during development).
All debug/info logs go to a timestamped file under logs/.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logging_config import setup_logging
from llm.router import LLMRouter
from agents.intent_parser_agent import IntentParserAgent
from orchestrator.orchestrator import Orchestrator


def run() -> None:
    log_file = setup_logging(log_dir="logs", session_label="orchestrator_test")
    print(f"Logs → {log_file}\n")

    # Router for the intake agent — free fallback because the user hasn't
    # provided credentials yet; those are collected during the conversation.
    intake_router = LLMRouter.from_collected_info(collected_info={}, max_tokens=800)

    # Orchestrator starts with no pipeline agents. They are registered after
    # intake completes and we know which provider/model the user wants.
    orc = Orchestrator(agents={}, db_path="workspace/test_jobs.db")

    job_id = orc.new_job(llm_router=intake_router)
    print(orc.get_opening_message(job_id))
    print()

    # ── Intake phase ──────────────────────────────────────────────────────────
    ready = False
    while not ready:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession interrupted.")
            return

        if not user_input:
            continue

        resp = orc.send_intake_message(job_id, user_input)
        print(f"\nAgent: {resp['message']}\n")
        ready = resp["ready"]

    # ── Build pipeline router from what the user just told us ─────────────────
    # collected_info now has llm_provider, llm_model, llm_api_key (if provided).
    # LLMRouter.from_collected_info falls back to free models if keys are absent.
    collected_info = orc._store.load(job_id).collected_info
    pipeline_router = LLMRouter.from_collected_info(collected_info)

    # Register pipeline agents with the user's chosen router
    orc._agents["intent_parser"] = IntentParserAgent(llm_router=pipeline_router)

    # ── Pipeline phase ────────────────────────────────────────────────────────
    print("\n--- Intake complete. Running pipeline ---\n")
    result = orc.run_pipeline(job_id)

    if result["status"] == "failed":
        print(f"\n✗ Pipeline failed at stage '{result['current_stage']}':")
        print(f"  {result['failure_reason']}")
        return

    # ── Display results ───────────────────────────────────────────────────────
    context = orc._store.load(job_id)
    if context is None:
        print("Error: could not load job context from store.")
        return

    print(f"\n✓ Pipeline finished  |  job_id={job_id}  |  {result['duration_s']}s")
    print(f"  Stages completed : {[s for s, r in context.stage_results.items() if r.get('status') != 'skipped']}")
    print(f"  Stages skipped   : {[s for s, r in context.stage_results.items() if r.get('status') == 'skipped']}")

    if context.parsed_intent:
        pi = context.parsed_intent
        print("\n" + "=" * 70)
        print("PARSED INTENT — summary")
        print("=" * 70)
        print(f"  task_type    : {pi.get('task_type')}")
        print(f"  expertise    : {pi.get('user_expertise_level')}")
        print(f"  runtime      : {pi.get('runtime')}")
        print(f"  dataset_url  : {(pi.get('dataset') or {}).get('url')}")
        print(f"  backbone     : {(pi.get('architecture') or {}).get('backbone')}")
        print(f"  use_lora     : {(pi.get('peft') or {}).get('use_lora')}")
        print(f"  use_qlora    : {(pi.get('peft') or {}).get('use_qlora')}")

    print("\n" + "=" * 70)
    print("FULL JOB CONTEXT")
    print("=" * 70 + "\n")
    print(json.dumps(context.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    run()

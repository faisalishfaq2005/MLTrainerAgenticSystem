"""
intake_manager_test.py
-----------------------
Interactive CLI test for IntakeManagerAgent.

Run from the project root:
    python -m tests.unit.intake_manager_test

Console shows ONLY the agent's messages and your input.
All debug / info logs go to a timestamped file under logs/.
"""

import sys
import os
import json

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logging_config import setup_logging
from llm.router import LLMRouter
from agents.intake_manager_agent import IntakeManagerAgent
from agents.intent_parser_agent import IntentParserAgent
from orchestrator.job_context import JobContext


def run_manual_orchestrator_test() -> None:
    log_file = setup_logging(log_dir="logs", session_label="intake_test")
    print(f"Logs → {log_file}\n")

    llm_router = LLMRouter.from_collected_info(collected_info={}, max_tokens=800)
    intake_manager_agent = IntakeManagerAgent(llm_router=llm_router)
    intent_parser_agent=IntentParserAgent(llm_router=llm_router)


    context = JobContext()

    print(intake_manager_agent.get_opening_message())
    print()

    ready = False
    while not ready:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession interrupted.")
            break

        if not user_input:
            continue

        resp = intake_manager_agent.converse(user_message=user_input)
        print(f"\nAgent: {resp['message']}\n")
        ready = resp["ready"]

    if ready:
        intake_manager_agent.finalize(context)
        print("\n--- Intake complete ---")
        print(f"Collected fields: {list(context.collected_info.keys())}")

    agent_parsed_intent=intent_parser_agent._execute(context=context)

    context.parsed_intent=agent_parsed_intent

    print("\n" + "="*80)
    print("JOB CONTEXT (PRETTY PRINTED)")
    print("="*80 + "\n")
    print(json.dumps(context.to_dict(), indent=2, default=str))

    


if __name__ == "__main__":
    run_manual_orchestrator_test()

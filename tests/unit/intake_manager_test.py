from llm.router import LLMRouter
from agents.intake_manager_agent import IntakeManagerAgent
from orchestrator.job_context import JobContext
import logging

import httpx

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    # Keep external library logs quiet; show only warnings/errors from them.
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Keep your project logs at requested verbosity.
    logging.getLogger("agents").setLevel(level)
    logging.getLogger("tool").setLevel(level)
    logging.getLogger("llm").setLevel(level)



def run_manual_intake_test():
    configure_logging(logging.INFO)
    llm_router_for_intake_manager = LLMRouter.from_collected_info(collected_info={},max_tokens=800)
    intake_manager_agent = IntakeManagerAgent(llm_router=llm_router_for_intake_manager)
    context = JobContext()

    print(intake_manager_agent.get_opening_message())
    ready = False
    while not ready:
        prompt = input("User: ")
        resp = intake_manager_agent.converse(user_message=prompt)
        response = resp["message"]
        ready = resp["ready"]

        print(response)

    intake_manager_agent.finalize(context)
    print("\nIntake complete. Collected info:")
    print(context.collected_info)


if __name__ == "__main__":
    run_manual_intake_test()
   

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING
import logging
import time

if TYPE_CHECKING:
    from orchestrator.queue_classes import Event_Queue, EventType

logger = logging.getLogger(__name__)
from orchestrator.job_context import JobContext

class AgentError(Exception):
    """Raised when an agent fails after all retries are exhausted."""
    def __init__(self,agent_name:str, stage:str, reason: str):
        self.agent_name=agent_name
        self.stage=stage
        self.reason=reason
        super().__init__(f"[{agent_name}] Stage '{stage}' failed: {reason}")


class BaseAgent(ABC):
    """
    Every agent inherits this class and implements run().
 
    The orchestrator calls agent.run(context) and gets back a dict.
    Agents do NOT talk to each other — they only read/write the context.
 
    Lifecycle:
        1. orchestrator calls run(context)
        2. run() calls _execute(context) — the subclass implements this
        3. result dict is returned to orchestrator
        4. orchestrator merges result into context and passes to next agent
    """
    def __init__(self, name: str, llm_router=None, event_queue: Optional["Event_Queue"] = None):
        self.name = name
        self.llm_router = llm_router
        self.event_queue = event_queue
        self.logger = logging.getLogger(f"agent.{name}")
        super().__init__()

    def run(self, context:JobContext) -> dict:
        self.logger.info(f"Starting agent: {self.name}")
        start = time.monotonic()

        try:
            result=self._execute(context)
            elapsed= time.monotonic() - start
            self.logger.info(f"Agent {self.name} completed in {elapsed:.1f}s")
            return {"status":"ok", "agent":self.name, **result}
        except AgentError:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start
            self.logger.error(f"Agent {self.name} failed after {elapsed:.1f}s: {exc}", exc_info=True)
            raise AgentError(agent_name=self.name, stage=self.name, reason=str(exc)) from exc
        
    
    @abstractmethod
    def _execute(self,context: JobContext) -> dict:
        """
        Subclasses implement the actual work here.
        Must return a dict of results that will be merged into the context.
 
        Example return values:
            dataset_agent  → {"data_report": {...}, "local_data_path": "/workspace/..."}
            config_agent   → {"final_config": {...}}
            codegen_agent  → {"train_script": "...", "requirements": "..."}
        """
    
    def _emit(self, event_type: "EventType", **data) -> None:
        """Put an event into the event queue if one is wired up."""
        if self.event_queue is None:
            return
        from orchestrator.queue_classes import Event
        self.event_queue.put(Event(event_type=event_type, data=data))

    def _require_context_keys(self, context: JobContext, *keys: str):  # noqa: F821
        """Helper — raise clearly if a required context field is missing."""
        for key in keys:
            if getattr(context, key, None) is None:
                raise AgentError(
                    agent_name=self.name,
                    stage=self.name,
                    reason=f"Required context field '{key}' is missing. "
                           f"Ensure the preceding agent populated it.",
                )

    # ── Retry-feedback helpers ────────────────────────────────────────────────
    # RetryHandler writes the previous attempt's error into
    # context.last_error_feedback[stage_name].  These two helpers let every
    # agent read that feedback and inject it into its LLM call without
    # duplicating the lookup logic.

    def _get_retry_feedback(self, context: JobContext):
        """
        Return the error string from the previous failed attempt for this
        agent's stage, or None if this is the first attempt.

        Use this when the agent builds a single string prompt:
            feedback = self._get_retry_feedback(context)
            if feedback:
                prompt += f"\\n\\nPREVIOUS ATTEMPT FAILED:\\n{feedback}"
        """
        return (context.last_error_feedback or {}).get(self.name)

    def _apply_retry_feedback_to_messages(self, messages: list, context: JobContext) -> list:
        """
        If there is retry feedback for this stage, append it as an extra
        user message at the end of the messages list so the LLM sees it
        as the most recent human turn before it responds.

        Use this when the agent passes a message-history list to the LLM:
            messages = self._apply_retry_feedback_to_messages(messages, context)
            response = self.llm_router.complete(..., message_history=messages)

        Returns a new list — the original is never mutated.
        """
        feedback = self._get_retry_feedback(context)
        if not feedback:
            return messages
        return list(messages) + [
            {
                "role": "user",
                "content": (
                    "Your previous response failed validation.\n"
                    f"Error: {feedback}\n\n"
                    "Please correct the mistake and try again."
                ),
            }
        ]

    


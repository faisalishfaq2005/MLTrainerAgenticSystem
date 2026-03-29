from abc import ABC, abstractmethod
from typing import Any
import logging
import time
 
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
    def __init__(self, name:str, llm_router=None):
        self.name=name
        self.llm_router=llm_router
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

    


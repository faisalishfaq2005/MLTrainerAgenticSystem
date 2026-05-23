"""
retry_handler.py
----------------
Per-stage retry configuration and execution logic.

Each stage has its own retry budget and backoff because they fail
for very different reasons:
  - intent_parser: LLM produced bad JSON → retry immediately, different seed
  - dataset:       Network flake → retry with backoff
  - codegen:       Syntax error → retry with error feedback injected
  - monitor:       Long-running poll → many retries with long gaps

The graph node calls RetryHandler.run_with_retry(agent, context, stage)
and gets back either (result_dict, None) on success or (None, error_dict)
on final failure.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

from agents.base_agent import AgentError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-stage configuration
# ---------------------------------------------------------------------------

@dataclass
class StageRetryConfig:
    max_attempts: int        # total attempts including the first
    base_delay:   float      # seconds to wait after first failure
    backoff:      float      # multiply delay by this after each failure
    max_delay:    float      # cap the delay at this many seconds
    # If True, the error message from the previous attempt is appended to
    # the context so the agent can self-correct on retry.
    inject_error_feedback: bool = False


# Stages that don't exist here use DEFAULT_CONFIG
STAGE_CONFIGS: dict[str, StageRetryConfig] = {
    "intake_manager": StageRetryConfig(
        max_attempts=1,      # interactive — never retry automatically
        base_delay=0, backoff=1, max_delay=0,
    ),
    "intent_parser": StageRetryConfig(
        max_attempts=3,      # LLM may produce bad JSON — retry quickly
        base_delay=2, backoff=1.5, max_delay=10,
        inject_error_feedback=True,
    ),
    "dataset": StageRetryConfig(
        max_attempts=3,      # network issues — retry with backoff
        base_delay=5, backoff=2.0, max_delay=30,
    ),
    "preprocessing": StageRetryConfig(
        max_attempts=2,
        base_delay=3, backoff=2.0, max_delay=15,
    ),
    "config": StageRetryConfig(
        max_attempts=3,      # LLM-driven — retry with feedback
        base_delay=2, backoff=1.5, max_delay=10,
        inject_error_feedback=True,
    ),
    "architecture": StageRetryConfig(
        max_attempts=3,
        base_delay=2, backoff=1.5, max_delay=10,
        inject_error_feedback=True,
    ),
    "codegen": StageRetryConfig(
        max_attempts=3,      # syntax errors → retry with error message injected
        base_delay=2, backoff=1.5, max_delay=10,
        inject_error_feedback=True,
    ),
    "monitor": StageRetryConfig(
        max_attempts=10,     # long-running — poll many times
        base_delay=30, backoff=1.2, max_delay=120,
    ),
    "deploy": StageRetryConfig(
        max_attempts=3,      # HF API flakes
        base_delay=5, backoff=2.0, max_delay=30,
    ),
}

DEFAULT_CONFIG = StageRetryConfig(
    max_attempts=2,
    base_delay=3, backoff=2.0, max_delay=20,
)


# ---------------------------------------------------------------------------
# RetryHandler
# ---------------------------------------------------------------------------

class RetryHandler:
    """
    Executes an agent with retries according to its StageRetryConfig.

    Usage (inside a LangGraph node):
        result, error = RetryHandler.run(
            agent=self.intent_parser,
            context=context,
            stage="intent_parser",
            retry_counts=state["retry_counts"],
        )
        if error:
            # agent exhausted all retries
            return {"pipeline_failed": True, "failure_reason": error["reason"]}
        # merge result into state
        context.parsed_intent = result["parsed_intent"]
    """

    @staticmethod
    def run(
        agent,
        context,
        stage: str,
        retry_counts: dict,
    ) -> tuple[Optional[dict], Optional[dict]]:
        """
        Run agent with retries. Returns (result, None) on success or
        (None, error_dict) when all attempts are exhausted.

        Args:
            agent:         Any BaseAgent subclass with a .run(context) method.
            context:       The current JobContext.
            stage:         Stage name key — must match STAGE_CONFIGS keys.
            retry_counts:  Mutable dict tracking how many attempts each stage
                           has made. Updated in place.

        Returns:
            (result_dict, None)  — agent succeeded
            (None, error_dict)   — agent failed, error_dict has "stage" and "reason"
        """
        cfg = STAGE_CONFIGS.get(stage, DEFAULT_CONFIG)
        attempts_so_far = retry_counts.get(stage, 0)
        last_error_msg: Optional[str] = None

        while attempts_so_far < cfg.max_attempts:
            attempt_num = attempts_so_far + 1
            retry_counts[stage] = attempt_num

            logger.info(
                f"[{stage}] Attempt {attempt_num}/{cfg.max_attempts}"
            )

            # On retry with feedback: attach the previous error to the context
            # so the agent's LLM prompt includes what went wrong
            if cfg.inject_error_feedback and last_error_msg and attempt_num > 1:
                RetryHandler._inject_feedback(context, stage, last_error_msg)

            try:
                result = agent.run(context)
                logger.info(f"[{stage}] Succeeded on attempt {attempt_num}")
                return result, None

            except AgentError as e:
                last_error_msg = e.reason
                attempts_so_far = attempt_num

                logger.warning(
                    f"[{stage}] Attempt {attempt_num} failed: {e.reason}"
                )

                # Record the error in the context for observability
                context.errors.append({
                    "stage":   stage,
                    "attempt": attempt_num,
                    "reason":  e.reason,
                    "ts":      time.time(),
                })

                # If we still have attempts left, wait before retrying
                if attempts_so_far < cfg.max_attempts:
                    delay = min(
                        cfg.base_delay * (cfg.backoff ** (attempts_so_far - 1)),
                        cfg.max_delay,
                    )
                    if delay > 0:
                        logger.info(f"[{stage}] Waiting {delay:.1f}s before retry...")
                        time.sleep(delay)

            except Exception as e:
                # Unexpected exceptions — wrap and fail immediately (no retry)
                reason = f"Unexpected error in {stage}: {type(e).__name__}: {e}"
                logger.error(f"[{stage}] {reason}", exc_info=True)
                context.errors.append({
                    "stage":   stage,
                    "attempt": attempt_num,
                    "reason":  reason,
                    "ts":      time.time(),
                })
                return None, {"stage": stage, "reason": reason}

        # Exhausted all attempts
        final_reason = (
            f"Stage '{stage}' failed after {cfg.max_attempts} attempts. "
            f"Last error: {last_error_msg}"
        )
        logger.error(f"[{stage}] {final_reason}")
        return None, {"stage": stage, "reason": final_reason}

    @staticmethod
    def _inject_feedback(context, stage: str, error_msg: str) -> None:
        """
        Write the previous attempt's error into context.last_error_feedback.

        Agents read this via BaseAgent._get_retry_feedback() or
        BaseAgent._apply_retry_feedback_to_messages() and include the error
        in their next LLM call so the model can self-correct.
        Agents that don't declare inject_error_feedback=True never see this.
        """
        context.last_error_feedback[stage] = error_msg
        logger.debug("[%s] Retry feedback stored for next attempt.", stage)
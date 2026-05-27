"""
orchestrator.py
---------------
The public API for running the ML training pipeline.

No external graph library. The pipeline is a simple ordered loop:
  1. Run the interactive intake conversation until the agent says "ready".
  2. Run intent_parser to convert raw_prompt → parsed_intent JSON.
  3. Determine the stage list from execution_plan.get_stages_for_task().
  4. Run each stage in order via RetryHandler (per-stage retry config).
  5. Checkpoint JobContext to SQLite after every completed stage.

Resume works because each completed stage is recorded in
context.stage_results — on resume the loop just skips those stages.

USAGE:
    from orchestrator.orchestrator import Orchestrator

    orc = Orchestrator(agents={"intent_parser": IntentParserAgent(router), ...})

    job_id = orc.new_job()
    print(orc.get_opening_message(job_id))

    while True:
        reply = orc.send_intake_message(job_id, input("You: "))
        print("Agent:", reply["message"])
        if reply["ready"]:
            break

    result = orc.run_pipeline(job_id)
    print(result)
"""

import logging
import time
import uuid
from typing import Optional

from orchestrator.job_context import JobContext
from orchestrator.retry_handler import RetryHandler
from orchestrator.execution_plan import get_stages_for_task, should_deploy
from storage.job_store import JobStore
from agents.intake_manager_agent import IntakeManagerAgent
from llm.router import LLMRouter
from orchestrator.queue_classes import Event, EventType, Event_Queue, Submission_Queue

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Owns the full lifecycle of every ML training job.

    Responsibilities:
      - Create and track jobs (job_id → JobContext in SQLite)
      - Drive the interactive intake phase (send_intake_message loop)
      - Run each pipeline stage sequentially with per-stage retries
      - Checkpoint context to SQLite after every stage
      - Resume interrupted jobs by skipping already-completed stages
      - Expose status and results to the API / UI layer

    Thread safety:
      Each job has its own JobContext object. SQLite uses WAL mode for
      concurrent reads. Do not share a single JobContext across threads.
    """

    def __init__(
        self,
        agents: Optional[dict] = None,
        db_path: str = "workspace/jobs.db",
        event_queue: Optional[Event_Queue] = None,
        submission_queue: Optional[Submission_Queue] = None,
    ):
        """
        Args:
            agents:  Dict of stage_name → agent instance. Stages without an
                     agent are skipped with a warning (useful while building).
                     {
                       "intent_parser":  IntentParserAgent(...),
                       "dataset":        DatasetAgent(...),
                       "preprocessing":  PreprocessingAgent(...),
                       "config":         ConfigAgent(...),
                       "architecture":   ArchitectureAgent(...),
                       "codegen":        CodegenAgent(...),
                       "monitor":        MonitorAgent(...),
                       "deploy":         DeployAgent(...),
                     }
            db_path: Path to the SQLite database for job checkpoints.
        """
        self._agents: dict = agents or {}
        self._store = JobStore(db_path=db_path)

        # In-memory: job_id → IntakeManagerAgent.
        # Only exists while intake is in progress; freed once intake completes.
        self._intake_agents: dict[str, IntakeManagerAgent] = {}
        self.event_queue = event_queue or Event_Queue()
        self.submission_queue = submission_queue or Submission_Queue()

        logger.info(
            "Orchestrator ready | agents=%s | db=%s",
            list(self._agents.keys()),
            db_path,
        )

    # ── Job lifecycle ──────────────────────────────────────────────────────────

    def new_job(self, llm_router: Optional[LLMRouter] = None) -> str:
        """
        Create a new job and return its job_id.

        Args:
            llm_router: LLMRouter for the intake agent. Defaults to the
                        free-fallback router (Groq → Ollama) when None.
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        context = JobContext(job_id=job_id)

        router = llm_router or LLMRouter.from_collected_info({}, max_tokens=800)
        self._intake_agents[job_id] = IntakeManagerAgent(llm_router=router, event_queue=self.event_queue)

        self._store.save(context, status="running")
        logger.info("New job created: %s", job_id)
        return job_id

    def get_opening_message(self, job_id: str) -> str:
        """Return the intake agent's opening greeting for this job."""
        agent = self._intake_agents.get(job_id)
        if agent is None:
            raise ValueError(f"No intake agent for job {job_id!r}. Call new_job() first.")
        return agent.get_opening_message()

    # ── Interactive intake ─────────────────────────────────────────────────────

    def send_intake_message(self, job_id: str, user_message: str) -> dict:
        """
        Forward one user message to the intake agent and return its reply.

        Returns:
            {"message": str, "ready": bool}

        When ready=True, call run_pipeline(job_id) to start training.
        """
        agent = self._intake_agents.get(job_id)
        if agent is None:
            raise ValueError(f"No intake agent for job {job_id!r}.")

        result = agent.converse(user_message)

        if result["ready"]:
            context = self._store.load(job_id) or JobContext(job_id=job_id)
            agent.finalize(context)
            self._store.save(context, status="running")
            del self._intake_agents[job_id]   # free memory
            logger.info("[%s] Intake complete — ready for pipeline.", job_id)

        return result

    # ── Pipeline execution ─────────────────────────────────────────────────────

    def run_pipeline(self, job_id: str) -> dict:
        """
        Run the full pipeline for a job that has completed intake.
        Blocks until the pipeline finishes (success or failure).

        Stage order:
          1. intent_parser  — always first; determines task_type
          2. task-specific  — from execution_plan.get_stages_for_task(task_type)

        Already-completed stages (in context.stage_results) are skipped,
        so calling run_pipeline on an interrupted job resumes correctly.

        Returns:
            {
              "job_id":         str,
              "status":         "completed" | "failed",
              "hf_model_url":   str | None,
              "hf_space_url":   str | None,
              "failure_reason": str | None,
              "current_stage":  str | None,
              "duration_s":     float,
              "errors":         list,
            }
        """
        context = self._store.load(job_id)
        if context is None:
            raise ValueError(f"Job {job_id!r} not found.")
        if context.collected_info is None:
            raise RuntimeError(
                f"Job {job_id!r} has not completed intake. "
                "Call send_intake_message() until ready=True first."
            )

        logger.info("[%s] Starting pipeline.", job_id)
        t0 = time.monotonic()
        self._emit(EventType.PIPELINE_START, job_id=job_id)

        try:
            # ── Phase 1: intent_parser ─────────────────────────────────────────
            if "intent_parser" not in context.stage_results:
                if self._run_stage(job_id, "intent_parser", context):
                    self._emit(EventType.PIPELINE_END, job_id=job_id, status="failed",
                               duration_s=round(time.monotonic() - t0, 2),
                               reason=context.failure_reason)
                    return self._build_result(context, time.monotonic() - t0)

            # ── Phase 2: task-specific stages ──────────────────────────────────
            task_type = (context.parsed_intent or {}).get("task_type", "")
            if not task_type:
                context.pipeline_failed = True
                context.failure_reason = "intent_parser returned no task_type."
                context.training_status = "failed"
                self._store.save(context, status="failed")
                self._emit(EventType.PIPELINE_END, job_id=job_id, status="failed",
                           duration_s=round(time.monotonic() - t0, 2),
                           reason=context.failure_reason)
                return self._build_result(context, time.monotonic() - t0)

            task_stages = get_stages_for_task(task_type)

            # Drop deploy if credentials or task type don't warrant it
            if "deploy" in task_stages and not should_deploy(task_type, context.parsed_intent):
                task_stages = [s for s in task_stages if s != "deploy"]

            for stage_name in task_stages:
                if stage_name in context.stage_results:
                    logger.info("[%s] Skipping completed stage: %s", job_id, stage_name)
                    continue
                if self._run_stage(job_id, stage_name, context):
                    self._emit(EventType.PIPELINE_END, job_id=job_id, status="failed",
                               duration_s=round(time.monotonic() - t0, 2),
                               reason=context.failure_reason)
                    return self._build_result(context, time.monotonic() - t0)

        except Exception as exc:
            logger.error("[%s] Unhandled exception in pipeline: %s", job_id, exc, exc_info=True)
            context.pipeline_failed = True
            context.failure_reason = f"Unhandled error: {type(exc).__name__}: {exc}"
            context.training_status = "failed"
            self._store.save(context, status="failed")
            self._emit(EventType.PIPELINE_END, job_id=job_id, status="failed",
                       duration_s=round(time.monotonic() - t0, 2),
                       reason=context.failure_reason)
            return self._build_result(context, time.monotonic() - t0)

        duration = time.monotonic() - t0
        context.training_status = "completed"
        context.current_stage = "completed"
        self._store.save(context, status="completed")
        logger.info("[%s] Pipeline completed in %.1fs.", job_id, duration)
        self._emit(EventType.PIPELINE_END, job_id=job_id, status="completed",
                   duration_s=round(duration, 2))
        return self._build_result(context, duration)

    def _run_stage(self, job_id: str, stage_name: str, context: JobContext) -> bool:
        """
        Run one pipeline stage with retries.

        Mutates context in place (merges agent output, updates retry_counts,
        records the stage in stage_results) and checkpoints to SQLite.

        Returns:
            True  — stage failed after all retries; pipeline should stop.
            False — stage succeeded or was skipped; continue to next stage.
        """
        agent = self._agents.get(stage_name)

        if agent is None:
            logger.warning(
                "[%s] Stage '%s' has no agent yet — skipping (placeholder).",
                job_id, stage_name,
            )
            context.stage_results[stage_name] = {"status": "skipped"}
            context.current_stage = stage_name
            self._emit(EventType.STAGE_SKIPPED, stage=stage_name, job_id=job_id)
            self._store.save(context, status="running")
            return False

        logger.info("[%s] ▶ Stage: %s", job_id, stage_name)
        context.current_stage = stage_name
        self._emit(EventType.STAGE_START, stage=stage_name, job_id=job_id)
        t_stage = time.monotonic()

        result, error = RetryHandler.run(
            agent=agent,
            context=context,
            stage=stage_name,
            retry_counts=context.retry_counts,  # mutated in-place by RetryHandler
            event_queue=self.event_queue,
        )

        if error:
            logger.error("[%s] ✗ Stage '%s' failed: %s", job_id, stage_name, error["reason"])
            context.pipeline_failed = True
            context.failure_reason = error["reason"]
            context.training_status = "failed"
            self._emit(EventType.STAGE_FAILED, stage=stage_name, job_id=job_id,
                       reason=error["reason"])
            self._store.save(context, status="failed")
            return True

        # Merge agent output fields back into context
        for key, val in result.items():
            if key not in ("status", "agent") and hasattr(context, key):
                setattr(context, key, val)

        context.stage_results[stage_name] = result
        # Clear feedback for this stage — it succeeded, stale error no longer relevant
        context.last_error_feedback.pop(stage_name, None)
        duration = round(time.monotonic() - t_stage, 2)
        logger.info("[%s] ✓ Stage: %s", job_id, stage_name)
        self._emit(EventType.STAGE_END, stage=stage_name, job_id=job_id, duration_s=duration)
        self._store.save(context, status="running")
        return False

    # ── Job status & info ──────────────────────────────────────────────────────

    def get_status(self, job_id: str) -> dict:
        """Return a lightweight status dict. Safe to call while the job runs."""
        context = self._store.load(job_id)
        if context is None:
            return {"job_id": job_id, "status": "not_found"}

        db_status = self._store.get_status(job_id) or "unknown"
        parsed = context.parsed_intent or {}

        return {
            "job_id":         job_id,
            "status":         db_status,
            "current_stage":  context.current_stage,
            "task_type":      parsed.get("task_type"),
            "errors":         context.errors,
            "hf_model_url":   context.hf_model_url,
            "failure_reason": context.failure_reason,
        }

    def get_logs(self, job_id: str) -> list[str]:
        """Return the training log lines collected so far."""
        context = self._store.load(job_id)
        return list(context.training_logs) if context else []

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return summary rows for recent jobs."""
        return self._store.list_jobs(limit=limit)

    # ── Resume / cancel ────────────────────────────────────────────────────────

    def resume_job(self, job_id: str) -> dict:
        """
        Resume a job that was interrupted (process crash, timeout, etc.).

        Loads the last SQLite checkpoint, resets failure flags and retry
        counts, then calls run_pipeline. Completed stages are skipped
        automatically because they are already in context.stage_results.
        """
        context = self._store.load(job_id)
        if context is None:
            raise ValueError(f"Job {job_id!r} not found.")

        db_status = self._store.get_status(job_id)
        if db_status == "completed":
            logger.info("[%s] Already completed — nothing to resume.", job_id)
            return self._build_result(context, 0.0)

        logger.info("[%s] Resuming from stage: %s", job_id, context.current_stage)

        # Reset failure state and give each incomplete stage a fresh retry budget
        context.pipeline_failed = False
        context.failure_reason = None
        context.retry_counts = {}
        self._store.save(context, status="running")

        return self.run_pipeline(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Mark a job as cancelled. Returns True if the job existed.
        Does not interrupt a currently-running pipeline invocation —
        use a threading.Event for cooperative cancellation if needed.
        """
        context = self._store.load(job_id)
        if context is None:
            return False
        context.training_status = "cancelled"
        context.current_stage = "cancelled"
        self._store.save(context, status="cancelled")
        logger.info("[%s] Cancelled.", job_id)
        return True

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _emit(self, event_type: EventType, **data) -> None:
        self.event_queue.put(Event(event_type=event_type, data=data))

    @staticmethod
    def _build_result(context: JobContext, duration_s: float) -> dict:
        return {
            "job_id":         context.job_id,
            "status":         "completed" if not context.pipeline_failed else "failed",
            "hf_model_url":   context.hf_model_url,
            "hf_space_url":   context.hf_space_url,
            "failure_reason": context.failure_reason,
            "current_stage":  context.current_stage,
            "duration_s":     round(duration_s, 2),
            "errors":         context.errors,
        }

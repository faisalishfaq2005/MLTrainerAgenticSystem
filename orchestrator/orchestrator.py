"""
orchestrator.py
---------------
The public API for running the ML training pipeline.

This is the single entry point that the API layer (api/main.py),
the UI (ui/app.py), and tests use. Nothing outside the orchestrator/
folder should import from graph.py, retry_handler.py, etc. directly.

USAGE:

    from orchestrator.orchestrator import Orchestrator

    # 1. Create the orchestrator (once per process)
    orc = Orchestrator()

    # 2. Create a new job and run the intake conversation
    job_id = orc.new_job()

    # Interactive loop (this is what the API/UI drives):
    print(orc.get_opening_message(job_id))
    while True:
        user_msg = input("User: ")
        reply = orc.send_intake_message(job_id, user_msg)
        print("Agent:", reply["message"])
        if reply["ready"]:
            break

    # 3. Run the pipeline (blocks until complete or failed)
    result = orc.run_pipeline(job_id)
    print(result)

    # 4. Check status of a running job
    status = orc.get_status(job_id)

    # 5. Resume a crashed job
    result = orc.resume_job(job_id)
"""

import logging
import time
import uuid
from typing import Optional

from orchestrator.pipeline_state import PipelineState, empty_state, state_to_context
from storage.job_store import JobStore
from orchestrator.graph import build_graph
from agents.b import IntakeManagerAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Owns the full lifecycle of every ML training job.

    Responsibilities:
      - Create and track jobs (job_id → state)
      - Drive the interactive intake phase (run_turn loop)
      - Hand off to LangGraph once intake is complete
      - Persist state after every stage
      - Expose status, logs, and results to the API layer

    Thread safety:
      Each job_id has its own LangGraph thread_id, so concurrent jobs
      are isolated. The SQLite store uses WAL mode for concurrent reads.
    """

    def __init__(
        self,
        agents: Optional[dict] = None,
        db_path: str = "workspace/jobs.db",
        bypass_validation: bool = False,
    ):
        """
        Args:
            agents:             Dict of stage_name → agent instance.
                                Pass None to run with placeholder nodes
                                (useful during development).
            db_path:            Path to the SQLite database file.
            bypass_validation:  If True, skip live API validation during
                                intake (for testing / demo mode).
        """
        self._agents = agents or {}
        self._store  = JobStore(db_path=db_path)
        self._graph  = build_graph(self._agents)
        self._bypass = bypass_validation

        # In-memory registry: job_id → IntakeManagerAgent
        # (only exists while intake is in progress)
        self._intake_agents: dict[str, IntakeManagerAgent] = {}

        # LangGraph thread config per job
        self._thread_configs: dict[str, dict] = {}

        logger.info(
            f"Orchestrator ready | "
            f"agents={list(self._agents.keys())} | "
            f"db={db_path}"
        )

    # ── Job lifecycle ─────────────────────────────────────────────────────────

    def new_job(self) -> str:
        """
        Create a new job and return its job_id.
        Initialises the intake agent and persists an empty state.
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        state  = empty_state(job_id)

        self._intake_agents[job_id]  = IntakeManagerAgent(bypass_validation=self._bypass)
        self._thread_configs[job_id] = {"configurable": {"thread_id": job_id}}

        self._store.save(state, status="running")
        logger.info(f"New job created: {job_id}")
        return job_id

    def get_opening_message(self, job_id: str) -> str:
        """Return the intake agent's opening greeting for this job."""
        agent = self._intake_agents.get(job_id)
        if agent is None:
            raise ValueError(f"No intake agent for job {job_id!r}. Call new_job() first.")
        return agent.get_opening_message()

    # ── Interactive intake ────────────────────────────────────────────────────

    def send_intake_message(self, job_id: str, user_message: str) -> dict:
        """
        Send one user message to the intake agent and get a reply.

        Returns:
            {
              "message": str,    the agent's reply
              "ready":   bool,   True = intake complete, call run_pipeline()
              "stage":   str,    current intake stage label
            }

        When ready=True, the collected_info has been written into the
        persisted state and run_pipeline() can be called.
        """
        agent = self._intake_agents.get(job_id)
        if agent is None:
            raise ValueError(f"No intake agent for job {job_id!r}.")

        result = agent.run_turn(user_message)

        if result["ready"]:
            # Finalise: write raw_prompt, conversation_history, collected_info
            # into a fresh JobContext, then persist in the store
            state = self._store.load(job_id) or empty_state(job_id)
            context = state_to_context(state)
            agent.finalise(context)

            # Write context fields back into state and persist
            from orchestrator.pipeline_state import context_to_state_patch
            patch = context_to_state_patch(context)
            for k, v in patch.items():
                state[k] = v  # type: ignore[literal-required]

            self._store.save(state, status="running")
            logger.info(f"[{job_id}] Intake complete — ready for pipeline.")

        return result

    # ── Pipeline execution ────────────────────────────────────────────────────

    def run_pipeline(self, job_id: str) -> dict:
        """
        Run the full LangGraph pipeline for a job that has completed intake.

        This method blocks until the pipeline finishes (success or failure).
        For non-blocking execution, run it in a background thread or async task.

        Returns:
            {
              "job_id":        str,
              "status":        "completed" | "failed",
              "hf_model_url":  str | None,
              "hf_space_url":  str | None,
              "failure_reason": str | None,
              "duration_s":    float,
            }
        """
        state = self._store.load(job_id)
        if state is None:
            raise ValueError(f"Job {job_id!r} not found in store.")
        if state.get("collected_info") is None:
            raise RuntimeError(
                f"Job {job_id!r} has not completed intake. "
                "Call send_intake_message() until ready=True first."
            )

        thread_config = self._thread_configs.get(
            job_id, {"configurable": {"thread_id": job_id}}
        )

        logger.info(f"[{job_id}] Starting pipeline...")
        t0 = time.monotonic()

        try:
            final_state: PipelineState = self._graph.invoke(
                state,
                config=thread_config,
            )
        except Exception as e:
            logger.error(f"[{job_id}] Graph raised unhandled exception: {e}", exc_info=True)
            state["pipeline_failed"]  = True
            state["failure_reason"]   = f"Unhandled graph error: {e}"
            state["current_stage"]    = "failed"
            state["training_status"]  = "failed"
            self._store.save(state, status="failed")
            return self._build_result(state, time.monotonic() - t0)

        duration = time.monotonic() - t0
        succeeded = not final_state.get("pipeline_failed", False)
        db_status = "completed" if succeeded else "failed"

        self._store.save(final_state, status=db_status)
        logger.info(
            f"[{job_id}] Pipeline {db_status} in {duration:.1f}s"
        )

        return self._build_result(final_state, duration)

    # ── Job status & info ─────────────────────────────────────────────────────

    def get_status(self, job_id: str) -> dict:
        """
        Return a lightweight status dict for a job.
        Safe to call while the job is running.

        Returns:
            {
              "job_id":        str,
              "status":        "running" | "completed" | "failed",
              "current_stage": str | None,
              "task_type":     str | None,
              "errors":        list,
            }
        """
        state = self._store.load(job_id)
        if state is None:
            return {"job_id": job_id, "status": "not_found"}

        parsed = state.get("parsed_intent") or {}
        db_status = self._store.get_status(job_id) or "unknown"

        return {
            "job_id":        job_id,
            "status":        db_status,
            "current_stage": state.get("current_stage"),
            "task_type":     parsed.get("task_type"),
            "errors":        state.get("errors", []),
            "hf_model_url":  state.get("hf_model_url"),
            "failure_reason": state.get("failure_reason"),
        }

    def get_logs(self, job_id: str) -> list[str]:
        """Return the training log lines collected so far."""
        state = self._store.load(job_id)
        if state is None:
            return []
        return list(state.get("training_logs", []))

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return summary rows for recent jobs."""
        return self._store.list_jobs(limit=limit)

    # ── Resume ────────────────────────────────────────────────────────────────

    def resume_job(self, job_id: str) -> dict:
        """
        Resume a job that was interrupted (process crash, timeout, etc.).

        LangGraph's MemorySaver checkpointer is in-process only, so after
        a restart we reload the last persisted state from SQLite and re-invoke
        the graph. LangGraph will start from the last completed node because
        the thread_id still has its checkpoint in the graph's memory
        (within the same process) — or it will restart from the beginning
        of the graph if the process was restarted.

        For true cross-process resume, swap MemorySaver for a persistent
        checkpointer (e.g. langgraph-checkpoint-sqlite or -postgres).
        """
        state = self._store.load(job_id)
        if state is None:
            raise ValueError(f"Job {job_id!r} not found.")

        db_status = self._store.get_status(job_id)
        if db_status == "completed":
            logger.info(f"[{job_id}] Already completed — nothing to resume.")
            return self._build_result(state, 0.0)

        logger.info(f"[{job_id}] Resuming from stage: {state.get('current_stage')}")

        # Reset the failure flag so the graph doesn't immediately exit
        state["pipeline_failed"] = False
        state["failure_reason"]  = None

        # Register the thread config if missing (new process)
        if job_id not in self._thread_configs:
            self._thread_configs[job_id] = {"configurable": {"thread_id": job_id}}

        return self.run_pipeline(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Mark a job as cancelled.
        Does not interrupt a currently-running graph invocation
        (use a threading.Event or asyncio.Event for that).
        Returns True if the job existed and was updated.
        """
        state = self._store.load(job_id)
        if state is None:
            return False
        state["training_status"] = "cancelled"
        state["current_stage"]   = "cancelled"
        self._store.save(state, status="cancelled")
        logger.info(f"[{job_id}] Cancelled.")
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_result(state: PipelineState, duration_s: float) -> dict:
        """Build the return dict from run_pipeline()."""
        succeeded = not state.get("pipeline_failed", False)
        return {
            "job_id":         state["job_id"],
            "status":         "completed" if succeeded else "failed",
            "hf_model_url":   state.get("hf_model_url"),
            "hf_space_url":   state.get("hf_space_url"),
            "failure_reason": state.get("failure_reason"),
            "current_stage":  state.get("current_stage"),
            "duration_s":     round(duration_s, 2),
            "errors":         state.get("errors", []),
        }
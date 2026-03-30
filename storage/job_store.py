"""
job_store.py
------------
SQLite-backed persistence for job state.

Every time a stage completes or fails, the orchestrator calls
JobStore.save(state) to checkpoint the full PipelineState.
If the process crashes or the server restarts, the job can be
resumed from the last saved checkpoint via JobStore.load(job_id).

Schema:
  jobs table:
    job_id      TEXT PRIMARY KEY
    status      TEXT   ("running"|"completed"|"failed"|"cancelled")
    created_at  REAL   (unix timestamp)
    updated_at  REAL
    state_json  TEXT   (full PipelineState serialised as JSON)

The state_json field holds the entire PipelineState dict, making it
trivial to add new fields — no migrations needed during development.
"""

import json
import sqlite3
import time
import logging
import os
from typing import Optional

from orchestrator.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

# Default path — override by passing db_path to JobStore.__init__
DEFAULT_DB_PATH = "workspace/jobs.db"


class JobStore:
    """
    Thread-safe SQLite job store.

    Usage:
        store = JobStore()

        # Save after every stage
        store.save(state, status="running")

        # Load to resume
        state = store.load("job_abc123")
        if state is None:
            print("Job not found")

        # Mark complete
        store.save(state, status="completed")

        # List recent jobs
        for row in store.list_jobs():
            print(row)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    # ── Schema init ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id      TEXT PRIMARY KEY,
                    status      TEXT    NOT NULL DEFAULT 'running',
                    created_at  REAL    NOT NULL,
                    updated_at  REAL    NOT NULL,
                    state_json  TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)"
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def save(self, state: PipelineState, status: str = "running") -> None:
        """
        Upsert the full PipelineState into the database.
        Call after every stage completion or failure.

        Args:
            state:  Current PipelineState dict.
            status: "running" | "completed" | "failed" | "cancelled"
        """
        now = time.time()
        state_json = json.dumps(state, default=str)  # default=str handles floats etc.

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO jobs (job_id, status, created_at, updated_at, state_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status     = excluded.status,
                    updated_at = excluded.updated_at,
                    state_json = excluded.state_json
            """, (state["job_id"], status, now, now, state_json))

        logger.debug(f"[job_store] Saved job {state['job_id']} status={status}")

    def load(self, job_id: str) -> Optional[PipelineState]:
        """
        Load a PipelineState from the database.
        Returns None if the job_id doesn't exist.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        if row is None:
            return None

        state_dict = json.loads(row[0])
        # Ensure lists are lists (JSON can't distinguish list vs None)
        for list_field in ("conversation_history", "training_logs", "errors"):
            if state_dict.get(list_field) is None:
                state_dict[list_field] = []
        for dict_field in ("stage_results", "retry_counts"):
            if state_dict.get(dict_field) is None:
                state_dict[dict_field] = {}

        logger.debug(f"[job_store] Loaded job {job_id}")
        return PipelineState(**state_dict)

    def get_status(self, job_id: str) -> Optional[str]:
        """Return just the status string for a job_id, or None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return row[0] if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """
        Return recent jobs as lightweight dicts (no full state).
        Useful for a dashboard or status endpoint.

        Returns list of {"job_id", "status", "created_at", "updated_at",
                          "current_stage", "task_type"} dicts.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT job_id, status, created_at, updated_at, state_json
                FROM jobs
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

        result = []
        for job_id, status, created_at, updated_at, state_json in rows:
            try:
                s = json.loads(state_json)
                task_type = None
                if s.get("parsed_intent"):
                    task_type = s["parsed_intent"].get("task_type")
                result.append({
                    "job_id":        job_id,
                    "status":        status,
                    "created_at":    created_at,
                    "updated_at":    updated_at,
                    "current_stage": s.get("current_stage"),
                    "task_type":     task_type,
                })
            except Exception:
                result.append({
                    "job_id": job_id,
                    "status": status,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "current_stage": None,
                    "task_type": None,
                })

        return result

    def delete(self, job_id: str) -> bool:
        """Delete a job record. Returns True if a row was deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE job_id = ?", (job_id,)
            )
        return cursor.rowcount > 0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Open a connection with WAL mode for concurrent reads."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn
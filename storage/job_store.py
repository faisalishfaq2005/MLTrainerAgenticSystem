"""
job_store.py
------------
SQLite-backed persistence for job state.

Every time a stage completes or fails, the orchestrator calls
JobStore.save(context) to checkpoint the full JobContext.
If the process crashes, the job can be resumed from the last checkpoint
via JobStore.load(job_id).

Schema:
  jobs table:
    job_id      TEXT PRIMARY KEY
    status      TEXT   ("running"|"completed"|"failed"|"cancelled")
    created_at  REAL   (unix timestamp — taken from context.created_at)
    updated_at  REAL   (wall-clock time of the last save)
    state_json  TEXT   (full JobContext serialised as JSON via to_dict())
"""

import json
import sqlite3
import time
import logging
import os
from typing import Optional

from orchestrator.job_context import JobContext

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "workspace/jobs.db"


class JobStore:
    """
    Thread-safe SQLite job store.

    Usage:
        store = JobStore()

        store.save(context, status="running")   # after every stage

        context = store.load("job_abc123")      # resume
        if context is None:
            print("Job not found")

        store.save(context, status="completed")
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

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

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, context: JobContext, status: str = "running") -> None:
        """
        Upsert the full JobContext into the database.
        Call after every stage completion or failure.
        """
        now = time.time()
        state_json = json.dumps(context.to_dict(), default=str)

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO jobs (job_id, status, created_at, updated_at, state_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status     = excluded.status,
                    updated_at = excluded.updated_at,
                    state_json = excluded.state_json
            """, (context.job_id, status, context.created_at, now, state_json))

        logger.debug("[job_store] Saved job %s status=%s", context.job_id, status)

    def load(self, job_id: str) -> Optional[JobContext]:
        """
        Load a JobContext from the database.
        Returns None if the job_id doesn't exist.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        if row is None:
            return None

        state_dict = json.loads(row[0])

        # Ensure list/dict fields are the right type even if JSON stored null
        for list_field in ("conversation_history", "training_logs", "errors"):
            if not isinstance(state_dict.get(list_field), list):
                state_dict[list_field] = []
        for dict_field in ("stage_results", "retry_counts"):
            if not isinstance(state_dict.get(dict_field), dict):
                state_dict[dict_field] = {}

        logger.debug("[job_store] Loaded job %s", job_id)
        return JobContext.from_dict(state_dict)

    def get_status(self, job_id: str) -> Optional[str]:
        """Return just the status string for a job_id, or None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return row[0] if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """
        Return recent jobs as lightweight dicts (no full state blob).
        Useful for a dashboard or status endpoint.
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
                    "job_id":        job_id,
                    "status":        status,
                    "created_at":    created_at,
                    "updated_at":    updated_at,
                    "current_stage": None,
                    "task_type":     None,
                })

        return result

    def delete(self, job_id: str) -> bool:
        """Delete a job record. Returns True if a row was deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE job_id = ?", (job_id,)
            )
        return cursor.rowcount > 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

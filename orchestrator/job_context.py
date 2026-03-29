"""
job_context.py
--------------
The shared data envelope passed through every agent in sequence.
Think of it as the "baton" in a relay race.

Each agent receives the context, reads what it needs, and writes its own results back.
The orchestrator persists this to SQLite after every stage so jobs can be resumed.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import uuid
import time


@dataclass
class JobContext:
    # ── Identity ───────────────────────────────────────────────────
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}")
    created_at: float = field(default_factory=time.time)

    # ── Raw input (set by intake, before intent parser) ─────────────
    raw_prompt: Optional[str] = None          # the user's original description
    conversation_history: list = field(default_factory=list)  # full intake chat
    collected_info: Optional[dict] = None # All validated fields gathered during intake conversation.

    # ── Parsed intent (set by intent_parser_agent) ──────────────────
    parsed_intent: Optional[dict] = None      # the universal JSON — see IntentSchema below

    # ── Dataset info (set by dataset_agent) ─────────────────────────
    data_report: Optional[dict] = None        # rows, columns, class balance, nulls, etc.
    local_data_path: Optional[str] = None     # workspace/job_id/data/

    # ── Preprocessing result (set by preprocessing_agent) ───────────
    preprocessed_data_path: Optional[str] = None
    preprocessing_report: Optional[dict] = None

    # ── Config (set by config_agent) ─────────────────────────────────
    final_config: Optional[dict] = None       # all HPs, locked user values + inferred

    # ── Architecture (set by architecture_agent) ─────────────────────
    architecture_spec: Optional[dict] = None  # backbone, head, modifiers, model_name

    # ── Code (set by codegen_agent) ──────────────────────────────────
    train_script: Optional[str] = None        # the generated train.py
    requirements_txt: Optional[str] = None

    # ── Runtime (set by orchestrator before execution) ───────────────
    runtime: Optional[str] = None             # "kaggle" | "modal"
    runtime_job_id: Optional[str] = None      # runtime's own job reference

    # ── Monitor (set by monitor_agent) ───────────────────────────────
    training_logs: list = field(default_factory=list)
    best_metric: Optional[float] = None
    training_status: Optional[str] = None     # "running" | "completed" | "failed"

    # ── Deploy (set by deploy_agent) ─────────────────────────────────
    hf_model_url: Optional[str] = None
    hf_space_url: Optional[str] = None

    # ── Pipeline state ───────────────────────────────────────────────
    current_stage: Optional[str] = None
    stage_results: dict = field(default_factory=dict)  # stage_name → result dict
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to dict for SQLite persistence."""
        import dataclasses, json
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JobContext":
        return cls(**d)
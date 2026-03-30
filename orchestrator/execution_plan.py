"""
execution_plan.py
-----------------
Defines which pipeline stages run for each task type.

Not all tasks need all agents. For example:
  - clustering has no deploy stage (unsupervised model, nothing to inference)
  - llm_finetuning skips the tabular preprocessing stage
  - Any task that the user explicitly marks "no deploy" skips deploy

This is a pure data file — no logic, no imports from agents.
The graph reads it to decide which nodes to include in each job's path.

HOW TO ADD A NEW TASK TYPE:
  1. Add the task name to TASK_STAGES with its list of stage names.
  2. If it needs a new stage not in any existing list, add the stage
     name here AND add the corresponding node in graph.py.
  3. Nothing else changes — the orchestrator picks it up automatically.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# The master stage sequence
# All possible stages in the order they can run.
# Not every task runs every stage — TASK_STAGES selects the subset.
# ---------------------------------------------------------------------------

ALL_STAGES = [
    "intake_manager",    # conversational intake (always runs, handled specially)
    "intent_parser",     # raw prompt → parsed_intent JSON
    "dataset",           # download + validate data
    "preprocessing",     # clean, encode, split
    "config",            # resolve all hyperparameters
    "architecture",      # pick or build model architecture
    "codegen",           # write train.py + requirements.txt
    "monitor",           # run training + watch logs
    "deploy",            # push to HuggingFace
]

# ---------------------------------------------------------------------------
# Per-task stage lists
# Each list is the ordered sequence of stages for that task type.
# "intake_manager" and "intent_parser" always run first and are not listed
# here — they are hardcoded as the graph entry path.
# ---------------------------------------------------------------------------

# Stages that run after intent_parser for each task type:
TASK_STAGES: dict[str, list[str]] = {

    # ── Tabular ──────────────────────────────────────────────────────────────
    "tabular_classification": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],
    "tabular_regression": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],

    # ── Image ─────────────────────────────────────────────────────────────────
    "image_classification": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],
    "image_regression": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],
    "object_detection": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],

    # ── NLP / Text ────────────────────────────────────────────────────────────
    "text_classification": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],
    "text_generation": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],
    "token_classification": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],
    "summarization": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],
    "translation": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],

    # ── LLM fine-tuning ───────────────────────────────────────────────────────
    # Skips tabular-style preprocessing — text datasets are tokenised inside
    # the training script itself using the model's own tokenizer.
    "llm_finetuning": [
        "dataset", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],

    # ── Time series ───────────────────────────────────────────────────────────
    "time_series_forecasting": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor", "deploy",
    ],

    # ── Clustering (unsupervised) ─────────────────────────────────────────────
    # No deploy stage — a clustering model has no single inference endpoint.
    # Results are saved to the workspace folder instead.
    "clustering": [
        "dataset", "preprocessing", "config", "architecture", "codegen",
        "monitor",           # no "deploy" here
    ],
}

# Tasks that never run the deploy stage regardless of user settings
NO_DEPLOY_TASKS: set[str] = {"clustering"}


# ---------------------------------------------------------------------------
# Public helpers — called by graph.py
# ---------------------------------------------------------------------------

def get_stages_for_task(task_type: str) -> list[str]:
    """
    Return the ordered list of post-intent-parser stages for this task.
    Falls back to a sensible default if the task_type is unrecognised.
    """
    if task_type not in TASK_STAGES:
        # Fallback: run everything. Better to run too much than skip something.
        import logging
        logging.getLogger(__name__).warning(
            f"Unknown task_type '{task_type}' — using default full stage list."
        )
        return TASK_STAGES["tabular_classification"]
    return TASK_STAGES[task_type]


def should_deploy(task_type: str, parsed_intent: Optional[dict] = None) -> bool:
    """
    Return True if the deploy stage should run for this job.

    Checks two conditions:
      1. The task type isn't in NO_DEPLOY_TASKS.
      2. The user didn't set parsed_intent.deploy to something falsy.
    """
    if task_type in NO_DEPLOY_TASKS:
        return False
    if parsed_intent:
        deploy = parsed_intent.get("deploy", {}) or {}
        # If there's no HF token and no repo name, skip deploy
        if not deploy.get("hf_token") and not deploy.get("hf_repo_name"):
            return False
    return True


def get_next_stage(current_stage: str, stage_list: list[str]) -> Optional[str]:
    """
    Given the current stage name and the ordered list for this job,
    return the name of the next stage, or None if we're at the end.
    """
    try:
        idx = stage_list.index(current_stage)
        return stage_list[idx + 1] if idx + 1 < len(stage_list) else None
    except ValueError:
        return None
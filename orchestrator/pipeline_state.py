"""
pipeline_state.py
-----------------
The LangGraph state TypedDict.

LangGraph needs its state to be a plain TypedDict (not a dataclass) because
it merges partial update dicts from each node into the state automatically.

Design:
  - JobContext (dataclass) is the rich Python object agents work with.
  - PipelineState (TypedDict) is what LangGraph sees — it holds the
    serialised JobContext + all the graph-level control fields.
  - The graph nodes convert back and forth using context_to_state()
    and state_to_context().

Why keep both? JobContext gives agents a clean typed interface.
PipelineState gives LangGraph a flat dict it can checkpoint and merge.
"""

from typing import TypedDict, Optional


class PipelineState(TypedDict):
    # ── Job identity ──────────────────────────────────────────────────────────
    job_id: str

    # ── JobContext fields (serialised — agents work with the dataclass) ───────
    # These mirror every field in JobContext exactly.
    # When a node runs, it reconstructs JobContext from these, calls the agent,
    # then writes the agent's output back into these fields.

    raw_prompt: Optional[str]
    conversation_history: list
    collected_info: Optional[dict]
    parsed_intent: Optional[dict]
    data_report: Optional[dict]
    local_data_path: Optional[str]
    preprocessed_data_path: Optional[str]
    preprocessing_report: Optional[dict]
    final_config: Optional[dict]
    architecture_spec: Optional[dict]
    train_script: Optional[str]
    requirements_txt: Optional[str]
    runtime_job_id: Optional[str]
    training_logs: list
    best_metric: Optional[float]
    training_status: Optional[str]
    hf_model_url: Optional[str]
    hf_space_url: Optional[str]

    # ── Graph control fields (only the orchestrator reads/writes these) ────────
    current_stage: Optional[str]
    # Name of the node currently executing. Set before each node runs.

    stage_results: dict
    # Map of stage_name → result dict returned by agent.run().
    # e.g. {"intent_parser": {"parsed_intent": {...}}, "dataset": {...}}

    errors: list
    # List of error dicts: {"stage": str, "attempt": int, "reason": str, "ts": float}

    retry_counts: dict
    # Map of stage_name → number of attempts so far. e.g. {"codegen": 2}

    should_skip_deploy: bool
    # Set True by execution_plan when the task type doesn't need HF deployment
    # (e.g. clustering — there's no model to inference from).

    pipeline_failed: bool
    # Set True when a stage exhausts all retries. Stops the graph.

    failure_reason: Optional[str]
    # Human-readable explanation of why the pipeline failed.


def empty_state(job_id: str) -> PipelineState:
    """Return a fully-initialised empty PipelineState for a new job."""
    return PipelineState(
        job_id=job_id,
        raw_prompt=None,
        conversation_history=[],
        collected_info=None,
        parsed_intent=None,
        data_report=None,
        local_data_path=None,
        preprocessed_data_path=None,
        preprocessing_report=None,
        final_config=None,
        architecture_spec=None,
        train_script=None,
        requirements_txt=None,
        runtime_job_id=None,
        training_logs=[],
        best_metric=None,
        training_status=None,
        hf_model_url=None,
        hf_space_url=None,
        current_stage=None,
        stage_results={},
        errors=[],
        retry_counts={},
        should_skip_deploy=False,
        pipeline_failed=False,
        failure_reason=None,
    )


def state_to_context(state: PipelineState):
    """
    Reconstruct a JobContext dataclass from the flat PipelineState dict.
    Called at the start of every node before running the agent.
    """
    from orchestrator.job_context import JobContext
    return JobContext(
        job_id=state["job_id"],
        raw_prompt=state["raw_prompt"],
        conversation_history=list(state["conversation_history"]),
        collected_info=state["collected_info"],
        parsed_intent=state["parsed_intent"],
        data_report=state["data_report"],
        local_data_path=state["local_data_path"],
        preprocessed_data_path=state["preprocessed_data_path"],
        preprocessing_report=state["preprocessing_report"],
        final_config=state["final_config"],
        architecture_spec=state["architecture_spec"],
        train_script=state["train_script"],
        requirements_txt=state["requirements_txt"],
        runtime_job_id=state["runtime_job_id"],
        training_logs=list(state["training_logs"]),
        best_metric=state["best_metric"],
        training_status=state["training_status"],
        hf_model_url=state["hf_model_url"],
        hf_space_url=state["hf_space_url"],
        current_stage=state["current_stage"],
        stage_results=dict(state["stage_results"]),
        errors=list(state["errors"]),
    )


def context_to_state_patch(context) -> dict:
    """
    Extract all JobContext fields as a partial state dict.
    Called after an agent runs to merge its output back into the state.
    Only returns fields that LangGraph should update (not graph-control fields).
    """
    return {
        "raw_prompt":              context.raw_prompt,
        "conversation_history":    list(context.conversation_history),
        "collected_info":          context.collected_info,
        "parsed_intent":           context.parsed_intent,
        "data_report":             context.data_report,
        "local_data_path":         context.local_data_path,
        "preprocessed_data_path":  context.preprocessed_data_path,
        "preprocessing_report":    context.preprocessing_report,
        "final_config":            context.final_config,
        "architecture_spec":       context.architecture_spec,
        "train_script":            context.train_script,
        "requirements_txt":        context.requirements_txt,
        "runtime_job_id":          context.runtime_job_id,
        "training_logs":           list(context.training_logs),
        "best_metric":             context.best_metric,
        "training_status":         context.training_status,
        "hf_model_url":            context.hf_model_url,
        "hf_space_url":            context.hf_space_url,
        "stage_results":           dict(context.stage_results),
        "errors":                  list(context.errors),
    }
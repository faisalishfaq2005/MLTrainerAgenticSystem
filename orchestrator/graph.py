"""
graph.py
--------
The LangGraph pipeline graph.

Every ML training job flows through this graph. Each node corresponds to
one pipeline stage. LangGraph handles the state machine, checkpointing,
and conditional branching.

GRAPH STRUCTURE:
                    ┌─────────────────────────────────────────────┐
                    │               ENTRY                         │
                    └──────────────────┬──────────────────────────┘
                                       │
                              [intake_manager]
                              (interactive loop)
                                       │ ready
                              [intent_parser]
                                       │ ok
                              [route_by_task]──────────────────────┐
                                       │                           │
                    ┌──────────────────▼─────────────────┐         │
                    │          [dataset]                  │         │
                    │          [preprocessing]*           │         │
                    │          [config]                   │  skip   │
                    │          [architecture]             │  deploy │
                    │          [codegen]                  │         │
                    │          [monitor]                  │         │
                    │          [deploy]*                  │         │
                    └──────────────────┬─────────────────┘         │
                                       │                           │
                              [pipeline_success]◄──────────────────┘
                                       │
                                      END

  * preprocessing is skipped for llm_finetuning
  * deploy is skipped for clustering and tasks with no HF credentials

FAILURE PATH:
  Any node can transition to [pipeline_failed] → END
  The RetryHandler inside each node handles per-stage retries before
  the node signals failure to the graph.

HOW TO ADD A NEW STAGE:
  1. Add an entry to STAGE_CONFIGS in retry_handler.py
  2. Add the stage name to TASK_STAGES in execution_plan.py
  3. Add a node function below (copy the pattern from any existing node)
  4. Call graph.add_node() and add the edge in build_graph()
  That's it — the conditional routing picks it up automatically.
"""

import logging
import time
from typing import Callable

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.pipeline_state import (
    PipelineState, empty_state,
    state_to_context, context_to_state_patch,
)
from orchestrator.retry_handler import RetryHandler
from orchestrator.execution_plan import (
    get_stages_for_task, should_deploy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------
# Every pipeline stage (except intake_manager which is interactive) follows
# the same pattern:
#   1. Reconstruct JobContext from PipelineState
#   2. Run the agent via RetryHandler
#   3. If success: merge agent output back into state, advance stage
#   4. If failure: set pipeline_failed=True and failure_reason

def _make_stage_node(stage_name: str, get_agent: Callable) -> Callable:
    """
    Factory that produces a LangGraph node function for a standard stage.

    Args:
        stage_name: The stage key (must match STAGE_CONFIGS and TASK_STAGES).
        get_agent:  Zero-arg callable that returns the agent instance.
                    Using a callable lets us do lazy instantiation — agents
                    are built once when the graph is compiled, not imported.

    Returns:
        A node function: (state: PipelineState) -> dict
    """
    _agent = None  # cached after first call 

    def node(state: PipelineState) -> dict:
        nonlocal _agent
        if _agent is None:
            _agent = get_agent()

        logger.info(f"▶ Stage: {stage_name} (job={state['job_id']})")

        context = state_to_context(state)
        context.current_stage = stage_name

        result, error = RetryHandler.run(
            agent=_agent,
            context=context,
            stage=stage_name,
            retry_counts=dict(state["retry_counts"]),  # mutable copy
        )

        if error:
            logger.error(f"✗ Stage {stage_name} failed: {error['reason']}")
            return {
                "current_stage":  stage_name,
                "pipeline_failed": True,
                "failure_reason":  error["reason"],
                "errors":          list(state["errors"]) + [error],
                "retry_counts":    context.__dict__.get("retry_counts", state["retry_counts"]),
            }

        # Merge agent output into context, then extract updated state fields
        for key, val in result.items():
            if key not in ("status", "agent") and hasattr(context, key):
                setattr(context, key, val)

        patch = context_to_state_patch(context)
        patch["current_stage"] = stage_name
        patch["retry_counts"] = state["retry_counts"]

        logger.info(f"✓ Stage: {stage_name}")
        return patch

    node.__name__ = f"node_{stage_name}"
    return node


# ---------------------------------------------------------------------------
# Routing functions (LangGraph conditional edges)
# ---------------------------------------------------------------------------

def route_after_intake(state: PipelineState) -> str:
    """After intake: always go to intent_parser (or fail if intake failed)."""
    if state["pipeline_failed"]:
        return "pipeline_failed"
    if state["collected_info"] is None:
        return "pipeline_failed"
    return "intent_parser"


def route_after_intent_parser(state: PipelineState) -> str:
    """After intent_parser: decide the execution plan for this task type."""
    if state["pipeline_failed"]:
        return "pipeline_failed"
    parsed = state.get("parsed_intent") or {}
    task_type = parsed.get("task_type", "")
    if not task_type:
        return "pipeline_failed"

    # Record the execution plan in state so downstream routers can use it
    # (we can't mutate state here, but graph.py reads the task_type from
    # parsed_intent directly in every subsequent routing function)
    return "dataset"  # first stage after intent_parser for all task types


def route_after_dataset(state: PipelineState) -> str:
    if state["pipeline_failed"]: return "pipeline_failed"
    parsed = state.get("parsed_intent") or {}
    task_type = parsed.get("task_type", "")
    # LLM fine-tuning skips the preprocessing stage
    if task_type == "llm_finetuning":
        return "config"
    return "preprocessing"


def route_after_preprocessing(state: PipelineState) -> str:
    if state["pipeline_failed"]: return "pipeline_failed"
    return "config"


def route_after_config(state: PipelineState) -> str:
    if state["pipeline_failed"]: return "pipeline_failed"
    return "architecture"


def route_after_architecture(state: PipelineState) -> str:
    if state["pipeline_failed"]: return "pipeline_failed"
    return "codegen"


def route_after_codegen(state: PipelineState) -> str:
    if state["pipeline_failed"]: return "pipeline_failed"
    return "monitor"


def route_after_monitor(state: PipelineState) -> str:
    if state["pipeline_failed"]: return "pipeline_failed"
    parsed = state.get("parsed_intent") or {}
    task_type = parsed.get("task_type", "")
    if state.get("should_skip_deploy") or not should_deploy(task_type, parsed):
        return "pipeline_success"
    return "deploy"


def route_after_deploy(state: PipelineState) -> str:
    if state["pipeline_failed"]: return "pipeline_failed"
    return "pipeline_success"


# ---------------------------------------------------------------------------
# Terminal nodes
# ---------------------------------------------------------------------------

def node_pipeline_success(state: PipelineState) -> dict:
    """Final node for successful completion."""
    job_id = state["job_id"]
    hf_url = state.get("hf_model_url") or "(not deployed)"
    logger.info(f"🎉 Job {job_id} completed successfully. Model: {hf_url}")
    return {
        "current_stage":   "completed",
        "training_status": "completed",
    }


def node_pipeline_failed(state: PipelineState) -> dict:
    """Final node for pipeline failure."""
    job_id = state["job_id"]
    reason = state.get("failure_reason") or "Unknown error"
    logger.error(f"💥 Job {job_id} FAILED at stage '{state.get('current_stage')}': {reason}")
    return {
        "current_stage":   "failed",
        "training_status": "failed",
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(agents: dict) -> StateGraph:
    """
    Build and return the compiled LangGraph pipeline.

    Args:
        agents: dict mapping stage_name → agent instance.
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
                Only "intake_manager" is handled separately (interactive).
                Agents not in the dict are skipped (useful while building).

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    g = StateGraph(PipelineState)

    # ── Standard stage nodes ─────────────────────────────────────────────────
    # Each node is built from the factory using the agent from the registry.
    # Stages not in `agents` get a pass-through node so the graph still runs.

    stage_names = [
        "intent_parser", "dataset", "preprocessing",
        "config", "architecture", "codegen", "monitor", "deploy",
    ]

    for stage in stage_names:
        if stage in agents:
            agent_instance = agents[stage]
            g.add_node(stage, _make_stage_node(stage, lambda a=agent_instance: a))
        else:
            # Placeholder node — logs a warning and passes through unchanged.
            # Remove when the real agent is implemented.
            def _passthrough(state: PipelineState, _s=stage) -> dict:
                logger.warning(f"⚠ Stage '{_s}' has no agent yet — skipping.")
                return {"current_stage": _s}
            _passthrough.__name__ = f"node_{stage}_placeholder"
            g.add_node(stage, _passthrough)

    # ── Terminal nodes ────────────────────────────────────────────────────────
    g.add_node("pipeline_success", node_pipeline_success)
    g.add_node("pipeline_failed",  node_pipeline_failed)

    # ── Entry point ───────────────────────────────────────────────────────────
    # intake_manager is interactive so it's handled by the Orchestrator class
    # directly before the graph is invoked. The graph starts at intent_parser.
    g.set_entry_point("intent_parser")

    # ── Edges ─────────────────────────────────────────────────────────────────
    g.add_conditional_edges(
        "intent_parser",
        route_after_intent_parser,
        {"dataset": "dataset", "pipeline_failed": "pipeline_failed"},
    )
    g.add_conditional_edges(
        "dataset",
        route_after_dataset,
        {
            "preprocessing": "preprocessing",
            "config":         "config",          # llm_finetuning bypass
            "pipeline_failed": "pipeline_failed",
        },
    )
    g.add_conditional_edges(
        "preprocessing",
        route_after_preprocessing,
        {"config": "config", "pipeline_failed": "pipeline_failed"},
    )
    g.add_conditional_edges(
        "config",
        route_after_config,
        {"architecture": "architecture", "pipeline_failed": "pipeline_failed"},
    )
    g.add_conditional_edges(
        "architecture",
        route_after_architecture,
        {"codegen": "codegen", "pipeline_failed": "pipeline_failed"},
    )
    g.add_conditional_edges(
        "codegen",
        route_after_codegen,
        {"monitor": "monitor", "pipeline_failed": "pipeline_failed"},
    )
    g.add_conditional_edges(
        "monitor",
        route_after_monitor,
        {
            "deploy":            "deploy",
            "pipeline_success":  "pipeline_success",
            "pipeline_failed":   "pipeline_failed",
        },
    )
    g.add_conditional_edges(
        "deploy",
        route_after_deploy,
        {"pipeline_success": "pipeline_success", "pipeline_failed": "pipeline_failed"},
    )

    # Terminal nodes → END
    g.add_edge("pipeline_success", END)
    g.add_edge("pipeline_failed",  END)

    return g.compile(checkpointer=MemorySaver())
from typing import Optional, Any
from enum import Enum
import queue
from dataclasses import dataclass, field


class OpType(Enum):
    USER_INPUT    = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT     = "interrupt"
    UNDO          = "undo"
    COMPACT       = "compact"


class EventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_END   = "pipeline_end"
    STAGE_START    = "stage_start"
    STAGE_END      = "stage_end"
    STAGE_FAILED   = "stage_failed"
    STAGE_SKIPPED  = "stage_skipped"
    STAGE_RETRY    = "stage_retry"
    LLM_CALL       = "llm_call"
    LLM_RESPONSE   = "llm_response"
    TOOL_CALL      = "tool_call"
    TOOL_RESULT    = "tool_result"
    AGENT_LOG      = "agent_log"


@dataclass
class Event:
    event_type: EventType
    data: dict = field(default_factory=dict)


@dataclass
class Submission:
    operation: OpType
    data: dict = field(default_factory=dict)


class Event_Queue:
    """Thread-safe queue for pipeline events (agent → UI)."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def put(self, event: Event) -> None:
        self._q.put_nowait(event)

    def get(self, timeout: float = 1.0) -> Optional[Event]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        self._q.task_done()

    def empty(self) -> bool:
        return self._q.empty()


class Submission_Queue:
    """Thread-safe queue for user submissions (UI → orchestrator)."""

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()

    def put(self, submission: Submission) -> None:
        self._q.put_nowait(submission)

    def get(self, timeout: float = 1.0) -> Optional[Submission]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        self._q.task_done()

    def empty(self) -> bool:
        return self._q.empty()

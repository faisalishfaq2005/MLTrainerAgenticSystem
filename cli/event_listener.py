"""
cli/event_listener.py
---------------------
Background thread that consumes events from the Event_Queue and renders
them to the terminal using Rich — giving real-time visibility into what
every agent is doing (tool calls, LLM calls, stage progress, retries).

Layout mirrors the feel of Claude Code / GitHub Copilot:

  >>  intent_parser
  |   -> calling LLM...
  |
  >>  intake_manager
  |   *  validate_hf_token
  |       token: hf_***abc
  |   OK  validate_hf_token   username=john_doe
  |   -> calling LLM...
  |   -> response received
  |
  OK  intent_parser   1.2s

  --  dataset  (no agent - placeholder)
"""

import threading
import time
from typing import Optional

from rich.console import Console

from orchestrator.queue_classes import Event, Event_Queue, EventType


class EventListener:
    """
    Consumes events from the shared Event_Queue and prints Rich-formatted
    output to the console in a daemon background thread.

    Usage:
        listener = EventListener(orc.event_queue, console)
        listener.start()
        orc.run_pipeline(job_id)   # events stream while this blocks
        listener.stop()
    """

    _INDENT = "  "
    _BAR    = "[dim]|[/dim]"

    def __init__(self, event_queue: Event_Queue, console: Console) -> None:
        self._queue   = event_queue
        self._console = console
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="EventListener"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal stop and wait for the thread to drain remaining events."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── Consumer loop ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running.is_set():
            event = self._queue.get(timeout=0.15)
            if event is None:
                continue
            self._render(event)
            self._queue.task_done()

        # Drain any events still in the queue after stop is signalled
        while True:
            event = self._queue.get(timeout=0.05)
            if event is None:
                break
            self._render(event)
            self._queue.task_done()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self, event: Event) -> None:
        t = event.event_type
        d = event.data
        p = self._console.print
        I = self._INDENT
        B = self._BAR

        if t == EventType.PIPELINE_START:
            p(f"\n{I}[bold]Pipeline starting[/bold]  [dim]{d.get('job_id', '')}[/dim]\n")

        elif t == EventType.STAGE_START:
            p(f"{I}[bold cyan]>>[/bold cyan]  [bold]{d.get('stage', '')}[/bold]")

        elif t == EventType.STAGE_SKIPPED:
            p(f"{I}[dim]--  {d.get('stage', '')}  (no agent - placeholder)[/dim]")

        elif t == EventType.LLM_CALL:
            p(f"{I}{B}  [dim]-> calling LLM...[/dim]")

        elif t == EventType.LLM_RESPONSE:
            summary = d.get("summary", "")
            if summary:
                p(f"{I}{B}  [dim]-> response received -- {summary}[/dim]")
            else:
                p(f"{I}{B}  [dim]-> response received[/dim]")

        elif t == EventType.TOOL_CALL:
            tool = d.get("tool_name", "")
            args = d.get("args") or {}
            p(f"{I}{B}  [yellow]Calling:[/yellow] [bold yellow]{tool}[/bold yellow]")
            for k, v in args.items():
                v_str = str(v)
                if len(v_str) > 70:
                    v_str = v_str[:67] + "..."
                p(f"{I}{B}    [dim]{k}:[/dim] {v_str}")

        elif t == EventType.TOOL_RESULT:
            tool    = d.get("tool_name", "")
            valid   = d.get("is_valid")
            summary = d.get("summary", "")
            if valid is True:
                status_tag = "[green]OK[/green]"
            elif valid is False:
                status_tag = "[red]FAIL[/red]"
            else:
                status_tag = "[dim]done[/dim]"
            p(f"{I}{B}  [bold]Tool Result:[/bold] [dim]{tool}[/dim]  {status_tag}")
            if summary:
                for part in summary.split("  "):
                    if part.strip():
                        p(f"{I}{B}    [dim]{part.strip()}[/dim]")

        elif t == EventType.STAGE_RETRY:
            attempt  = d.get("attempt", "?")
            max_att  = d.get("max_attempts", "?")
            delay    = d.get("delay", 0)
            error    = d.get("error", "")
            short_e  = (error[:90] + "...") if len(error) > 90 else error
            p(
                f"{I}{B}  [yellow]~[/yellow]  "
                f"[yellow]retrying in {delay:.0f}s[/yellow]  "
                f"[dim](attempt {attempt}/{max_att})[/dim]"
            )
            if short_e:
                p(f"{I}{B}    [dim]{short_e}[/dim]")

        elif t == EventType.STAGE_END:
            stage = d.get("stage", "")
            dur   = d.get("duration_s")
            dur_s = f"  [dim]{dur:.1f}s[/dim]" if dur is not None else ""
            p(f"{I}[green]OK[/green]  [bold]{stage}[/bold]{dur_s}\n")

        elif t == EventType.STAGE_FAILED:
            stage  = d.get("stage", "")
            reason = d.get("reason", "")
            short_r = (reason[:100] + "...") if len(reason) > 100 else reason
            p(f"{I}[red]!![/red]  [bold red]{stage}[/bold red]  [dim]{short_r}[/dim]\n")

        elif t == EventType.PIPELINE_END:
            status = d.get("status", "")
            dur    = d.get("duration_s")
            if status == "completed":
                dur_s = f"  [dim]{dur:.1f}s[/dim]" if dur is not None else ""
                p(f"\n{I}[bold green]Pipeline complete[/bold green]{dur_s}\n")
            else:
                reason = d.get("reason", "")
                short_r = (reason[:100] + "...") if len(reason) > 100 else reason
                p(f"\n{I}[bold red]Pipeline failed[/bold red]  [dim]{short_r}[/dim]\n")

        elif t == EventType.AGENT_LOG:
            msg   = d.get("message", "")
            level = d.get("level", "info").lower()
            if level == "error":
                p(f"{I}{B}  [red]{msg}[/red]")
            elif level == "warning":
                p(f"{I}{B}  [yellow]{msg}[/yellow]")
            else:
                p(f"{I}{B}  [dim]{msg}[/dim]")

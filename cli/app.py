"""
cli/app.py
----------
Entry point for the ML Trainer Agent CLI.

After install:   mltrainer
Direct run:      python -m cli.app

Session persistence
-------------------
The active job_id is written to workspace/.session when a job starts and
deleted when it completes.  On the next startup the CLI reads that file,
checks the database, and offers the user two choices:

  Mid-pipeline interruption  →  resume from the last completed stage
  Mid-intake interruption    →  intake state is in-memory only so it is
                                lost; user is informed and must start fresh
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from logging_config import setup_logging
from llm.router import LLMRouter
from agents.intent_parser_agent import IntentParserAgent
from agents.base_agent import AgentError
from orchestrator.orchestrator import Orchestrator
from orchestrator.job_context import JobContext
from cli.event_listener import EventListener

console = Console()
VERSION = "0.1.0"

SESSION_FILE = Path("workspace/.session")


# ── Session helpers ────────────────────────────────────────────────────────────

def _save_session(job_id: str) -> None:
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(job_id, encoding="utf-8")


def _clear_session() -> None:
    SESSION_FILE.unlink(missing_ok=True)


def _read_session() -> str | None:
    if SESSION_FILE.exists():
        return SESSION_FILE.read_text(encoding="utf-8").strip() or None
    return None


# ── Display helpers ────────────────────────────────────────────────────────────

def _banner() -> None:
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("ML Trainer Agent  ", "bold white"),
                (f"v{VERSION}\n", "dim"),
                ("Train · Deploy · Automate", "dim cyan"),
            ),
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()


def _agent(text: str) -> None:
    console.print("  [bold green]Agent[/bold green]")
    for line in text.strip().split("\n"):
        console.print(f"  {line}")
    console.print()


def _divider() -> None:
    console.print(f"  [dim]{'─' * 60}[/dim]\n")


def _result_row(label: str, value) -> None:
    console.print(f"  [green]OK[/green]  [bold]{label:<14}[/bold] {value}")


import re as _re


# Each row: (type-name substrings, label, hint).
# Matched against type(e).__name__.lower() for every exception in the chain,
# innermost first.  "connection" catches Connection*, APIConnection*, etc.
_EXC_TYPE_MAP: list[tuple[tuple[str, ...], str, str]] = [
    (("auth", "authentication"),          # AuthError, AuthenticationError, OAuthError…
     "Authentication error", "Check your API key or token."),
    (("ratelimit", "rate_limit", "toomanyrequests"),
     "Rate limit", "Wait a moment, then try again."),
    (("timeout", "timedout"),
     "Timeout", "The request timed out. Try again."),
    (("connection", "networkerror"),      # ConnectionError, ConnectionRefusedError,
     "Network error",                      # APIConnectionError, ConnectionResetError…
     "Check your internet connection and try again."),
    (("gaierror", "herror", "oserror"),   # socket.gaierror / low-level OS errors
     "Network error", "Check your internet connection and try again."),
    (("badrequest", "bad_request", "invalidrequest"),
     "Bad request", "Check your model name and request settings."),
    (("internalservererror", "serviceunavailable", "overloaded"),
     "Provider error", "The LLM provider is having issues. Try again."),
    (("jsondecodeerror", "invaliddecoding"),
     "LLM format error", "The model returned an unexpected format. Try again."),
]


def _exc_chain(exc: BaseException) -> list[BaseException]:
    """Return [outermost, …, innermost] without cycles."""
    chain: list[BaseException] = []
    node: BaseException | None = exc
    while node is not None and node not in chain:
        chain.append(node)
        node = node.__cause__ or node.__context__
    return chain


def _classify_exception(exc: BaseException) -> tuple[str, str]:
    """
    Classify an exception by type — not by string-matching its message.
    Walks the full __cause__/__context__ chain, checking the innermost
    (most specific) exception first.  Always returns a (label, hint) pair;
    the final fallback handles every exception not covered by _EXC_TYPE_MAP.
    """
    for e in reversed(_exc_chain(exc)):
        t = type(e).__name__.lower()

        for keywords, label, hint in _EXC_TYPE_MAP:
            if any(k in t for k in keywords):
                return label, hint

        # stdlib network base-classes whose names may not contain "connection"
        # (e.g. TimeoutError, BrokenPipeError, plain ConnectionError)
        if isinstance(e, (ConnectionError, TimeoutError)):
            return "Network error", "Check your internet connection and try again."

    return "Error", "See the log file for full details."


def _clean_message(exc: BaseException) -> str:
    """
    Return a readable one-line summary by taking the innermost exception's
    message and stripping repetitive 'Module.ClassName: ' / 'XException - '
    wrappers that providers stack when re-raising.
    No fixed list of class names — the regex handles any provider.
    """
    # Use innermost exception — it has the cleanest, most specific message
    innermost = _exc_chain(exc)[-1]
    msg = str(innermost)

    # Strip patterns like "litellm.SomeError: ", "openai.APIError: ", "GroqException - "
    msg = _re.sub(r'(\w+\.)*\w+(Error|Exception|Warning|Fault)(?::\s*|\s+-\s*)', '', msg)
    msg = msg.strip()
    return (msg[:200] + "...") if len(msg) > 200 else msg or repr(innermost)


def _show_error(exc: BaseException) -> None:
    """Display a clean two-line error. Never shows a raw traceback."""
    label, hint = _classify_exception(exc)
    short = _clean_message(exc)
    console.print(f"\n  [bold red]{label}[/bold red]  [dim]{short}[/dim]")
    console.print(f"  [dim]{hint}[/dim]\n")


# ── Agent registration ─────────────────────────────────────────────────────────

def _register_pipeline_agents(orc: Orchestrator, collected_info: dict) -> None:
    """
    Build the pipeline LLM router and register all available pipeline agents.
    All agents receive the shared event_queue so they can stream events to the UI.
    Add new agents here as they are implemented.
    """
    pipeline_router = LLMRouter.from_collected_info(collected_info)
    eq = orc.event_queue
    orc._agents["intent_parser"] = IntentParserAgent(llm_router=pipeline_router, event_queue=eq)
    # orc._agents["dataset"]       = DatasetAgent(llm_router=pipeline_router, event_queue=eq)
    # orc._agents["config"]        = ConfigAgent(llm_router=pipeline_router, event_queue=eq)
    # ... add agents as they are built


# ── Resume logic ───────────────────────────────────────────────────────────────

def _check_resume(orc: Orchestrator) -> tuple[str, str] | None:
    """
    Check workspace/.session for an unfinished job.

    Returns (job_id, kind) where kind is:
      "pipeline" — intake was complete, pipeline was interrupted → resumable
      "intake"   — user quit during intake → state is lost, must restart
    Returns None if no unfinished session exists.
    """
    job_id = _read_session()
    if not job_id:
        return None

    status = orc._store.get_status(job_id)
    if status is None or status in ("completed", "cancelled"):
        _clear_session()
        return None

    context = orc._store.load(job_id)
    if context is None:
        _clear_session()
        return None

    if context.collected_info is not None:
        return (job_id, "pipeline")
    return (job_id, "intake")


def _handle_pipeline_resume(orc: Orchestrator, job_id: str) -> bool:
    """
    Show resume prompt and run the pipeline if the user chooses to resume.
    Returns True if we resumed (caller should not start a new job).
    Returns False if the user chose to start fresh.
    """
    context = orc._store.load(job_id)
    parsed  = context.parsed_intent or {}
    task    = parsed.get("task_type", "unknown")
    stage   = context.current_stage or "—"
    done    = [s for s, r in context.stage_results.items()
               if r.get("status") != "skipped"]

    console.print(
        Panel(
            "\n".join([
                f"  [bold]job_id      [/bold]  [dim]{job_id}[/dim]",
                f"  [bold]task_type   [/bold]  {task}",
                f"  [bold]last stage  [/bold]  {stage}",
                f"  [bold]completed   [/bold]  {', '.join(done) or '—'}",
            ]),
            title="[yellow]Unfinished session found[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()

    try:
        choice = console.input(
            "  [dim][R][/dim] Resume   [dim][N][/dim] New session   "
            "[bold blue]→[/bold blue]  "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]Goodbye.[/dim]\n")
        sys.exit(0)

    console.print()

    if choice != "r":
        _clear_session()
        return False

    # Resume: register agents and run from last checkpoint
    _register_pipeline_agents(orc, context.collected_info)

    _divider()
    with console.status("  [bold]Resuming pipeline…[/bold]", spinner="dots"):
        result = orc.resume_job(job_id)

    _show_pipeline_result(orc, job_id, result)
    _clear_session()
    return True


def _handle_intake_resume() -> None:
    """
    Intake state is held in IntakeManagerAgent._state which is in-memory only.
    There is nothing to restore — inform the user and fall through to a fresh start.
    """
    console.print(
        "  [yellow]⚠[/yellow]  Your last session was interrupted during intake.\n"
        "     Conversation progress cannot be recovered — starting fresh.\n"
    )
    _clear_session()


# ── Pipeline result display ────────────────────────────────────────────────────

def _show_pipeline_result(orc: Orchestrator, job_id: str, result: dict) -> None:
    if result["status"] == "failed":
        console.print(
            f"\n  [bold red]✗  Pipeline failed[/bold red]"
            f"  [dim]{result['failure_reason']}[/dim]\n"
        )
        return

    context = orc._store.load(job_id)
    console.print()
    _result_row("job_id",       f"[dim]{job_id}[/dim]")
    _result_row("finished in",  f"[dim]{result['duration_s']}s[/dim]")
    completed = [s for s, r in (context.stage_results if context else {}).items()
                 if r.get("status") != "skipped"]
    _result_row("stages run",   f"[dim]{', '.join(completed) or '—'}[/dim]")
    console.print()

    if context and context.parsed_intent:
        pi   = context.parsed_intent
        ds   = pi.get("dataset")      or {}
        arch = pi.get("architecture") or {}
        peft = pi.get("peft")         or {}

        rows = [
            ("task_type", pi.get("task_type")),
            ("expertise", pi.get("user_expertise_level")),
            ("runtime",   pi.get("runtime")),
            ("dataset",   ds.get("url")),
            ("backbone",  arch.get("backbone")),
            ("LoRA",      peft.get("use_lora")),
            ("QLoRA",     peft.get("use_qlora")),
        ]
        body = "\n".join(
            f"  [bold]{label:<10}[/bold]  {value}"
            for label, value in rows
            if value is not None
        )
        console.print(
            Panel(body, title="[bold]Parsed Intent[/bold]",
                  border_style="green", padding=(1, 2))
        )

    console.print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log_file = setup_logging(log_dir="logs", session_label="cli")

    _banner()
    console.print(f"  [dim]Logs → {log_file}[/dim]\n")

    intake_router = LLMRouter.from_collected_info({}, max_tokens=800)
    orc = Orchestrator(agents={}, db_path="workspace/jobs.db")

    listener = EventListener(orc.event_queue, console)
    listener.start()

    # ── Resume check ───────────────────────────────────────────────────────────
    resume = _check_resume(orc)

    if resume:
        job_id, kind = resume
        if kind == "pipeline":
            if _handle_pipeline_resume(orc, job_id):
                listener.stop()
                return   # done — pipeline resumed and finished
            # user chose "new session", fall through
        else:
            _handle_intake_resume()
            # fall through to fresh intake

    # ── Fresh start ────────────────────────────────────────────────────────────
    job_id = orc.new_job(llm_router=intake_router)
    _save_session(job_id)

    _agent(orc.get_opening_message(job_id))

    # ── Intake conversation ────────────────────────────────────────────────────
    ready = False
    while not ready:
        try:
            user_input = console.input("  [bold blue]You »[/bold blue]  ").strip()
        except (EOFError, KeyboardInterrupt):
            listener.stop()
            console.print("\n  [dim]Goodbye. Run [bold]mltrainer[/bold] again to resume.[/dim]\n")
            sys.exit(0)

        if not user_input:
            continue

        console.print()
        # Plain indicator so event output can stream through
        console.print("  [dim]thinking...[/dim]")

        try:
            resp = orc.send_intake_message(job_id, user_input)
            time.sleep(0.15)   # let listener flush buffered events
            _agent(resp["message"])
            ready = resp["ready"]
        except (AgentError, Exception) as exc:
            time.sleep(0.15)
            _show_error(exc)

    # ── Pipeline ───────────────────────────────────────────────────────────────
    _divider()

    collected_info = orc._store.load(job_id).collected_info
    _register_pipeline_agents(orc, collected_info)

    result = orc.run_pipeline(job_id)

    # Let the listener drain any final events before showing the summary panel
    listener.stop()

    _show_pipeline_result(orc, job_id, result)
    _clear_session()


if __name__ == "__main__":
    main()

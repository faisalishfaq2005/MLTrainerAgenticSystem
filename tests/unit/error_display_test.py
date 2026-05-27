"""
tests/unit/error_display_test.py
---------------------------------
Tests the error-display system in cli/app.py across every realistic
exception category:

  Section 1 — _classify_exception  : asserts correct (label) for each type
  Section 2 — _clean_message        : asserts noisy prefixes are stripped
  Section 3 — _show_error (visual)  : prints what the user actually sees

Run from project root:
    python -m tests.unit.error_display_test
"""

import sys
import os
import socket
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rich.console import Console

import cli.app as _app
from cli.app import _classify_exception, _clean_message, _show_error
from agents.base_agent import AgentError


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_exc(name: str, msg: str = "", base: type = Exception) -> BaseException:
    """Dynamically create an exception class with the given name."""
    return type(name, (base,), {})(msg)


def _chain(*exceptions: BaseException) -> BaseException:
    """
    Link exceptions as a __cause__ chain: exceptions[0] caused by
    exceptions[1] caused by … caused by exceptions[-1].
    Returns exceptions[0] (the outermost).
    """
    for i in range(len(exceptions) - 1):
        exceptions[i].__cause__ = exceptions[i + 1]
    return exceptions[0]


# ── Section 1 — _classify_exception ───────────────────────────────────────────

# Format: (description, exception_to_classify, expected_label)
CLASSIFY_CASES: list[tuple[str, BaseException, str]] = [

    # ── Network — stdlib types ──────────────────────────────────────────────
    ("socket.gaierror  (real-world errno 11001 — the error the user saw)",
     socket.gaierror(11001, "getaddrinfo failed"),
     "Network error"),

    ("ConnectionRefusedError  (server not running)",
     ConnectionRefusedError("Connection refused"),
     "Network error"),

    ("ConnectionResetError  (server dropped the connection)",
     ConnectionResetError("Connection reset by peer"),
     "Network error"),

    ("ConnectionAbortedError",
     ConnectionAbortedError("Software caused connection abort"),
     "Network error"),

    ("BrokenPipeError  (write to closed socket)",
     BrokenPipeError("Broken pipe"),
     "Network error"),

    ("TimeoutError  (builtin stdlib)",
     TimeoutError("Request timed out"),
     "Timeout"),

    ("socket.timeout",
     socket.timeout("timed out"),
     "Timeout"),

    # ── Network — provider / litellm custom names ───────────────────────────
    ("APIConnectionError  (litellm-style name)",
     _make_exc("APIConnectionError", "Failed to connect to api.openai.com"),
     "Network error"),

    ("NetworkError  (generic custom name)",
     _make_exc("NetworkError", "Could not reach server"),
     "Network error"),

    # ── Auth ────────────────────────────────────────────────────────────────
    ("AuthenticationError  (litellm/openai-style name)",
     _make_exc("AuthenticationError", "Incorrect API key provided: sk-***"),
     "Authentication error"),

    ("AuthError  (partial name still matches)",
     _make_exc("AuthError", "Bad credentials"),
     "Authentication error"),

    # ── Rate limit ──────────────────────────────────────────────────────────
    ("RateLimitError  (litellm-style)",
     _make_exc("RateLimitError", "429 Too Many Requests"),
     "Rate limit"),

    ("TooManyRequestsError",
     _make_exc("TooManyRequestsError", "Quota exceeded"),
     "Rate limit"),

    # ── Provider errors ─────────────────────────────────────────────────────
    ("InternalServerError  (500 from provider)",
     _make_exc("InternalServerError", "500 Internal Server Error"),
     "Provider error"),

    ("ServiceUnavailableError  (503)",
     _make_exc("ServiceUnavailableError", "503 Service Unavailable"),
     "Provider error"),

    ("OverloadedError  (Anthropic style)",
     _make_exc("OverloadedError", "Overloaded"),
     "Provider error"),

    # ── Bad request ─────────────────────────────────────────────────────────
    ("BadRequestError  (context length exceeded)",
     _make_exc("BadRequestError", "context_length_exceeded"),
     "Bad request"),

    ("InvalidRequestError  (openai legacy name)",
     _make_exc("InvalidRequestError", "Invalid model: gpt-5"),
     "Bad request"),

    # ── JSON / format ────────────────────────────────────────────────────────
    ("json.JSONDecodeError  (LLM returned non-JSON)",
     json.JSONDecodeError("Expecting value", "not json at all", 0),
     "LLM format error"),

    # ── Unknown / generic fallback ───────────────────────────────────────────
    ("ValueError  (unexpected internal error)",
     ValueError("some unexpected internal error"),
     "Error"),

    ("KeyError  (missing dict key)",
     KeyError("missing_field"),
     "Error"),

    ("RuntimeError  (general crash)",
     RuntimeError("something blew up"),
     "Error"),

    ("PermissionError  (file system — should NOT be Network)",
     PermissionError("Access denied to workspace/jobs.db"),
     "Error"),

    # ── Chained exceptions  (real-world paths through the codebase) ─────────
    ("AgentError -> socket.gaierror  [the exact error the user saw]",
     _chain(
         AgentError("intake_manager", "INTAKE_MANAGER_AGENT",
                    "Unexpected error calling groq/llama-3.3-70b-versatile: "
                    "InternalServerError: litellm.InternalServerError: "
                    "GroqException - [Errno 11001] getaddrinfo failed"),
         _make_exc("LLMRouterError", "Unexpected error: [Errno 11001] getaddrinfo failed"),
         socket.gaierror(11001, "getaddrinfo failed"),
     ),
     "Network error"),

    ("AgentError -> AuthenticationError  (bad API key)",
     _chain(
         AgentError("intent_parser", "intent_parser",
                    "litellm.AuthenticationError: Incorrect API key"),
         _make_exc("AuthenticationError", "Incorrect API key provided: sk-***"),
     ),
     "Authentication error"),

    ("AgentError -> RateLimitError",
     _chain(
         AgentError("intent_parser", "intent_parser", "429 Too Many Requests"),
         _make_exc("RateLimitError", "Too Many Requests"),
     ),
     "Rate limit"),

    ("AgentError -> TimeoutError  (slow provider)",
     _chain(
         AgentError("codegen", "codegen", "Request timed out"),
         TimeoutError("The request to api.groq.com timed out after 30s"),
     ),
     "Timeout"),

    ("AgentError -> InternalServerError  (provider 500)",
     _chain(
         AgentError("dataset", "dataset", "500 Internal Server Error"),
         _make_exc("InternalServerError", "500 Internal Server Error from groq"),
     ),
     "Provider error"),

    ("AgentError -> ValueError  (unknown inner — falls back to Error)",
     _chain(
         AgentError("config", "config", "unexpected state"),
         ValueError("dict changed size during iteration"),
     ),
     "Error"),

    ("Three-level chain: AgentError -> LLMRouterError -> ConnectionRefusedError",
     _chain(
         AgentError("deploy", "deploy", "connection failed"),
         _make_exc("LLMRouterError", "connection failed"),
         ConnectionRefusedError("Connection refused"),
     ),
     "Network error"),
]


def run_classify_tests(console: Console) -> int:
    console.print("\n[bold]1. _classify_exception — label assertions[/bold]\n")
    failures = 0
    for desc, exc, expected_label in CLASSIFY_CASES:
        label, _hint = _classify_exception(exc)
        ok = label == expected_label
        if not ok:
            failures += 1
        mark = "[green]PASS[/green]" if ok else f"[red]FAIL[/red]  expected=[yellow]{expected_label}[/yellow]  got=[red]{label}[/red]"
        console.print(f"  {mark}  [dim]{desc}[/dim]")
    return failures


# ── Section 2 — _clean_message ────────────────────────────────────────────────

# Format: (description, exception, substring_that_must_appear_in_output)
CLEAN_CASES: list[tuple[str, BaseException, str]] = [
    ("litellm.X: openai.Y: actual message — both prefixes stripped",
     Exception("litellm.InternalServerError: openai.APIError: actual message"),
     "actual message"),

    ("GroqException - payload — dash separator also stripped",
     Exception("GroqException - [Errno 11001] getaddrinfo failed"),
     "getaddrinfo failed"),

    ("Three nested prefixes",
     Exception("litellm.InternalServerError: litellm.InternalServerError: "
               "InternalServerError: GroqException - real error here"),
     "real error here"),

    ("No prefixes — message passed through unchanged",
     Exception("Plain message with no class names"),
     "Plain message with no class names"),

    ("Very long message — truncated to 200 chars",
     Exception("X" * 300),
     "..."),

    ("Chained: innermost message used (socket.gaierror)",
     _chain(
         AgentError("a", "b", "Unexpected error: InternalServerError: [Errno 11001] getaddrinfo failed"),
         _make_exc("LLMRouterError", "[Errno 11001] getaddrinfo failed"),
         socket.gaierror(11001, "getaddrinfo failed"),
     ),
     "getaddrinfo failed"),
]


def run_clean_tests(console: Console) -> int:
    console.print("\n[bold]2. _clean_message — prefix stripping[/bold]\n")
    failures = 0
    for desc, exc, expected_part in CLEAN_CASES:
        result = _clean_message(exc)
        ok = expected_part in result
        if not ok:
            failures += 1
        mark = "[green]PASS[/green]" if ok else f"[red]FAIL[/red]  expected [yellow]{expected_part!r}[/yellow] in output"
        console.print(f"  {mark}  [dim]{desc}[/dim]")
        if not ok:
            console.print(f"         got: [dim]{result!r}[/dim]")
    return failures


# ── Section 3 — _show_error visual ────────────────────────────────────────────

VISUAL_CASES: list[tuple[str, BaseException]] = [
    ("Real-world: network (the error the user encountered)",
     _chain(
         AgentError("intake_manager", "INTAKE_MANAGER_AGENT",
                    "Unexpected error calling groq/llama-3.3-70b-versatile: "
                    "InternalServerError: litellm.InternalServerError: "
                    "InternalServerError: GroqException - [Errno 11001] getaddrinfo failed"),
         _make_exc("LLMRouterError", "Unexpected error: [Errno 11001] getaddrinfo failed"),
         socket.gaierror(11001, "getaddrinfo failed"),
     )),

    ("Authentication failure",
     _chain(
         AgentError("intent_parser", "intent_parser",
                    "litellm.AuthenticationError: Incorrect API key provided"),
         _make_exc("AuthenticationError", "Incorrect API key provided: sk-***"),
     )),

    ("Rate limit hit",
     _chain(
         AgentError("intent_parser", "intent_parser", "429 Too Many Requests"),
         _make_exc("RateLimitError", "429 Too Many Requests - rate limit exceeded"),
     )),

    ("Request timeout",
     _chain(
         AgentError("codegen", "codegen", "timed out"),
         TimeoutError("The request to api.groq.com timed out after 30s"),
     )),

    ("Provider 500 error",
     _make_exc("InternalServerError",
               "litellm.InternalServerError: InternalServerError: 500 Service Unavailable")),

    ("Context-length exceeded",
     _make_exc("BadRequestError",
               "litellm.BadRequestError: context_length_exceeded. Max tokens: 4096")),

    ("JSON decode failure (LLM returned bad format)",
     _chain(
         AgentError("intent_parser", "intent_parser", "LLM returned invalid JSON"),
         json.JSONDecodeError("Expecting value", "{bad json", 5),
     )),

    ("Unknown / completely unrecognised exception",
     PermissionError("Access denied to workspace/jobs.db")),

    ("Generic RuntimeError",
     RuntimeError("Something went wrong in the pipeline")),

    ("Long deeply-nested message with real class names",
     Exception("litellm.InternalServerError: openai.APIError: "
               "anthropic.InternalServerError: " + "A" * 200)),
]


def run_visual_tests(console: Console) -> None:
    console.print("\n[bold]3. _show_error — visual output (verify each looks clean)[/bold]\n")
    divider = "  " + "-" * 62
    console.print(divider)
    for desc, exc in VISUAL_CASES:
        console.print(f"  [bold dim]{desc}[/bold dim]")
        _show_error(exc)
        console.print(divider)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    console = Console()
    _app.console = console   # route _show_error output to this console

    console.print("\n[bold cyan]Error Display Test Suite[/bold cyan]")
    console.print("[dim]cli/app.py -> _classify_exception / _clean_message / _show_error[/dim]")

    failures = 0
    failures += run_classify_tests(console)
    failures += run_clean_tests(console)
    run_visual_tests(console)

    console.print()
    if failures == 0:
        console.print("[bold green]All assertions passed.[/bold green]\n")
    else:
        console.print(f"[bold red]{failures} assertion(s) failed — see FAIL lines above.[/bold red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

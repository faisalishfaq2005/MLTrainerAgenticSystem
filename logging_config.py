"""
logging_config.py
-----------------
Centralised logging setup for the ML Trainer Agentic System.

Call setup_logging() once at the start of any entry point (test runner,
CLI, API server). After that, every logger in the project writes to a
timestamped file under logs/ — nothing goes to the console.

Sensitive data (API keys, tokens) is masked before any record is written,
using both a regex filter on raw log text and awareness of common key prefixes.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Patterns to mask in log output
# ---------------------------------------------------------------------------
# Each tuple is (compiled pattern, masked replacement).
# Patterns keep the first 4 and last 4 characters for debugging; middle is ***.
_MASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    # HuggingFace tokens:  hf_xxxxxxx
    (re.compile(r'(hf_[A-Za-z0-9]{4})[A-Za-z0-9]{4,}([A-Za-z0-9]{4})'), r'\1***\2'),
    # Anthropic keys:      sk-ant-xxx...
    (re.compile(r'(sk-ant-[A-Za-z0-9]{4})[A-Za-z0-9\-]{4,}([A-Za-z0-9]{4})'), r'\1***\2'),
    # OpenAI keys:         sk-xxx...
    (re.compile(r'(sk-[A-Za-z0-9]{4})[A-Za-z0-9]{4,}([A-Za-z0-9]{4})'), r'\1***\2'),
    # Groq keys:           gsk_xxx...
    (re.compile(r'(gsk_[A-Za-z0-9]{4})[A-Za-z0-9]{4,}([A-Za-z0-9]{4})'), r'\1***\2'),
    # Google / Gemini keys: AIza...
    (re.compile(r'(AIza[A-Za-z0-9]{4})[A-Za-z0-9\-_]{4,}([A-Za-z0-9]{4})'), r'\1***\2'),
    # Generic quoted secrets in JSON payloads: "api_key": "...", "token": "..."
    (re.compile(r'("(?:api_key|token|key|authorization)":\s*")[^"]{8,}(")', re.IGNORECASE), r'\1***\2'),
]


def _mask_sensitive(text: str) -> str:
    for pattern, replacement in _MASK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Logging filter that scrubs sensitive data from every record
# ---------------------------------------------------------------------------

class SensitiveDataFilter(logging.Filter):
    """
    Applied to the file handler. Intercepts every LogRecord and masks
    API keys / tokens before the formatter writes them to disk.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # getMessage() expands %-style format args into the final string.
            formatted = record.getMessage()
        except Exception:
            formatted = str(record.msg)

        masked = _mask_sensitive(formatted)

        # Replace msg with the fully-expanded, masked string and clear args so
        # the formatter does not try to %-expand it a second time.
        record.msg = masked
        record.args = ()
        return True


# ---------------------------------------------------------------------------
# Pretty formatter
# ---------------------------------------------------------------------------

_FILE_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-45s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str = "logs",
    level: int = logging.DEBUG,
    session_label: Optional[str] = None,
) -> str:
    """
    Configure the root logger to write everything to a timestamped log file.

    No StreamHandler is added — the console stays clean so test runners can
    display only the agent / user conversation via print().

    Args:
        log_dir:        Directory for log files (created if absent). Relative
                        paths are resolved from the current working directory.
        level:          Minimum log level written to the file. Default DEBUG
                        (captures everything from the project).
        session_label:  Optional label embedded in the filename, e.g.
                        "intake_test". Defaults to "session".

    Returns:
        Absolute path to the log file created for this session.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    label = session_label or "session"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(log_path / f"{label}_{ts}.log")

    root = logging.getLogger()

    # Remove any previously installed handlers (e.g. from basicConfig calls
    # inside imported libraries that run at import time).
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    root.setLevel(level)

    # --- File handler: pretty, with sensitive-data masking ---
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(fmt=_FILE_FORMAT, datefmt=_DATE_FORMAT)
    )
    file_handler.addFilter(SensitiveDataFilter())
    root.addHandler(file_handler)

    # --- Silence noisy third-party libraries ---
    # We only want their warnings/errors in the log, not their debug spam.
    _quiet = (
        "litellm", "LiteLLM", "httpx", "httpcore",
        "urllib3", "requests", "openai", "anthropic",
    )
    for name in _quiet:
        logging.getLogger(name).setLevel(logging.WARNING)

    # --- Keep project loggers at full verbosity ---
    for name in ("agents", "tool", "llm", "orchestrator", "storage"):
        logging.getLogger(name).setLevel(level)

    return log_file

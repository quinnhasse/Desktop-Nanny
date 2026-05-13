"""Action handlers: alert, block, and log."""

from __future__ import annotations

import datetime
import logging
import os
import subprocess
from typing import List

logger = logging.getLogger(__name__)


def alert(label: str, confidence: float) -> None:
    """Print a terminal alert and send a desktop notification if plyer is available.

    Args:
        label: Detected object class name.
        confidence: Detection confidence score (0–1).
    """
    msg = f"[ALERT] {label} detected (conf={confidence:.2f})"
    print(msg)
    try:
        from plyer import notification  # type: ignore[import]

        notification.notify(
            title="Desktop Nanny",
            message=msg,
            timeout=3,
        )
    except Exception:  # noqa: BLE001
        pass  # plyer unavailable or display not connected — skip silently


def block(label: str, process_names: List[str] | None = None) -> None:
    """Attempt to kill processes listed in *process_names*.

    Args:
        label: Detected object class name (used for log messages).
        process_names: List of process names to terminate. If ``None`` or empty,
            a warning is logged but no processes are killed.
    """
    if not process_names:
        logger.warning("block action triggered for '%s' but no process_names configured", label)
        return

    for name in process_names:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/f", "/im", name], check=False, capture_output=True)
            else:
                subprocess.run(["pkill", "-f", name], check=False, capture_output=True)
            logger.info("blocked process '%s' (triggered by %s detection)", name, label)
        except FileNotFoundError:
            logger.warning("kill utility not found, cannot block '%s'", name)


def log_detection(label: str, confidence: float, log_path: str = "nanny.log") -> None:
    """Append a timestamped detection record to *log_path*.

    Args:
        label: Detected object class name.
        confidence: Detection confidence score (0–1).
        log_path: Path to the log file. Created on first write.
    """
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    line = f"{ts}  {label}  conf={confidence:.4f}\n"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line)


ACTION_MAP = {
    "alert": lambda label, conf, _cfg: alert(label, conf),
    "log": lambda label, conf, cfg: log_detection(
        label, conf, cfg.get("log_path", "nanny.log")
    ),
    "block": lambda label, conf, cfg: block(label, cfg.get("process_names")),
}


def dispatch(action: str, label: str, confidence: float, cfg: dict) -> None:
    """Dispatch a single named action.

    Args:
        action: One of ``"alert"``, ``"log"``, or ``"block"``.
        label: Detected object class name.
        confidence: Detection confidence score (0–1).
        cfg: Full rule config dict (may contain ``log_path``, ``process_names``, etc.).

    Raises:
        ValueError: If *action* is not a known action type.
    """
    handler = ACTION_MAP.get(action)
    if handler is None:
        raise ValueError(f"unknown action '{action}'; expected one of {list(ACTION_MAP)}")
    handler(label, confidence, cfg)

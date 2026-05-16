"""Tests for actions.py — alert, block, log, and dispatch."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from actions import alert, block, dispatch, log_detection


# ── alert ─────────────────────────────────────────────────────────────────────

def test_alert_prints(capsys):
    alert("person", 0.85)
    captured = capsys.readouterr()
    assert "person" in captured.out
    assert "0.85" in captured.out


def test_alert_skips_plyer_gracefully(capsys):
    with patch("actions.notification", side_effect=AttributeError):
        alert("cell phone", 0.7)
    captured = capsys.readouterr()
    assert "cell phone" in captured.out


# ── log_detection ─────────────────────────────────────────────────────────────

def test_log_detection_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        log_detection("laptop", 0.9, log_path=log_path)
        assert os.path.exists(log_path)


def test_log_detection_writes_label_and_confidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        log_detection("laptop", 0.9234, log_path=log_path)
        content = open(log_path).read()
        assert "laptop" in content
        assert "0.9234" in content


def test_log_detection_appends():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        log_detection("person", 0.8, log_path=log_path)
        log_detection("laptop", 0.6, log_path=log_path)
        lines = open(log_path).readlines()
        assert len(lines) == 2


# ── block ─────────────────────────────────────────────────────────────────────

def test_block_no_process_names_logs_warning(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="actions"):
        block("cell phone", process_names=None)
    assert "no process_names" in caplog.text


def test_block_calls_pkill_on_unix():
    with patch("actions.subprocess.run") as mock_run, \
         patch("actions.os.name", "posix"):
        block("cell phone", process_names=["slack"])
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pkill" in args or "taskkill" in args


# ── dispatch ──────────────────────────────────────────────────────────────────

def test_dispatch_alert():
    with patch("actions.alert") as mock_alert:
        dispatch("alert", "person", 0.9, {})
        mock_alert.assert_called_once_with("person", 0.9)


def test_dispatch_log():
    with patch("actions.log_detection") as mock_log:
        dispatch("log", "laptop", 0.7, {"log_path": "test.log"})
        mock_log.assert_called_once_with("laptop", 0.7, "test.log")


def test_dispatch_block():
    with patch("actions.block") as mock_block:
        dispatch("block", "phone", 0.8, {"process_names": ["chrome"]})
        mock_block.assert_called_once_with("phone", ["chrome"])


def test_dispatch_unknown_action_raises():
    with pytest.raises(ValueError, match="unknown action"):
        dispatch("explode", "person", 0.9, {})

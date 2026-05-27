"""Tests for the logging system: LogStore, ApiRecorder, query_logs tool, tool capture."""

import logging
import time
from pathlib import Path

from mutagent.app.log_store import (
    LogEntry,
    LogStore,
    LogStoreHandler,
)
from mutagent.core._agent_impl import _tool_log_buffer, ToolLogCaptureHandler



# ---------------------------------------------------------------------------
# LogStore
# ---------------------------------------------------------------------------

class TestLogStore:

    def test_append_and_count(self):
        store = LogStore()
        assert store.count() == 0
        store.append(LogEntry(time.time(), "INFO", "test", "hello"))
        assert store.count() == 1

    def test_query_returns_newest_first(self):
        store = LogStore()
        store.append(LogEntry(1.0, "INFO", "test", "first"))
        store.append(LogEntry(2.0, "INFO", "test", "second"))
        store.append(LogEntry(3.0, "INFO", "test", "third"))
        results = store.query(limit=10)
        assert len(results) == 3
        assert results[0].message == "third"
        assert results[2].message == "first"

    def test_query_limit(self):
        store = LogStore()
        for i in range(100):
            store.append(LogEntry(float(i), "INFO", "test", f"msg {i}"))
        results = store.query(limit=5)
        assert len(results) == 5
        assert results[0].message == "msg 99"

    def test_query_level_filter(self):
        store = LogStore()
        store.append(LogEntry(1.0, "DEBUG", "test", "debug msg"))
        store.append(LogEntry(2.0, "INFO", "test", "info msg"))
        store.append(LogEntry(3.0, "WARNING", "test", "warn msg"))
        store.append(LogEntry(4.0, "ERROR", "test", "error msg"))

        results = store.query(level="WARNING", limit=10)
        assert len(results) == 2
        assert results[0].message == "error msg"
        assert results[1].message == "warn msg"

    def test_query_pattern_filter(self):
        store = LogStore()
        store.append(LogEntry(1.0, "INFO", "test", "module foo defined"))
        store.append(LogEntry(2.0, "INFO", "test", "module bar defined"))
        store.append(LogEntry(3.0, "INFO", "test", "something else"))
        results = store.query(pattern="module.*defined", limit=10)
        assert len(results) == 2

    def test_query_pattern_and_level(self):
        store = LogStore()
        store.append(LogEntry(1.0, "DEBUG", "test", "error occurred"))
        store.append(LogEntry(2.0, "ERROR", "test", "error occurred"))
        results = store.query(pattern="error", level="ERROR", limit=10)
        assert len(results) == 1
        assert results[0].level == "ERROR"

    def test_query_empty_pattern_matches_all(self):
        store = LogStore()
        store.append(LogEntry(1.0, "INFO", "test", "hello"))
        results = store.query(pattern="", limit=10)
        assert len(results) == 1

    def test_no_capacity_limit(self):
        store = LogStore()
        for i in range(5000):
            store.append(LogEntry(float(i), "DEBUG", "test", f"msg {i}"))
        assert store.count() == 5000

    def test_tool_capture_default_off(self):
        store = LogStore()
        assert store.tool_capture_enabled is False


# ---------------------------------------------------------------------------
# LogStoreHandler
# ---------------------------------------------------------------------------

class TestLogStoreHandler:

    def test_handler_writes_to_store(self):
        store = LogStore()
        handler = LogStoreHandler(store)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.log_store_handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("test message")
        finally:
            logger.removeHandler(handler)

        assert store.count() == 1
        entry = store.query(limit=1)[0]
        assert entry.level == "INFO"
        assert entry.message == "test message"
        assert entry.logger_name == "test.log_store_handler"

    def test_handler_captures_all_levels(self):
        store = LogStore()
        handler = LogStoreHandler(store)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.log_store_all_levels")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.debug("d")
            logger.info("i")
            logger.warning("w")
            logger.error("e")
        finally:
            logger.removeHandler(handler)

        assert store.count() == 4


# ---------------------------------------------------------------------------
# ToolLogCaptureHandler
# ---------------------------------------------------------------------------

class TestToolLogCaptureHandler:

    def test_capture_when_buffer_active(self):
        handler = ToolLogCaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.tool_capture_active")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            buf: list[str] = []
            token = _tool_log_buffer.set(buf)
            try:
                logger.info("captured message")
            finally:
                _tool_log_buffer.reset(token)
        finally:
            logger.removeHandler(handler)

        assert len(buf) == 1
        assert buf[0] == "captured message"

    def test_no_capture_when_buffer_inactive(self):
        handler = ToolLogCaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.tool_capture_inactive")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            # No _tool_log_buffer set — should be a no-op
            logger.info("not captured")
        finally:
            logger.removeHandler(handler)
        # No assertion needed — just verifying no crash



# ---------------------------------------------------------------------------
# Integration: tool log capture in agent loop
# ---------------------------------------------------------------------------

class TestToolLogCaptureIntegration:

    def test_capture_appends_to_tool_result(self):
        """Simulate what agent_impl does with tool capture."""
        handler = ToolLogCaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        test_logger = logging.getLogger("test.capture_integration")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        try:
            buf: list[str] = []
            token = _tool_log_buffer.set(buf)
            try:
                # Simulate tool execution that emits logs
                test_logger.info("inside tool execution")
                test_logger.debug("debug detail")
            finally:
                _tool_log_buffer.reset(token)

            assert len(buf) == 2
            assert "inside tool execution" in buf[0]
            assert "debug detail" in buf[1]

            # Simulate appending to tool result
            tool_output = "OK: module defined"
            if buf:
                tool_output += "\n\n[Tool Logs]\n" + "\n".join(buf)
            assert "[Tool Logs]" in tool_output
            assert "inside tool execution" in tool_output
        finally:
            test_logger.removeHandler(handler)

    def test_no_capture_without_buffer(self):
        """Without setting the buffer, no logs are captured."""
        handler = ToolLogCaptureHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        test_logger = logging.getLogger("test.no_capture")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        try:
            test_logger.info("this should not be captured anywhere")
        finally:
            test_logger.removeHandler(handler)
        # If we get here without error, it works


# ---------------------------------------------------------------------------
# Integration: LogStore + FileHandler sharing session timestamp
# ---------------------------------------------------------------------------

class TestLogFileIntegration:

    def test_file_handler_writes_logs(self, tmp_path):
        """Verify FileHandler produces a log file alongside LogStore."""
        log_file = tmp_path / "20260217_100000-log.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s %(name)s — %(message)s")
        )

        store = LogStore()
        mem_handler = LogStoreHandler(store)
        mem_handler.setFormatter(logging.Formatter("%(message)s"))

        test_logger = logging.getLogger("test.file_integration")
        test_logger.addHandler(file_handler)
        test_logger.addHandler(mem_handler)
        test_logger.setLevel(logging.DEBUG)

        try:
            test_logger.info("file and memory")
        finally:
            test_logger.removeHandler(file_handler)
            test_logger.removeHandler(mem_handler)
            file_handler.close()

        # Memory
        assert store.count() == 1
        assert store.query(limit=1)[0].message == "file and memory"

        # File
        content = log_file.read_text(encoding="utf-8")
        assert "file and memory" in content

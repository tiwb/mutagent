"""JSONL logging helpers for LLM proxy traffic."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path(".mutagent/logs/proxy")


class ProxyLogger:
    """Append proxy call metadata to daily JSONL files."""

    def __init__(self, log_dir: Path | str = DEFAULT_LOG_DIR):
        self.log_dir = Path(log_dir)
        self._current_date = ""
        self._file: Any = None

    def log_call(
        self,
        *,
        client_format: str,
        backend_format: str,
        model: str,
        backend_model: str,
        provider: str,
        translated: bool,
        request_meta: dict[str, Any],
        response_meta: dict[str, Any],
        usage: dict[str, int],
        duration_ms: int,
    ) -> None:
        record = {
            "type": "proxy_call",
            "ts": datetime.now(timezone.utc).isoformat(),
            "client_format": client_format,
            "backend_format": backend_format,
            "model": model,
            "backend_model": backend_model,
            "provider": provider,
            "translated": translated,
            "request": request_meta,
            "response": response_meta,
            "usage": usage,
            "duration_ms": duration_ms,
        }
        today = datetime.now().strftime("%Y-%m-%d")
        self._ensure_file(today)
        try:
            assert self._file is not None
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception:
            logger.warning("Failed to write llmproxy log", exc_info=True)

    def _ensure_file(self, date_str: str) -> None:
        if self._current_date == date_str and self._file is not None:
            return
        self.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{date_str}.jsonl"
        self._file = path.open("a", encoding="utf-8")
        self._current_date = date_str

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None
        self._current_date = ""


def read_log_file(date_str: str, log_dir: Path | str = DEFAULT_LOG_DIR) -> list[dict[str, Any]]:
    path = Path(log_dir) / f"{date_str}.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed llmproxy log line from %s", path)
    return records


def get_summary(date_str: str, log_dir: Path | str = DEFAULT_LOG_DIR) -> dict[str, Any]:
    records = read_log_file(date_str, log_dir)
    if not records:
        return {"date": date_str, "total_calls": 0}

    total_input = sum(r.get("usage", {}).get("input_tokens", 0) for r in records)
    total_output = sum(r.get("usage", {}).get("output_tokens", 0) for r in records)
    total_duration = sum(int(r.get("duration_ms", 0)) for r in records)
    by_model: dict[str, int] = {}
    for record in records:
        model = str(record.get("model", "unknown"))
        by_model[model] = by_model.get(model, 0) + 1

    return {
        "date": date_str,
        "total_calls": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_duration_ms": total_duration,
        "avg_duration_ms": total_duration // len(records),
        "by_model": by_model,
    }

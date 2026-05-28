from __future__ import annotations

from mutagent.llmproxy.logging import ProxyLogger, get_summary, read_log_file


def test_proxy_logger_writes_and_summarizes(tmp_path):
    logger = ProxyLogger(tmp_path)
    logger.log_call(
        client_format="openai",
        backend_format="anthropic",
        model="alias-model",
        backend_model="claude-sonnet-4-20250514",
        provider="anthropic",
        translated=True,
        request_meta={"stream": False},
        response_meta={"status_code": 200, "stream": False},
        usage={"input_tokens": 10, "output_tokens": 4},
        duration_ms=123,
    )
    logger.close()

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    date_str = files[0].stem

    records = read_log_file(date_str, tmp_path)
    assert len(records) == 1
    assert records[0]["translated"] is True
    assert records[0]["backend_model"] == "claude-sonnet-4-20250514"

    summary = get_summary(date_str, tmp_path)
    assert summary["total_calls"] == 1
    assert summary["total_input_tokens"] == 10
    assert summary["total_output_tokens"] == 4
    assert summary["avg_duration_ms"] == 123

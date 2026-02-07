from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_env(name: str, value: str) -> str:
    key = name.upper()
    if any(s in key for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASS", "CREDENTIAL")):
        return "***REDACTED***"
    if len(value) > 300:
        return value[:300] + "...(truncated)"
    return value


def _safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, default=str, indent=2)


def _summarize(data: Any, max_list: int = 8, max_chars: int = 4000) -> Any:
    if data is None or isinstance(data, (bool, int, float, str)):
        if isinstance(data, str) and len(data) > 1200:
            return data[:1200] + "...(truncated)"
        return data
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            out[str(k)] = _summarize(v, max_list=max_list, max_chars=max_chars)
        s = _safe_json_dumps(out)
        if len(s) > max_chars:
            return {"_summary": s[:max_chars] + "...(truncated)"}
        return out
    if isinstance(data, (list, tuple, set)):
        seq = list(data)
        head = seq[:max_list]
        summarized = [_summarize(v, max_list=max_list, max_chars=max_chars) for v in head]
        if len(seq) > max_list:
            summarized.append({"_truncated_items": len(seq) - max_list})
        return summarized
    s = str(data)
    if len(s) > 1200:
        s = s[:1200] + "...(truncated)"
    return s


class PipelineDebugReporter:
    """
    High-performance async debug reporter for pipeline-only introspection.
    Active only when DEBUG=true. Writes one markdown file per run.
    """

    def __init__(self, run_id: str):
        self.enabled = (os.getenv("DEBUG", "false") or "").strip().lower() in {"1", "true", "yes", "on"}
        self.run_id = run_id
        self._queue: asyncio.Queue[str] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._closed = False
        self._last_step_ts: float | None = None
        self.file_path: str | None = None
        self._dropped = 0

        if not self.enabled:
            return

        out_dir = Path(os.getenv("DEBUG_LOG_DIR", "/tmp/pipeline_debug"))
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.file_path = str(out_dir / f"pipeline_report_{run_id}_{ts}.md")
        self._queue = asyncio.Queue(maxsize=2000)
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        buffer: list[str] = []
        while True:
            item = await self._queue.get()
            if item == "__PIPELINE_REPORT_CLOSE__":
                break
            buffer.append(item)
            if len(buffer) >= 12:
                await self._flush(buffer)
                buffer.clear()

        if buffer:
            await self._flush(buffer)

    async def _flush(self, chunks: list[str]) -> None:
        if not self.file_path or not chunks:
            return
        payload = "".join(chunks)
        await asyncio.to_thread(self._append_text, payload)

    def _append_text(self, content: str) -> None:
        if not self.file_path:
            return
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(content)

    async def initialize(self, claim: str) -> None:
        if not self.enabled:
            return
        env_vars = {k: _redact_env(k, v) for k, v in sorted(os.environ.items())}
        await self.log_step(
            step_name="Start time",
            description="Pipeline run initialized",
            input_data={"run_id": self.run_id},
            output_data={"start_time_utc": _utc_now_iso(), "log_file": self.file_path},
        )
        await self.log_step(
            step_name="Available environment variables",
            description="Container environment snapshot for debugging",
            input_data=None,
            output_data=env_vars,
        )
        await self.log_step(
            step_name="Received claim",
            description="Incoming claim payload",
            input_data={"claim": claim},
            output_data={"claim_length": len(claim or "")},
        )

    async def log_step(
        self,
        step_name: str,
        description: str,
        input_data: Any,
        output_data: Any,
    ) -> None:
        if not self.enabled or self._closed or self._queue is None:
            return

        now = asyncio.get_event_loop().time()
        elapsed = 0.0 if self._last_step_ts is None else max(0.0, now - self._last_step_ts)
        self._last_step_ts = now
        ts = _utc_now_iso()

        body = [
            f"## [{ts}] {step_name}\n",
            f"- Description: {description}\n",
            f"- Elapsed since previous step: {elapsed:.3f}s\n",
            "### Input\n",
            "```json\n",
            _safe_json_dumps(_summarize(input_data)),
            "\n```\n",
            "### Output\n",
            "```json\n",
            _safe_json_dumps(_summarize(output_data)),
            "\n```\n\n",
        ]
        try:
            self._queue.put_nowait("".join(body))
        except asyncio.QueueFull:
            self._dropped += 1

    async def close(self) -> None:
        if not self.enabled or self._closed or self._queue is None:
            return
        self._closed = True
        if self._dropped:
            await self.log_step(
                step_name="Report summary",
                description="Dropped entries due to backpressure",
                input_data=None,
                output_data={"dropped_entries": self._dropped},
            )
        await self._queue.put("__PIPELINE_REPORT_CLOSE__")
        if self._writer_task is not None:
            await self._writer_task

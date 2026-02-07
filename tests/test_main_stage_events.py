import pytest

from app import main as worker_main


class _FakeMessage:
    def __init__(self, value):
        self.value = value


class _FakeConsumer:
    def __init__(self, values):
        self._values = values
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._values):
            raise StopAsyncIteration
        value = self._values[self._idx]
        self._idx += 1
        return _FakeMessage(value)


class _FakeProducer:
    def __init__(self):
        self.sent = []

    async def send(self, topic, value):
        self.sent.append((topic, value))


class _FakePipeline:
    async def run(self, post_text, domain, round_id, top_k, stage_callback=None):  # noqa: ANN001
        if stage_callback:
            await stage_callback("started", {"x": 1})
            await stage_callback("retrieval_done", {"x": 2})
            await stage_callback("completed", {"status": "completed"})
        return {
            "status": "completed",
            "ranked": [
                {
                    "statement": "evidence",
                    "source_url": "https://who.int/a",
                    "final_score": 0.9,
                    "sem_score": 0.7,
                    "kg_score": 0.5,
                }
            ],
            "facts": [{"statement": "evidence"}],
            "semantic_candidates_count": 1,
            "kg_candidates_count": 1,
            "debug_counts": {"kg_in_ranked": 1},
            "kg_signal_count": 1,
            "kg_signal_sum_top5": 0.5,
            "verdict": {
                "verdict": "TRUE",
                "confidence": 0.9,
                "truthfulness_percent": 95.0,
                "rationale": "ok",
                "key_findings": [],
                "claim_breakdown": [],
                "evidence_map": [],
            },
            "llm": {"degraded_mode": False},
        }


@pytest.mark.asyncio
async def test_single_completed_event(monkeypatch):
    fake_consumer = _FakeConsumer(
        [
            {
                "job_id": "j1",
                "post": {"post_id": "p1", "room_id": "r1", "text": "claim"},
                "assigned_worker_group": "health",
            }
        ]
    )
    fake_producer = _FakeProducer()

    monkeypatch.setattr(worker_main, "_kafka_consumer", fake_consumer)
    monkeypatch.setattr(worker_main, "_kafka_producer", fake_producer)
    monkeypatch.setattr(worker_main, "CorrectivePipeline", lambda: _FakePipeline())

    await worker_main.process_jobs()

    completed_results = [
        payload
        for _, payload in fake_producer.sent
        if payload.get("status") == "completed" and payload.get("event_type") != "job.stage"
    ]
    completed_stages = [
        payload
        for _, payload in fake_producer.sent
        if payload.get("event_type") == "job.stage" and payload.get("job", {}).get("stage") == "completed"
    ]

    assert len(completed_results) == 1
    assert len(completed_stages) == 1
    assert completed_results[0]["pipeline_status"] == "completed"

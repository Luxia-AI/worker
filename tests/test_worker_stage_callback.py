from unittest.mock import AsyncMock, patch

import pytest

from app import main as worker_main


class _PipelineWithStages:
    async def run(self, post_text, domain, round_id, top_k, stage_callback=None):  # noqa: ANN001
        if stage_callback:
            await stage_callback("started", {"claim_preview": post_text[:20]})
            await stage_callback("search_done", {"queries_total": 2})
            await stage_callback("completed", {"status": "completed"})
        return {
            "status": "completed",
            "ranked": [],
            "facts": [],
            "semantic_candidates_count": 0,
            "kg_candidates_count": 0,
            "kg_signal_count": 0,
            "kg_signal_sum_top5": 0.0,
            "verdict": {
                "verdict": "UNVERIFIABLE",
                "confidence": 0.5,
                "truthfulness_percent": 50.0,
                "rationale": "test",
                "key_findings": [],
                "claim_breakdown": [],
                "evidence_map": [],
            },
            "llm": {"degraded_mode": False},
        }


@pytest.mark.asyncio
async def test_worker_verify_includes_stage_events_and_emits_callbacks() -> None:
    request = worker_main.VerifyRequest(
        job_id="job-1",
        claim="Vitamin C does not support immune health",
        room_id="health-room",
        client_id="client_demo",
        client_claim_id="claim-1",
    )

    with (
        patch.object(worker_main, "_get_pipeline", AsyncMock(return_value=_PipelineWithStages())),
        patch.object(worker_main, "_emit_stage_callback", AsyncMock()) as emit_stage_mock,
    ):
        response = await worker_main.worker_verify(request)

    assert response["status"] == "completed"  # nosec
    assert "stage_events" in response  # nosec
    stage_names = [str(item.get("stage") or "") for item in response["stage_events"]]
    assert stage_names == ["started", "search_done", "completed"]  # nosec
    assert emit_stage_mock.await_count == 3  # nosec

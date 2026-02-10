import pytest

from app.services.corrective.pipeline import CorrectivePipeline


class _FailingEntityExtractor:
    async def annotate_entities(self, _facts):  # noqa: ANN001
        raise RuntimeError("forced failure for fallback path")


@pytest.mark.asyncio
async def test_fallback_entities_remove_negation_verbs_and_synthetic_bigrams():
    pipeline = CorrectivePipeline.__new__(CorrectivePipeline)
    pipeline.entity_extractor = _FailingEntityExtractor()
    pipeline.log_manager = None

    entities = await pipeline._extract_claim_entities(
        "Vaccines do not cause autism or the flu.",
        round_id="test-round",
    )

    lowered = {e.lower() for e in entities}
    assert "vaccines" in lowered
    assert "autism" in lowered
    assert "flu" in lowered
    assert "not" not in lowered
    assert "cause" not in lowered
    assert all(" " not in e for e in lowered)


@pytest.mark.asyncio
async def test_fallback_entities_drop_reporting_and_noise_tokens():
    pipeline = CorrectivePipeline.__new__(CorrectivePipeline)
    pipeline.entity_extractor = _FailingEntityExtractor()
    pipeline.log_manager = None

    entities = await pipeline._extract_claim_entities(
        "The body cleanses itself through the liver and kidneys, and this is scientifically supported.",
        round_id="test-round",
    )
    lowered = {e.lower() for e in entities}
    assert "liver" in lowered
    assert "kidneys" in lowered
    assert "scientifically" not in lowered
    assert "supported" not in lowered
    assert "through" not in lowered
    assert "itself" not in lowered

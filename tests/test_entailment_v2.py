from app.services.verdict.v2.entailment import DeterministicEntailmentVerifier


class _FakeConfig:
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}


class _FakeInnerModel:
    config = _FakeConfig()


class _FakeCrossEncoder:
    model = _FakeInnerModel()

    def __init__(self, out):
        self._out = out

    def predict(self, _pairs):
        return self._out


def test_entailment_v2_parses_multiclass_logits_without_scalar_error():
    verifier = DeterministicEntailmentVerifier()
    verifier._model = _FakeCrossEncoder([[3.0, -1.0, -2.0]])
    verifier._enabled = True
    verifier._model_unavailable = False
    verifier._load_model = lambda: True

    probs = verifier.score_pair(
        "Vaccines are safe and effective.",
        "Vaccines are dangerous and ineffective.",
    )
    assert 0.0 <= probs["entail"] <= 1.0
    assert 0.0 <= probs["contradict"] <= 1.0
    assert 0.0 <= probs["neutral"] <= 1.0
    assert abs((probs["entail"] + probs["contradict"] + probs["neutral"]) - 1.0) < 1e-6


def test_entailment_v2_scalar_output_still_supported():
    verifier = DeterministicEntailmentVerifier()
    verifier._model = _FakeCrossEncoder([0.8])
    verifier._enabled = True
    verifier._model_unavailable = False
    verifier._load_model = lambda: True

    probs = verifier.score_pair(
        "Dietary supplements require pre-market FDA effectiveness approval.",
        "FDA does not determine dietary supplement effectiveness before marketing.",
    )
    assert abs((probs["entail"] + probs["contradict"] + probs["neutral"]) - 1.0) < 1e-6

from dataclasses import dataclass

from app.services.ranking.evidence_snapshot import make_evidence_snapshot_id


@dataclass
class _Ev:
    id: int
    domain: str
    source_url: str
    text: str


def test_snapshot_id_is_stable_and_order_sensitive():
    ev1 = _Ev(1, "a.com", "https://a.com/1", "alpha")
    ev2 = _Ev(2, "b.com", "https://b.com/2", "beta")

    snap_a = make_evidence_snapshot_id([ev1, ev2], salt="claim").snapshot_id
    snap_b = make_evidence_snapshot_id([ev1, ev2], salt="claim").snapshot_id
    snap_c = make_evidence_snapshot_id([ev2, ev1], salt="claim").snapshot_id

    assert snap_a == snap_b
    assert snap_a != snap_c

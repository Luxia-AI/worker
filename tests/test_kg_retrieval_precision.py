from app.services.kg.kg_retrieval import _canonicalize_relation_label, _relation_type, _role_alignment_score


def test_canonicalize_relation_label_maps_synonyms():
    assert _canonicalize_relation_label("is associated with") == "associated_with"
    assert _canonicalize_relation_label("leads to") == "causes"
    assert _canonicalize_relation_label("supports") == "supports"


def test_relation_type_is_assigned():
    assert _relation_type("associated_with") == "associative"
    assert _relation_type("causes") == "causal_or_effect"
    assert _relation_type("unknown_relation") == "other"


def test_role_alignment_score_prefers_subject_object_anchor_hits():
    anchors = ["vitamin c", "immune health"]
    high = _role_alignment_score("Vitamin C", "immune health", anchors)
    low = _role_alignment_score("Vitamin A", "bone density", anchors)
    assert high > low

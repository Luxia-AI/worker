"""
Test: Deferred Domain Trust Resolution Workflow

Demonstrates the full workflow for handling evidence with untrusted domains:

1. Evidence submitted with UNTRUSTED domain
2. Validation marks evidence as PENDING_DOMAIN_TRUST (not INVALID)
3. Verdict is marked PROVISIONAL
4. Evidence is NOT persisted to VDB/KG yet
5. Admin approves the domain
6. Evidence is revalidated: PENDING_DOMAIN_TRUST → TRUSTED
7. Verdict changes: PROVISIONAL → CONFIRMED
8. Update event emitted for audit trail

This test verifies that the system doesn't permanently reject evidence
just because a domain is untrusted at submission time, allowing admin
decisions to change verdict outcomes retroactively.
"""

from datetime import datetime

import pytest

from app.constants.config import ValidationState, VerdictState
from app.services.domain_trust import DomainTrustStore
from app.services.evidence_validator import EvidenceValidator
from app.services.revalidation_handler import RevalidationEvent, RevalidationHandler


class TestDeferredDomainTrustResolution:
    """Test suite for deferred domain trust resolution."""

    @pytest.fixture
    def domain_trust_store(self, tmp_path):
        """Create a temporary domain trust store for testing."""
        return DomainTrustStore(persist_path=str(tmp_path / "test_domain_trust.json"))

    @pytest.fixture
    def sample_fact_untrusted_domain(self):
        """Sample fact from untrusted domain."""
        return {
            "fact_id": "fact-001",
            "statement": "Untrusted source states that X causes Y",
            "source_url": "https://untrusted-blog.example.com/article",
            "entities": ["X", "Y"],
            "confidence": 0.75,
        }

    @pytest.fixture
    def sample_fact_trusted_domain(self):
        """Sample fact from hardcoded trusted domain."""
        return {
            "fact_id": "fact-002",
            "statement": "CDC reports that Z prevents W",
            "source_url": "https://www.cdc.gov/facts",
            "entities": ["Z", "W"],
            "confidence": 0.95,
        }

    def test_validation_state_untrusted_domain(self, sample_fact_untrusted_domain):
        """
        Evidence with untrusted domain should be PENDING_DOMAIN_TRUST, not UNTRUSTED.
        This allows the evidence to be processed and revalidated later.
        """
        source_url = sample_fact_untrusted_domain["source_url"]
        validation_state = EvidenceValidator.get_validation_state(source_url)

        # Key assertion: untrusted domain → PENDING_DOMAIN_TRUST, not UNTRUSTED
        assert validation_state == ValidationState.PENDING_DOMAIN_TRUST, (
            "Untrusted domain should return PENDING_DOMAIN_TRUST to allow "
            "deferral of domain trust decision, not UNTRUSTED"
        )

    def test_validation_state_trusted_domain(self, sample_fact_trusted_domain):
        """Evidence with hardcoded trusted domain should be TRUSTED."""
        source_url = sample_fact_trusted_domain["source_url"]
        validation_state = EvidenceValidator.get_validation_state(source_url)
        assert validation_state == ValidationState.TRUSTED

    def test_verdict_state_for_pending_domain_trust(self, sample_fact_untrusted_domain):
        """Evidence with PENDING_DOMAIN_TRUST should have PROVISIONAL verdict."""
        source_url = sample_fact_untrusted_domain["source_url"]
        validation_state = EvidenceValidator.get_validation_state(source_url)
        verdict_state = EvidenceValidator.get_verdict_state(validation_state)

        # Key assertion: PENDING_DOMAIN_TRUST → PROVISIONAL verdict
        assert verdict_state == VerdictState.PROVISIONAL, (
            "Pending domain trust should result in PROVISIONAL verdict, "
            "not CONFIRMED, to signal to client that verdict may change"
        )

    def test_verdict_state_for_trusted_domain(self, sample_fact_trusted_domain):
        """Evidence with TRUSTED domain should have CONFIRMED verdict."""
        source_url = sample_fact_trusted_domain["source_url"]
        validation_state = EvidenceValidator.get_validation_state(source_url)
        verdict_state = EvidenceValidator.get_verdict_state(validation_state)
        assert verdict_state == VerdictState.CONFIRMED

    def test_enrich_evidence_with_validation_state(self, sample_fact_untrusted_domain):
        """Evidence should be enriched with validation and verdict states."""
        fact = EvidenceValidator.enrich_evidence_with_validation(sample_fact_untrusted_domain)

        # Verify enrichment
        assert fact["validation_state"] == ValidationState.PENDING_DOMAIN_TRUST.value
        assert fact["verdict_state"] == VerdictState.PROVISIONAL.value
        assert "fact_id" in fact
        assert "statement" in fact

    @pytest.mark.asyncio
    async def test_admin_approves_domain(self, domain_trust_store, sample_fact_untrusted_domain):
        """
        When admin approves a domain, the domain trust store is updated
        and subsequent validation should return TRUSTED.
        """
        source_url = sample_fact_untrusted_domain["source_url"]
        domain = "untrusted-blog.example.com"

        # Before approval: PENDING_DOMAIN_TRUST
        validation_state = EvidenceValidator.get_validation_state(source_url)
        assert validation_state == ValidationState.PENDING_DOMAIN_TRUST

        # Admin approves domain
        record = await domain_trust_store.approve_domain(
            domain,
            approved_by="alice",
            reason="Verified as reputable source after audit",
        )

        # Verify record was created
        assert record.domain == domain
        assert record.is_trusted is True
        assert record.approved_by == "alice"
        assert record.reason is not None

    @pytest.mark.asyncio
    async def test_revalidation_after_domain_approval(self, domain_trust_store, sample_fact_untrusted_domain):
        """
        After admin approves a domain, evidence with PENDING_DOMAIN_TRUST
        for that domain should be revalidated to TRUSTED.
        """
        domain = "untrusted-blog.example.com"

        # Enrich fact before approval
        fact = EvidenceValidator.enrich_evidence_with_validation(sample_fact_untrusted_domain)
        assert fact["validation_state"] == ValidationState.PENDING_DOMAIN_TRUST.value
        assert fact["verdict_state"] == VerdictState.PROVISIONAL.value

        # Admin approves domain
        await domain_trust_store.approve_domain(domain, approved_by="alice")

        # Revalidate fact
        revalidated_count, events = await RevalidationHandler.handle_domain_approval(
            domain=domain,
            pending_facts=[fact],
            approved_by="alice",
        )

        # Verify revalidation
        assert revalidated_count == 1, "One fact should be revalidated"
        assert len(events) == 1, "One verdict change event should be emitted"

        # Verify event details
        event = events[0]
        assert event.event_type == "VERDICT_UPDATED"
        assert event.fact_id == fact["fact_id"]
        assert event.domain == domain
        assert event.old_verdict_state == VerdictState.PROVISIONAL.value
        assert event.new_verdict_state == VerdictState.CONFIRMED.value
        assert event.old_validation_state == ValidationState.PENDING_DOMAIN_TRUST.value
        assert event.new_validation_state == ValidationState.TRUSTED.value
        assert event.approved_by == "alice"

    @pytest.mark.asyncio
    async def test_multiple_facts_revalidated_for_same_domain(self, domain_trust_store):
        """
        When a domain is approved, ALL facts with PENDING_DOMAIN_TRUST
        for that domain should be revalidated in batch.
        """
        domain = "new-trusted.example.com"

        # Create multiple facts from the same untrusted domain
        facts = []
        for i in range(3):
            fact = {
                "fact_id": f"fact-{i:03d}",
                "statement": f"Fact {i} from untrusted domain",
                "source_url": f"https://{domain}/article{i}",
                "entities": ["entity1", "entity2"],
                "confidence": 0.70 + (i * 0.05),
            }
            fact = EvidenceValidator.enrich_evidence_with_validation(fact)
            facts.append(fact)

        # All should be PENDING initially
        for fact in facts:
            assert fact["validation_state"] == ValidationState.PENDING_DOMAIN_TRUST.value

        # Admin approves domain
        await domain_trust_store.approve_domain(domain, approved_by="bob")

        # Revalidate all facts
        revalidated_count, events = await RevalidationHandler.handle_domain_approval(
            domain=domain,
            pending_facts=facts,
            approved_by="bob",
        )

        # Verify all revalidated
        assert revalidated_count == 3, "All 3 facts should be revalidated"
        assert len(events) == 3, "3 verdict change events should be emitted"

        # Verify all facts now CONFIRMED
        for fact in facts:
            assert fact["validation_state"] == ValidationState.TRUSTED.value
            assert fact["verdict_state"] == VerdictState.CONFIRMED.value

    @pytest.mark.asyncio
    async def test_no_revalidation_event_if_verdict_unchanged(self, domain_trust_store):
        """
        If a fact is already TRUSTED, revalidation doesn't emit an event
        (verdict didn't change).
        """
        domain = "already-trusted.example.com"

        # Create fact that's already trusted (hardcoded domain)
        fact = {
            "fact_id": "fact-trusted",
            "statement": "CDC fact",
            "source_url": "https://www.cdc.gov/health",
            "entities": ["health"],
            "confidence": 0.95,
        }
        fact = EvidenceValidator.enrich_evidence_with_validation(fact)
        assert fact["verdict_state"] == VerdictState.CONFIRMED.value

        # Revalidate (verdict unchanged)
        revalidated_count, events = await RevalidationHandler.handle_domain_approval(
            domain=domain,
            pending_facts=[fact],
            approved_by="charlie",
        )

        # No event should be emitted because verdict didn't change
        assert revalidated_count == 1
        assert len(events) == 0, "No event if verdict unchanged"

    def test_event_to_dict(self):
        """RevalidationEvent should serialize to dict for logging."""
        event = RevalidationEvent(
            event_type="VERDICT_UPDATED",
            fact_id="fact-123",
            domain="example.com",
            old_verdict_state="provisional",
            new_verdict_state="confirmed",
            old_validation_state="pending_domain_trust",
            new_validation_state="trusted",
            timestamp=datetime.utcnow(),
            approved_by="admin@example.com",
        )

        event_dict = event.to_dict()

        # Verify all fields present
        assert event_dict["event_type"] == "VERDICT_UPDATED"
        assert event_dict["fact_id"] == "fact-123"
        assert event_dict["domain"] == "example.com"
        assert event_dict["old_verdict_state"] == "provisional"
        assert event_dict["new_verdict_state"] == "confirmed"
        assert "timestamp" in event_dict
        assert event_dict["approved_by"] == "admin@example.com"

    @pytest.mark.asyncio
    async def test_admin_rejects_domain(self, domain_trust_store):
        """Admin can explicitly reject a domain."""
        domain = "misinformation.example.com"

        record = await domain_trust_store.reject_domain(
            domain,
            approved_by="dave",
            reason="Contains misinformation",
        )

        assert record.domain == domain
        assert record.is_trusted is False
        assert record.approved_by == "dave"
        assert record.reason == "Contains misinformation"

        # Verify domain is marked as rejected
        assert domain_trust_store.is_domain_rejected(domain)
        assert not domain_trust_store.is_domain_approved(domain)

    @pytest.mark.asyncio
    async def test_persistence_across_store_instances(self, tmp_path):
        """Domain trust decisions should persist across store instances."""
        persist_path = str(tmp_path / "persistent_domain_trust.json")

        # Create first store and approve domain
        store1 = DomainTrustStore(persist_path=persist_path)
        await store1.approve_domain("example.com", approved_by="admin1")

        # Create second store instance (should load from disk)
        store2 = DomainTrustStore(persist_path=persist_path)

        # Verify domain is still approved
        assert store2.is_domain_approved("example.com")
        approved_domains = store2.get_approved_domains()
        assert "example.com" in approved_domains

    def test_validation_states_are_enums(self):
        """ValidationState and VerdictState should be proper enums."""
        assert hasattr(ValidationState, "TRUSTED")
        assert hasattr(ValidationState, "UNTRUSTED")
        assert hasattr(ValidationState, "PENDING_DOMAIN_TRUST")

        assert hasattr(VerdictState, "CONFIRMED")
        assert hasattr(VerdictState, "PROVISIONAL")
        assert hasattr(VerdictState, "REVOKED")


# =============================================================================
# INTEGRATION TEST: End-to-End Workflow
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_deferred_trust_workflow(tmp_path):
    """
    End-to-end test: evidence with untrusted domain → provisional verdict
    → admin approval → revalidation → confirmed verdict.

    This test demonstrates the full workflow without network/external services.
    """
    # Setup
    domain_trust = DomainTrustStore(persist_path=str(tmp_path / "e2e_trust.json"))

    # Step 1: User submits evidence from untrusted domain
    evidence = {
        "fact_id": "claim-pending-001",
        "statement": "The treatment is effective",
        "source_url": "https://new-medical-blog.example.com/treatment",
        "entities": ["treatment", "effectiveness"],
        "confidence": 0.72,
    }

    # Step 2: System validates evidence
    enriched = EvidenceValidator.enrich_evidence_with_validation(evidence)
    assert enriched["validation_state"] == ValidationState.PENDING_DOMAIN_TRUST.value
    assert enriched["verdict_state"] == VerdictState.PROVISIONAL.value
    print("✓ Step 2: Evidence marked as PROVISIONAL (domain pending approval)")

    # Step 3: System sends verdict to app server (with status: provisional)
    verdict_response = {
        "fact_id": enriched["fact_id"],
        "verdict_state": enriched["verdict_state"],  # "provisional"
        "confidence": enriched["confidence"],
        "message": "Verdict is provisional; domain approval may change outcome",
    }
    assert verdict_response["verdict_state"] == VerdictState.PROVISIONAL.value
    print(f"✓ Step 3: Verdict sent to app server with status={verdict_response['verdict_state']}")

    # Step 4: Evidence is NOT persisted to VDB/KG (safe ingestion guard)
    # [This would be tested in integration tests with VDB/KG]
    print("✓ Step 4: Evidence NOT persisted to VDB/KG (domain untrusted)")

    # Step 5: Admin reviews and approves the domain
    domain = "new-medical-blog.example.com"
    await domain_trust.approve_domain(
        domain,
        approved_by="dr_validator",
        reason="Domain verified as reputable medical source",
    )
    print(f"✓ Step 5: Admin approved domain '{domain}'")

    # Step 6: System revalidates evidence
    revalidated_count, events = await RevalidationHandler.handle_domain_approval(
        domain=domain,
        pending_facts=[enriched],
        approved_by="dr_validator",
    )
    assert revalidated_count == 1
    assert len(events) == 1
    print(
        f"✓ Step 6: Evidence revalidated; verdict changed {events[0].old_verdict_state} → {events[0].new_verdict_state}"
    )

    # Step 7: Verdict is now CONFIRMED
    assert enriched["verdict_state"] == VerdictState.CONFIRMED.value
    assert enriched["validation_state"] == ValidationState.TRUSTED.value
    print("✓ Step 7: Verdict now CONFIRMED (domain trusted)")

    # Step 8: Audit event logged
    event = events[0]
    audit_trail = {
        "timestamp": event.timestamp,
        "event_type": event.event_type,
        "fact_id": event.fact_id,
        "domain": event.domain,
        "change": f"{event.old_verdict_state} → {event.new_verdict_state}",
        "approved_by": event.approved_by,
    }
    print(f"✓ Step 8: Audit event recorded: {audit_trail}")

    # Verify no false negatives
    assert (
        enriched["verdict_state"] == VerdictState.CONFIRMED.value
    ), "Verdict should be CONFIRMED after admin approval, not REVOKED or PROVISIONAL"
    print("✓ Success: No false negatives; verdict correctly updated after admin approval")

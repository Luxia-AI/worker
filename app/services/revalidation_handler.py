"""
Revalidation Handler: Event-driven revalidation when admin approves domains.

When an admin approves a domain, trigger revalidation of all evidence that was
previously marked as PENDING_DOMAIN_TRUST for that domain. This allows verdicts
to be updated retroactively without silently changing them.

Workflow:
1. Admin approves domain via API
2. Emit DOMAIN_APPROVED event
3. Find all facts with PENDING_DOMAIN_TRUST for that domain
4. Revalidate those facts
5. If verdict changes from PROVISIONAL to CONFIRMED, emit UPDATE_VERDICT event
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger
from app.services.evidence_validator import EvidenceValidator

logger = get_logger(__name__)


class RevalidationEvent:
    """Event emitted when a verdict is updated due to domain approval."""

    def __init__(
        self,
        event_type: str,
        fact_id: str,
        domain: str,
        old_verdict_state: str,
        new_verdict_state: str,
        old_validation_state: str,
        new_validation_state: str,
        timestamp: datetime,
        approved_by: str,
    ):
        self.event_type = event_type
        self.fact_id = fact_id
        self.domain = domain
        self.old_verdict_state = old_verdict_state
        self.new_verdict_state = new_verdict_state
        self.old_validation_state = old_validation_state
        self.new_validation_state = new_validation_state
        self.timestamp = timestamp
        self.approved_by = approved_by

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dict for logging/persistence."""
        return {
            "event_type": self.event_type,
            "fact_id": self.fact_id,
            "domain": self.domain,
            "old_verdict_state": self.old_verdict_state,
            "new_verdict_state": self.new_verdict_state,
            "old_validation_state": self.old_validation_state,
            "new_validation_state": self.new_validation_state,
            "timestamp": self.timestamp.isoformat(),
            "approved_by": self.approved_by,
        }


class RevalidationHandler:
    """
    Handles revalidation of evidence when admin approves domains.

    Non-blocking: failures in updating facts don't block the approval process.
    All changes are logged with timestamps for auditability.
    """

    @staticmethod
    async def handle_domain_approval(
        domain: str,
        pending_facts: List[Dict[str, Any]],
        approved_by: str,
    ) -> tuple[int, List[RevalidationEvent]]:
        """
        Process domain approval and revalidate pending facts.

        Args:
            domain: Domain that was approved
            pending_facts: Facts previously marked PENDING_DOMAIN_TRUST for this domain
            approved_by: Admin username who approved

        Returns:
            (count_revalidated, list_of_events) where events track verdict changes
        """
        if not pending_facts:
            logger.info(f"[RevalidationHandler] No pending facts for domain {domain}")
            return 0, []

        logger.info(
            f"[RevalidationHandler] Processing approval for domain '{domain}': " f"{len(pending_facts)} pending facts"
        )

        events = []
        revalidated_count = 0

        for fact in pending_facts:
            try:
                source_url = fact.get("source_url", "")
                fact_id = fact.get("fact_id", "unknown")

                # Get old states (before approval)
                old_validation_state = fact.get("validation_state", "unknown")
                old_verdict_state = fact.get("verdict_state", "unknown")

                # Revalidate (should now return TRUSTED for this domain)
                new_validation_state = EvidenceValidator.get_validation_state(source_url)
                new_verdict_state = EvidenceValidator.get_verdict_state(new_validation_state)

                # Update fact
                fact["validation_state"] = new_validation_state.value
                fact["verdict_state"] = new_verdict_state.value

                # Track change if verdict state changed
                if old_verdict_state != new_verdict_state.value:
                    event = RevalidationEvent(
                        event_type="VERDICT_UPDATED",
                        fact_id=fact_id,
                        domain=domain,
                        old_verdict_state=old_verdict_state,
                        new_verdict_state=new_verdict_state.value,
                        old_validation_state=old_validation_state,
                        new_validation_state=new_validation_state.value,
                        timestamp=datetime.utcnow(),
                        approved_by=approved_by,
                    )
                    events.append(event)

                    logger.info(
                        f"[RevalidationHandler] Fact {fact_id}: "
                        f"verdict {old_verdict_state} → {new_verdict_state.value}"
                    )

                revalidated_count += 1

            except Exception as e:
                logger.warning(f"[RevalidationHandler] Failed to revalidate fact {fact.get('fact_id')}: {e}")

        logger.info(f"[RevalidationHandler] Revalidated {revalidated_count} facts, " f"{len(events)} verdicts changed")

        return revalidated_count, events

    @staticmethod
    async def emit_verdict_update_events(
        events: List[RevalidationEvent],
        log_manager: Optional[Any] = None,
    ) -> None:
        """
        Emit verdict update events to log system and external event bus.

        Non-blocking: failures in emission don't affect the main flow.
        All events are persisted with full audit trail.

        Args:
            events: List of RevalidationEvent to emit
            log_manager: Optional LogManager for structured logging
        """
        for event in events:
            try:
                # Log event for auditability
                logger.info(
                    f"[RevalidationHandler] Event: {event.event_type} "
                    f"fact_id={event.fact_id}, domain={event.domain}, "
                    f"verdict {event.old_verdict_state} → {event.new_verdict_state} "
                    f"(approved_by={event.approved_by})"
                )

                # Emit to LogManager if available
                if log_manager:
                    await log_manager.add_log(
                        level="INFO",
                        message=f"Verdict updated: {event.fact_id}",
                        module=__name__,
                        request_id=f"revalidation-{event.domain}",
                        context=event.to_dict(),
                    )

                # TODO: Emit to external event bus (Kafka, event store, etc.)
                # This allows subscribers to react to verdict changes
                # await emit_to_event_bus(event)

            except Exception as e:
                logger.warning(f"[RevalidationHandler] Failed to emit event for fact {event.fact_id}: {e}")

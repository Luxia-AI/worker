"""
Domain Trust Management: Track admin-approved domains with timestamps and revision history.

Maintains persistent storage of domain trust decisions to support deferred trust resolution:
- When admin approves a domain, it's added to the trusted set
- Previously PENDING_DOMAIN_TRUST evidence can be revalidated
- All changes are timestamped for auditability
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DomainTrustRecord:
    """Record of a domain trust decision."""

    domain: str
    is_trusted: bool
    approved_by: str  # Admin username or "system"
    approved_at: datetime
    reason: Optional[str] = None
    revision_id: str = field(default_factory=lambda: str(datetime.utcnow().timestamp()))


class DomainTrustStore:
    """
    In-memory domain trust store with JSON persistence.

    Tracks:
    - Dynamic domain trust approvals (beyond the hardcoded TRUSTED_DOMAINS config)
    - Approval timestamps
    - Revision history for auditability

    Non-blocking: failures in persistence don't block the pipeline.
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize domain trust store.

        Args:
            persist_path: Optional JSON file path for persistence
        """
        self.persist_path = Path(persist_path) if persist_path else Path("/tmp/domain_trust.json")
        self._lock = asyncio.Lock()

        # In-memory store: domain -> DomainTrustRecord
        self._approved_domains: Dict[str, DomainTrustRecord] = {}
        self._rejected_domains: Dict[str, DomainTrustRecord] = {}

        # Load from disk if exists
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load domain trust records from JSON file (non-blocking)."""
        if not self.persist_path.exists():
            logger.info(f"[DomainTrustStore] No existing trust store at {self.persist_path}")
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            # Reconstruct records
            for domain, record_dict in data.get("approved", {}).items():
                record_dict["approved_at"] = datetime.fromisoformat(record_dict["approved_at"])
                self._approved_domains[domain] = DomainTrustRecord(**record_dict)

            for domain, record_dict in data.get("rejected", {}).items():
                record_dict["approved_at"] = datetime.fromisoformat(record_dict["approved_at"])
                self._rejected_domains[domain] = DomainTrustRecord(**record_dict)

            logger.info(
                f"[DomainTrustStore] Loaded {len(self._approved_domains)} approved, "
                f"{len(self._rejected_domains)} rejected domains"
            )
        except Exception as e:
            logger.warning(f"[DomainTrustStore] Failed to load from disk: {e}")

    def _save_to_disk(self) -> None:
        """Persist domain trust records to JSON file (non-blocking)."""
        try:
            data = {
                "approved": {
                    domain: {
                        "domain": record.domain,
                        "is_trusted": record.is_trusted,
                        "approved_by": record.approved_by,
                        "approved_at": record.approved_at.isoformat(),
                        "reason": record.reason,
                        "revision_id": record.revision_id,
                    }
                    for domain, record in self._approved_domains.items()
                },
                "rejected": {
                    domain: {
                        "domain": record.domain,
                        "is_trusted": record.is_trusted,
                        "approved_by": record.approved_by,
                        "approved_at": record.approved_at.isoformat(),
                        "reason": record.reason,
                        "revision_id": record.revision_id,
                    }
                    for domain, record in self._rejected_domains.items()
                },
            }

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"[DomainTrustStore] Persisted to {self.persist_path}")
        except Exception as e:
            logger.warning(f"[DomainTrustStore] Failed to persist: {e}")

    async def approve_domain(self, domain: str, approved_by: str, reason: Optional[str] = None) -> DomainTrustRecord:
        """
        Record admin approval of a domain.

        Args:
            domain: Domain name (e.g., "example.com")
            approved_by: Admin username or "system"
            reason: Optional reason for approval

        Returns:
            DomainTrustRecord with timestamp
        """
        async with self._lock:
            record = DomainTrustRecord(
                domain=domain,
                is_trusted=True,
                approved_by=approved_by,
                approved_at=datetime.utcnow(),
                reason=reason,
            )
            self._approved_domains[domain] = record

            # Remove from rejected if previously rejected
            self._rejected_domains.pop(domain, None)

            # Persist changes (non-blocking)
            self._save_to_disk()

            logger.info(f"[DomainTrustStore] Approved domain '{domain}' by {approved_by}")
            return record

    async def reject_domain(self, domain: str, approved_by: str, reason: Optional[str] = None) -> DomainTrustRecord:
        """
        Record admin rejection of a domain.

        Args:
            domain: Domain name
            approved_by: Admin username
            reason: Optional reason for rejection

        Returns:
            DomainTrustRecord with timestamp
        """
        async with self._lock:
            record = DomainTrustRecord(
                domain=domain,
                is_trusted=False,
                approved_by=approved_by,
                approved_at=datetime.utcnow(),
                reason=reason,
            )
            self._rejected_domains[domain] = record

            # Remove from approved if previously approved
            self._approved_domains.pop(domain, None)

            # Persist changes
            self._save_to_disk()

            logger.info(f"[DomainTrustStore] Rejected domain '{domain}' by {approved_by}")
            return record

    def is_domain_approved(self, domain: str) -> bool:
        """Check if domain was dynamically approved by admin."""
        return domain in self._approved_domains

    def is_domain_rejected(self, domain: str) -> bool:
        """Check if domain was explicitly rejected by admin."""
        return domain in self._rejected_domains

    def get_approved_domains(self) -> Set[str]:
        """Get all dynamically approved domains."""
        return set(self._approved_domains.keys())

    def get_rejected_domains(self) -> Set[str]:
        """Get all rejected domains."""
        return set(self._rejected_domains.keys())

    def get_record(self, domain: str) -> Optional[DomainTrustRecord]:
        """Get trust record for a domain (if any dynamic decision exists)."""
        return self._approved_domains.get(domain) or self._rejected_domains.get(domain)


# Global singleton instance
_domain_trust_store: Optional[DomainTrustStore] = None


def get_domain_trust_store() -> DomainTrustStore:
    """Get or create the global domain trust store."""
    global _domain_trust_store
    if _domain_trust_store is None:
        _domain_trust_store = DomainTrustStore(persist_path="/tmp/luxia_domain_trust.json")
    return _domain_trust_store

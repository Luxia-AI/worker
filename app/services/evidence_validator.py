"""
Evidence Validator: Determine domain trust validation state and verdict state.

Supports deferred domain trust resolution:
- Evidence with untrusted domains are marked PENDING_DOMAIN_TRUST (not INVALID)
- Verdicts are marked PROVISIONAL when domain approval is pending
- When admin approves a domain, evidence can be revalidated
"""

from typing import Any, Dict

from app.constants.config import TRUSTED_DOMAINS, ValidationState, VerdictState
from app.core.logger import get_logger
from app.services.common.url_helpers import extract_domain
from app.services.domain_trust import get_domain_trust_store

logger = get_logger(__name__)


class EvidenceValidator:
    """
    Validate evidence domain trust and determine validation/verdict states.

    Logic:
    1. Check if domain is in hardcoded TRUSTED_DOMAINS config
    2. Check if domain was dynamically approved by admin
    3. Check if domain was explicitly rejected
    4. Default: UNTRUSTED (but still process, mark as PENDING_DOMAIN_TRUST)
    """

    @staticmethod
    def get_validation_state(source_url: str) -> ValidationState:
        """
        Determine validation state for a piece of evidence based on domain.

        Args:
            source_url: URL of the evidence source

        Returns:
            ValidationState: TRUSTED, UNTRUSTED, or PENDING_DOMAIN_TRUST

        Logic:
        - If domain in TRUSTED_DOMAINS config → TRUSTED
        - Else if domain approved by admin → TRUSTED
        - Else if domain rejected by admin → UNTRUSTED
        - Else → PENDING_DOMAIN_TRUST (allow processing, await admin decision)
        """
        domain = extract_domain(source_url)
        if not domain:
            logger.warning(f"[EvidenceValidator] Could not extract domain from {source_url}")
            return ValidationState.UNTRUSTED

        # Check hardcoded trusted domains
        if domain in TRUSTED_DOMAINS:
            return ValidationState.TRUSTED

        # Check dynamic admin approvals
        domain_trust = get_domain_trust_store()

        if domain_trust.is_domain_approved(domain):
            return ValidationState.TRUSTED

        if domain_trust.is_domain_rejected(domain):
            return ValidationState.UNTRUSTED

        # Default: domain awaiting admin decision
        # Do NOT mark as UNTRUSTED; instead return PENDING_DOMAIN_TRUST
        # so evidence can be processed and re-validated later
        return ValidationState.PENDING_DOMAIN_TRUST

    @staticmethod
    def get_verdict_state(validation_state: ValidationState) -> VerdictState:
        """
        Determine verdict state based on validation state.

        Args:
            validation_state: ValidationState from get_validation_state()

        Returns:
            VerdictState: CONFIRMED, PROVISIONAL, or REVOKED

        Logic:
        - TRUSTED validation → CONFIRMED verdict (domain was trusted at verdict time)
        - PENDING_DOMAIN_TRUST → PROVISIONAL (verdict depends on future admin decision)
        - UNTRUSTED → REVOKED or CONFIRMED (depends on historical state)
        """
        if validation_state == ValidationState.TRUSTED:
            return VerdictState.CONFIRMED
        elif validation_state == ValidationState.PENDING_DOMAIN_TRUST:
            return VerdictState.PROVISIONAL
        else:
            # UNTRUSTED: verdict is not confirmed without domain trust
            return VerdictState.REVOKED

    @staticmethod
    def enrich_evidence_with_validation(fact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a fact dict with validation_state and verdict_state.

        Args:
            fact: Fact dictionary with 'source_url' key

        Returns:
            Enhanced fact with 'validation_state' and 'verdict_state' keys
        """
        source_url = fact.get("source_url", "")
        validation_state = EvidenceValidator.get_validation_state(source_url)
        verdict_state = EvidenceValidator.get_verdict_state(validation_state)

        fact["validation_state"] = validation_state.value
        fact["verdict_state"] = verdict_state.value

        return fact

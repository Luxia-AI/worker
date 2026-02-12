"""Canonical trusted-domain configuration and helpers."""

from __future__ import annotations

from urllib.parse import urlparse

TRUSTED_ROOT_DOMAINS: set[str] = {
    "statnews.com",
    "healthaffairs.org",
    "kff.org",
    "ourworldindata.org",
    "ihme.org",
    "healthdata.org",
    "idsociety.org",
    "acc.org",
    "heart.org",
    "plos.org",
    "cochrane.org",
    "cochranelibrary.com",
    "sciencedirect.com",
    "sciencemag.org",
    "science.org",
    "nature.com",
    "bmj.com",
    "thelancet.com",
    "amanetwork.com",
    "jamanetwork.com",
    "nejm.org",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "mayoclinic.org",
    "doctorswithoutborders.org",
    "redcross.org",
    "icrc.org",
    "reliefweb.int",
    "worldbank.org",
    "unaids.org",
    "unfpa.org",
    "undp.org",
    "unicef.org",
    "health.gov.za",
    "saude.gov.br",
    "icmr.gov.in",
    "mohfw.gov.in",
    "pmda.go.jp",
    "health.gov.lk",
    "healthhub.sg",
    "moh.gov.sg",
    "health.govt.nz",
    "tga.gov.au",
    "healthdirect.gov.au",
    "health.gov.au",
    "canada.ca",
    "euclinicaltrials.eu",
    "ecdc.europa.eu",
    "ema.europa.eu",
    "gov.uk",
    "cks.nice.org.uk",
    "nice.org.uk",
    "nhs.uk",
    "uspreventiveservicestaskforce.org",
    "effectivehealthcare.ahrq.gov",
    "ahrq.gov",
    "health.gov",
    "hhs.gov",
    "cancer.gov",
    "api.fda.gov",
    "open.fda.gov",
    "fda.gov",
    "clinicaltrials.gov",
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "medlineplus.gov",
    "nlm.nih.gov",
    "nih.gov",
    "cdc.gov",
    "iarc.fr",
    "who.int",
}


def _extract_domain(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}", scheme="https")
    domain = (parsed.netloc or parsed.path or "").strip().lower()
    if "@" in domain:
        domain = domain.split("@", 1)[-1]
    if ":" in domain:
        domain = domain.split(":", 1)[0]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def is_trusted_domain(url: str) -> bool:
    """Return True when URL/domain matches canonical trusted domain list."""
    domain = _extract_domain(url)
    if not domain:
        return False
    if domain in TRUSTED_ROOT_DOMAINS:
        return True
    return any(domain.endswith(f".{trusted}") for trusted in TRUSTED_ROOT_DOMAINS)

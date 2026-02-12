from app.config.trusted_domains import TRUSTED_ROOT_DOMAINS, is_trusted_domain


def test_root_domain_match() -> None:
    assert is_trusted_domain("https://who.int/health-topics")
    assert is_trusted_domain("https://www.who.int/news")


def test_subdomain_suffix_match() -> None:
    assert "nih.gov" in TRUSTED_ROOT_DOMAINS
    assert is_trusted_domain("https://pubmed.ncbi.nlm.nih.gov/12345/")
    assert is_trusted_domain("https://pmc.ncbi.nlm.nih.gov/articles/PMC1/")
    assert is_trusted_domain("https://api.fda.gov/drug/event.json")


def test_non_listed_domain_fails() -> None:
    assert not is_trusted_domain("https://example.com/health")
    assert not is_trusted_domain("https://totallynotcdc.org")


def test_case_insensitive_and_bare_host() -> None:
    assert is_trusted_domain("HTTPS://WWW.CDC.GOV/clean-hands")
    assert is_trusted_domain("PUBMED.NCBI.NLM.NIH.GOV")

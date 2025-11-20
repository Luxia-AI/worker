"""
URL validation, parsing, and normalization utilities.
"""

import re
from typing import Optional
from urllib.parse import parse_qs, urlparse


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent matching and deduplication.

    Removes:
        - Trailing slashes
        - Query parameters
        - URL fragments
        - 'www' prefix

    Args:
        url: URL to normalize

    Returns:
        Normalized URL
    """
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        # Reconstruct without query and fragment
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        # Remove trailing slash
        normalized = normalized.rstrip("/")
        # Remove www prefix for comparison
        normalized = normalized.replace("://www.", "://")
        return normalized.lower()
    except Exception:
        return url.lower()


def is_valid_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid URL format
    """
    if not url or not isinstance(url, str):
        return False

    # Basic URL pattern
    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(url_pattern, url.strip(), re.IGNORECASE))


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.

    Args:
        url: Full URL

    Returns:
        Domain (e.g., 'example.com') or None if invalid
    """
    if not is_valid_url(url):
        return None

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def get_domain_root(url: str) -> Optional[str]:
    """
    Get root domain from URL (e.g., 'example.com' from 'subdomain.example.com').

    Args:
        url: Full URL

    Returns:
        Root domain or None if invalid
    """
    domain = extract_domain(url)
    if not domain:
        return None

    # Split and take last two parts (domain + TLD)
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def is_domain_in_list(url: str, domain_list: list) -> bool:
    """
    Check if URL's domain is in allowed domain list.

    Args:
        url: URL to check
        domain_list: List of allowed domains

    Returns:
        True if domain matches any in list
    """
    url_domain = extract_domain(url)
    if not url_domain:
        return False

    # Normalize domain_list
    normalized_list = [d.lower().replace("www.", "") for d in domain_list]

    # Check exact match
    if url_domain in normalized_list:
        return True

    # Check root domain match
    url_root = get_domain_root(url)
    for allowed in normalized_list:
        allowed_root = allowed.split(".")[-2:] if "." in allowed else allowed
        allowed_root = ".".join(allowed_root) if isinstance(allowed_root, list) else allowed_root
        if url_root and allowed_root and url_root == allowed_root:
            return True

    return False


def get_query_param(url: str, param: str) -> Optional[str]:
    """
    Extract query parameter from URL.

    Args:
        url: Full URL
        param: Parameter name

    Returns:
        Parameter value or None
    """
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        values = params.get(param, [])
        return values[0] if values else None
    except Exception:
        return None


def remove_query_params(url: str) -> str:
    """
    Remove all query parameters from URL.

    Args:
        url: URL with potential query params

    Returns:
        URL without query params
    """
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except Exception:
        return url


def is_accessible_url(url: str) -> bool:
    """
    Check if URL is likely to be accessible (not PDF, video, etc).

    Args:
        url: URL to check

    Returns:
        False if URL appears to be non-HTML content
    """
    url_lower = url.lower()
    blocked_extensions = [
        ".pdf",
        ".mp3",
        ".mp4",
        ".mov",
        ".avi",
        ".zip",
        ".exe",
        ".pptx",
        ".xls",
        ".doc",
    ]

    for ext in blocked_extensions:
        if ext in url_lower:
            return False

    return True


def dedup_urls(urls: list) -> list:
    """
    Deduplicate URLs after normalization.

    Args:
        urls: List of URLs

    Returns:
        Deduplicated URLs (in original form, not normalized)
    """
    seen = set()
    result = []

    for url in urls:
        normalized = normalize_url(url)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(url)  # Keep original URL

    return result

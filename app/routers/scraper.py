"""
Scraper API routes for WHO/IRIS/IMSEAR content extraction.

Endpoints:
  POST /scraper/scrape - Scrape a URL and return structured content
  GET /scraper/metrics - Get scraper performance metrics
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from app.core.logger import get_logger
from app.services.scraper import ScrapeResult, WHOScraper

logger = get_logger(__name__)

router = APIRouter()

# Global scraper instance
_scraper: Optional[WHOScraper] = None


def set_scraper(scraper: WHOScraper) -> None:
    """Initialize the scraper instance."""
    global _scraper
    _scraper = scraper


class ScrapeRequest(BaseModel):
    """Request model for scraping."""

    url: HttpUrl
    timeout: Optional[float] = 60.0  # seconds


@router.post("/scraper/scrape", response_model=ScrapeResult, tags=["Scraper"])
async def scrape_url(request: ScrapeRequest):
    """
    Scrape content from a WHO/IRIS/IMSEAR URL.

    Returns structured scrape result with extracted text content.
    """
    if not _scraper:
        raise HTTPException(status_code=503, detail="Scraper not initialized")

    try:
        # Use the scraper instance
        async with _scraper:
            result = await _scraper.scrape(str(request.url))

        return result

    except Exception as e:
        logger.error(f"Scrape request failed for {request.url}: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@router.get("/scraper/metrics", tags=["Scraper"])
async def get_scraper_metrics():
    """
    Get scraper performance metrics.

    Returns success rates by domain, average latency by method, and error counts.
    """
    if not _scraper:
        raise HTTPException(status_code=503, detail="Scraper not initialized")

    metrics = _scraper.metrics

    return {
        "total_attempts": metrics.total_count,
        "success_count": metrics.success_count,
        "success_rate": metrics.success_count / metrics.total_count if metrics.total_count > 0 else 0,
        "success_rate_by_domain": metrics.get_success_rate_by_domain(),
        "avg_latency_by_method": metrics.get_avg_latency_by_method(),
        "error_counts": metrics.get_error_counts(),
    }

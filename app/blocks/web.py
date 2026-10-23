"""Web Block - Real HTTP scraping via httpx + BeautifulSoup"""

import asyncio
import re
from typing import Any, Dict
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.core.universal_base import UniversalBlock
from app.core.url_guard import UnsafeURLError, validate_public_url

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_TIMEOUT = 15.0
_MAX_TEXT = 20_000


def _clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)[:_MAX_TEXT]


def _extract_links(soup: BeautifulSoup, base_url: str) -> list:
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.startswith("http") and href not in seen:
            seen.add(href)
            links.append({"text": a.get_text(strip=True)[:100], "url": href})
        if len(links) >= 50:
            break
    return links


def _meta(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": f"og:{name}"})
    return tag["content"].strip() if tag and tag.get("content") else ""


class WebBlock(UniversalBlock):
    """Web scraping and structured HTML content extraction"""

    name = "web"
    version = "2.0"
    description = "Fetch and extract content from any URL"
    layer = 3
    tags = ["domain", "web", "scraping"]
    requires = []

    ui_schema = {
        "input": {
            "type": "url",
            "accept": None,
            "placeholder": "Enter URL to scrape...",
            "multiline": False,
        },
        "output": {
            "type": "text",
            "fields": [
                {"name": "title", "type": "text", "label": "Title"},
                {"name": "text", "type": "markdown", "label": "Content"},
                {"name": "links", "type": "array", "label": "Links"},
            ],
        },
        "quick_actions": [
            {"icon": "🌐", "label": "Scrape URL", "prompt": "Extract content from URL"}
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        url = ""
        if isinstance(input_data, str):
            url = input_data
        elif isinstance(input_data, dict):
            url = (input_data.get("url") or input_data.get("text") or
                   input_data.get("input") or params.get("url", ""))
        else:
            url = params.get("url", "")

        operation = params.get("operation", "fetch")  # fetch | extract_links | extract_text | html_parse

        # SSRF guard: reject non-http(s) schemes and hosts that resolve to
        # private / loopback / link-local addresses before making any request.
        try:
            url = await asyncio.to_thread(validate_public_url, url)
        except UnsafeURLError as e:
            return {"status": "error", "error": str(e)}

        try:
            async with httpx.AsyncClient(
                headers=_HEADERS,
                timeout=_TIMEOUT,
                follow_redirects=False,
            ) as client:
                resp = await client.get(url)
                # Follow redirects manually so every hop is SSRF-validated —
                # a public URL must not bounce the request to an internal host.
                for _ in range(5):
                    if not resp.is_redirect or not resp.headers.get("location"):
                        break
                    nxt = urljoin(str(resp.url), resp.headers["location"])
                    try:
                        nxt = await asyncio.to_thread(validate_public_url, nxt)
                    except UnsafeURLError as e:
                        return {"status": "error", "error": f"Unsafe redirect: {e}"}
                    resp = await client.get(nxt)
                if resp.is_redirect:
                    return {"status": "error", "error": f"Too many redirects: {url}"}
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")

                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    return {
                        "status": "success",
                        "url": str(resp.url),
                        "content_type": content_type,
                        "text": f"[Binary content — {len(resp.content)} bytes]",
                        "title": "",
                        "links": [],
                    }

                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else _meta(soup, "title")
                description = _meta(soup, "description")

                if operation == "extract_links":
                    return {
                        "status": "success",
                        "url": str(resp.url),
                        "title": title,
                        "links": _extract_links(soup, str(resp.url)),
                    }

                text = _clean_text(soup)
                links = _extract_links(soup, str(resp.url)) if operation in ("fetch", "html_parse") else []

                return {
                    "status": "success",
                    "url": str(resp.url),
                    "title": title,
                    "description": description,
                    "text": text,
                    "links": links,
                    "word_count": len(text.split()),
                    "status_code": resp.status_code,
                }

        except httpx.TimeoutException:
            return {"status": "error", "error": f"Timeout fetching {url}"}
        except httpx.HTTPStatusError as e:
            return {"status": "error", "error": f"HTTP {e.response.status_code}: {url}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

"""Search Block - duckduckgo-search (works from cloud) + Serper fallback"""

import asyncio
import os
from typing import Any, Dict
from urllib.parse import urlparse

import httpx

from app.core.universal_base import UniversalBlock

_TIMEOUT = 20.0
_SERPER_URL = "https://google.serper.dev/search"


async def _search_duckduckgo(query: str, num: int) -> list:
    from ddgs import DDGS

    loop = asyncio.get_event_loop()

    def _sync_search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=num))

    raw = await loop.run_in_executor(None, _sync_search)

    results = []
    for item in raw:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "snippet": item.get("body", ""),
            "display_url": urlparse(item.get("href", "")).netloc,
            "source": "duckduckgo",
        })
    return results


async def _search_serper(query: str, num: int, api_key: str) -> list:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            _SERPER_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("organic", [])[:num]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "display_url": urlparse(item.get("link", "")).netloc,
            "position": item.get("position"),
            "source": "serper",
        })
    return results


class SearchBlock(UniversalBlock):
    """Real-time web search — DuckDuckGo HTML (no API key) or Serper API"""

    name = "search"
    version = "2.0"
    description = "Search the web via DuckDuckGo (no key required) or Serper API"
    layer = 3
    tags = ["domain", "search", "web"]
    requires = []
    required_input_one_of = ["query", "text", "input"]

    ui_schema = {
        "input": {
            "type": "text",
            "accept": None,
            "placeholder": "Search the web...",
            "multiline": False,
        },
        "output": {
            "type": "list",
            "fields": [{"name": "results", "type": "array", "label": "Results"}],
        },
        "quick_actions": [{"icon": "🔍", "label": "Search", "prompt": "Search for"}],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        query = ""
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            query = (input_data.get("query") or input_data.get("text") or
                     input_data.get("input") or params.get("query", ""))
        else:
            query = params.get("query", "")

        num = min(int(params.get("num_results", 10)), 20)

        if not query or not query.strip():
            return {"status": "error", "error": "Query is required"}

        serper_key = os.getenv("SERPER_API_KEY", "")
        provider = "serper" if serper_key else "duckduckgo"

        try:
            if serper_key:
                results = await _search_serper(query.strip(), num, serper_key)
            else:
                results = await _search_duckduckgo(query.strip(), num)

            return {
                "status": "success",
                "query": query,
                "results": results,
                "total": len(results),
                "provider": provider,
            }

        except httpx.TimeoutException:
            return {"status": "error", "error": "Search timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e), "provider": provider}

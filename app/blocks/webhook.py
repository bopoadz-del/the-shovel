"""Webhook Block - Outgoing webhooks"""
from app.core.universal_base import UniversalBlock
from app.core.url_guard import UnsafeURLError, validate_public_url
from typing import Dict, Any, List
import asyncio
import hmac
import hashlib

class WebhookBlock(UniversalBlock):
    """Outgoing webhooks with retries and signatures"""
    name = "webhook"
    version = "1.0.0"
    # `requires = ["config", "queue"]` previously named two blocks that
    # don't exist in BLOCK_REGISTRY — `config` was never built, and there
    # is no queue block (async_processor exists but uses a different
    # interface). Drop both: webhooks send synchronously inline.
    requires = []
    layer = 5  # Integration layer
    tags = ["webhook", "http", "integration"]
    allow_empty_input = True
    skip_input_validation_actions = ["list", "register", "send", "trigger"]
    default_config = {
        "timeout": 30,
        "retries": 3,
        "verify_ssl": True
    }

    def __init__(self, hal_block, config: Dict[str, Any]):
        super().__init__(hal_block, config)
        self.secret = config.get("secret", "")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        # Registered webhooks
        self.endpoints = {}  # name -> {url, events, secret}

    @property
    def queue_block(self):
        # The platform's async_processor block is the closest analogue if
        # we ever want fire-and-forget delivery; today it stays None and
        # `_trigger_event` falls through to its synchronous-send branch.
        return self.get_dep("async_processor")
        
    async def process(self, input_data: Dict, params: Dict = None) -> Dict:
        action = (params or {}).get("action") or (input_data.get("action") if isinstance(input_data, dict) else None)
        if action == "register":
            return await self._register_webhook(input_data)
        elif action == "send":
            return await self._send_webhook(input_data)
        elif action == "trigger":
            return await self._trigger_event(input_data)
        elif action == "list":
            return await self._list_webhooks()
        return {"error": "Unknown action"}
    
    async def _register_webhook(self, data: Dict) -> Dict:
        """Register a webhook endpoint"""
        name = data.get("name")
        url = data.get("url")
        events = data.get("events", ["*"])  # ["user.created", "payment.received"]
        secret = data.get("secret", self.secret)
        
        self.endpoints[name] = {
            "name": name,
            "url": url,
            "events": events,
            "secret": secret,
            "created": asyncio.get_event_loop().time()
        }
        
        return {
            "registered": True,
            "name": name,
            "url": url,
            "events": events
        }
    
    async def _send_webhook(self, data: Dict) -> Dict:
        """Send webhook to specific URL"""
        url = data.get("url")
        payload = data.get("payload", {})
        secret = data.get("secret", self.secret)
        headers = data.get("headers", {})

        # SSRF guard — only POST to a public host (not loopback / internal).
        try:
            url = await asyncio.to_thread(validate_public_url, url)
        except UnsafeURLError as e:
            return {"error": f"Unsafe webhook URL: {e}", "url": url}

        # Add signature
        payload_str = str(payload)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers.update({
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": str(int(asyncio.get_event_loop().time()))
        })
        
        # Send with retries
        for attempt in range(self.max_retries):
            try:
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=False,
                    ) as resp:
                        if resp.status < 400:
                            return {
                                "sent": True,
                                "url": url,
                                "status": resp.status,
                                "attempt": attempt + 1
                            }
                        else:
                            error_body = await resp.text()
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return {
                                "error": f"HTTP {resp.status}: {error_body}",
                                "url": url,
                                "attempts": self.max_retries
                            }
                            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "error": f"Failed after {self.max_retries} attempts: {str(e)}",
                    "url": url
                }

        # Reached only if the retry loop never ran (max_retries <= 0).
        return {"error": "Webhook not sent — max_retries must be >= 1", "url": url}

    async def _trigger_event(self, data: Dict) -> Dict:
        """Trigger event to all registered webhooks"""
        event = data.get("event")  # e.g., "user.created"
        payload = data.get("payload", {})
        
        results = []
        
        for name, endpoint in self.endpoints.items():
            # Check if webhook subscribed to this event
            if "*" not in endpoint["events"] and event not in endpoint["events"]:
                continue
            
            # Queue or send directly
            if self.queue_block:
                await self.queue_block.execute({
                    "action": "enqueue",
                    "job_type": "webhook",
                    "payload": {
                        "url": endpoint["url"],
                        "payload": {**payload, "event": event},
                        "secret": endpoint["secret"]
                    }
                })
                results.append({"name": name, "status": "queued"})
            else:
                result = await self._send_webhook({
                    "url": endpoint["url"],
                    "payload": {**payload, "event": event},
                    "secret": endpoint["secret"]
                })
                results.append({"name": name, **result})
        
        return {
            "event": event,
            "triggered": len(results),
            "results": results
        }
    
    async def _list_webhooks(self) -> Dict:
        """List registered webhooks"""
        return {
            "webhooks": [
                {"name": w["name"], "url": w["url"], "events": w["events"]}
                for w in self.endpoints.values()
            ],
            "count": len(self.endpoints)
        }
    
    def health(self) -> Dict:
        h = {"name": self.name, "version": self.version}
        h["endpoints"] = len(self.endpoints)
        h["max_retries"] = self.max_retries
        return h

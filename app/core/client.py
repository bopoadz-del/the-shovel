"""Cerebrum client for interacting with blocks."""

from typing import Any, Dict, Optional
import aiohttp


class CerebrumClient:
    """Client for interacting with the Cerebrum Blocks API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def execute_block(self, block_name: str, input_data: Any, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a block with the given input."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "block": block_name,
                "input": input_data,
                "params": params or {}
            }
            async with session.post(
                f"{self.base_url}/execute",
                json=payload,
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def ingest(self, file_path: Optional[str] = None, url: Optional[str] = None, 
                     base64_data: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Ingest data into the system."""
        import aiohttp
        import aiofiles
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            
            if file_path:
                async with aiofiles.open(file_path, "rb") as f:
                    content = await f.read()
                data.add_field("file", content, filename=file_path.split("/")[-1])
            
            if url:
                data.add_field("url", url)
            
            if base64_data:
                data.add_field("base64_data", base64_data)
            
            if metadata:
                import json
                data.add_field("metadata", json.dumps(metadata))
            
            async with session.post(
                f"{self.base_url}/ingest",
                data=data,
                headers=self.headers
            ) as response:
                return await response.json()
    
    async def get_block_info(self, block_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about blocks."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/blocks"
            if block_name:
                url = f"{url}/{block_name}"
            
            async with session.get(url, headers=self.headers) as response:
                return await response.json()

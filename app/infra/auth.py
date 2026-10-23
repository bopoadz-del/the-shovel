from app.infra.lego_base import LegoBlock
from typing import Dict, Any, Optional, List
import os
import hashlib
import time
import secrets
from enum import Enum

class Role(Enum):
    ADMIN = "admin"      # Full access, billing, user management
    PRO = "pro"          # 50k requests/month, all blocks
    BASIC = "basic"      # 1k requests/month, core blocks only
    READONLY = "readonly"  # View-only, no mutations

class AuthBlock(LegoBlock):
    """
    Authentication & Authorization Block - API keys, rate limiting, RBAC
    Multi-tenant ready for Block Store
    """
    
    name = "auth"
    version = "1.0.0"
    requires = ["memory"]  # Stores API keys and rate limit counters
    layer = 1  # Security layer
    tags = ["security", "auth", "core"]
    default_config = {
        "master_key": None,
        "rate_limit_default": 100,
        "rate_limit_window": 3600
    }
    
    def __init__(self, hal_block, config: Dict[str, Any]):
        super().__init__(hal_block, config)
        self.memory_block = None  # Wired by assembler
        
        # Rate limiting config
        self.rate_limits = {
            Role.ADMIN: {"requests": 1000000, "window": 3600},  # 1M/hour
            Role.PRO: {"requests": 50000, "window": 3600},        # 50k/hour
            Role.BASIC: {"requests": 1000, "window": 3600},       # 1k/hour
            Role.READONLY: {"requests": 500, "window": 3600}     # 500/hour
        }
        
        self.block_permissions = {
            Role.ADMIN: ["*"],  # All blocks including admin
            Role.PRO: ["chat", "vector", "storage", "ocr", "image", "voice", "queue"],
            Role.BASIC: ["chat", "vector", "storage"],
            # "vector_search" removed with the block itself — the grant named a
            # block that no longer exists, so no capability is lost here.
            Role.READONLY: ["chat_readonly"]  # No mutations
        }
        
        # Master key for admin operations
        self.master_key = config.get("master_key", secrets.token_hex(32))
    
    async def initialize(self):
        """Init auth system"""
        print("[auth] Auth Block initialized")
        print(f"   Roles: {list(Role._value2member_map_.keys())}")
        print(f"   Rate limits: PRO={self.rate_limits[Role.PRO]['requests']}/hour")
        
        # Create default admin key if none exists
        if self.memory_block:
            admin_exists = await self.memory_block.execute({
                "action": "exists",
                "key": "auth:keys:admin"
            })
            if not admin_exists.get("exists"):
                await self.create_api_key("admin_default", Role.ADMIN, "system")
        
        return True
    
    async def execute(self, input_data: Dict) -> Dict:
        """Auth operations"""
        action = input_data.get("action")
        
        if action == "validate":
            return await self._validate_key(input_data.get("api_key"))
        elif action == "check_rate_limit":
            return await self._check_rate_limit(input_data.get("api_key"))
        elif action == "check_permission":
            return await self._check_permission(
                input_data.get("api_key"), 
                input_data.get("block")
            )
        elif action == "create_key":
            return await self.create_api_key(
                input_data.get("name"),
                Role(input_data.get("role", "basic")),
                input_data.get("owner")
            )
        elif action == "revoke_key":
            return await self._revoke_key(input_data.get("api_key"))
        elif action == "rotate_key":
            return await self._rotate_key(input_data.get("api_key"))
        elif action == "get_usage":
            return await self._get_usage(input_data.get("api_key"))
        elif action == "list_keys":
            return await self._list_keys(input_data.get("owner"))
        
        return {"error": f"Unknown action: {action}"}
    
    async def _validate_key(self, api_key: str) -> Dict:
        """Validate API key format and existence"""
        if not api_key:
            return {"valid": False, "reason": "no_key_provided"}
        
        # The development key (app.core.auth.dev_key): from the environment
        # only, in a development environment only; never a value in the source.
        from app.core.auth import dev_key

        dev = dev_key()
        if dev is not None and api_key == dev:
            return {"valid": True, "role": "admin", "owner": "dev", "name": "dev_key"}
        
        # Check if revoked/blocked
        if self.memory_block:
            revoked = await self.memory_block.execute({
                "action": "exists",
                "key": f"auth:revoked:{api_key}"
            })
            if revoked.get("exists"):
                return {"valid": False, "reason": "key_revoked"}
            
            # Get key metadata
            key_data = await self.memory_block.execute({
                "action": "get",
                "key": f"auth:keys:{api_key}"
            })
            
            if not key_data.get("hit"):
                return self._not_found()
            
            metadata = key_data.get("value", {})
            return {
                "valid": True,
                "role": metadata.get("role"),
                "owner": metadata.get("owner"),
                "created": metadata.get("created"),
                "name": metadata.get("name")
            }
        
        return self._not_found()
    

    @staticmethod
    def _not_found() -> Dict:
        """An unknown key. In development with no development key configured,
        the refusal says so and names the variable (fail closed, never a
        built-in value)."""
        from app.core.auth import DEV_KEY_ENV, dev_key, is_dev_environment

        if is_dev_environment() and dev_key() is None:
            return {"valid": False, "reason": "dev_key_not_configured",
                    "message": f"No development key is configured: set {DEV_KEY_ENV} to use one in this environment."}
        return {"valid": False, "reason": "key_not_found"}
    async def _check_rate_limit(self, api_key: str) -> Dict:
        """Check if request is within rate limit"""
        validation = await self._validate_key(api_key)
        if not validation.get("valid"):
            return {"allowed": False, "reason": "invalid_key"}
        
        role = Role(validation.get("role", "basic"))
        limit = self.rate_limits[role]
        
        if not self.memory_block:
            # No memory = unlimited (dev mode)
            return {"allowed": True, "remaining": 999999}
        
        # Window-based counter (sliding window approximation)
        window_key = f"auth:ratelimit:{api_key}:{int(time.time() / limit['window'])}"
        
        current = await self.memory_block.execute({
            "action": "get",
            "key": window_key
        })
        
        count = current.get("value", {}).get("count", 0) if current.get("hit") else 0
        
        if count >= limit["requests"]:
            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "limit": limit["requests"],
                "window_seconds": limit["window"],
                "retry_after": limit["window"] - (time.time() % limit["window"])
            }
        
        # Increment counter
        await self.memory_block.execute({
            "action": "set",
            "key": window_key,
            "value": {"count": count + 1, "last_request": time.time()},
            "ttl": limit["window"] + 60  # Slight buffer
        })
        
        return {
            "allowed": True,
            "remaining": limit["requests"] - count - 1,
            "limit": limit["requests"],
            "reset_time": int(time.time() / limit["window"] + 1) * limit["window"]
        }
    
    async def _check_permission(self, api_key: str, block_name: str) -> Dict:
        """Check if key can access specific block"""
        validation = await self._validate_key(api_key)
        if not validation.get("valid"):
            return {"allowed": False, "reason": "invalid_key"}
        
        role = Role(validation.get("role", "basic"))
        allowed_blocks = self.block_permissions.get(role, [])
        
        # Admin has wildcard
        if "*" in allowed_blocks or block_name in allowed_blocks:
            return {
                "allowed": True,
                "role": role.value,
                "block": block_name
            }
        
        # Check for readonly variant - only if the exact block is in the allowed list
        if role == Role.READONLY and block_name in allowed_blocks:
            return {
                "allowed": True,
                "role": role.value,
                "block": block_name,
                "mode": "readonly"
            }
        
        return {
            "allowed": False,
            "reason": "insufficient_permissions",
            "role": role.value,
            "required_role": "pro"  # Suggest upgrade
        }
    
    async def create_api_key(self, name: str, role: Role, owner: str) -> Dict:
        """Generate new API key"""
        # Generate secure key
        key_bytes = secrets.token_bytes(32)
        api_key = f"cb_{hashlib.sha256(key_bytes).hexdigest()[:24]}"
        
        metadata = {
            "name": name,
            "role": role.value,
            "owner": owner,
            "created": time.time(),
            "active": True
        }
        
        if self.memory_block:
            # Store key metadata
            await self.memory_block.execute({
                "action": "set",
                "key": f"auth:keys:{api_key}",
                "value": metadata,
                "ttl": 0  # No expiry unless revoked
            })
            
            # Add to owner's key list
            owner_keys = await self.memory_block.execute({
                "action": "get",
                "key": f"auth:owner:{owner}"
            })
            
            keys_list = owner_keys.get("value", {}).get("keys", []) if owner_keys.get("hit") else []
            keys_list.append(api_key)
            
            await self.memory_block.execute({
                "action": "set",
                "key": f"auth:owner:{owner}",
                "value": {"keys": keys_list},
                "ttl": 0
            })
        
        return {
            "api_key": api_key,
            "name": name,
            "role": role.value,
            "owner": owner,
            "message": "Save this key - it won't be shown again"
        }
    
    async def _revoke_key(self, api_key: Optional[str]) -> Dict:
        """Revoke an API key"""
        if not api_key:
            return {"revoked": False, "reason": "no_key_provided"}
        if not self.memory_block:
            return {"revoked": False, "reason": "no_memory_backend"}
        
        # Add to revoked list (prevents reuse)
        await self.memory_block.execute({
            "action": "set",
            "key": f"auth:revoked:{api_key}",
            "value": {"revoked_at": time.time()},
            "ttl": 86400 * 30  # Keep revocation record for 30 days
        })
        
        # Delete key metadata
        await self.memory_block.execute({
            "action": "delete",
            "key": f"auth:keys:{api_key}"
        })
        
        return {"revoked": True, "api_key": api_key[:8] + "..."}
    
    async def _get_usage(self, api_key: Optional[str]) -> Dict:
        """Get usage statistics for a key"""
        if not api_key:
            return {"error": "no_key_provided"}
        if not self.memory_block:
            return {"usage": "untracked"}
        
        # Get all rate limit windows for this key
        # (This is simplified - real impl would scan or use counters)
        return {
            "api_key": api_key[:8] + "...",
            "status": "active",  # Would check revoked status
            "current_hour_requests": "tracked_in_memory"
        }
    
    async def _list_keys(self, owner: Optional[str]) -> Dict:
        """List all keys for an owner"""
        if not self.memory_block:
            return {"keys": []}
        
        target_owner = owner or "system"
        owner_data = await self.memory_block.execute({
            "action": "get",
            "key": f"auth:owner:{target_owner}"
        })
        
        if not owner_data.get("hit"):
            return {"keys": []}
        
        keys = owner_data.get("value", {}).get("keys", [])
        key_details = []
        
        for key in keys:
            metadata = await self.memory_block.execute({
                "action": "get",
                "key": f"auth:keys:{key}"
            })
            if metadata.get("hit"):
                m = metadata.get("value", {})
                key_details.append({
                    "key": key,
                    "name": m.get("name"),
                    "role": m.get("role"),
                    "created": m.get("created"),
                    "preview": key[:8] + "..."
                })
        
        return {"keys": key_details, "count": len(key_details)}
    
    async def _rotate_key(self, api_key: Optional[str]) -> Dict:
        """Rotate an API key: revoke old and create new with same metadata"""
        if not api_key:
            return {"error": "no_key_provided"}
        if not self.memory_block:
            return {"error": "no_memory_backend"}
        
        # Get old metadata
        old_data = await self.memory_block.execute({
            "action": "get",
            "key": f"auth:keys:{api_key}"
        })
        
        if not old_data.get("hit"):
            return {"error": "key_not_found"}
        
        metadata = old_data.get("value", {})
        
        # Revoke old
        await self._revoke_key(api_key)
        
        # Create new with same metadata
        new_key = await self.create_api_key(
            metadata.get("name", "rotated"),
            Role(metadata.get("role", "basic")),
            metadata.get("owner")
        )
        
        return {
            "rotated": True,
            "old_key": api_key[:8] + "...",
            "new_api_key": new_key.get("api_key"),
            "message": "Save this key - it won't be shown again"
        }
    
    def health(self) -> Dict:
        h = super().health()
        h["auth_method"] = "api_key_bearer"
        h["rate_limiting"] = True
        h["rbac_roles"] = len(Role)
        h["master_key_preview"] = (self.master_key or "none")[:8] + "..."
        return h

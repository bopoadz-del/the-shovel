"""Base LegoBlock class for Cerebrum Blocks."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import time
import uuid


class LegoBlock(ABC):
    """Base class for all Cerebrum Blocks.
    
    Universal Features:
    - name: Unique identifier for the block
    - version: Block version string
    - requires: List of block names this block depends on
    - layer: Initialization layer (0=infra, 1=security, 2=core, etc.)
    """
    
    name: str = "base"
    version: str = "1.0.0"
    requires: List[str] = []
    layer: int = 99  # Default: unassigned layer
    
    def __init__(self, hal_block=None, config: Optional[Dict] = None):
        self.hal = hal_block
        self.config = config or {}
        self.execution_count = 0
        self.total_execution_time = 0
        self.initialized = False
        self._dependencies: Dict[str, 'LegoBlock'] = {}
    
    @abstractmethod
    async def execute(self, input_data: Dict) -> Dict:
        """Execute the block with the given input."""
        pass
    
    async def initialize(self) -> bool:
        """Initialize the block. Override for custom init."""
        self.initialized = True
        return True
    
    def inject(self, block_name: str, block_instance: 'LegoBlock'):
        """Receive dependency injection from assembler.
        
        Sets self.{name}_block attribute for easy access.
        """
        self._dependencies[block_name] = block_instance
        attr_name = f"{block_name}_block"
        setattr(self, attr_name, block_instance)
    
    def get_dependency(self, name: str) -> Optional['LegoBlock']:
        """Get injected dependency by name."""
        return self._dependencies.get(name)
    
    def health(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "name": self.name,
            "version": self.version,
            "initialized": self.initialized,
            "execution_count": self.execution_count,
            "healthy": True,
            "dependencies": list(self._dependencies.keys())
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get block statistics."""
        avg_time = self.total_execution_time / self.execution_count if self.execution_count > 0 else 0
        return {
            "name": self.name,
            "version": self.version,
            "execution_count": self.execution_count,
            "total_execution_time_ms": self.total_execution_time,
            "avg_execution_time_ms": round(avg_time, 2)
        }

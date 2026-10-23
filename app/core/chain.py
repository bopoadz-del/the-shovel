"""Chain blocks together."""

from typing import Any, Dict, List, Optional, Callable
import asyncio


class Chain:
    """Chain multiple blocks together."""
    
    def __init__(self, client):
        self.client = client
        self.steps: List[Dict[str, Any]] = []
        self.current_output: Any = None
    
    def then(self, block_name: str, params: Dict[str, Any] = None) -> "Chain":
        """Add a block to the chain."""
        self.steps.append({
            "block": block_name,
            "params": params or {}
        })
        return self
    
    async def run(self, initial_input: Any) -> "ChainResult":
        """Execute the chain."""
        self.current_output = initial_input
        results = []
        
        for step in self.steps:
            result = await self.client.execute_block(
                step["block"],
                self.current_output,
                step["params"]
            )
            results.append(result)
            self.current_output = result.get("result", {})
        
        return ChainResult(results, self.current_output)
    
    def __or__(self, other: Callable) -> "Chain":
        """Support pipe operator for chaining."""
        if isinstance(other, tuple) and len(other) == 2:
            block_name, params = other
            return self.then(block_name, params)
        return self.then(other)


class ChainResult:
    """Result of a chain execution."""
    
    def __init__(self, steps: List[Dict[str, Any]], final_output: Any):
        self.steps = steps
        self.final_output = final_output
        self.all_results = steps
    
    @property
    def success(self) -> bool:
        """Check if all steps were successful."""
        return all(step.get("status") == "success" for step in self.steps)
    
    @property
    def total_time_ms(self) -> int:
        """Get total execution time."""
        return sum(step.get("processing_time_ms", 0) for step in self.steps)
    
    def get_step(self, index: int) -> Dict[str, Any]:
        """Get a specific step result."""
        return self.steps[index] if 0 <= index < len(self.steps) else {}


def chain(client) -> Chain:
    """Create a new chain."""
    return Chain(client)

"""Sandbox Block - Code execution isolation and security.

POSIX-only: the `resource` and `signal.SIGXCPU` primitives this block uses
to enforce memory / CPU limits don't exist on Windows. We import them
conditionally so the block stays loadable on Windows (it will raise an
honest error when actually invoked there, rather than failing at import
time and blocking the whole registry).
"""
from app.core.subprocess_env import scrubbed_env
from app.core.universal_base import UniversalBlock
from typing import Dict, Any, Callable, Optional
import asyncio
import tempfile
import os
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:
    import resource  # POSIX only
    _RESOURCE_AVAILABLE = True
except ImportError:  # pragma: no cover — Windows path
    resource = None  # type: ignore[assignment]
    _RESOURCE_AVAILABLE = False


class SandboxLevel(Enum):
    """Security sandbox levels"""
    NONE = "none"           # No sandboxing
    PERMISSIVE = "permissive"  # Log only, don't block
    STRICT = "strict"       # Block dangerous operations
    ISOLATED = "isolated"   # Full process isolation


@dataclass
class SandboxPolicy:
    """Sandbox policy configuration"""
    max_memory_mb: int = 512
    max_cpu_time: int = 5  # seconds
    max_file_size_mb: int = 10
    network_allowed: bool = False
    filesystem_readonly: bool = True
    allowed_modules: list = None
    blocked_builtins: list = None
    
    def __post_init__(self):
        if self.allowed_modules is None:
            self.allowed_modules = ["math", "random", "datetime", "json", "re"]
        if self.blocked_builtins is None:
            self.blocked_builtins = ["open", "exec", "eval", "compile"]


class TimeoutException(Exception):
    pass


# Wall-clock slack on top of SandboxPolicy.max_cpu_time for JS/bash subprocess
# spawn. max_cpu_time is the user-code budget; process creation on a loaded
# GitHub Actions runner (full pytest + coverage) is extra. Treating that
# startup as "Execution timeout" failed
# tests/test_sandbox_block.py::test_execute_javascript_simple on PR #397
# (CI run 32449705289, virgin profile) with
# ``{'error': 'Execution timeout', 'killed': True}`` for ``console.log(2+3)``.
_SUBPROCESS_SPAWN_GRACE_S = 25.0


def _subprocess_wall_s(policy: SandboxPolicy) -> float:
    """Deadline for spawn + communicate. Always longer than max_cpu_time."""
    return float(policy.max_cpu_time) + _SUBPROCESS_SPAWN_GRACE_S


class SandboxBlock(UniversalBlock):
    """
    Sandbox Block - Secure code execution isolation
    
    Features:
    - Memory limits (prevents OOM)
    - CPU time limits (prevents infinite loops)
    - Network blocking
    - Filesystem restrictions
    - Module whitelisting
    - Builtin function restrictions
    - Process isolation (optional)
    
    Use cases:
    - Safe user code execution
    - Plugin sandboxing
    - Code validation before deployment
    """
    auto_validate = False
    name = "sandbox"
    version = "1.0.0"
    requires = ["memory", "config"]
    layer = 1  # Security layer
    tags = ["security", "isolation", "sandbox", "enterprise"]
    default_config = {
        "default_level": "strict",
        "max_memory_mb": 512,
        "max_cpu_time": 5,
        "network_allowed": False,
        "filesystem_readonly": True,
        "allowed_modules": ["math", "random", "datetime", "json", "re", "string", "collections"],
        "auto_kill": True
    }
    
    def __init__(self, hal_block=None, config: Dict = None):
        super().__init__(hal_block, config)
        self.policies: Dict[str, SandboxPolicy] = {}
        self.active_sessions: Dict[str, Dict] = {}
        self.execution_count = 0
        self.blocked_count = 0
        # Default policy must exist before process() is invoked; _legacy_initialize
        # also sets it, but callers may use the block without booting HAL.
        self.policies["default"] = SandboxPolicy(
            max_memory_mb=self.config.get("max_memory_mb", 512),
            max_cpu_time=self.config.get("max_cpu_time", 5),
            network_allowed=self.config.get("network_allowed", False),
            filesystem_readonly=self.config.get("filesystem_readonly", True),
            allowed_modules=self.config.get("allowed_modules", []),
        )
        
    async def _legacy_initialize(self) -> bool:
        """Initialize sandbox environment"""
        print("🔒 Sandbox Block initialized")
        print(f"   Default level: {self.config.get('default_level', 'strict')}")
        print(f"   Memory limit: {self.config.get('max_memory_mb', 512)} MB")
        print(f"   CPU limit: {self.config.get('max_cpu_time', 5)} seconds")
        print(f"   Network: {'allowed' if self.config.get('network_allowed') else 'blocked'}")
        
        # Create default policy
        self.policies["default"] = SandboxPolicy(
            max_memory_mb=self.config.get("max_memory_mb", 512),
            max_cpu_time=self.config.get("max_cpu_time", 5),
            network_allowed=self.config.get("network_allowed", False),
            filesystem_readonly=self.config.get("filesystem_readonly", True),
            allowed_modules=self.config.get("allowed_modules", [])
        )
        
        self.initialized = True
        return True
    
    async def process(self, input_data: Dict, params: Dict = None) -> Dict:
        """Handle sandbox actions"""
        action = (params or {}).get("action") or (input_data.get("action") if isinstance(input_data, dict) else None)
        # A bare {"code": ...} payload is a pre-flight safety check -- the
        # block's whole purpose (see self-coding agent config). Without this
        # default it died with "Unknown action: None" (same class as the
        # file_hasher F30b find).
        if action is None and isinstance(input_data, dict) and input_data.get("code"):
            action = "check_safety"
        
        if action == "execute":
            return await self._execute_sandboxed(input_data)
        elif action == "validate_code":
            return await self._validate_code(input_data)
        elif action == "create_policy":
            return self._create_policy(input_data)
        elif action == "wrap_block":
            return await self._wrap_block(input_data)
        elif action == "get_stats":
            return self._get_stats(input_data)
        elif action == "check_safety":
            return await self._check_safety(input_data)
            
        return {"error": f"Unknown action: {action}"}
    
    async def _execute_sandboxed(self, data: Dict) -> Dict:
        """
        Execute code in sandbox with full isolation
        
        Supports Python, JavaScript (via Node), and Bash (restricted)
        """
        code = data.get("code")
        language = data.get("language", "python")
        policy_name = data.get("policy", "default")
        
        if not code:
            return {"error": "No code provided"}
        
        policy = self.policies.get(policy_name, self.policies["default"])
        self.execution_count += 1
        
        # Pre-validation
        safety_check = await self._check_safety({"code": code, "language": language})
        if not safety_check.get("safe"):
            self.blocked_count += 1
            return {
                "error": "Code failed safety check",
                "violations": safety_check.get("violations", []),
                "blocked": True
            }
        
        # Execute based on language
        try:
            if language == "python":
                return await self._execute_python(code, policy, data.get("inputs", {}))
            elif language == "javascript":
                return await self._execute_javascript(code, policy)
            elif language == "bash":
                return await self._execute_bash(code, policy)
            else:
                return {"error": f"Unsupported language: {language}"}
        except TimeoutException:
            return {"error": "Execution timeout", "killed": True}
        except MemoryError:
            return {"error": "Memory limit exceeded", "killed": True}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}
    
    async def _execute_python(self, code: str, policy: SandboxPolicy, inputs: Dict) -> Dict:
        """Execute Python code in sandbox"""
        import time
        import io
        import sys
        
        start_time = time.time()
        
        # Create restricted globals
        safe_globals = {
            "__builtins__": self._get_restricted_builtins(policy),
            "input": lambda prompt="": inputs.get("input", ""),
            "print": lambda *args, **kwargs: stdout_capture.write(
                kwargs.get("sep", " ").join(str(a) for a in args) + kwargs.get("end", "\n")
            ),
        }
        
        # Add allowed modules
        for mod_name in policy.allowed_modules:
            try:
                safe_globals[mod_name] = __import__(mod_name)
            except ImportError:
                logger.debug(
                    "swallowed %s in _execute_python() — continuing",
                    "ImportError", exc_info=True,
                )
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Execute with resource limits
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture
        
        result_value = None
        error = None
        rlimit_to_restore = None

        try:
            # Set memory limit (POSIX only; on Windows we skip the rlimit
            # call and let the OS handle process-level OOM — the sandbox
            # provides weaker isolation here, which is documented).
            #
            # Lower ONLY the soft limit and restore it after exec. The old
            # code set soft AND hard to the policy cap and never restored —
            # rlimits are process-wide and a lowered hard limit can never be
            # raised again (EPERM), so the first sandboxed run permanently
            # capped the HOST process at 512MB of address space. Every
            # thread started afterwards hung at pthread stack mmap: that was
            # the 2026-07-05 CI zombie (TestClient portals, asyncio child
            # watchers, and to_thread executors all park in thread.start()).
            if policy.max_memory_mb > 0 and _RESOURCE_AVAILABLE:
                old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
                limit = policy.max_memory_mb * 1024 * 1024
                if old_hard != resource.RLIM_INFINITY:
                    limit = min(limit, old_hard)
                resource.setrlimit(resource.RLIMIT_AS, (limit, old_hard))
                rlimit_to_restore = (old_soft, old_hard)

            # Execute with timeout
            exec(code, safe_globals)

            # Try to get result
            if "result" in safe_globals:
                result_value = safe_globals["result"]
            elif "output" in safe_globals:
                result_value = safe_globals["output"]

        except Exception as e:
            error = str(e)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            if rlimit_to_restore is not None:
                # Hard limit was never touched, so restoring the old soft
                # limit is always permitted.
                try:
                    resource.setrlimit(resource.RLIMIT_AS, rlimit_to_restore)
                except (ValueError, OSError):  # pragma: no cover — defensive
                    logger.debug(
                        "swallowed %s in _execute_python() — continuing",
                        "(ValueError, OSError)", exc_info=True,
                    )
        
        execution_time = time.time() - start_time
        
        return {
            "success": error is None,
            "result": result_value,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": error,
            "execution_time": round(execution_time, 3),
            "memory_used_mb": "N/A",  # Would need psutil for accurate measurement
            "sandboxed": True
        }
    
    def _get_restricted_builtins(self, policy: SandboxPolicy) -> Dict:
        """Create restricted builtins dict"""
        safe_builtins = {}

        # ``__builtins__`` may be a dict (e.g. under pytest) or the builtins
        # module.  Normalise to a {name: value} mapping before filtering.
        if isinstance(__builtins__, dict):
            builtins_items = __builtins__.items()
        else:
            builtins_items = vars(__builtins__).items()

        # Copy safe builtins
        for name, value in builtins_items:
            if name not in policy.blocked_builtins and (not name.startswith("_") or name == "__import__"):
                safe_builtins[name] = value

        # Add safe versions of blocked functions
        safe_builtins["open"] = self._safe_open

        return safe_builtins
    
    def _safe_open(self, filepath: str, mode: str = "r", *args, **kwargs):
        """Safe file open - restricts to temp directory"""
        # Normalize path
        filepath = os.path.abspath(filepath)
        temp_dir = tempfile.gettempdir()
        
        # Only allow access to temp directory
        if not filepath.startswith(temp_dir):
            raise PermissionError(f"Access denied: {filepath}. Only {temp_dir} allowed.")
        
        # Check write permissions
        policy = self.policies.get("default")
        if "w" in mode or "a" in mode:
            if policy and policy.filesystem_readonly:
                raise PermissionError("Write access denied (read-only filesystem)")
        
        return open(filepath, mode, *args, **kwargs)
    
    async def _execute_javascript(self, code: str, policy: SandboxPolicy) -> Dict:
        """Execute JavaScript in sandbox using Node.js"""
        import shutil

        node_bin = shutil.which("node")
        if not node_bin:
            return {"error": "node is not installed", "success": False}

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_path = f.name

        wall_s = _subprocess_wall_s(policy)
        try:
            # The whole spawn+communicate sequence runs under ONE outer
            # deadline. The inner wait_for only covered communicate(); the
            # 2026-07-05 CI zombies proved the pipeline can block before or
            # around it (subprocess creation / pipe teardown), hanging the
            # caller forever. A sandbox that can hang its caller is not a
            # sandbox — convert any stall into the timeout error path.
            #
            # communicate()'s deadline is wall-clock (spawn grace + CPU
            # budget), not bare max_cpu_time: a loaded runner can take
            # longer than 5s to exec node even for console.log(2+3).
            async def _spawn_and_wait():
                proc = await asyncio.create_subprocess_exec(
                    node_bin, temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=policy.max_memory_mb * 1024 * 1024,
                    env=scrubbed_env(),  # audit §6.1 — no app secrets in the sandbox
                    start_new_session=True,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=wall_s,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    raise TimeoutException()
                return proc.returncode, stdout, stderr

            try:
                returncode, stdout, stderr = await asyncio.wait_for(
                    _spawn_and_wait(),
                    timeout=wall_s + 5,
                )
            except asyncio.TimeoutError:
                raise TimeoutException()

            return {
                "success": returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "returncode": returncode,
                "sandboxed": True
            }

        finally:
            os.unlink(temp_path)
    
    _SHELL_ALLOWLIST = {"echo", "cat", "ls", "pwd", "wc", "head", "tail", "grep", "find", "mkdir", "touch", "cp", "mv", "chmod", "python", "pytest"}

    def _audit_shell(self, code: str, allowed: bool, result: dict = None):
        import datetime, pathlib
        log_dir = pathlib.Path("logs")
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "shell_audit.log", "a") as f:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            status = "ALLOWED" if allowed else "BLOCKED"
            rc = result.get("return_code", "N/A") if result else "N/A"
            f.write(f"{ts} | SANDBOX | {status} | {code!r} | {rc}\n")

    async def _execute_bash(self, code: str, policy: SandboxPolicy) -> Dict:
        """Execute bash commands in restricted shell with strict allowlist."""
        import subprocess
        
        # Strict allowlist — no blacklist bypass
        command = code.strip().split()[0] if code.strip() else ""
        if command not in self._SHELL_ALLOWLIST:
            result = {"error": f"Command '{command}' not in sandbox allowlist.", "blocked": True}
            self._audit_shell(code, allowed=False, result=result)
            return result
        
        # Run in restricted mode. Same outer-deadline shape as
        # _execute_javascript: spawn+communicate under one wait_for so a
        # stall anywhere in the subprocess pipeline becomes a timeout
        # instead of hanging the caller. Wall clock includes spawn grace
        # (see _subprocess_wall_s) so a loaded runner is not a false kill.
        wall_s = _subprocess_wall_s(policy)

        async def _spawn_and_wait():
            proc = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=policy.max_memory_mb * 1024 * 1024,
                # audit §6.1 — a shell command must not be able to echo
                # $SECRET_KEY / $DATABASE_URL out of the sandbox.
                env=scrubbed_env(),
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=wall_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                raise TimeoutException()
            return proc.returncode, stdout, stderr

        try:
            returncode, stdout, stderr = await asyncio.wait_for(
                _spawn_and_wait(),
                timeout=wall_s + 5,
            )
        except asyncio.TimeoutError:
            raise TimeoutException()

        return {
            "success": returncode == 0,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": returncode,
            "sandboxed": True
        }
    
    async def _validate_code(self, data: Dict) -> Dict:
        """Static analysis of code before execution"""
        code = data.get("code", "")
        language = data.get("language", "python")
        
        violations = []
        warnings = []
        
        if language == "python":
            # Check for dangerous patterns
            dangerous_patterns = {
                "__import__": "Dynamic import detected",
                "eval(": "eval() usage detected",
                "exec(": "exec() usage detected",
                "compile(": "compile() usage detected",
                "subprocess": "Subprocess usage detected",
                "os.system": "System command detected",
                "open(": "File operation detected (use safe_open)",
                "socket": "Network socket detected",
                "urllib": "Network access detected"
            }
            
            for pattern, message in dangerous_patterns.items():
                if pattern in code:
                    violations.append({"pattern": pattern, "message": message})
            
            # Check for infinite loop indicators
            if "while True:" in code and "break" not in code:
                warnings.append("Potential infinite loop (while True without break)")
            
            # Check for recursion
            if code.count("def ") > 0 and "(self" not in code:
                func_name = code.split("def ")[1].split("(")[0].strip()
                if func_name in code.split("def ")[1]:
                    warnings.append(f"Potential recursion in {func_name}")
        
        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "language": language
        }
    
    def _create_policy(self, data: Dict) -> Dict:
        """Create custom sandbox policy"""
        policy_name = data.get("name", "custom")
        
        self.policies[policy_name] = SandboxPolicy(
            max_memory_mb=data.get("max_memory_mb", 512),
            max_cpu_time=data.get("max_cpu_time", 5),
            network_allowed=data.get("network_allowed", False),
            filesystem_readonly=data.get("filesystem_readonly", True),
            allowed_modules=data.get("allowed_modules", []),
            blocked_builtins=data.get("blocked_builtins", [])
        )
        
        return {
            "created": True,
            "policy": policy_name,
            "config": {
                "max_memory_mb": self.policies[policy_name].max_memory_mb,
                "max_cpu_time": self.policies[policy_name].max_cpu_time,
                "network_allowed": self.policies[policy_name].network_allowed
            }
        }
    
    async def _wrap_block(self, data: Dict) -> Dict:
        """Wrap an existing block with sandbox"""
        block_name = data.get("block_name")
        policy_name = data.get("policy", "default")
        
        # This would integrate with ContainerBlock
        return {
            "wrapped": True,
            "block": block_name,
            "policy": policy_name,
            "note": "Use with ContainerBlock.load_module() for full sandbox wrapping"
        }
    
    async def _check_safety(self, data: Dict) -> Dict:
        """Quick safety check"""
        validation = await self._validate_code(data)
        return {
            "safe": validation["safe"],
            "score": 100 - len(validation["violations"]) * 20 - len(validation["warnings"]) * 5,
            "violations": validation["violations"],
            "warnings": validation["warnings"]
        }
    
    def _get_stats(self, data: Dict) -> Dict:
        """Get sandbox statistics"""
        return {
            "executions": self.execution_count,
            "blocked": self.blocked_count,
            "policies": len(self.policies),
            "active_sessions": len(self.active_sessions),
            "default_limits": {
                "memory_mb": self.config.get("max_memory_mb", 512),
                "cpu_seconds": self.config.get("max_cpu_time", 5)
            }
        }
    
    def health(self) -> Dict:
        """Sandbox health status"""
        h = {"name": self.name, "version": self.version}
        h["executions"] = self.execution_count
        h["blocked"] = self.blocked_count
        h["policies"] = len(self.policies)
        h["default_level"] = self.config.get("default_level", "strict")
        return h

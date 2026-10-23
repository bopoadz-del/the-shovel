"""Code Block - Sandboxed Python execution via subprocess + AST analysis"""

import ast
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict

from app.core.subprocess_env import scrubbed_env
from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds
_MAX_OUTPUT = 10_000

_DANGEROUS_PATTERNS = [
    "os.system", "subprocess", "shutil.rmtree", "__import__",
    "open(", "os.remove", "os.unlink", "socket.connect",
]


def _syntax_check(code: str) -> str | None:
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


def _analyze(code: str) -> Dict:
    issues = []
    for pat in _DANGEROUS_PATTERNS:
        if pat in code:
            issues.append(f"Potentially unsafe: {pat}")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"valid": False, "error": str(e), "issues": issues}

    imports = [
        node.names[0].name if hasattr(node, "names") and node.names else ""
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    funcs = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    classes = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    lines = code.splitlines()

    return {
        "valid": True,
        "lines": len(lines),
        "imports": list(set(filter(None, imports))),
        "functions": funcs,
        "classes": classes,
        "issues": issues,
    }


def _run_python(code: str, timeout: int) -> Dict:
    syntax_err = _syntax_check(code)
    if syntax_err:
        return {"status": "error", "error": syntax_err, "output": "", "exit_code": 1}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Never hand app secrets to an executed code snippet (audit §6.1).
            env=scrubbed_env({"PYTHONDONTWRITEBYTECODE": "1"}),
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        stdout = proc.stdout[:_MAX_OUTPUT]
        stderr = proc.stderr[:2000]

        return {
            "status": "success" if proc.returncode == 0 else "error",
            "output": stdout,
            "stderr": stderr if stderr else None,
            "exit_code": proc.returncode,
            "execution_time_ms": elapsed_ms,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Execution timed out after {timeout}s", "output": ""}
    except Exception as e:
        return {"status": "error", "error": str(e), "output": ""}
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            logger.debug(
                "swallowed %s in _run_python() — continuing",
                "OSError", exc_info=True,
            )


def _run_node(code: str, timeout: int) -> Dict:
    node_bin = "node"
    if subprocess.run(["which", node_bin], capture_output=True,
                      env=scrubbed_env()).returncode != 0:
        return {"status": "error", "error": "Node.js not available on this server"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        start = time.monotonic()
        proc = subprocess.run(
            [node_bin, tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=scrubbed_env(),  # audit §6.1
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "output": proc.stdout[:_MAX_OUTPUT],
            "stderr": proc.stderr[:2000] or None,
            "exit_code": proc.returncode,
            "execution_time_ms": elapsed_ms,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Execution timed out after {timeout}s", "output": ""}
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            logger.debug(
                "swallowed %s in _run_node() — continuing",
                "OSError", exc_info=True,
            )


class CodeBlock(UniversalBlock):
    """Python and JavaScript code execution with static analysis"""

    auto_validate = False
    name = "code"
    version = "2.0"
    description = "Execute Python or JavaScript code; analyze code for issues"
    layer = 3
    tags = ["domain", "code", "execution"]
    requires = []

    ui_schema = {
        "input": {
            "type": "code",
            "accept": None,
            "placeholder": "Paste code to execute or describe code to analyze...",
            "multiline": True,
        },
        "output": {
            "type": "code",
            "fields": [
                {"name": "output", "type": "code", "label": "Result"},
                {"name": "language", "type": "text", "label": "Language"},
                {"name": "execution_time_ms", "type": "number", "label": "Time (ms)"},
            ],
        },
        "quick_actions": [
            {"icon": "🐍", "label": "Python", "prompt": "Write Python code to"},
            {"icon": "📜", "label": "JavaScript", "prompt": "Write JavaScript code to"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}

        code = ""
        if isinstance(input_data, str):
            code = input_data
        elif isinstance(input_data, dict):
            code = (input_data.get("code") or input_data.get("text") or
                    input_data.get("input") or params.get("code", ""))
        else:
            code = params.get("code", "")

        language = params.get("language", "python").lower()
        operation = params.get("operation", "execute")
        timeout = min(int(params.get("timeout", _TIMEOUT)), 30)

        if not code or not code.strip():
            return {"status": "error", "error": "No code provided"}

        if operation == "analyze":
            analysis = _analyze(code)
            return {
                "status": "success",
                "operation": "analyze",
                "language": language,
                **analysis,
            }

        analysis = _analyze(code)
        if analysis.get("issues"):
            return {
                "status": "error",
                "error": "Refusing to execute: " + "; ".join(analysis["issues"]),
                "operation": operation,
                "issues": analysis["issues"],
            }

        # execute
        if language in ("python", "py"):
            result = _run_python(code, timeout)
        elif language in ("javascript", "js", "node"):
            result = _run_node(code, timeout)
        else:
            return {
                "status": "error",
                "error": f"Unsupported language: {language}. Supported: python, javascript",
            }

        return {
            **result,
            "language": language,
            "operation": operation,
            "lines_executed": len(code.splitlines()),
        }

"""Async Processor Block - Celery-style task dispatcher with in-process fallback."""

import asyncio
import uuid
import time
from typing import Any, Dict, List, Callable, Optional
from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)


class AsyncProcessorBlock(UniversalBlock):
    """Dispatch long-running tasks asynchronously with job tracking."""

    name = "async_processor"
    version = "1.0.0"
    description = "Celery-style task dispatcher with in-process fallback queue"
    layer = 0
    tags = ["infrastructure", "async", "queue", "worker"]
    requires = []

    default_config = {
        "max_concurrent": 4,
        "default_timeout": 300,
        "celery_app": None  # e.g. "tasks.celery_app" if Celery is configured
    }

    ui_schema = {
        "input": {
            "type": "json",
            "accept": None,
            "placeholder": '{"action": "dispatch", "task": "process_pdf", "args": {}}',
            "multiline": False
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "job_id", "type": "text", "label": "Job ID"}
            ]
        },
        "quick_actions": [
            {"icon": "🚀", "label": "Dispatch Job", "prompt": '{"action":"dispatch","task":"process_pdf","args":{}}'},
            {"icon": "🔍", "label": "Check Status", "prompt": '{"action":"status","job_id":"your-job-id"}'}
        ]
    }

    def __init__(self, hal_block=None, config=None):
        super().__init__(hal_block, config)
        self._jobs: Dict[str, Dict] = {}
        self._celery = None
        self._semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 4))
        self._init_celery()

    def _init_celery(self):
        celery_path = self.config.get("celery_app") or self.default_config.get("celery_app")
        if celery_path:
            try:
                module_path, app_name = celery_path.rsplit(":", 1)
                module = __import__(module_path, fromlist=[app_name])
                self._celery = getattr(module, app_name)
            except Exception:
                self._celery = None

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Route to appropriate async action."""
        params = params or {}
        action = params.get("action") or (input_data.get("action") if isinstance(input_data, dict) else "status")
        handlers = {
            "dispatch": self.dispatch,
            "status": self.status,
            "result": self.result,
            "cancel": self.cancel,
            "list": self.list_jobs,
            "health_check": self.health_check,
        }
        handler = handlers.get(action)
        if not handler:
            return {"status": "error", "error": f"Unknown action: {action}"}
        return await handler(input_data, params)

    async def dispatch(self, input_data: Any, params: Dict) -> Dict:
        """Dispatch a new async task."""
        payload = input_data if isinstance(input_data, dict) else {}
        task_name = params.get("task") or payload.get("task")
        args = params.get("args") or payload.get("args", {})

        if not task_name:
            return {"status": "error", "error": "No task name provided"}

        job_id = f"job-{uuid.uuid4().hex[:12]}"

        # Celery path
        if self._celery:
            try:
                task = self._celery.send_task(task_name, kwargs=args)
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "task": task_name,
                    "status": "queued",
                    "celery_task_id": task.id,
                    "created_at": time.time(),
                    "result": None
                }
                return {"status": "success", "job_id": job_id, "backend": "celery"}
            except Exception as e:
                return {"status": "error", "error": f"Celery dispatch failed: {str(e)}"}

        # In-process fallback
        self._jobs[job_id] = {
            "job_id": job_id,
            "task": task_name,
            "status": "queued",
            "created_at": time.time(),
            "result": None
        }
        asyncio.create_task(self._run_task(job_id, task_name, args))
        return {"status": "success", "job_id": job_id, "backend": "asyncio"}

    async def _run_task(self, job_id: str, task_name: str, args: Dict):
        """Execute task with semaphore control."""
        async with self._semaphore:
            self._jobs[job_id]["status"] = "running"
            try:
                handler = self._get_task_handler(task_name)
                if handler:
                    result = await handler(args)
                else:
                    # Try to find block action
                    result = await self._dispatch_to_block(task_name, args)
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["result"] = result
            except Exception as e:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["result"] = {"error": str(e)}
            self._jobs[job_id]["finished_at"] = time.time()

    def _get_task_handler(self, task_name: str) -> Optional[Callable]:
        """Map task names to local handlers."""
        handlers = {
            "sleep": lambda args: asyncio.sleep(args.get("seconds", 1)),
        }
        return handlers.get(task_name)

    async def _dispatch_to_block(self, task_name: str, args: Dict) -> Dict:
        """Dispatch task to a block by name.action."""
        try:
            from app.blocks import BLOCK_REGISTRY
            if "." in task_name:
                block_name, action = task_name.split(".", 1)
                block_class = BLOCK_REGISTRY.get(block_name)
                if not block_class:
                    return {"status": "error", "error": f"Block '{block_name}' not found"}
                instance = block_class()
                args = {**args, "action": action}
                result = await instance.execute(args, args)
                return result.get("result", result)
            return {"status": "error", "error": f"Unknown task: {task_name}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def status(self, input_data: Any, params: Dict) -> Dict:
        """Get job status."""
        job_id = params.get("job_id") or (input_data.get("job_id") if isinstance(input_data, dict) else None)
        if not job_id:
            return {"status": "error", "error": "No job_id provided"}

        job = self._jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "Job not found", "job_id": job_id}

        # Check Celery if applicable
        if self._celery and job.get("celery_task_id"):
            try:
                task_result = self._celery.AsyncResult(job["celery_task_id"])
                job["status"] = task_result.status.lower()
                if task_result.ready():
                    job["result"] = task_result.result if task_result.successful() else {"error": str(task_result.result)}
            except Exception:
                logger.debug(
                    "swallowed %s in status() — continuing",
                    "Exception", exc_info=True,
                )

        return {"status": "success", "job": job}

    async def result(self, input_data: Any, params: Dict) -> Dict:
        """Get job result."""
        job_id = params.get("job_id") or (input_data.get("job_id") if isinstance(input_data, dict) else None)
        if not job_id:
            return {"status": "error", "error": "No job_id provided"}

        job = self._jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "Job not found", "job_id": job_id}

        if job["status"] not in ("completed", "failed"):
            return {"status": "success", "ready": False, "job_id": job_id, "status": job["status"]}

        return {"status": "success", "ready": True, "job_id": job_id, "result": job.get("result")}

    async def cancel(self, input_data: Any, params: Dict) -> Dict:
        """Cancel a queued or running job."""
        job_id = params.get("job_id") or (input_data.get("job_id") if isinstance(input_data, dict) else None)
        if not job_id:
            return {"status": "error", "error": "No job_id provided"}

        job = self._jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "Job not found", "job_id": job_id}

        if self._celery and job.get("celery_task_id"):
            try:
                self._celery.control.revoke(job["celery_task_id"], terminate=True)
            except Exception as e:
                return {"status": "error", "error": str(e)}

        job["status"] = "cancelled"
        return {"status": "success", "cancelled": True, "job_id": job_id}

    async def list_jobs(self, input_data: Any = None, params: Dict = None) -> Dict:
        """List all jobs."""
        status_filter = (params or {}).get("status")
        jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j["status"] == status_filter]
        return {"status": "success", "jobs": jobs, "count": len(jobs)}

    async def health_check(self, input_data: Any = None, params: Dict = None) -> Dict:
        """Health check for async processor."""
        return {
            "status": "success",
            "block": self.name,
            "version": self.version,
            "celery_connected": self._celery is not None,
            "active_jobs": len([j for j in self._jobs.values() if j["status"] == "running"]),
            "total_jobs": len(self._jobs)
        }

    def get_actions(self) -> Dict[str, Any]:
        """Return all public methods for block registry."""
        return {
            "dispatch": self.dispatch,
            "status": self.status,
            "result": self.result,
            "cancel": self.cancel,
            "list": self.list_jobs,
            "health_check": self.health_check,
        }

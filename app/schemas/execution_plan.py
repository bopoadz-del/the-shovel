"""Execution-plan schemas — Reasoning Engine Plan 5.

An ExecutionPlan is the reasoner's PLAN-phase output: an ordered list of
typed steps the PlanExecutor knows how to run.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One executable step.

    `type` selects a handler in PlanExecutor (compute_cpm, resource_histogram,
    gantt, compress, generate_code). `args` carries step-specific parameters.
    `output_key` names where the result lands in session.data — defaults to a
    handler-specific key when blank.
    """
    type: str = Field(min_length=1)
    args: Dict[str, Any] = Field(default_factory=dict)
    output_key: str = ""
    description: str = ""


class ExecutionPlan(BaseModel):
    """The reasoner's plan for one user request."""
    understanding: str = ""        # the UNDERSTAND-phase restatement
    steps: List[PlanStep] = Field(default_factory=list)


class StepResult(BaseModel):
    """Outcome of running one PlanStep."""
    type: str
    output_key: str
    status: str                    # 'success' | 'error'
    error: str = ""
    output: Any = None             # this step's computed value (or a compact
                                   # summary of it when large)


class PlanRunResult(BaseModel):
    """Outcome of running a whole ExecutionPlan."""
    status: str                    # 'success' | 'partial' | 'error'
    step_results: List[StepResult] = Field(default_factory=list)

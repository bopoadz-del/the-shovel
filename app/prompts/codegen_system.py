"""Code-generation system prompt — Reasoning Engine Plan 4.

Pure string building. No AI, no I/O — unit-testable without an API key.
"""

from typing import Any, Dict, Optional

_CONTRACT = """\
You are a Python code generator for a construction-project intelligence \
platform. Write a SHORT Python snippet that solves the described task.

HARD RULES:
- Assign the final answer to a variable named `result`.
- Use ONLY the input variables listed below — they are already in scope.
- You MAY import ONLY from these modules: `math`, `statistics`, `datetime`, \
`json`, `numpy`, `sympy`, `pint`, `app.lib.pm_computations`, `app.schemas.cpm`.
- Do NOT read files, make network calls, or import anything else.
  - `numpy` is allowed for array/numeric/vector calculations.
  - `sympy` is allowed for auditable symbolic formulas.
  - `pint` is allowed for unit-aware calculations.
  - `scipy` is NOT allowed.
- Return ONLY a single fenced ```python code block — no prose.

PROJECT-MANAGEMENT LIBRARY — `app.lib.pm_computations`:
Its functions take and return Pydantic models from `app.schemas.cpm`. Read
result fields with ATTRIBUTE access (`out.project_duration`), never `out[...]`.

- compute_cpm(data: CPMInput) -> CPMOutput
- resource_histogram(results, activities, period_unit="week") -> ResourceHistogram
- gantt_data(results: list[CPMResult]) -> list[GanttBar]
- compress_schedule(data: CPMInput, reductions: dict[str, int]) -> CPMOutput

Models in `app.schemas.cpm`:
- Activity(id: str, duration: int, name="", predecessors: list[Dependency]=[])
- Dependency(predecessor_id: str, type="FS", lag=0)
- CPMInput(activities: list[Activity], project_start=None)
- CPMOutput fields: .project_duration (int), .critical_path (list[str]),
  .results (list[CPMResult], each with .id .duration .total_float .is_critical)

Example — duration of activities A(3d) -> B(5d) -> C(2d) in series:
```python
from app.lib.pm_computations import compute_cpm
from app.schemas.cpm import CPMInput, Activity, Dependency
data = CPMInput(activities=[
    Activity(id="A", duration=3),
    Activity(id="B", duration=5, predecessors=[Dependency(predecessor_id="A")]),
    Activity(id="C", duration=2, predecessors=[Dependency(predecessor_id="B")]),
])
result = compute_cpm(data).project_duration
```
"""


def build_codegen_prompt(
    task: str,
    variables: Dict[str, Any],
    *,
    prior_code: Optional[str] = None,
    prior_error: Optional[str] = None,
) -> str:
    """Assemble the system prompt for one code-generation attempt.

    `prior_code` / `prior_error` are supplied on a retry so the LLM can see
    what failed and why; they are omitted on the first attempt.
    """
    if variables:
        var_lines = "\n".join(
            f"  - {k} = {v!r}" for k, v in variables.items()
        )
        var_block = f"INPUT VARIABLES (already in scope):\n{var_lines}"
    else:
        var_block = "INPUT VARIABLES: none."

    parts = [_CONTRACT, f"TASK:\n{task}", var_block]

    if prior_code is not None and prior_error is not None:
        parts.append(
            "YOUR PREVIOUS ATTEMPT FAILED. Fix it.\n"
            f"Previous code:\n```python\n{prior_code}\n```\n"
            f"Error:\n{prior_error}"
        )

    return "\n\n".join(parts)

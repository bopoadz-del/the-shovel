"""Artifact contract — Roadmap V2 · Epic 4 (conversational UI).

A reply is prose; anything inspectable it produces is an *artifact* attached to
the reply and rendered in the side panel (Claude-style) — on demand, never
auto-filled.

Artifact shape: {"type": <one of ARTIFACT_TYPES>, "title": str, "payload": ...}
"""

import json
from typing import Any, Dict, List, Tuple

ARTIFACT_TYPES = {"text", "code", "table", "schedule", "file", "link"}


def make_artifact(type_: str, title: str, payload: Any) -> Dict[str, Any]:
    if type_ not in ARTIFACT_TYPES:
        raise ValueError(
            f"Unknown artifact type '{type_}'. Allowed: {sorted(ARTIFACT_TYPES)}"
        )
    return {"type": type_, "title": title, "payload": payload}


def text_artifact(title: str, text: str) -> Dict[str, Any]:
    return make_artifact("text", title, {"text": str(text)})


def code_artifact(title: str, code: str, language: str = "text") -> Dict[str, Any]:
    return make_artifact("code", title, {"code": str(code), "language": language})


def table_artifact(title: str, columns: List[Any], rows: List[List[Any]]) -> Dict[str, Any]:
    return make_artifact("table", title, {
        "columns": list(columns),
        "rows": [list(r) for r in rows],
    })


def link_artifact(title: str, links: List[Dict[str, str]]) -> Dict[str, Any]:
    return make_artifact("link", title, {"links": links})


def file_artifact(title: str, files: List[Dict[str, Any]]) -> Dict[str, Any]:
    return make_artifact("file", title, {"files": files})


def _rows_from_dicts(items: List[Dict]) -> Tuple[List[str], List[List[Any]]]:
    """Turn a list of flat dicts into (columns, rows) for a table artifact."""
    columns: List[str] = []
    for it in items:
        for k in it:
            if k not in columns:
                columns.append(k)
    rows = [[it.get(c, "") for c in columns] for it in items]
    return columns, rows


def _stringify(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, default=str)[:4000]
    except Exception:
        return str(data)[:4000]


def result_to_artifacts(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort: derive artifacts from a block's inner result dict.

    The panel stays empty unless the result actually carries something
    inspectable — this never fabricates content.
    """
    artifacts: List[Dict[str, Any]] = []
    if not isinstance(result, dict):
        return artifacts

    # generated code
    code = result.get("generated_code")
    if isinstance(code, str) and code.strip():
        artifacts.append(code_artifact("Generated code", code, "python"))

    # panels → tables / text
    for panel in result.get("panels", []) or []:
        if not isinstance(panel, dict):
            continue
        title = panel.get("title") or panel.get("type") or "Panel"
        data = panel.get("data")
        items = panel.get("line_items")
        if items is None and isinstance(data, dict):
            items = data.get("items")
        if isinstance(items, list) and items and all(isinstance(i, dict) for i in items):
            cols, rows = _rows_from_dicts(items)
            artifacts.append(table_artifact(title, cols, rows))
        elif data:
            artifacts.append(text_artifact(title, _stringify(data)))

    # explicit tables
    for i, tbl in enumerate(result.get("tables", []) or []):
        if isinstance(tbl, dict) and tbl.get("rows"):
            artifacts.append(table_artifact(
                tbl.get("title", f"Table {i + 1}"),
                tbl.get("columns", []), tbl.get("rows", []),
            ))

    # files
    files = result.get("files")
    if isinstance(files, list) and files:
        norm = [f if isinstance(f, dict) else {"name": str(f)} for f in files]
        artifacts.append(file_artifact("Files", norm))
    elif result.get("file_path"):
        artifacts.append(file_artifact("File", [{"name": result["file_path"]}]))

    # links
    links = result.get("links")
    if isinstance(links, list) and links:
        norm = [
            l if isinstance(l, dict) else {"label": str(l), "url": str(l)}
            for l in links
        ]
        artifacts.append(link_artifact("Links", norm))

    # extracted text — only when nothing richer was found
    text = result.get("text") or result.get("extracted_text")
    if isinstance(text, str) and text.strip() and not artifacts:
        artifacts.append(text_artifact("Extracted text", text[:4000]))

    return artifacts

"""Stable file sharding for multi-worker ingestion.

Workers set INGEST_TOTAL_SHARDS / INGEST_SHARD_INDEX (or CLI equivalents).
Assignment uses hashlib.sha256 — never Python's randomized hash().
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def file_identity(
    *,
    drive_file_id: Optional[str] = None,
    source_path: Optional[str] = None,
    document_id: Optional[str] = None,
) -> str:
    """Preferred identity: Drive id → normalized path → document id."""
    if drive_file_id:
        return str(drive_file_id).strip()
    if source_path:
        return str(source_path).replace("\\", "/").strip().lower()
    if document_id:
        return str(document_id).strip()
    raise ValueError("file_identity requires drive_file_id, source_path, or document_id")


def stable_shard(identity: str, total_shards: int) -> int:
    """Return shard index in [0, total_shards) for a stable identity string."""
    if total_shards < 1:
        raise ValueError(f"total_shards must be >= 1, got {total_shards}")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return int(digest, 16) % total_shards


def belongs_to_shard(identity: str, shard_index: int, total_shards: int) -> bool:
    validate_shard_params(shard_index, total_shards)
    return stable_shard(identity, total_shards) == shard_index


def validate_shard_params(shard_index: int, total_shards: int) -> None:
    if total_shards < 1:
        raise ValueError(f"total_shards must be >= 1, got {total_shards}")
    if shard_index < 0 or shard_index >= total_shards:
        raise ValueError(
            f"shard_index must be in [0, {total_shards}), got {shard_index}"
        )


def resolve_shard_env(
    *,
    total_shards: Optional[int] = None,
    shard_index: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve CLI args with env fallbacks (INGEST_* preferred, P1B_SHARD_* legacy)."""
    if total_shards is None:
        raw = os.getenv("INGEST_TOTAL_SHARDS") or os.getenv("P1B_SHARD_COUNT") or "1"
        total_shards = int(raw)
    if shard_index is None:
        raw = os.getenv("INGEST_SHARD_INDEX") or os.getenv("P1B_SHARD_INDEX") or "0"
        shard_index = int(raw)
    validate_shard_params(shard_index, total_shards)
    return total_shards, shard_index


def resolve_dry_run(cli_flag: bool = False) -> bool:
    if cli_flag:
        return True
    raw = (os.getenv("INGEST_DRY_RUN") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_limit_offset(
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Tuple[int, int]:
    """Resolve optional limit/offset; 0 limit means no cap."""
    if limit is None:
        limit = int(os.getenv("INGEST_LIMIT") or "0")
    if offset is None:
        offset = int(os.getenv("INGEST_OFFSET") or "0")
    return max(0, limit), max(0, offset)


def filter_by_shard(
    items: Sequence[Any],
    *,
    identity_fn,
    shard_index: int,
    total_shards: int,
) -> List[Any]:
    """Keep items whose stable identity maps to this shard."""
    validate_shard_params(shard_index, total_shards)
    if total_shards == 1:
        return list(items)
    return [
        item for item in items
        if belongs_to_shard(identity_fn(item), shard_index, total_shards)
    ]


def apply_offset_limit(
    items: Sequence[Any],
    *,
    offset: int = 0,
    limit: int = 0,
) -> List[Any]:
    """Apply offset/limit AFTER shard filtering so assignment stays stable."""
    offset = max(0, offset)
    sliced = list(items[offset:])
    if limit and limit > 0:
        return sliced[:limit]
    return sliced


def extension_breakdown(paths: Iterable[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for p in paths:
        ext = Path(p).suffix.lower() or "(none)"
        counts[ext] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def shard_distribution(
    identities: Sequence[str],
    total_shards: int,
) -> List[int]:
    """Count how many identities land in each shard [0..total_shards)."""
    validate_shard_params(0, total_shards)
    dist = [0] * total_shards
    for ident in identities:
        dist[stable_shard(ident, total_shards)] += 1
    return dist


def dry_run_report(
    *,
    total_discovered: int,
    supported_items: Sequence[Any],
    unsupported_count: int,
    shard_index: int,
    total_shards: int,
    identity_fn,
    path_fn,
    already_indexed_count: int = 0,
) -> Dict[str, Any]:
    """Build the dry-run summary dict (no DB side effects)."""
    validate_shard_params(shard_index, total_shards)
    identities = [identity_fn(item) for item in supported_items]
    assigned = [
        item for item in supported_items
        if belongs_to_shard(identity_fn(item), shard_index, total_shards)
    ]
    assigned_paths = [path_fn(item) for item in assigned]
    assigned_ids = [identity_fn(item) for item in assigned]
    return {
        "total_files_discovered": total_discovered,
        "total_supported_files": len(supported_items),
        "total_skipped_unsupported": unsupported_count,
        "already_indexed_skipped": already_indexed_count,
        "total_shards": total_shards,
        "shard_index": shard_index,
        "files_assigned_to_this_shard": len(assigned),
        "extension_breakdown": extension_breakdown(assigned_paths),
        "first_20_identities": assigned_ids[:20],
        "per_shard_counts": shard_distribution(identities, total_shards),
        "dry_run": True,
        "db_writes": False,
    }


def shard_manifest_path(shard_index: int, total_shards: int) -> Path:
    return Path(f"manifests/ingest_shard_{shard_index}_of_{total_shards}.json")


def empty_shard_manifest(
    *,
    shard_index: int,
    total_shards: int,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "shard_index": shard_index,
        "total_shards": total_shards,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "unsupported": 0,
        "already_indexed": 0,
        "durations_sec": {},
        "results": [],
    }
    if extra:
        base.update(dict(extra))
    return base

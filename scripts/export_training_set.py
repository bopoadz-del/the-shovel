#!/usr/bin/env python3
"""Export training data for the LoRA fine-tune (PR 3a).

Pulls (user_message, assistant_response) pairs from ``agent_memory.db``
and writes a JSONL file ready for ``scripts/finetune_router.py``. Also
accepts an optional ``--seed-data`` JSONL the operator can prepend
(curated construction Q&A pairs from a domain expert).

Output schema (one row per line):

    {"instruction": "user question text", "response": "assistant answer text",
     "source": "agent_memory" | "seed_file", "conversation_id": "..."}

No GPU, no ML dependencies — pure SQLite + JSON. Fully testable on this
container; ``tests/test_export_training_set.py`` covers it.

CLI:
    python scripts/export_training_set.py \\
        --out data/learning/training_set.jsonl \\
        [--seed-data path/to/seed.jsonl] \\
        [--min-message-chars 20] \\
        [--max-message-chars 2000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Iterator, Optional

# Resolve the `app` package when invoked directly (not via `python -m`).
# Adding the repo root to sys.path lets `from app.core import agent_memory`
# work regardless of how the script is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def collect_from_agent_memory(
    min_chars: int = 20,
    max_chars: int = 2000,
) -> Iterator[dict]:
    """Yield (user_message, assistant_response) pairs from agent_memory.

    Pairs are formed by walking each conversation's messages in order
    and joining a user message with the immediately-following assistant
    response. Short or excessively-long messages are filtered — fine-tune
    examples below 20 chars are usually meaningless ("ok", "thanks") and
    above 2000 chars usually mean a paste of upstream context that won't
    generalize. Both bounds are CLI-tunable.
    """
    from app.core import agent_memory

    agent_memory._ensure_db()
    convs = agent_memory.list_conversations()
    for conv in convs:
        messages = agent_memory.get_messages(conv["id"])
        last_user: Optional[dict] = None
        for msg in messages:
            role = (msg.get("role") or "").lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                last_user = msg if min_chars <= len(content) <= max_chars else None
            elif role == "assistant" and last_user is not None:
                if min_chars <= len(content) <= max_chars:
                    yield {
                        "instruction": (last_user.get("content") or "").strip(),
                        "response": content,
                        "source": "agent_memory",
                        "conversation_id": conv["id"],
                    }
                last_user = None  # consume the pair regardless


def collect_from_seed_file(path: str) -> Iterator[dict]:
    """Read a JSONL file of pre-curated training rows.

    Expected schema per row: ``{"instruction": "...", "response": "..."}``.
    Anything else is preserved verbatim (in case the operator wants to
    add metadata like ``topic``, ``expert``, etc.).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"seed file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("seed line %d malformed, skipping: %s", i, exc)
                continue
            if not row.get("instruction") or not row.get("response"):
                logger.warning(
                    "seed line %d missing instruction/response, skipping", i
                )
                continue
            row.setdefault("source", "seed_file")
            yield row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.getenv("DATA_DIR", "./data"), "learning", "training_set.jsonl"
        ),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--seed-data",
        default=None,
        help="Optional path to a JSONL of curated Q&A pairs to prepend",
    )
    parser.add_argument(
        "--min-message-chars", type=int, default=20,
        help="Skip messages shorter than this (default: 20)",
    )
    parser.add_argument(
        "--max-message-chars", type=int, default=2000,
        help="Skip messages longer than this (default: 2000)",
    )
    parser.add_argument(
        "--no-agent-memory", action="store_true",
        help="Skip the agent_memory.db source (seed file only)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_seed = 0
    n_memory = 0
    with open(args.out, "w", encoding="utf-8") as out:
        if args.seed_data:
            for row in collect_from_seed_file(args.seed_data):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_seed += 1
        if not args.no_agent_memory:
            for row in collect_from_agent_memory(
                min_chars=args.min_message_chars,
                max_chars=args.max_message_chars,
            ):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_memory += 1

    logger.info(
        "wrote %s rows to %s (seed=%d, agent_memory=%d)",
        n_seed + n_memory, args.out, n_seed, n_memory,
    )
    if (n_seed + n_memory) == 0:
        logger.warning(
            "ZERO rows exported. Either provide --seed-data or accumulate "
            "chat history first."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

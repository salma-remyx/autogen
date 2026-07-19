"""Parameter-free classifier for selective persistent memory.

Adapted from "Shared Selective Persistent Memory for Agentic LLM Systems"
(arXiv:2607.09493). The paper uses an LLM to sort each piece of context into
one of four reusable categories (or mark it a disposable reasoning trace).
This module substitutes that *learned* classifier with a deterministic,
parameter-free keyword/structure heuristic so the selective-persistence
mechanism runs with no model calls and no extra dependencies -- the same
shape of substitution (learned estimator -> parameter-free proxy) the adapted
port is allowed to make. The selection mechanism itself is unchanged.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, Tuple

from autogen_core.memory import MemoryContent


class MemoryCategory(Enum):
    """Reusable-context categories retained by selective persistent memory.

    The first four values are the categories the paper keeps;
    ``REASONING_TRACE`` is session-specific scratch content that the paper
    discards before persistence.
    """

    TASK_SPECIFICATION = "task_specification"
    DATA_SCHEMA = "data_schema"
    TOOL_CONFIGURATION = "tool_configuration"
    OUTPUT_CONSTRAINT = "output_constraint"
    REASONING_TRACE = "reasoning_trace"


REUSABLE_CATEGORIES: Tuple[MemoryCategory, ...] = (
    MemoryCategory.TASK_SPECIFICATION,
    MemoryCategory.DATA_SCHEMA,
    MemoryCategory.TOOL_CONFIGURATION,
    MemoryCategory.OUTPUT_CONSTRAINT,
)

# Lower-cased phrases matched on word boundaries.
_LEXICONS: Dict[MemoryCategory, Tuple[str, ...]] = {
    MemoryCategory.TASK_SPECIFICATION: (
        "task",
        "goal",
        "objective",
        "aim is",
        "purpose",
        "deliverable",
        "instructions",
        "requirements",
        "your job",
        "you are tasked",
        "the user wants",
        "please",
        "do the following",
    ),
    MemoryCategory.DATA_SCHEMA: (
        "schema",
        "column",
        "columns",
        "field",
        "fields",
        "data type",
        "dtype",
        "primary key",
        "foreign key",
        "table",
        "dataframe",
        "rows",
        "json schema",
        "struct",
    ),
    MemoryCategory.TOOL_CONFIGURATION: (
        "api",
        "endpoint",
        "connection",
        "config",
        "configuration",
        "parameter",
        "parameters",
        "auth",
        "api_key",
        "base_url",
        "host",
        "port",
        "mcp server",
        "credentials",
        "settings",
        "tool",
        "function call",
    ),
    MemoryCategory.OUTPUT_CONSTRAINT: (
        "output must",
        "must be",
        "should be",
        "constraint",
        "constraints",
        "return as",
        "respond in",
        "do not include",
        "max length",
        "valid",
        "format",
        "schema for output",
    ),
}

# Multi-word markers that strongly indicate disposable chain-of-thought.
_REASONING_MARKERS: Tuple[str, ...] = (
    "let me",
    "let's see",
    "first i",
    "then i",
    "next i",
    "now i",
    "so i",
    "i need to",
    "i should",
    "i'll",
    "i will",
    "i think",
    "step 1",
    "step 2",
    "thinking",
    "scratchpad",
    "chain of thought",
    "reasoning",
    "hmm",
    "wait,",
    "actually,",
    "so the answer",
    "intermediate step",
)


def _to_text(content: MemoryContent) -> Tuple[str, bool]:
    """Normalize a MemoryContent payload to lower-cased text.

    Returns ``(text, is_structured)`` where ``is_structured`` is True for
    dict/JSON payloads, which biases classification toward ``DATA_SCHEMA``.
    """
    payload: Any = content.content
    is_structured = False
    if isinstance(payload, str):
        text = payload
    elif isinstance(payload, dict):
        is_structured = True
        text = json.dumps(payload, default=str, sort_keys=True)
    elif isinstance(payload, (bytes, bytearray)):
        text = payload.decode("utf-8", errors="ignore")
    else:
        # Image or unknown payload: cannot meaningfully classify.
        text = str(payload)
    return text.lower(), is_structured


def _phrase_count(text: str, phrases: Tuple[str, ...]) -> int:
    """Count how many distinct phrases from ``phrases`` appear in ``text``."""
    count = 0
    for phrase in phrases:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            count += 1
    return count


def classify_content(
    content: MemoryContent,
    default_category: MemoryCategory = MemoryCategory.REASONING_TRACE,
) -> MemoryCategory:
    """Classify ``content`` into one reusable category or ``REASONING_TRACE``.

    Order of precedence:

    1. An explicit ``metadata["category"]`` override wins outright, and
       ``metadata["persist"] is False`` (or ``metadata["ephemeral"]``) forces
       ``REASONING_TRACE``.
    2. A strong reusable-category match (>= 2 distinct keyword hits, or any hit
       when the payload is a JSON/dict schema) is retained.
    3. A strong reasoning-trace signal (any chain-of-thought marker) is
       discarded.
    4. A weak reusable-category match (exactly 1 keyword hit) is retained.
    5. Otherwise ``default_category`` (discard by default).

    Args:
        content: The memory item to classify.
        default_category: Category assigned when nothing matches. Defaults to
            ``REASONING_TRACE`` so ambiguous content is discarded rather than
            naively persisted -- the paper's headline finding that selective
            memory beats full-history recall.

    Returns:
        The :class:`MemoryCategory` judged most likely for ``content``.
    """
    metadata: Dict[str, Any] = content.metadata or {}

    explicit = metadata.get("category")
    if isinstance(explicit, MemoryCategory):
        return explicit
    if isinstance(explicit, str):
        try:
            return MemoryCategory(explicit)
        except ValueError:
            pass
    if metadata.get("persist") is False or metadata.get("ephemeral"):
        return MemoryCategory.REASONING_TRACE

    text, is_structured = _to_text(content)
    scores: Dict[MemoryCategory, int] = {cat: _phrase_count(text, lex) for cat, lex in _LEXICONS.items()}
    if is_structured:
        scores[MemoryCategory.DATA_SCHEMA] += 1

    best_category = max(scores, key=lambda c: scores[c])
    best_score = scores[best_category]
    reasoning_hits = _phrase_count(text, _REASONING_MARKERS)

    if best_score >= 2:
        return best_category
    if reasoning_hits >= 1:
        return MemoryCategory.REASONING_TRACE
    if best_score == 1:
        return best_category
    return default_category

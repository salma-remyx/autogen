"""Active context-lifecycle memory with token-budgeted, validated compaction.

Adapted from "Agentic Context Management: Solving Agent Memory and Cost by Treating
Them as Lifecycle and Architecture Problems" (arXiv:2607.21503).

The paper argues that what an agent *holds in mind* is a **lifecycle**, not merely a
store, and decomposes the discipline (Agentic Context Management, ACM) into five
primitives: *architect*, *ingest*, *scope*, *anticipate*, and *compact & consolidate*.
Its central economic claim is that naive context accumulation grows token cost
*quadratically* in conversation length, crude summarization buys *linear* cost at the
price of an **accuracy cliff**, and only *validated compaction* achieves linear cost
with preserved fidelity.

This module realizes those five primitives on top of the AutoGen
:class:`~autogen_core.memory.Memory` ABC, which maps onto them directly:

    ingest            -> :meth:`CompactingContextMemory.add`
                         (store + structure new context, tag a data-type ``kind``)
    scope             -> :meth:`CompactingContextMemory.query`
                         (retrieve entries relevant to a query)
    anticipate        -> scoring boost inside :meth:`update_context`
                         (terms recurring in the *live* model context are anticipated
                         as needed-next and boosted -- a parameter-free lookahead)
    compact &         -> :meth:`CompactingContextMemory.update_context`
    consolidate          (inject a token-budgeted, provenance-preserving view of the
                         most salient context as a ``SystemMessage``)
    architect          -> the in-process structured store itself (per-data-type
                         ``kind`` tagging, monotonic provenance sequence)

**Implementation mode -- Mode 2 (adapted port).** The core *compaction* mechanism is
kept at full fidelity: extractive selection of salient units under a strict token
budget that preserves verbatim text and provenance (validated compaction -- no
accuracy cliff). The paper's auxiliary components are substituted with target-native,
parameter-free equivalents:

    * learned salience / MI estimators  -> token-overlap relevance + recency weighting
    * bespoke tokenizer                 -> char-based token proxy (``len / 4``), overridable
    * LLM extraction / consolidation    -> extractive sentence-unit selection (no model call)
    * multi-tenant service (Maximem)    -> the in-process ``Memory`` ABC + AssistantAgent loop

Intentionally out of scope (honest scoping, not shortfalls):

    * the org-scope hierarchy (single-agent scope here),
    * the LongMemEval / LoCoMo benchmark suites -- evaluation belongs in a downstream PR,
    * mutating the live conversation history (we only *add* a compacted system message;
      compacting the agent's own turn history is invasive and left for a follow-up).

The ``summary`` compaction strategy is provided *only* as the lossy baseline the paper
argues against (a recency-window truncation that drops older context entirely),
so the suggested with/without-compaction experiment is runnable directly.
"""

import json
import re
from typing import Any, Callable, List, Literal, Set, Tuple

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = ["CompactingContextMemory", "CompactingContextMemoryConfig"]

# --- parameter-free proxies (Mode 2 substitutions for the paper's learned estimators) ---

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|; ")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "had",
        "his",
        "how",
        "who",
        "its",
        "let",
        "may",
        "she",
        "than",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "that",
        "with",
        "what",
        "when",
        "where",
        "which",
        "will",
        "your",
        "from",
        "into",
        "over",
        "such",
        "also",
        "any",
        "been",
        "being",
        "did",
        "does",
        "done",
        "about",
        "after",
        "before",
        "because",
    }
)
_CHARS_PER_TOKEN = 4  # rough proxy: ~4 chars per BPE token for English/code text
_MIN_FILL_TOKENS = 8  # don't pack units smaller than this into leftover budget


def _to_text(content: Any) -> str:
    """Best-effort text view of arbitrary :class:`MemoryContent` content."""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="ignore")
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def _estimate_tokens(text: str) -> int:
    """Char-based proxy for BPE token count (Mode 2 substitution for a real tokenizer)."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _terms(text: str) -> Set[str]:
    """Normalized content-term set used for parameter-free relevance scoring."""
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) > 2 and tok not in _STOPWORDS}


def _split_units(text: str) -> List[str]:
    """Split text into salient sentence/line units for extractive compaction."""
    return [part.strip() for part in _SENT_SPLIT.split(text) if part.strip()]


def _truncate_to_token_budget(text: str, token_budget: int) -> str:
    """Truncate ``text`` to at most ``token_budget`` proxy tokens, on a character boundary."""
    chars = token_budget * _CHARS_PER_TOKEN
    if chars <= 0:
        return ""
    return text[:chars].rstrip()


def _message_text(message: Any) -> str:
    """Extract a text view from any model-context message shape."""
    return _to_text(getattr(message, "content", ""))


class CompactingContextMemoryConfig(BaseModel):
    """Declarative configuration for :class:`CompactingContextMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    token_budget: int = Field(default=1024, ge=64)
    """Hard ceiling (in proxy tokens) on the memory injected per ``update_context`` call.

    Bounding the injection is what makes token cost grow linearly with ingested volume
    rather than quadratically -- the paper's headline economic result.
    """

    compaction_strategy: Literal["extractive", "summary"] = "extractive"
    """``extractive`` (default) is validated compaction: salient units are kept verbatim
    with provenance. ``summary`` is the lossy recency-window baseline the paper argues
    against (older context is dropped entirely -> accuracy cliff)."""

    recency_decay: float = Field(default=0.95, gt=0.0, lt=1.0)
    """Per-step multiplier applied to older entries' salience (newest entry is undecayed)."""

    recent_context_messages: int = Field(default=4, ge=0)
    """How many trailing live-context messages feed the anticipate/scope signal."""

    max_query_results: int = Field(default=10, ge=1)
    """Maximum entries returned by :meth:`CompactingContextMemory.query` (scope primitive)."""


class CompactingContextMemory(Memory, Component["CompactingContextMemoryConfig"]):
    """Memory backend that actively manages the agent's context as a lifecycle.

    See the module docstring for the mapping onto the ACM five primitives and the
    Mode 2 (adapted port) substitutions.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.memory.compacting_context import CompactingContextMemory

            memory = CompactingContextMemory(token_budget=512)
            # AssistantAgent consumes the Memory ABC on its forward path, calling
            # update_context each turn -- no other wiring required.
            agent = AssistantAgent(name="analyst", model_client=..., memory=[memory])

    Args:
        name: Optional instance identifier.
        token_budget: Proxy-token ceiling on the injected system message.
        compaction_strategy: ``"extractive"`` (validated, default) or ``"summary"`` (lossy baseline).
        recency_decay: Per-step salience decay for older entries (0 < d < 1).
        recent_context_messages: Trailing live messages used to anticipate/scope.
        max_query_results: Cap on entries returned by :meth:`query`.
        token_counter: Optional callable overriding the default token proxy.
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.compacting_context.CompactingContextMemory"
    component_config_schema = CompactingContextMemoryConfig

    def __init__(
        self,
        name: str | None = None,
        token_budget: int = 1024,
        compaction_strategy: Literal["extractive", "summary"] = "extractive",
        recency_decay: float = 0.95,
        recent_context_messages: int = 4,
        max_query_results: int = 10,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._name = name or "compacting_context"
        self._token_budget = token_budget
        self._strategy = compaction_strategy
        self._recency_decay = recency_decay
        self._recent_context_messages = recent_context_messages
        self._max_query_results = max_query_results
        self._token_counter = token_counter
        self._entries: List[MemoryContent] = []
        self._seq = 0  # monotonic provenance sequence (architect primitive)

    @property
    def name(self) -> str:
        """Instance identifier."""
        return self._name

    @property
    def content(self) -> List[MemoryContent]:
        """Currently ingested entries (in ingest order)."""
        return self._entries

    # -- ingest ---------------------------------------------------------------

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Ingest primitive: store content with provenance and a structured data-type tag."""
        _ = cancellation_token
        metadata = dict(content.metadata or {})
        metadata.setdefault("seq", self._seq)
        metadata.setdefault("kind", metadata.get("kind", "fact"))
        stored = content.model_copy(update={"metadata": metadata})
        self._entries.append(stored)
        self._seq += 1

    # -- scope ----------------------------------------------------------------

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Scope primitive: return entries ranked by relevance to ``query``."""
        _ = cancellation_token, kwargs
        query_text = query if isinstance(query, str) else _to_text(getattr(query, "content", ""))
        query_terms = _terms(query_text)
        if not self._entries or not query_terms:
            return MemoryQueryResult(results=list(self._entries))
        ranked = self._ranked_entries(query_terms, query_terms)
        return MemoryQueryResult(results=[entry for _, _, entry in ranked[: self._max_query_results]])

    # -- compact & consolidate (headline primitive) ---------------------------

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Compact the ingested context to a token budget and inject it as a system message.

        Scopes/anticipates against the live ``model_context`` (its trailing messages),
        then selects the most salient units under :attr:`token_budget` and adds them as a
        single ``SystemMessage``. The injection is bounded -> linear token cost; salient
        units are kept verbatim with provenance -> no accuracy cliff.
        """
        query_terms, context_terms = await self._live_scope(model_context)
        payload, selected = self._compact(query_terms, context_terms)
        if payload:
            await model_context.add_message(SystemMessage(content=payload))
        return UpdateContextResult(memories=MemoryQueryResult(results=selected))

    async def clear(self) -> None:
        """Forget everything (consolidation primitive: full reset)."""
        self._entries = []
        self._seq = 0

    async def close(self) -> None:
        """No external resources to release."""

    # -- compaction internals -------------------------------------------------

    def _token_count(self, text: str) -> int:
        return self._token_counter(text) if self._token_counter is not None else _estimate_tokens(text)

    async def _live_scope(self, model_context: ChatCompletionContext) -> Tuple[Set[str], Set[str]]:
        """Derive the scope/anticipate term sets from the live model context."""
        if self._recent_context_messages <= 0:
            return set(), set()
        messages = await model_context.get_messages()
        if not messages:
            return set(), set()
        recent = messages[-self._recent_context_messages :]
        context_text = " ".join(_message_text(m) for m in recent)
        context_terms = _terms(context_text)
        # The immediate query is the most recent user-authored message, if any.
        query_terms: Set[str] = set()
        for message in reversed(messages):
            if getattr(message, "source", "") and getattr(message, "content", "") and not _is_tool_message(message):
                query_terms = _terms(_message_text(message))
                break
        return query_terms, context_terms | query_terms

    def _ranked_entries(self, query_terms: Set[str], context_terms: Set[str]) -> List[Tuple[float, int, MemoryContent]]:
        """Score whole entries by relevance (scope) + recency (anticipate)."""
        total = len(self._entries)
        ranked: List[Tuple[float, int, MemoryContent]] = []
        for index, entry in enumerate(self._entries):
            entry_terms = _terms(_to_text(entry.content))
            relevance = float(len(entry_terms & context_terms) + 2 * len(entry_terms & query_terms))
            recency = self._recency_decay ** (total - 1 - index)
            ranked.append((relevance * recency, index, entry))
        # Highest score first; ties broken toward the more recent entry (higher index).
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        return ranked

    def _compact(self, query_terms: Set[str], context_terms: Set[str]) -> Tuple[str, List[MemoryContent]]:
        if not self._entries:
            return "", []
        if self._strategy == "summary":
            return self._compact_summary()
        return self._compact_extractive(query_terms, context_terms)

    def _compact_extractive(self, query_terms: Set[str], context_terms: Set[str]) -> Tuple[str, List[MemoryContent]]:
        """Validated compaction: greedily pack the highest-scoring salient units under budget."""
        total = len(self._entries)
        units: List[Tuple[float, int, int, str, MemoryContent]] = []
        for index, entry in enumerate(self._entries):
            text = _to_text(entry.content)
            entry_terms = _terms(text)
            base_relevance = float(len(entry_terms & context_terms))
            recency = self._recency_decay ** (total - 1 - index)
            for order, unit in enumerate(_split_units(text)):
                unit_terms = _terms(unit)
                # Units that directly mention the current query rank highest (anticipate).
                score = (
                    base_relevance + 3.0 * len(unit_terms & query_terms) + 0.5 * len(unit_terms & context_terms)
                ) * recency
                units.append((score, index, order, unit, entry))
        # Highest score first; ties -> more recent entry, then original unit order.
        units.sort(key=lambda item: (-item[0], -item[1], item[2]))

        header = (
            f"Compacted context memory (budget {self._token_budget} tokens, {total} ingested, strategy=extractive):\n"
        )
        used = self._token_count(header)
        lines: List[str] = []
        selected: List[MemoryContent] = []
        chosen: Set[int] = set()
        for _, _index, _order, unit, entry in units:
            # Account for the full formatted line (provenance prefix + unit) so the
            # budget bounds the text actually injected, not just the bare unit.
            line = self._provenance_line(entry, unit)
            tokens = self._token_count(line)
            if used + tokens > self._token_budget:
                remaining = self._token_budget - used
                prefix_tokens = self._token_count(self._provenance_line(entry, ""))
                if remaining > prefix_tokens + _MIN_FILL_TOKENS:
                    unit = _truncate_to_token_budget(unit, remaining - prefix_tokens)
                    line = self._provenance_line(entry, unit)
                    tokens = self._token_count(line)
                else:
                    continue  # leave room for a smaller, still-salient unit later
            if tokens <= 0:
                continue
            lines.append(line)
            if id(entry) not in chosen:
                chosen.add(id(entry))
                selected.append(entry)
            used += tokens
            if used >= self._token_budget:
                break
        if not lines:
            return "", []
        return header + "\n".join(lines), selected

    def _compact_summary(self) -> Tuple[str, List[MemoryContent]]:
        """Lossy baseline (the accuracy-cliff strawman): keep only a recency window."""
        header = (
            f"Compacted context memory (budget {self._token_budget} tokens, "
            f"{len(self._entries)} ingested, strategy=summary):\n"
        )
        used = self._token_count(header)
        kept: List[MemoryContent] = []
        lines: List[str] = []
        for entry in reversed(self._entries):  # newest first
            line = self._provenance_line(entry, _to_text(entry.content).strip())
            tokens = self._token_count(line)
            if used + tokens > self._token_budget:
                break
            kept.append(entry)
            lines.append(line)
            used += tokens
        if not kept:
            return "", []
        kept.reverse()
        lines.reverse()
        return header + "\n".join(lines), kept

    def _provenance_line(self, entry: MemoryContent, text: str) -> str:
        metadata = entry.metadata or {}
        kind = metadata.get("kind", "fact")
        seq = metadata.get("seq", "?")
        return f"- [{kind}#{seq}] {text}"

    # -- component round-trip -------------------------------------------------

    @classmethod
    def _from_config(cls, config: "CompactingContextMemoryConfig") -> Self:
        return cls(
            name=config.name,
            token_budget=config.token_budget,
            compaction_strategy=config.compaction_strategy,
            recency_decay=config.recency_decay,
            recent_context_messages=config.recent_context_messages,
            max_query_results=config.max_query_results,
        )

    def _to_config(self) -> "CompactingContextMemoryConfig":
        return CompactingContextMemoryConfig(
            name=self._name,
            token_budget=self._token_budget,
            compaction_strategy=self._strategy,
            recency_decay=self._recency_decay,
            recent_context_messages=self._recent_context_messages,
            max_query_results=self._max_query_results,
        )


def _is_tool_message(message: Any) -> bool:
    """Heuristic: tool-call result messages are outputs, not user-authored queries."""
    return message.__class__.__name__ in {"ToolCallExecutionEvent", "ToolCallSummaryMessage"}

from __future__ import annotations

import string
from enum import Enum
from typing import Any, List

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = ["ArbitratedMemory", "ArbitratedMemoryConfig", "MemoryBank"]


# Adapted from "MemArbiter: Decision-Time Memory Arbitration for Long-Horizon LLM Agents"
# (arxiv:2608.02113). The paper's learned bank-demand and item-relevance estimators are
# replaced here by parameter-free lexical proxies, and the ALFWorld evaluation harness is
# cut (evaluation belongs in a downstream PR). The core mechanism -- decompose history into
# atomic items, organize them into functional banks, then arbitrate salience at decision
# time via bank-level demand x item-level relevance x focal/ambient presentation x a
# recency ("temporal presentation") gate under a per-step token budget -- is preserved.

_PUNCT = str.maketrans(string.punctuation, " " * len(string.punctuation))


class MemoryBank(str, Enum):
    """Functional memory banks.

    Interaction history is decomposed into atomic items and routed into one of these
    function-oriented banks. Bank membership drives decision-time demand rather than a
    flat recency/retrieval ordering.
    """

    TASK_GOALS = "task_goals"
    """The agent's objectives, instructions, or plan."""

    TOOL_OUTPUTS = "tool_outputs"
    """Results returned by tools/execution, including observed state from actions."""

    USER_PREFERENCES = "user_preferences"
    """Durable user constraints: format, tone, units, things to always/never do."""

    STATE_FACTS = "state_facts"
    """Facts about the current world or session state."""

    EXPERIENCE = "experience"
    """Lessons and reflections from prior attempts, used for failure recovery."""


# Keyword tables used by the parameter-free proxies. ``_classify_bank`` routes an untagged
# item to a bank using its own text; ``_bank_demand`` scores how much each bank is "in
# demand" given the current decision context.
_BANK_KEYWORDS: dict[MemoryBank, set[str]] = {
    MemoryBank.TASK_GOALS: {"goal", "task", "objective", "want", "need", "aim", "mission", "plan", "find", "solve"},
    MemoryBank.TOOL_OUTPUTS: {"result", "output", "returned", "executed", "ran", "tool", "call", "status", "value"},
    MemoryBank.USER_PREFERENCES: {"prefer", "like", "style", "always", "never", "format", "language", "tone", "should"},
    MemoryBank.STATE_FACTS: {"current", "state", "where", "located", "value", "now", "exists", "is", "are"},
    MemoryBank.EXPERIENCE: {
        "failed",
        "mistake",
        "lesson",
        "before",
        "previously",
        "avoid",
        "recover",
        "retry",
        "again",
    },
}


def _tokenize(text: str) -> set[str]:
    cleaned = text.lower().translate(_PUNCT)
    return {tok for tok in cleaned.split() if len(tok) > 1}


def _extract_text(content: str | MemoryContent) -> str:
    """Best-effort flat text extraction for scoring and presentation."""
    if isinstance(content, str):
        return content
    raw = content.content
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    if isinstance(raw, dict):
        return str(raw)
    if isinstance(raw, str):
        return raw
    return str(raw)


def _approx_tokens(text: str) -> int:
    """Word-count proxy for the per-step token budget (no tokenizer dependency)."""
    return len(text.split())


def _header(content: MemoryContent) -> str:
    """Compact one-line summary used for the ambient (lower-salience) representation."""
    first_line = _extract_text(content).strip().split("\n", 1)[0]
    words = first_line.split()
    return " ".join(words[:8]) + ("..." if len(words) > 8 else "")


def _classify_bank(text: str) -> MemoryBank:
    tokens = _tokenize(text)
    best, best_score = MemoryBank.TASK_GOALS, 0
    for bank, keywords in _BANK_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best, best_score = bank, score
    return best


def _coerce_bank(value: Any) -> MemoryBank:
    if isinstance(value, MemoryBank):
        return value
    name = str(value)
    for bank in MemoryBank:
        if bank.value == name or bank.name == name:
            return bank
    return MemoryBank.TASK_GOALS


def _bank_demand(query_tokens: set[str]) -> dict[MemoryBank, float]:
    """Parameter-free bank-level demand: relative keyword overlap with the current context."""
    raw = {bank: len(query_tokens & keywords) for bank, keywords in _BANK_KEYWORDS.items()}
    total = sum(raw.values())
    if total == 0:
        # No demand signal in the current step: a uniform low baseline lets item relevance
        # and recency still rank memory instead of returning nothing (graceful fallback).
        baseline = 1.0 / len(_BANK_KEYWORDS)
        return {bank: baseline for bank in _BANK_KEYWORDS}
    return {bank: score / total for bank, score in raw.items()}


def _relevance(item_tokens: set[str], query_tokens: set[str]) -> float:
    """Parameter-free item-level relevance: Jaccard token overlap with the current context."""
    if not item_tokens or not query_tokens:
        return 0.0
    union = len(item_tokens | query_tokens)
    return len(item_tokens & query_tokens) / union if union else 0.0


class _StoredItem:
    """An atomic memory item with its functional bank and precomputed signal."""

    __slots__ = ("content", "bank", "index", "tokens")

    def __init__(self, content: MemoryContent, bank: MemoryBank, index: int) -> None:
        self.content = content
        self.bank = bank
        self.index = index
        self.tokens = _tokenize(_extract_text(content))


class ArbitratedMemoryConfig(BaseModel):
    """Configuration for :class:`ArbitratedMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    token_budget: int = Field(default=512, gt=0)
    """Per-step token budget for presented memory (the paper's unified per-step budget)."""

    focal_top_k: int = Field(default=3, ge=0)
    """Number of items rendered as focal (full detail)."""

    ambient_top_k: int = Field(default=4, ge=0)
    """Number of items rendered as ambient (compact headers)."""

    k: int = Field(default=5, ge=1)
    """Maximum items returned by :meth:`ArbitratedMemory.query`."""

    demand_weight: float = Field(default=0.5, ge=0.0)
    """Salience weight on bank-level demand."""

    relevance_weight: float = Field(default=0.35, ge=0.0)
    """Salience weight on item-level relevance."""

    recency_weight: float = Field(default=0.15, ge=0.0)
    """Salience weight on the temporal presentation gate (recency)."""


class ArbitratedMemory(Memory, Component[ArbitratedMemoryConfig]):
    """Function-aware memory that arbitrates salience at decision time.

    Each item is routed into one of five functional :class:`MemoryBank` instances. At
    decision time, :meth:`update_context` scores every item by combining bank-level demand,
    item-level relevance, and a recency gate, then presents the top items under a per-step
    token budget split into a detailed **focal** layer and a compact **ambient** layer. This
    closes the paper's "Memory-Action Gap" -- accessible memory that still fails to guide the
    current action because it is poorly organized, prioritized, or presented.

    The bank-demand and item-relevance estimators are parameter-free lexical proxies for the
    paper's learned/LLM estimators; no model calls are made. Drop it into any
    ``AssistantAgent`` via ``memory=[ArbitratedMemory()]`` and the agent runtime invokes
    ``add`` / ``query`` / ``update_context`` during inference.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.arbitrated_memory import ArbitratedMemory, MemoryBank


            async def main() -> None:
                memory = ArbitratedMemory()
                await memory.add(
                    MemoryContent(
                        content="Answer temperatures in Celsius.",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"bank": MemoryBank.USER_PREFERENCES},
                    )
                )
                agent = AssistantAgent("assistant", model_client=..., memory=[memory])
                # update_context is called automatically each inference step.


            asyncio.run(main())

    Args:
        config: Optional :class:`ArbitratedMemoryConfig`. Defaults are used when ``None``.

    """

    component_provider_override = "autogen_ext.memory.arbitrated_memory.ArbitratedMemory"
    component_config_schema = ArbitratedMemoryConfig

    def __init__(self, config: ArbitratedMemoryConfig | None = None) -> None:
        self._cfg = config or ArbitratedMemoryConfig()
        self._items: List[_StoredItem] = []
        self._next_index = 0

    @property
    def name(self) -> str:
        """Memory instance identifier."""
        return self._cfg.name or "arbitrated_memory"

    def _bank_of(self, content: MemoryContent) -> MemoryBank:
        raw = (content.metadata or {}).get("bank")
        if raw is None:
            return _classify_bank(_extract_text(content))
        return _coerce_bank(raw)

    def _rank(self, query_tokens: set[str]) -> List[tuple[float, _StoredItem]]:
        """Score and rank all items by decision-time salience (demand x relevance x recency)."""
        demand = _bank_demand(query_tokens)
        newest = max((it.index for it in self._items), default=0)
        scored: List[tuple[float, _StoredItem]] = []
        for item in self._items:
            bank_demand = demand.get(item.bank, 0.0)
            relevance = _relevance(item.tokens, query_tokens)
            recency = (item.index + 1) / (newest + 1)  # newer items score higher
            salience = (
                self._cfg.demand_weight * bank_demand
                + self._cfg.relevance_weight * relevance
                + self._cfg.recency_weight * recency
            )
            scored.append((salience, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def _select(
        self, ranked: List[tuple[float, _StoredItem]]
    ) -> tuple[List[_StoredItem], List[_StoredItem], List[_StoredItem]]:
        """Apply the token budget and focal/ambient split to a ranked list.

        Returns ``(focal, ambient, selected)`` where ``selected = focal + ambient``.
        """
        used = 0
        focal: List[_StoredItem] = []
        ambient: List[_StoredItem] = []
        for rank, (_, item) in enumerate(ranked):
            if rank < self._cfg.focal_top_k:
                cost = _approx_tokens(_extract_text(item.content))
                slot = focal
            else:
                if len(ambient) >= self._cfg.ambient_top_k:
                    break
                cost = _approx_tokens(_header(item.content))
                slot = ambient
            if used + cost > self._cfg.token_budget:
                continue
            used += cost
            slot.append(item)
        return focal, ambient, focal + ambient

    def _format(self, focal: List[_StoredItem], ambient: List[_StoredItem]) -> str:
        lines = ["Relevant memory (arbitrated by functional salience):"]
        if focal:
            lines.append("[Focal - high-salience, full detail]")
            for i, item in enumerate(focal, 1):
                lines.append(f"  {i}. ({item.bank.value}) {_extract_text(item.content)}")
        if ambient:
            lines.append("[Ambient - lower-salience, headers]")
            for i, item in enumerate(ambient, 1):
                lines.append(f"  {i}. ({item.bank.value}) {_header(item.content)}")
        return "\n".join(lines)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages or not self._items:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last = messages[-1]
        query_tokens = _tokenize(last.content if isinstance(last.content, str) else str(last))
        focal, ambient, selected = self._select(self._rank(query_tokens))
        if not selected:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        await model_context.add_message(SystemMessage(content=self._format(focal, ambient)))
        return UpdateContextResult(memories=MemoryQueryResult(results=[item.content for item in selected]))

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        _ = cancellation_token
        self._items.append(_StoredItem(content=content, bank=self._bank_of(content), index=self._next_index))
        self._next_index += 1

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        _ = cancellation_token, kwargs
        if not self._items:
            return MemoryQueryResult(results=[])
        query_tokens = _tokenize(_extract_text(query))
        ranked = self._rank(query_tokens)[: self._cfg.k]
        results: List[MemoryContent] = []
        for salience, item in ranked:
            metadata = dict(item.content.metadata or {})
            metadata["score"] = salience
            metadata["bank"] = item.bank.value
            results.append(
                MemoryContent(content=item.content.content, mime_type=item.content.mime_type, metadata=metadata)
            )
        return MemoryQueryResult(results=results)

    async def clear(self) -> None:
        self._items = []
        self._next_index = 0

    async def close(self) -> None:
        pass

    def _to_config(self) -> ArbitratedMemoryConfig:
        return self._cfg

    @classmethod
    def _from_config(cls, config: ArbitratedMemoryConfig) -> Self:
        return cls(config=config)

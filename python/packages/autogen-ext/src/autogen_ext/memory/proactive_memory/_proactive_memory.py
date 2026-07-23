"""Proactive memory — memory as an active intervention mechanism.

Adapted (Mode 2) from "Remember When It Matters: Proactive Memory Agent for
Long-Horizon Agents" (arxiv:2607.08716). The paper studies memory not as passive
retrieval but as an *active intervention*: a memory module runs alongside an
unmodified action agent, maintains a *structured memory bank*, and decides —
each turn — whether to *inject a memory-grounded reminder or remain silent*.
This "selective intervention" combats *behavioral state decay*, the failure mode
where decision-relevant state (open subgoals, constraints, prior failed attempts)
gets buried in a long trajectory and stops influencing decisions.

Kept at full fidelity:

* The structured memory bank with a **status / knowledge / procedural** taxonomy.
* **BM25 retrieval** over the bank (:mod:`._bm25`).
* The **selective intervention** decision — inject only when decision-relevant
  state is *buried*, otherwise stay silent. The paper's ablations show this beats
  passive bank exposure, always-on injection, advisor-only guidance, and general
  retrieval.

Substituted (Mode 2):

* The paper's **learned** intervention policy (a trained Qwen3.5-27B that decides
  when to intervene) is replaced by a parameter-free **"is-it-buried" heuristic**:
  a bank entry is surfaced only when it is BM25-relevant to the latest turn *and*
  its decision-relevant tokens are largely absent from the recent context window.
  This is a parameter-free proxy for the learned "should I intervene" signal.
* The separate Terminal-Bench / τ²-Bench evaluation harness is out of scope —
  evaluation belongs in a downstream PR.

The module is a drop-in :class:`~autogen_core.memory.Memory`, so any agent that
already wires a memory list invokes the proactive policy automatically: the
framework calls :meth:`update_context` before each inference. Registering it as
one of an agent's ``memory=[...]`` is the integration call site.
"""

from __future__ import annotations

import logging
from typing import Any, List

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel
from typing_extensions import Self

from ._bm25 import BM25, tokenize_text

logger = logging.getLogger(__name__)

# Structured memory bank taxonomy, per the paper.
MEMORY_CATEGORIES: tuple[str, ...] = ("status", "knowledge", "procedural")
DEFAULT_CATEGORY = "knowledge"

# An entry counts as "buried" when at least this fraction of its decision-relevant
# tokens are absent from the recent context the action agent is operating on.
_BURIED_OVERLAP_THRESHOLD = 0.5


class ProactiveMemoryConfig(BaseModel):
    """Configuration for :class:`ProactiveMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    k: int = 3
    """Maximum number of reminders to inject in a single intervention."""

    recency_window: int = 6
    """Number of most-recent context messages inspected to decide if state is buried."""

    score_threshold: float = 0.0
    """Minimum BM25 score for a bank entry to be considered decision-relevant."""

    bm25_k1: float = 1.5
    """BM25 term-frequency saturation."""

    bm25_b: float = 0.75
    """BM25 length-normalization strength."""


class ProactiveMemory(Memory, Component[ProactiveMemoryConfig]):
    """A memory that selectively injects buried, decision-relevant state.

    Unlike passive memories (e.g. :class:`~autogen_core.memory.ListMemory`, which
    always appends every entry), :class:`ProactiveMemory` *decides whether to
    intervene*. On each :meth:`update_context` call it:

    1. Builds a query from the latest trajectory turn.
    2. BM25-ranks bank entries by relevance to that turn.
    3. For each relevant entry, checks whether its decision-relevant tokens are
       already reflected in the recent context window.
    4. Injects a concise reminder only for entries that are relevant **and**
       buried — and stays **silent** otherwise (no context mutation, so the
       framework emits no memory event).

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.proactive_memory import ProactiveMemory


            async def main() -> None:
                memory = ProactiveMemory(name="long_horizon_state")
                await memory.add(
                    MemoryContent(
                        content="The final answer must be a single word with no punctuation.",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"category": "status"},
                    )
                )
                # After many turns, that constraint is buried — at the next
                # update_context, ProactiveMemory re-surfaces it as a reminder,
                # or stays silent if it is already present in recent context.


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance.
        k: Maximum reminders injected per intervention.
        recency_window: Recent messages inspected for the "buried" check.
        score_threshold: Minimum BM25 relevance to intervene on.
        bm25_k1: BM25 ``k1``.
        bm25_b: BM25 ``b``.
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.proactive_memory.ProactiveMemory"
    component_config_schema = ProactiveMemoryConfig

    def __init__(
        self,
        name: str | None = None,
        *,
        k: int = 3,
        recency_window: int = 6,
        score_threshold: float = 0.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self._name = name or "proactive_memory"
        self._k = k
        self._recency_window = recency_window
        self._score_threshold = score_threshold
        self._k1 = bm25_k1
        self._b = bm25_b
        self._entries: List[MemoryContent] = []

    @property
    def name(self) -> str:
        """Memory instance identifier."""
        return self._name

    @property
    def content(self) -> List[MemoryContent]:
        """Current bank entries in insertion order."""
        return self._entries

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Selectively inject buried, decision-relevant state as a reminder.

        Mutates ``model_context`` only when at least one bank entry is both
        relevant to the latest turn and buried in the trajectory. Otherwise this
        is a no-op (the proactive policy chooses to *remain silent*).
        """
        messages = await model_context.get_messages()
        if not messages or not self._entries:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last = messages[-1]
        query_text = last.content if isinstance(last.content, str) else str(last.content)

        ranked = self._rank(query_text)
        recent_text = self._recent_text(messages)

        injected: List[MemoryContent] = []
        for entry, score in ranked:
            if score < self._score_threshold:
                continue
            if self._is_buried(self._extract_text(entry), recent_text):
                injected.append(entry)
            if len(injected) >= self._k:
                break

        if injected:
            lines = [f"- [{self._category(e)}] {self._extract_text(e)}" for e in injected]
            reminder = (
                "Proactive memory reminder — the following decision-relevant context is "
                "not reflected in the recent trajectory; keep it in mind:\n" + "\n".join(lines)
            )
            await model_context.add_message(SystemMessage(content=reminder))

        return UpdateContextResult(memories=MemoryQueryResult(results=injected))

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """General retrieval: top-k bank entries by BM25 relevance to ``query``.

        This is the passive retrieval baseline. The selective intervention
        filtering lives in :meth:`update_context`.
        """
        _ = cancellation_token, kwargs
        ranked = self._rank(self._extract_text(query))
        results = [entry for entry, score in ranked if score >= self._score_threshold][: self._k]
        return MemoryQueryResult(results=results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a structured entry to the memory bank.

        Entries optionally carry a ``metadata["category"]`` of ``"status"``,
        ``"knowledge"``, or ``"procedural"`` (defaults to ``"knowledge"``).
        """
        _ = cancellation_token
        self._entries.append(content)

    async def clear(self) -> None:
        """Clear the memory bank."""
        self._entries = []

    async def close(self) -> None:
        """Release resources (none held)."""

    def _rank(self, query: str) -> List[tuple[MemoryContent, float]]:
        """Return bank entries ranked by BM25 relevance to ``query`` (desc)."""
        if not self._entries:
            return []
        docs = [self._extract_text(e) for e in self._entries]
        scores = BM25(docs, k1=self._k1, b=self._b).score(query)
        ranked = sorted(zip(self._entries, scores, strict=False), key=lambda pair: pair[1], reverse=True)
        return ranked

    def _recent_text(self, messages: List[Any]) -> str:
        """Concatenate the text of the most-recent ``recency_window`` messages."""
        if self._recency_window > 0:
            window = messages[-self._recency_window :]
        else:
            window = messages
        parts: List[str] = []
        for m in window:
            c = getattr(m, "content", m)
            parts.append(c if isinstance(c, str) else str(c))
        return " \n ".join(parts)

    def _is_buried(self, entry_text: str, recent_text: str) -> bool:
        """Parameter-free proxy for the learned "should I intervene" signal.

        An entry is buried when at least half of its decision-relevant tokens are
        absent from the recent context the action agent is operating on.
        """
        entry_tokens = set(tokenize_text(entry_text))
        if not entry_tokens:
            return False
        recent_tokens = set(tokenize_text(recent_text))
        overlap = len(entry_tokens & recent_tokens) / len(entry_tokens)
        return overlap < _BURIED_OVERLAP_THRESHOLD

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        if isinstance(content_item, str):
            return content_item
        c = content_item.content
        return c if isinstance(c, str) else str(c)

    def _category(self, entry: MemoryContent) -> str:
        meta = entry.metadata or {}
        category = meta.get("category", DEFAULT_CATEGORY)
        return category if isinstance(category, str) and category in MEMORY_CATEGORIES else DEFAULT_CATEGORY

    def _to_config(self) -> ProactiveMemoryConfig:
        return ProactiveMemoryConfig(
            name=self._name,
            k=self._k,
            recency_window=self._recency_window,
            score_threshold=self._score_threshold,
            bm25_k1=self._k1,
            bm25_b=self._b,
        )

    @classmethod
    def _from_config(cls, config: ProactiveMemoryConfig) -> Self:
        return cls(
            name=config.name,
            k=config.k,
            recency_window=config.recency_window,
            score_threshold=config.score_threshold,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
        )

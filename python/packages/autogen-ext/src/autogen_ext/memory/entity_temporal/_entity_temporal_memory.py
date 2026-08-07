"""Entity-temporal memory with zero-token (no-LLM) retrieval operations.

Adapted from *Zero-Mem: Zero-Token Memory Operations for LLM Agents*
(https://arxiv.org/abs/2607.29377v1).

Mode 2 (adapted port). The paper's core mechanism is kept at full fidelity:
original interaction traces are preserved verbatim as the source of record and
organized into two complementary views -- an *entity-context graph* that exposes
connections across interactions, and a *temporal hierarchy* (a recency gradient)
that preserves conversational locality. For each query the two views are weighed
against each other, both are retrieved from, and a deterministic calibration step
discards low-relevance and redundant evidence before the result is injected into
the model context. No step in ``add`` / ``query`` / ``update_context`` invokes an
LLM; only the consuming agent's own final response generation does, which is the
paper's "final-QA reader".

Auxiliary components are replaced with parameter-free, dependency-free proxies so
the module runs with no extra installs and no model download:

* entity extraction (NER / LLM in the paper) -> lexical term extraction with a
  small built-in stopword list;
* embeddings / encoder similarity -> term-overlap scoring over an inverted
  entity index (no encoder, no token consumption);
* deterministic "discard conflicting evidence" calibration -> relevance-threshold
  filtering plus near-duplicate suppression. True semantic contradiction
  detection requires generation and is intentionally out of scope.
"""

import re
from typing import Any

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

# Minimal built-in English stopword list so term extraction has no runtime deps.
_STOPWORDS: set[str] = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it",
    "its", "just", "no", "not", "of", "on", "or", "our", "should", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EntityTemporalMemoryConfig(BaseModel):
    """Configuration for :class:`EntityTemporalMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    traces: list[MemoryContent] = Field(default_factory=list)
    """Traces preserved as the source of record (round-tripped by the component loader)."""

    top_k: int = Field(default=5, ge=1, description="Maximum number of traces to return per query.")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum combined view score for a trace to survive calibration."
    )
    dup_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Jaccard term overlap at which two traces are treated as redundant evidence.",
    )


class EntityTemporalMemory(Memory, Component[EntityTemporalMemoryConfig]):
    """Zero-token memory organized into entity-context and temporal views.

    Traces are stored verbatim and indexed two ways: an inverted entity index
    (term -> trace ids) forms the *entity-context graph* that connects
    interactions sharing entities, while insertion order forms the *temporal
    hierarchy*. At query time a query-dependent weight blends the two views --
    entity-dense queries lean on the graph, vague queries lean on recency -- and
    deterministic calibration (relevance threshold + near-duplicate suppression)
    prunes the result before it is injected as a system message.

    No LLM is called during memory operations.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.entity_temporal import EntityTemporalMemory


            async def main() -> None:
                memory = EntityTemporalMemory(name="assistant_memory")
                await memory.add(MemoryContent(content="User prefers metric units", mime_type=MemoryMimeType.TEXT))
                await memory.add(MemoryContent(content="The Paris trip is in April", mime_type=MemoryMimeType.TEXT))
                results = await memory.query("What units does the user prefer?")
                print([str(m.content) for m in results.results])


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance.
        traces: Optional initial traces to index at construction time.
        top_k: Maximum number of traces returned per query.
        score_threshold: Minimum combined view score to survive calibration.
        dup_threshold: Jaccard overlap at which two traces are treated as redundant.
    """

    component_config_schema = EntityTemporalMemoryConfig
    component_provider_override = "autogen_ext.memory.entity_temporal.EntityTemporalMemory"

    def __init__(
        self,
        name: str | None = None,
        traces: list[MemoryContent] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        dup_threshold: float = 0.85,
    ) -> None:
        self._name = name or "entity_temporal_memory"
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._dup_threshold = dup_threshold
        self._traces: list[MemoryContent] = []
        self._trace_terms: list[set[str]] = []
        self._entity_index: dict[str, set[int]] = {}
        for content in traces or []:
            self._add(content)

    @property
    def name(self) -> str:
        """Get the memory instance identifier."""
        return self._name

    @property
    def content(self) -> list[MemoryContent]:
        """Get the stored traces in insertion order (the source of record)."""
        return self._traces

    @staticmethod
    def _content_text(content: MemoryContent) -> str:
        value = content.content
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            import json

            return json.dumps(value, sort_keys=True, default=str)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) >= 2 and token not in _STOPWORDS}

    def _add(self, content: MemoryContent) -> None:
        trace_id = len(self._traces)
        self._traces.append(content)
        terms = self._terms(self._content_text(content))
        self._trace_terms.append(terms)
        for term in terms:
            self._entity_index.setdefault(term, set()).add(trace_id)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a trace to memory and update both views. Makes no LLM call."""
        _ = cancellation_token
        self._add(content)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve traces by weighing the entity and temporal views. Makes no LLM call.

        Args:
            query: A string or :class:`MemoryContent` to retrieve against.
            cancellation_token: Optional token to cancel operation (unused).
            **kwargs: ``top_k`` optionally overrides the configured maximum.

        Returns:
            Calibrated, ranked traces (most relevant first).
        """
        _ = cancellation_token
        top_k = int(kwargs.pop("top_k", self._top_k))
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, MemoryContent):
            query_text = self._content_text(query)
        else:
            raise TypeError("'query' must be either a string or MemoryContent")

        n = len(self._traces)
        if n == 0 or not query_text.strip():
            return MemoryQueryResult(results=[])

        query_terms = self._terms(query_text)

        # Query-dependent view weighting: density of query terms present anywhere
        # in the entity index. Entity-dense queries lean on the graph (capped so
        # the temporal view always contributes); vague queries lean on recency.
        matched = len(query_terms & self._entity_index.keys())
        w_entity = min(0.8, matched / (matched + 2.0)) if matched else 0.0
        w_temporal = 1.0 - w_entity
        denom = len(query_terms) or 1

        scored: list[tuple[int, float]] = []
        for i in range(n):
            entity_score = len(query_terms & self._trace_terms[i]) / denom
            temporal_score = (i + 1) / n  # newest trace -> 1.0
            combined = w_entity * entity_score + w_temporal * temporal_score
            if combined > self._score_threshold:
                scored.append((i, combined))

        # Deterministic calibration: rank by combined score, then suppress
        # near-duplicate evidence (redundancy proxy for "discard conflicting").
        scored.sort(key=lambda item: item[1], reverse=True)
        picked: list[int] = []
        for i, _score in scored:
            terms_i = self._trace_terms[i]
            is_redundant = False
            for j in picked:
                union = terms_i | self._trace_terms[j]
                if union and len(terms_i & self._trace_terms[j]) / len(union) >= self._dup_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                picked.append(i)
            if len(picked) >= top_k:
                break

        return MemoryQueryResult(results=[self._traces[i] for i in picked])

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject calibrated memories relevant to the last message as a system message."""
        messages = await model_context.get_messages()
        last_message = str(messages[-1].content) if messages else ""
        results = await self.query(last_message)
        if results.results:
            lines = [f"- {self._content_text(memory)}" for memory in results.results]
            block = "Relevant memory (entity + temporal views, zero-token retrieval):\n" + "\n".join(lines)
            await model_context.add_message(SystemMessage(content=block))
        return UpdateContextResult(memories=results)

    async def clear(self) -> None:
        """Clear all traces and both views."""
        self._traces = []
        self._trace_terms = []
        self._entity_index = {}

    async def close(self) -> None:
        """No external resources to release."""

    @classmethod
    def _from_config(cls, config: EntityTemporalMemoryConfig) -> Self:
        return cls(
            name=config.name,
            traces=config.traces,
            top_k=config.top_k,
            score_threshold=config.score_threshold,
            dup_threshold=config.dup_threshold,
        )

    def _to_config(self) -> EntityTemporalMemoryConfig:
        return EntityTemporalMemoryConfig(
            name=self._name,
            traces=list(self._traces),
            top_k=self._top_k,
            score_threshold=self._score_threshold,
            dup_threshold=self._dup_threshold,
        )

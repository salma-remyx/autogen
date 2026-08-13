"""Affect-sensitive, conflict-aware memory for AutoGen agents.

Adapted from *PsychoAgent: An Affect-Sensitive Cognitive Architecture for
Conflict-Aware Memory in LLM Agents* (arXiv:2608.07438).

The architecture separates *factual* and *affective* traces and retrieves them
in two stages -- first by topical relevance, then by affective salience -- so
that emotionally significant, conflict-critical memories can enter the prompt
without sacrificing semantic fit. A conflict-aware executive promotes
unresolved-conflict traces during re-ranking.

Mode-2 adaptation (what was substituted):

* The paper's learned semantic-relevance estimator is replaced by a
  parameter-free lexical-overlap proxy (cosine over normalized token counts),
  so the memory needs no embedding backend -- it is dependency-free, like
  :class:`~autogen_core.memory.ListMemory`.
* Affective salience (``metadata["affect"]``, a float) and the conflict flag
  (``metadata["conflict"]``, a bool) are carried in ``MemoryContent.metadata``
  and supplied by the caller -- e.g. derived from simulated task outcomes --
  which is exactly the "affect tag" routing the paper describes for ``add()``.
* The paper's separate benchmark / human-rater evaluation is intentionally cut;
  evaluation belongs in a downstream PR.

The factual/affective partition + two-stage retrieval + conflict-aware merge are
kept at full fidelity.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "about",
        "that",
        "this",
        "from",
        "have",
        "was",
        "were",
        "are",
        "but",
        "not",
        "had",
        "has",
    }
)


class AffectiveMemoryConfig(BaseModel):
    """Configuration for :class:`AffectiveMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    top_k: int = 5
    """Maximum number of memories to surface after the two-stage retrieval."""

    min_similarity: float = 0.0
    """Minimum lexical-similarity score required to pass the semantic gate (stage 1)."""

    affect_weight: float = 0.25
    """Weight applied to affective salience during re-ranking (stage 2). Set to 0 for a semantic-only baseline."""

    conflict_weight: float = 0.3
    """Boost applied to unresolved-conflict memories during re-ranking (stage 2). Set to 0 for a non-conflict-aware baseline."""

    affect_key: str = "affect"
    """Metadata key holding the affective-salience score (float, defaults to 0.0)."""

    conflict_key: str = "conflict"
    """Metadata key holding the unresolved-conflict flag (bool, defaults to False)."""

    memory_contents: List[MemoryContent] = Field(default_factory=list)
    """Memories seeded into the store at construction time."""


class AffectiveMemory(Memory, Component[AffectiveMemoryConfig]):
    """Affect-sensitive, conflict-aware memory.

    Memories are partitioned implicitly by their affect/conflict metadata.
    Retrieval runs in two stages: (1) a semantic-relevance gate that keeps the
    ``top_k`` topically-fitting traces, then (2) a salience re-rank that lets
    affectively significant and conflict-critical traces surface within that
    gate. This mirrors the PsychoAgent executive controller, which preserves
    topical fit while allowing emotionally important traces to enter the prompt.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory import AffectiveMemory, AffectiveMemoryConfig


            async def main() -> None:
                memory = AffectiveMemory(AffectiveMemoryConfig())
                # A factual, emotionally neutral trace.
                await memory.add(
                    MemoryContent(content="Deploy the payment service to production.", mime_type=MemoryMimeType.TEXT)
                )
                # An affective, conflict-critical trace.
                await memory.add(
                    MemoryContent(
                        content="The team argued bitterly about the rollback plan and never resolved it.",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"affect": 1.0, "conflict": True},
                    )
                )
                results = (await memory.query("How do we handle the deployment rollback?")).results
                print(results[0].metadata)  # conflict-critical trace is promoted to the top


            asyncio.run(main())

    Args:
        config: Optional :class:`AffectiveMemoryConfig`. Defaults to a sensible configuration.

    """

    component_type = "memory"
    component_config_schema = AffectiveMemoryConfig
    component_provider_override = "autogen_ext.memory.affective.AffectiveMemory"

    def __init__(self, config: AffectiveMemoryConfig | None = None) -> None:
        self._config = config or AffectiveMemoryConfig()
        self._contents: List[MemoryContent] = list(self._config.memory_contents)

    @property
    def name(self) -> str:
        """Get the memory instance identifier."""
        return self._config.name or "affective_memory"

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Update ``model_context`` with affect-sensitive, conflict-aware memories.

        Derives a query from the last message in ``model_context``, runs the
        two-stage retrieval, and injects the ranked traces as a
        :class:`~autogen_core.models.SystemMessage` whose entries are tagged with
        their affect/conflict state so the mechanism is inspectable.

        Args:
            model_context: The context to update. Mutated if relevant memories exist.

        Returns:
            UpdateContextResult containing the retrieved memories.
        """
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        query_results = await self.query(query_text)

        if query_results.results:
            lines: List[str] = []
            for index, memory in enumerate(query_results.results, 1):
                affect = self._affect_of(memory)
                conflict = self._conflict_of(memory)
                lines.append(f"{index}. [affect={affect:.2f} conflict={conflict}] {self._text_of(memory)}")
            memory_context = (
                "\nRelevant memory (affect-sensitive, conflict-aware retrieval):\n"
                + "\n".join(lines)
                + "\n"
            )
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a memory, routing it via its affect/conflict metadata.

        Args:
            content: The memory content to store. ``metadata`` may carry
                ``affect`` (float) and ``conflict`` (bool) tags; absent tags read
                back as ``0.0`` / ``False``.
            cancellation_token: Optional token to cancel the operation (ignored).
        """
        _ = cancellation_token
        if content.metadata is None:
            content.metadata = {}
        self._contents.append(content)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Two-stage affect-sensitive retrieval.

        Stage 1 filters memories by lexical similarity to ``query`` (the semantic
        gate) and keeps the ``top_k`` topically-fitting traces. Stage 2 re-ranks
        those survivors by a salience score that blends semantic fit with
        affective significance and an unresolved-conflict boost.

        Args:
            query: Query content item.
            cancellation_token: Optional token to cancel the operation (ignored).
            **kwargs: Additional implementation-specific parameters (ignored).

        Returns:
            MemoryQueryResult containing re-ranked memories with ``score`` and
            ``rank_score`` populated in their metadata.
        """
        _ = cancellation_token, kwargs
        query_text = self._extract_text(query)

        # Stage 1: semantic-relevance gate.
        scored: List[Tuple[float, MemoryContent]] = []
        for memory in self._contents:
            similarity = self._similarity(query_text, self._text_of(memory))
            if similarity < self._config.min_similarity:
                continue
            scored.append((similarity, memory))
        scored.sort(key=lambda item: item[0], reverse=True)
        candidates = scored[: self._config.top_k]

        # Stage 2: salience re-rank with the conflict-aware boost.
        candidates.sort(key=lambda item: self._rank_score(item[0], item[1]), reverse=True)

        results: List[MemoryContent] = []
        for similarity, memory in candidates:
            metadata: Dict[str, Any] = dict(memory.metadata) if memory.metadata else {}
            metadata["score"] = similarity
            metadata["rank_score"] = self._rank_score(similarity, memory)
            results.append(
                MemoryContent(content=memory.content, mime_type=memory.mime_type, metadata=metadata)
            )
        return MemoryQueryResult(results=results)

    async def clear(self) -> None:
        """Clear all stored memories."""
        self._contents = []

    async def close(self) -> None:
        """Clean up resources. Nothing to release for this in-memory store."""

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from a query."""
        if isinstance(content_item, str):
            return content_item
        return self._text_of(content_item)

    def _text_of(self, content: MemoryContent) -> str:
        """Extract searchable text from a stored memory."""
        value = content.content
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, dict):
            return str(value).lower()
        # Remaining ContentType option is Image.
        raise ValueError(f"Cannot extract text from content of type {type(value).__name__}")

    def _tokens(self, text: str) -> Counter[str]:
        """Tokenize text into a term-frequency counter, dropping stopwords."""
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1 and token not in _STOPWORDS
        ]
        return Counter(tokens)

    def _similarity(self, left: str, right: str) -> float:
        """Cosine similarity over normalized token-count vectors (parameter-free semantic proxy)."""
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        dot = sum(count * right_tokens.get(token, 0) for token, count in left_tokens.items())
        left_norm = math.sqrt(sum(count * count for count in left_tokens.values()))
        right_norm = math.sqrt(sum(count * count for count in right_tokens.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _affect_of(self, content: MemoryContent) -> float:
        """Read the affective-salience score from metadata (default 0.0)."""
        metadata: Dict[str, Any] = content.metadata or {}
        return float(metadata.get(self._config.affect_key, 0.0))

    def _conflict_of(self, content: MemoryContent) -> bool:
        """Read the unresolved-conflict flag from metadata (default False)."""
        metadata: Dict[str, Any] = content.metadata or {}
        return bool(metadata.get(self._config.conflict_key, False))

    def _rank_score(self, similarity: float, content: MemoryContent) -> float:
        """Salience score blending semantic fit, affect, and the conflict boost."""
        conflict_boost = self._config.conflict_weight if self._conflict_of(content) else 0.0
        return similarity + self._config.affect_weight * self._affect_of(content) + conflict_boost

    def _to_config(self) -> AffectiveMemoryConfig:
        """Serialize the memory to its declarative configuration."""
        config = self._config.model_copy(deep=True)
        config.memory_contents = [content.model_copy(deep=True) for content in self._contents]
        return config

    @classmethod
    def _from_config(cls, config: AffectiveMemoryConfig) -> Self:
        """Deserialize the memory from its declarative configuration."""
        return cls(config=config.model_copy(deep=True))

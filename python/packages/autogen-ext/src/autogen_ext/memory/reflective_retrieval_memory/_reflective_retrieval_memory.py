"""Reflective retrieval memory — learns retrieval strategies from past queries.

Adapted (Mode 2 port) from:

    RRM: Experience-Driven Reflective Retrieval Memory for Long-Horizon
    Multimodal Reasoning (arXiv:2607.28156v1).

The paper's core mechanism is a *reflective experience memory* that distills
transferable procedural retrieval knowledge from historical task trajectories,
converts retrieved experiences into *query-level guidance*, and regulates them
through a *lifecycle* (usage frequency, reuse feedback, temporal decay). This
module keeps that core at full fidelity while substituting the paper's
auxiliaries with target-native equivalents:

- The multimodal video setting and entity-centric multimodal memory graph are
  replaced with text retrieval over any existing :class:`~autogen_core.memory.Memory`
  backend (the factual store). The reflective layer is modality-agnostic.
- The paper's learned failure-diagnosis mechanism is replaced by a
  parameter-free token-overlap proxy (Jaccard similarity) that retrieves the
  experiences relevant to a new query. No model call is required to record or
  apply a lesson.
- The paper's separate benchmark suite (Video-MME-Long, M3-Bench) is out of
  scope; evaluation belongs in a downstream PR.

Answer generation is conditioned only on the factual evidence returned by the
underlying store — reflective experiences only *transform the query*, never the
facts themselves.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

__all__ = ["ReflectiveRetrievalMemory", "ReflectiveRetrievalMemoryConfig"]

# Short, high-frequency terms that carry little retrieval signal.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "what",
        "which",
        "who",
        "when",
        "i",
        "you",
        "my",
        "me",
        "we",
        "it",
        "this",
        "that",
        "these",
        "those",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "please",
        "tell",
        "get",
    }
)


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase alphanumeric token set, dropping stopwords and single characters.

    This is the parameter-free proxy for the paper's experience-retrieval signal.
    """
    return frozenset(
        token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1 and token not in _STOPWORDS
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap between two token sets; 0.0 when either is empty."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class ReflectiveRetrievalMemoryConfig(BaseModel):
    """Configuration for :class:`ReflectiveRetrievalMemory`.

    The ``underlying_memory`` is a runtime-injected factual store (like a database
    connection) and is excluded from JSON serialization; the declarative knobs below
    fully round-trip through :meth:`ReflectiveRetrievalMemory._to_config`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "reflective_retrieval_memory"
    """Identifier for this memory instance."""

    underlying_memory: Memory | None = None
    """Factual memory backend whose queries are guided by reflective experiences."""

    similarity_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    """Minimum query/experience token overlap for an experience to be consulted."""

    max_guidance_lessons: int = Field(default=3, ge=1)
    """Maximum number of experiences whose guidance may augment a single query."""

    min_lifecycle_score: float = Field(default=0.05, ge=0.0)
    """Experiences whose lifecycle weight falls below this floor are pruned."""

    decay_half_life_days: float = Field(default=30.0, gt=0.0)
    """Halving distance for temporal decay of an experience's influence."""

    guidance_prefix: str = "Retrieval guidance from past experience:"
    """Prefix prepended to the appended guidance block when augmenting a query."""


@dataclass
class _Experience:
    """A single reflective lesson: a reusable retrieval strategy."""

    id: str
    query_signature: str
    """Raw text describing the retrieval situation this lesson applies to."""

    guidance: str
    """Query-level guidance applied when this lesson is consulted."""

    signature_tokens: frozenset[str]
    usage_count: int = 0
    feedback_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_used_at: float | None = None


class ReflectiveRetrievalMemory(Memory, Component[ReflectiveRetrievalMemoryConfig]):
    """Memory that learns and reuses retrieval strategies across queries.

    ``ReflectiveRetrievalMemory`` wraps an existing factual :class:`~autogen_core.memory.Memory`
    backend (e.g. :class:`~autogen_core.memory.ListMemory`, ``ChromaDBVectorMemory``,
    ``RedisMemory``) and adds a *reflective experience memory* layer on top. After a
    poorly-formulated query is diagnosed, the agent records a lesson about how to
    retrieve for that situation. On subsequent queries whose situation overlaps a
    stored lesson, the lesson's guidance is appended to the query before it reaches
    the factual store, biasing retrieval toward what worked before.

    Lessons are regulated by a lifecycle combining **usage frequency** (how often a
    lesson's guidance is applied), **reuse feedback** (rated helpful/unhelpful), and
    **temporal decay** (influence halves every ``decay_half_life_days``). Experiences
    whose combined weight drops below ``min_lifecycle_score`` are pruned, keeping the
    reflective store free of redundant or stale strategies.

    Recording a lesson uses the standard :meth:`Memory.add` interface — pass a
    :class:`~autogen_core.memory.MemoryContent` whose ``metadata`` marks it as a
    ``reflective_lesson`` with ``query_signature`` and ``guidance`` fields — or the
    ergonomic :meth:`reflect_on_query` helper.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
            from autogen_ext.memory.reflective_retrieval_memory import ReflectiveRetrievalMemory


            async def main() -> None:
                facts = ListMemory()
                memory = ReflectiveRetrievalMemory(underlying_memory=facts)

                # Store a lesson learned about retrieving for this kind of query.
                memory.reflect_on_query(
                    query="how do I configure api options",
                    guidance="search for 'settings' and 'config', not just 'options'",
                    helpful=True,
                )
                await memory.add(
                    MemoryContent(content="API config lives in settings.json", mime_type=MemoryMimeType.TEXT)
                )

                # Later query is augmented with the recorded guidance before hitting the facts.
                result = await memory.query("configure api options")
                print([m.content for m in result.results])


            asyncio.run(main())

    Args:
        config: Configuration; defaults to :class:`ReflectiveRetrievalMemoryConfig`.
    """

    component_type = "memory"
    component_config_schema = ReflectiveRetrievalMemoryConfig
    component_provider_override = "autogen_ext.memory.reflective_retrieval_memory.ReflectiveRetrievalMemory"

    def __init__(self, config: ReflectiveRetrievalMemoryConfig | None = None) -> None:
        """Initialize the reflective retrieval memory."""
        self.config = config or ReflectiveRetrievalMemoryConfig()
        self._underlying: Memory | None = self.config.underlying_memory
        self._experiences: List[_Experience] = []
        self._next_id = 0

    @property
    def experiences(self) -> List[_Experience]:
        """Snapshot of the currently stored reflective lessons (newest last)."""
        return list(self._experiences)

    def reflect_on_query(self, query: str, guidance: str, *, helpful: bool | None = None) -> _Experience:
        """Record a reusable retrieval lesson for a query situation.

        This is the ergonomic path for the paper's "store a lesson learned about query
        formulation" step.

        Args:
            query: A query (or description) characterizing the retrieval situation.
            guidance: Query-level guidance to apply on similar future queries.
            helpful: Optional initial reuse feedback for the lesson.

        Returns:
            The created experience, exposing its ``id`` for later feedback.
        """
        experience = self._register_experience(signature=query, guidance=guidance)
        if helpful is not None:
            experience.feedback_score += 1.0 if helpful else -1.0
        return experience

    def record_feedback(self, lesson_id: str, helpful: bool) -> None:
        """Adjust a lesson's reuse-feedback lifecycle signal.

        Args:
            lesson_id: Identifier returned by :meth:`reflect_on_query`.
            helpful: ``True`` rewards the strategy; ``False`` suppresses it.

        Raises:
            KeyError: If ``lesson_id`` does not match a stored experience.
        """
        for experience in self._experiences:
            if experience.id == lesson_id:
                experience.feedback_score += 1.0 if helpful else -1.0
                return
        raise KeyError(lesson_id)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Guide retrieval for the last message and inject factual results as context.

        The query derived from the last message is augmented with reflective guidance,
        then resolved against the factual store. Only factual evidence is injected.

        Args:
            model_context: The context to update with retrieved factual memories.

        Returns:
            UpdateContextResult containing the factual memories that were injected.
        """
        messages = await model_context.get_messages()
        last_message = str(messages[-1].content) if messages else ""
        results = await self.query(last_message)
        if results.results:
            await model_context.add_message(SystemMessage(content="\n\n".join(str(m.content) for m in results.results)))
        return UpdateContextResult(memories=results)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Augment the query with reflective guidance, then query the factual store.

        Factual results come solely from the underlying memory; reflective experiences
        only transform the query. Consulted lessons accrue usage-frequency credit.

        Args:
            query: The query, or a :class:`~autogen_core.memory.MemoryContent`.
            cancellation_token: Optional token forwarded to the underlying store.
            **kwargs: Forwarded to the underlying memory's ``query``.

        Returns:
            Factual :class:`~autogen_core.memory.MemoryQueryResult` from the underlying
            store (empty if no underlying store is configured).
        """
        query_str = query if isinstance(query, str) else str(query.content)
        now = time.time()
        augmented, consulted = self._augment(query_str, now)
        if self._underlying is not None:
            results = await self._underlying.query(augmented, cancellation_token, **kwargs)
        else:
            results = MemoryQueryResult(results=[])
        for experience in consulted:
            experience.usage_count += 1
            experience.last_used_at = now
        self._maybe_prune(now)
        return results

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add content, dispatching reflective lessons vs. factual evidence.

        A :class:`~autogen_core.memory.MemoryContent` whose ``metadata`` is a
        ``reflective_lesson`` (or carries a ``guidance`` key) is stored as an
        experience; any other content is forwarded to the factual store. Adding
        factual content when no underlying store is configured is a no-op.

        Args:
            content: The memory content to store.
            cancellation_token: Optional token forwarded to the underlying store.
        """
        metadata: Dict[str, Any] = content.metadata or {}
        if metadata.get("type") == "reflective_lesson" or "guidance" in metadata:
            self._register_experience(
                signature=str(metadata.get("query_signature", content.content)),
                guidance=str(metadata.get("guidance", "")),
                feedback=float(metadata.get("feedback_score", 0.0)),
            )
            return
        if self._underlying is not None:
            await self._underlying.add(content, cancellation_token)

    async def clear(self) -> None:
        """Drop all reflective experiences and clear the underlying factual store."""
        self._experiences = []
        if self._underlying is not None:
            await self._underlying.clear()

    async def close(self) -> None:
        """Release the underlying factual store's resources."""
        if self._underlying is not None:
            await self._underlying.close()

    def _register_experience(self, signature: str, guidance: str, feedback: float = 0.0) -> _Experience:
        experience = _Experience(
            id=f"lesson_{self._next_id}",
            query_signature=signature,
            guidance=guidance,
            signature_tokens=_tokenize(signature),
            usage_count=0,
            feedback_score=feedback,
            created_at=time.time(),
        )
        self._next_id += 1
        self._experiences.append(experience)
        return experience

    def _augment(self, query_str: str, now: float) -> tuple[str, List[_Experience]]:
        """Build the guidance-augmented query and the experiences that contributed."""
        if not query_str.strip() or not self._experiences:
            return query_str, []
        query_tokens = _tokenize(query_str)
        scored: List[tuple[float, _Experience]] = []
        for experience in self._experiences:
            similarity = _jaccard(query_tokens, experience.signature_tokens)
            if similarity >= self.config.similarity_threshold:
                relevance = similarity * self._lifecycle_weight(experience, now)
                scored.append((relevance, experience))
        if not scored:
            return query_str, []
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [experience for _, experience in scored[: self.config.max_guidance_lessons]]
        guidance_block = "\n".join(f"- {e.guidance}" for e in chosen)
        augmented = f"{query_str}\n{self.config.guidance_prefix}\n{guidance_block}"
        return augmented, chosen

    def _lifecycle_weight(self, experience: _Experience, now: float) -> float:
        """Combined weight from usage frequency, reuse feedback, and temporal decay."""
        age_days = max(0.0, (now - experience.created_at) / 86400.0)
        decay = 0.5 ** (age_days / self.config.decay_half_life_days)
        usage = min(1.0 + 0.25 * experience.usage_count, 2.0)
        feedback = max(0.1, 1.0 + experience.feedback_score)
        return usage * feedback * decay

    def _maybe_prune(self, now: float) -> None:
        """Evict experiences whose lifecycle weight has fallen below the floor."""
        if not self._experiences:
            return
        floor = self.config.min_lifecycle_score
        self._experiences = [e for e in self._experiences if self._lifecycle_weight(e, now) >= floor]

    def _to_config(self) -> ReflectiveRetrievalMemoryConfig:
        return ReflectiveRetrievalMemoryConfig(
            name=self.config.name,
            underlying_memory=self._underlying,
            similarity_threshold=self.config.similarity_threshold,
            max_guidance_lessons=self.config.max_guidance_lessons,
            min_lifecycle_score=self.config.min_lifecycle_score,
            decay_half_life_days=self.config.decay_half_life_days,
            guidance_prefix=self.config.guidance_prefix,
        )

    @classmethod
    def _from_config(cls, config: ReflectiveRetrievalMemoryConfig) -> Self:
        return cls(config=config)

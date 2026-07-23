"""Dual-memory backend with cross-memory resonance.

Adapted from *OpsMem: Dual-Memory Reasoning with Cross-Memory Resonance for
Failure Diagnosis* (arXiv:2607.11357). Mode 2 adapted port.

OpsMem maintains two coordinated stores during iterative diagnosis:

* **Short-term memory (STM)** — the evolving working state for the current
  incident (evidence gathered, hypotheses under consideration).
* **Long-term memory (LTM)** — reusable operational experience consolidated
  from previously solved incidents.

It then uses **cross-memory resonance (CMR)** to activate the LTM experience
that is relevant to the current STM state, conditioning the next step on the
short-term state *plus* the activated long-term experience, and
**consolidates** solved working state back into LTM for future reuse.

This module ports that mechanism onto AutoGen's :class:`~autogen_core.memory.Memory`
ABC:

* ``add`` accumulates into STM (the working state).
* ``update_context`` is the CMR activation step — it derives a state query
  from the current model context (the live STM) plus any STM notes,
  activates the resonating LTM entries, and injects them into the context.
* ``consolidate`` promotes the current STM into LTM as reusable experience.
* ``add_to_long_term`` seeds LTM directly with known-good experience.

The diagnosis-specific multi-agent loop from the paper is intentionally
decoupled: this is a generic memory backend any AutoGen agent can consume.
OpsMem's learned resonance estimator is substituted with the parameter-free
:class:`~autogen_ext.memory.dual_memory.TokenOverlapResonanceScorer` (see
``_resonance``); the scorer is pluggable so an embedding-based estimator can
be dropped in without changing the backend.

Example:

    .. code-block:: python

        import asyncio
        from autogen_core.memory import MemoryContent, MemoryMimeType
        from autogen_core.model_context import BufferedChatCompletionContext
        from autogen_core.models import UserMessage
        from autogen_ext.memory.dual_memory import DualMemory


        async def main() -> None:
            memory = DualMemory(name="ops", k=3, score_threshold=0.05)

            # Seed long-term memory with reusable operational experience.
            await memory.add_to_long_term(
                MemoryContent(
                    content="Restart the redis pod when latency spikes after a deploy.",
                    mime_type=MemoryMimeType.TEXT,
                    metadata={"incident": "INC-42"},
                )
            )

            context = BufferedChatCompletionContext(buffer_size=10)
            await context.add_message(UserMessage(content="redis latency is spiking", source="user"))

            # Cross-memory resonance activates the relevant experience and
            # injects it into the context.
            await memory.update_context(context)

            # Record the current working state, then consolidate it for reuse.
            await memory.add(MemoryContent(content="Rolling back deploy resolved the spike.", mime_type=MemoryMimeType.TEXT))
            await memory.consolidate()


        asyncio.run(main())
"""

import json
import logging
from typing import Any, List

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

from ._resonance import ResonanceScorer, get_scorer

logger = logging.getLogger(__name__)


class DualMemoryConfig(BaseModel):
    """Configuration for the :class:`DualMemory` component."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    k: int = 3
    """Maximum number of long-term memories to activate per query."""

    score_threshold: float = 0.0
    """Minimum resonance score for a long-term memory to be activated."""

    remove_stop_words: bool = True
    """Whether the resonance scorer drops common stop words."""

    scorer: str = "token_overlap"
    """Name of the resonance scorer to use."""

    long_term_contents: List[MemoryContent] = Field(default_factory=list)
    """Operational experience preloaded into long-term memory."""


class DualMemory(Memory, Component[DualMemoryConfig]):
    """A dual-memory (STM + LTM) backend that activates experience via cross-memory resonance.

    See the module docstring for the OpsMem mapping and the Mode 2 substitutions.
    """

    component_config_schema = DualMemoryConfig
    component_provider_override = "autogen_ext.memory.dual_memory.DualMemory"

    def __init__(
        self,
        name: str | None = None,
        *,
        k: int = 3,
        score_threshold: float = 0.0,
        remove_stop_words: bool = True,
        scorer: str = "token_overlap",
        long_term_contents: List[MemoryContent] | None = None,
    ) -> None:
        self._name = name or "default_dual_memory"
        self._k = k
        self._score_threshold = score_threshold
        self._remove_stop_words = remove_stop_words
        self._scorer_name = scorer
        self._scorer: ResonanceScorer = get_scorer(scorer, remove_stop_words=remove_stop_words)
        self._stm: List[MemoryContent] = []
        self._ltm: List[MemoryContent] = list(long_term_contents) if long_term_contents is not None else []

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._name

    @property
    def short_term_memory(self) -> List[MemoryContent]:
        """The current working state (STM)."""
        return self._stm

    @property
    def long_term_memory(self) -> List[MemoryContent]:
        """Consolidated, reusable operational experience (LTM)."""
        return self._ltm

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add content to short-term (working) memory.

        Args:
            content: The working-state content to record.
            cancellation_token: Optional token to cancel operation (unused).
        """
        _ = cancellation_token
        self._stm.append(content)

    async def add_to_long_term(
        self, content: MemoryContent, cancellation_token: CancellationToken | None = None
    ) -> None:
        """Seed long-term memory directly with reusable operational experience.

        Args:
            content: The experience content to store in LTM.
            cancellation_token: Optional token to cancel operation (unused).
        """
        _ = cancellation_token
        self._ltm.append(content)

    async def consolidate(self, *, clear_short_term: bool = True) -> None:
        """Promote the current short-term working state into long-term memory.

        Duplicates (by normalized text) already present in LTM are skipped, so
        consolidating the same solved incident more than once does not bloat
        the long-term store.

        Args:
            clear_short_term: If True, reset STM after promotion (the incident
                is resolved and its experience is now reusable).
        """
        existing = {self._text_of(item) for item in self._ltm}
        for item in self._stm:
            text = self._text_of(item)
            if text and text not in existing:
                self._ltm.append(item)
                existing.add(text)
        if clear_short_term:
            self._stm = []

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Activate long-term memories by cross-memory resonance with *query*.

        Args:
            query: A state query (text or :class:`MemoryContent`).
            cancellation_token: Optional token to cancel operation (unused).
            **kwargs: Additional parameters (ignored).

        Returns:
            Up to ``k`` LTM entries whose resonance score meets
            ``score_threshold``, highest-scoring first, with the score attached
            to each result's metadata.
        """
        _ = cancellation_token, kwargs
        query_text = self._extract_text(query)
        scored = [(self._scorer.score(query_text, self._text_of(item)), item) for item in self._ltm]
        activated = [(score, item) for score, item in scored if score >= self._score_threshold]
        activated.sort(key=lambda pair: pair[0], reverse=True)
        results = [self._with_score(item, score) for score, item in activated[: self._k]]
        return MemoryQueryResult(results=results)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Activate resonating long-term memory and inject it into *model_context*.

        The current state query is derived from the live model context (the
        evolving STM conversation) combined with any STM notes; this is the
        OpsMem cross-memory resonance activation step.

        Args:
            model_context: The context to update. Mutated in place when
                resonating memories are found.

        Returns:
            UpdateContextResult containing the activated memories.
        """
        messages = await model_context.get_messages()
        state_parts: List[str] = [self._message_text(message) for message in messages]
        state_parts.extend(self._text_of(item) for item in self._stm)
        state_query = " ".join(part for part in state_parts if part).strip()
        if not state_query:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        results = await self.query(state_query)
        if results.results:
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(results.results, 1)]
            memory_context = (
                "\nActivated long-term memory (cross-memory resonance):\n" + "\n".join(memory_strings) + "\n"
            )
            await model_context.add_message(SystemMessage(content=memory_context))
        return UpdateContextResult(memories=results)

    async def clear(self) -> None:
        """Clear both short-term and long-term memory."""
        self._stm = []
        self._ltm = []

    async def clear_short_term(self) -> None:
        """Reset the short-term working state, preserving long-term memory."""
        self._stm = []

    async def close(self) -> None:
        """Clean up resources. No external resources are held."""
        return None

    @classmethod
    def _from_config(cls, config: DualMemoryConfig) -> Self:
        return cls(
            name=config.name,
            k=config.k,
            score_threshold=config.score_threshold,
            remove_stop_words=config.remove_stop_words,
            scorer=config.scorer,
            long_term_contents=list(config.long_term_contents),
        )

    def _to_config(self) -> DualMemoryConfig:
        return DualMemoryConfig(
            name=self._name,
            k=self._k,
            score_threshold=self._score_threshold,
            remove_stop_words=self._remove_stop_words,
            scorer=self._scorer_name,
            long_term_contents=list(self._ltm),
        )

    @staticmethod
    def _text_of(content: MemoryContent) -> str:
        """Extract searchable text from a :class:`MemoryContent` item."""
        value = content.content
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True).lower()
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                logger.debug("Skipping non-utf8 bytes memory content during resonance scoring")
                return ""
        # Image and other types carry no lexical signal for the overlap scorer.
        return ""

    def _extract_text(self, query: str | MemoryContent) -> str:
        if isinstance(query, str):
            return query
        return self._text_of(query)

    @staticmethod
    def _message_text(message: Any) -> str:
        """Extract plain text from a model message, if it carries any."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _with_score(content: MemoryContent, score: float) -> MemoryContent:
        """Return a copy of *content* with the resonance score in its metadata."""
        metadata = dict(content.metadata) if content.metadata else {}
        metadata["score"] = score
        return content.model_copy(update={"metadata": metadata})

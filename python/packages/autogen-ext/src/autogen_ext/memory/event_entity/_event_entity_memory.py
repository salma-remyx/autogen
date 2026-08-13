"""Event-entity indexed memory with query-time dynamic-hyperedge retrieval.

This memory backend adapts the *event-entity index* and *query-time entity-join*
retrieval of SAG (SQL-Retrieval Augmented Generation, arXiv:2608.12129) to
AutoGen's :class:`~autogen_core.memory.Memory` ABC. It drops in wherever a
:class:`~autogen_core.memory.Memory` is consumed (e.g.
``AssistantAgent(memory=[EventEntityMemory()])``), and is reached through the
same ``update_context`` hook the other ``autogen_ext.memory`` backends use.

What is kept from the paper (full fidelity)
-------------------------------------------
* **Event-entity index built in ``add``.** Every chunk is stored once as a
  semantically complete *event* paired with the *entities* it mentions. The
  ``event <-> entities`` pairing is a latent hyperedge: it preserves the chunk's
  n-ary relations without decomposing them into subject-predicate-object triples.
* **Query-time dynamic-hyperedge retrieval in ``query`` / ``update_context``.**
  Entities drawn from the query are treated as *join keys*. Seed chunks are the
  ones that share an entity with the query, and the query-scoped neighbourhood is
  grown by following further shared-entity joins up to ``max_hops``. Every piece
  of evidence returned is the original chunk text, exactly as in the paper, so
  the downstream model reasons over verbatim evidence rather than graph triples.

What is substituted (Mode 2 adapted port)
-----------------------------------------
The paper's SQL / Elasticsearch / OceanBase deployment backends are collapsed to
an in-process store -- the same trade-off the other ``autogen_ext.memory``
backends make. The paper's *learned* entity/event extractor is replaced by a
parameter-free proxy: named-entity-style candidate spans (capitalised token
runs) are detected heuristically, and callers may supply explicit entities
through ``metadata["entities"]`` for the highest-fidelity path. The chunk text
itself serves as the event representation (the paper notes that evidence always
remains the original chunk), so no learned event rewriter is needed.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

logger = logging.getLogger(__name__)

# Sentence-initial function words that the capitalised-span heuristic would
# otherwise mistake for entities (e.g. the leading "The" in "The Eiffel Tower").
_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "for",
    "nor",
    "of",
    "in",
    "on",
    "at",
    "to",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "when",
    "where",
    "how",
    "why",
}

# A maximal run of capitalised tokens, e.g. "Marie Curie", "Eiffel Tower", "IBM".
_CAP_RUN_RE = re.compile(r"[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*")


class EventEntityMemoryConfig(BaseModel):
    """Configuration for :class:`EventEntityMemory`."""

    k: int = Field(default=3, ge=1, description="Maximum number of chunks returned per query.")
    max_hops: int = Field(
        default=2,
        ge=1,
        description="Entity-join hops used to grow the query neighbourhood. ``1`` returns only seed chunks "
        "that directly share an entity with the query; larger values chain more joins for multi-hop evidence.",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Minimum join score for a chunk to be returned. ``None`` returns the top ``k`` regardless of score.",
    )
    entities_key: str = Field(
        default="entities",
        description="Metadata key holding caller-supplied entity strings for a chunk. When present, "
        "these replace the heuristic extraction for that chunk.",
    )
    min_entity_len: int = Field(
        default=3,
        ge=1,
        description="Minimum character length for a heuristic candidate entity (after stopword trimming).",
    )


@dataclass
class _EventRecord:
    """A single indexed chunk: its text, original content, and extracted entities."""

    id: int
    text: str
    mime_type: MemoryMimeType | str
    entities: Set[str] = field(default_factory=set)


class EventEntityMemory(Memory, Component[EventEntityMemoryConfig]):
    """A structured-retrieval memory that indexes chunks as event-entity pairs and
    retrieves a query-scoped neighbourhood by joining on shared entities.

    This is an adapted port of SAG's dynamic-hyperedge retrieval
    (arXiv:2608.12129). See the module docstring for what is kept at full
    fidelity and what is substituted.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.event_entity import EventEntityMemory, EventEntityMemoryConfig


            async def main() -> None:
                memory = EventEntityMemory(EventEntityMemoryConfig(k=4, max_hops=2))
                await memory.add(
                    MemoryContent(
                        content="Marie Curie discovered radium while working in Paris.",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"entities": ["Marie Curie", "radium", "Paris"]},
                    )
                )
                assistant = AssistantAgent(name="assistant", model_client=..., memory=[memory])
                await memory.close()


            asyncio.run(main())
    """

    component_config_schema = EventEntityMemoryConfig
    component_provider_override = "autogen_ext.memory.event_entity.EventEntityMemory"

    def __init__(self, config: EventEntityMemoryConfig | None = None) -> None:
        self._config = config or EventEntityMemoryConfig()
        self._events: List[_EventRecord] = []
        self._entity_index: Dict[str, Set[int]] = defaultdict(set)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject the query-scoped neighbourhood into ``model_context`` before inference."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        query_results = await self.query(query_text)
        if query_results.results:
            lines = [f"{i}. {memory.content}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memory content (event-entity neighbourhood):\n" + "\n".join(lines)
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Index a chunk as an event paired with its entities."""
        text = self._extract_text(content)
        event_id = len(self._events)
        record = _EventRecord(
            id=event_id, text=text, mime_type=content.mime_type, entities=set(self._collect_entities(content, text))
        )
        self._events.append(record)
        for entity in record.entities:
            self._entity_index[entity].add(event_id)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve a query-scoped neighbourhood by joining indexed chunks on shared entities.

        Seed chunks share at least one entity with the query; ``max_hops`` then
        grows the neighbourhood along further shared-entity joins, which is what
        lets multi-hop evidence chain together even when no single chunk matches
        the whole query.
        """
        query_entities = self._extract_query_entities(query)
        if not query_entities or not self._events:
            return MemoryQueryResult(results=[])

        # Hop 1: seed chunks sharing an entity with the query. Score = number of
        # shared query entities (join strength).
        scored: Dict[int, float] = {}
        for entity in query_entities:
            for event_id in self._events_for_entity(entity):
                scored[event_id] = scored.get(event_id, 0.0) + 1.0

        # Hops 2..max_hops: grow the neighbourhood along shared-entity joins,
        # penalising deeper hops so direct evidence ranks first.
        frontier: Set[int] = set(scored)
        visited: Set[int] = set(frontier)
        for hop in range(2, self._config.max_hops + 1):
            if not frontier:
                break
            penalty = 1.0 / (hop - 1)
            next_frontier: Set[int] = set()
            for event_id in frontier:
                for entity in self._events[event_id].entities:
                    for neighbour in self._events_for_entity(entity):
                        if neighbour in visited or neighbour in next_frontier:
                            continue
                        next_frontier.add(neighbour)
                        scored[neighbour] = scored.get(neighbour, 0.0) + penalty
            visited |= next_frontier
            frontier = next_frontier

        return MemoryQueryResult(results=self._build_results(scored))

    async def clear(self) -> None:
        """Drop every indexed event and the entity join index."""
        self._events.clear()
        self._entity_index.clear()

    async def close(self) -> None:
        """Release resources. The in-process store holds nothing external."""

    def _events_for_entity(self, entity: str) -> Set[int]:
        return self._entity_index.get(entity) or set()

    def _build_results(self, scored: Dict[int, float]) -> List[MemoryContent]:
        threshold = self._config.score_threshold
        results: List[MemoryContent] = []
        for event_id, score in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0])):
            if threshold is not None and score < threshold:
                continue
            record = self._events[event_id]
            metadata: Dict[str, Any] = {"score": score, "entities": sorted(record.entities), "id": str(event_id)}
            results.append(MemoryContent(content=record.text, mime_type=record.mime_type, metadata=metadata))
            if len(results) >= self._config.k:
                break
        return results

    def _collect_entities(self, content: MemoryContent, text: str) -> List[str]:
        """Prefer caller-supplied entities; fall back to the heuristic extractor."""
        supplied = content.metadata.get(self._config.entities_key) if content.metadata else None
        if supplied is not None:
            return _normalize_entities(supplied)
        return self._heuristic_entities(text)

    def _extract_query_entities(self, query: str | MemoryContent) -> List[str]:
        if isinstance(query, MemoryContent):
            supplied = query.metadata.get(self._config.entities_key) if query.metadata else None
            if supplied is not None:
                return _normalize_entities(supplied)
            query_text = self._extract_text(query)
        else:
            query_text = query
        return self._heuristic_entities(query_text)

    def _heuristic_entities(self, text: str) -> List[str]:
        """Parameter-free named-entity proxy: maximal runs of capitalised tokens.

        Replaces the paper's learned extractor. Reliable on text with proper-noun
        casing; for lowercase prose, supply entities explicitly via
        ``metadata["entities"]``.
        """
        min_len = self._config.min_entity_len
        seen: Set[str] = set()
        ordered: List[str] = []
        for match in _CAP_RUN_RE.finditer(text):
            tokens = match.group(0).split()
            # Trim a leading function word such as a sentence-initial "The".
            while tokens and tokens[0].lower() in _STOPWORDS:
                tokens = tokens[1:]
            if not tokens:
                continue
            entity = " ".join(tokens).lower()
            if len(entity) < min_len or entity in seen:
                continue
            seen.add(entity)
            ordered.append(entity)
        return ordered

    def _extract_text(self, content_item: str | MemoryContent) -> str:
        if isinstance(content_item, str):
            return content_item
        content = content_item.content
        mime_type = content_item.mime_type
        if mime_type in (MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN):
            return str(content)
        if mime_type == MemoryMimeType.JSON:
            if isinstance(content, dict):
                return str(content)
            raise ValueError("JSON content must be a dict")
        raise ValueError(f"Unsupported content type for event-entity indexing: {mime_type}")

    def _to_config(self) -> EventEntityMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: EventEntityMemoryConfig) -> Self:
        return cls(config=config)


def _normalize_entities(raw: object) -> List[str]:
    """Normalize a caller-supplied entity iterable to deduplicated lowercase strings."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in raw:  # type: ignore[union-attr]
        entity = str(item).strip().lower()
        if entity and entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    return ordered

from typing import List

import pytest
from autogen_core import Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage

from autogen_ext.memory import EventEntityMemory, EventEntityMemoryConfig
from autogen_ext.memory.event_entity import EventEntityMemory as EventEntityMemoryDirect

# Building blocks of a small multi-hop chain. The query mentions only "Alice";
# the other chunks are reachable only through shared entities (the join keys).
_CHAIN = [
    MemoryContent(
        content="Alice joined the Apollo project in 1969.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"entities": ["Alice", "Apollo project"]},
    ),
    MemoryContent(
        content="The Apollo project landed humans on the Moon.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"entities": ["Apollo project", "Moon"]},
    ),
    MemoryContent(
        content="The Moon is Earth's only natural satellite.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"entities": ["Moon"]},
    ),
    MemoryContent(
        content="Bob likes baking bread on weekends.",  # unrelated
        mime_type=MemoryMimeType.TEXT,
        metadata={"entities": ["Bob"]},
    ),
]


async def _build_chain(config: EventEntityMemoryConfig) -> EventEntityMemory:
    memory = EventEntityMemory(config)
    await memory.clear()
    for chunk in _CHAIN:
        await memory.add(chunk)
    return memory


def _texts(memory_results: List[MemoryContent]) -> List[str]:
    return [str(r.content) for r in memory_results]


@pytest.mark.asyncio
async def test_seed_only_when_max_hops_is_one() -> None:
    """max_hops=1 returns only chunks directly sharing an entity with the query."""
    memory = await _build_chain(EventEntityMemoryConfig(k=5, max_hops=1))
    try:
        results = _texts((await memory.query("Alice")).results)
    finally:
        await memory.close()

    assert any("Alice" in t for t in results)
    assert not any("landed humans" in t for t in results)  # hop-2 evidence excluded
    assert not any("Bob" in t for t in results)  # unrelated


@pytest.mark.asyncio
async def test_max_hops_grows_neighborhood_along_shared_entities() -> None:
    """Increasing max_hops chains more shared-entity joins, surfacing multi-hop evidence."""
    memory = await _build_chain(EventEntityMemoryConfig(k=5, max_hops=3))
    try:
        results = _texts((await memory.query("Alice")).results)
    finally:
        await memory.close()

    assert any("Alice" in t for t in results)  # hop 1 (seed)
    assert any("landed humans" in t for t in results)  # hop 2 via "Apollo project"
    assert any("natural satellite" in t for t in results)  # hop 3 via "Moon"
    assert not any("Bob" in t for t in results)  # never reachable


@pytest.mark.asyncio
async def test_heuristic_extraction_joins_on_capitalized_spans() -> None:
    """Without caller-supplied entities, capitalized spans act as the join keys."""
    memory = EventEntityMemory(EventEntityMemoryConfig(k=3, max_hops=1))
    await memory.clear()
    await memory.add(MemoryContent(content="Paris hosts the Eiffel Tower.", mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content="London is known for the Tower Bridge.", mime_type=MemoryMimeType.TEXT))
    try:
        results = _texts((await memory.query("What about Paris?")).results)
    finally:
        await memory.close()

    assert any("Eiffel" in t for t in results)
    assert not any("London" in t for t in results)


@pytest.mark.asyncio
async def test_score_threshold_filters_weak_joins() -> None:
    """A high threshold drops the penalized deeper-hop chunks."""
    memory = await _build_chain(EventEntityMemoryConfig(k=5, max_hops=3, score_threshold=0.9))
    try:
        results = _texts((await memory.query("Alice")).results)
    finally:
        await memory.close()

    assert any("Alice" in t for t in results)  # seed score 1.0 >= 0.9
    assert any("landed humans" in t for t in results)  # hop-2 score 1.0 >= 0.9
    assert not any("natural satellite" in t for t in results)  # hop-3 score 0.5 < 0.9


@pytest.mark.asyncio
async def test_update_context_injects_neighborhood_into_model_context() -> None:
    """update_context exercises the same context-injection path AssistantAgent uses."""
    memory = EventEntityMemory(EventEntityMemoryConfig(k=2, max_hops=2))
    await memory.clear()
    await memory.add(MemoryContent(content="Paris hosts the Eiffel Tower.", mime_type=MemoryMimeType.TEXT))

    context = BufferedChatCompletionContext()
    await context.add_message(UserMessage(content="What about Paris?", source="user"))

    try:
        result = await memory.update_context(context)
        assert result.memories.results  # retrieved the joined neighbourhood

        messages = await context.get_messages()
        injected = [str(m.content) for m in messages]
        assert any("event-entity neighbourhood" in m for m in injected)  # SystemMessage wrapping was added
        assert any("Eiffel" in m for m in injected)  # verbatim chunk text, not a decomposed triple
    finally:
        await memory.close()


def test_component_round_trip() -> None:
    """The memory round-trips through the declarative Component loader."""
    memory = EventEntityMemory(EventEntityMemoryConfig(k=5, max_hops=3))
    model = memory.dump_component()
    assert model.provider == "autogen_ext.memory.event_entity.EventEntityMemory"

    loaded = Component.load_component(model, Memory)
    assert isinstance(loaded, EventEntityMemory)
    # Public re-export and direct submodule import are the same class.
    assert EventEntityMemory is EventEntityMemoryDirect

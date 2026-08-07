"""Integration tests for :class:`EntityTemporalMemory`.

These tests drive the memory through the public ``Memory`` ABC (defined in the
non-new ``autogen_core.memory`` module) and a real ``BufferedChatCompletionContext``
from ``autogen_core.model_context`` -- the same surface ``AssistantAgent`` consumes.
They also exercise the ``Component`` loader round-trip. No model client is
constructed anywhere: memory operations make zero LLM calls by construction.
"""

import pytest
from autogen_core import ComponentModel
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage

# Imported via the package re-export to exercise the wiring in
# autogen_ext/memory/__init__.py (the call site).
from autogen_ext.memory import EntityTemporalMemory


@pytest.mark.asyncio
async def test_entity_temporal_empty() -> None:
    """An empty memory returns no results and injects nothing."""
    memory = EntityTemporalMemory(name="empty")
    context = BufferedChatCompletionContext(buffer_size=3)

    query_results = await memory.query("anything")
    assert query_results.results == []

    update_results = await memory.update_context(context)
    assert update_results.memories.results == []
    assert await context.get_messages() == []


@pytest.mark.asyncio
async def test_entity_temporal_entity_view_lifts_older_match() -> None:
    """An entity-dense query ranks an older matching trace above a newer unrelated one."""
    memory = EntityTemporalMemory()
    await memory.add(MemoryContent(content="user prefers metric units", mime_type=MemoryMimeType.TEXT))  # older
    await memory.add(MemoryContent(content="weather sunny today", mime_type=MemoryMimeType.TEXT))  # newer

    results = await memory.query("metric units")

    # The older metric-units trace wins because the entity view dominates the
    # query-dependent weighting; both traces survive calibration.
    assert [str(r.content) for r in results.results] == ["user prefers metric units", "weather sunny today"]


@pytest.mark.asyncio
async def test_entity_temporal_temporal_fallback_for_vague_query() -> None:
    """A query that shares no entities falls back to the temporal (recency) view."""
    memory = EntityTemporalMemory()
    await memory.add(MemoryContent(content="alpha beta", mime_type=MemoryMimeType.TEXT))  # older
    await memory.add(MemoryContent(content="gamma delta", mime_type=MemoryMimeType.TEXT))  # newer

    results = await memory.query("zzz unrelated")

    # No entity overlap -> pure recency ordering, newest first.
    assert [str(r.content) for r in results.results] == ["gamma delta", "alpha beta"]


@pytest.mark.asyncio
async def test_entity_temporal_calibration_suppresses_near_duplicates() -> None:
    """Deterministic calibration drops redundant evidence (near-duplicate term sets)."""
    memory = EntityTemporalMemory()
    await memory.add(MemoryContent(content="the user likes tea", mime_type=MemoryMimeType.TEXT))  # older
    await memory.add(MemoryContent(content="user likes the tea", mime_type=MemoryMimeType.TEXT))  # newer

    results = await memory.query("tea")

    # Identical term sets -> Jaccard >= dup_threshold -> only the higher-scored trace survives.
    assert len(results.results) == 1
    assert str(results.results[0].content) == "user likes the tea"


@pytest.mark.asyncio
async def test_entity_temporal_update_context_injects_system_message() -> None:
    """``update_context`` retrieves calibrated traces and injects them as a system message."""
    memory = EntityTemporalMemory()
    await memory.add(MemoryContent(content="user prefers metric units", mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content="weather sunny today", mime_type=MemoryMimeType.TEXT))

    context = BufferedChatCompletionContext(buffer_size=5)
    await context.add_message(UserMessage(content="which units does the user prefer", source="user"))

    update_results = await memory.update_context(context)
    messages = await context.get_messages()

    # The original user message plus one injected system message.
    assert len(messages) == 2
    assert isinstance(messages[-1], SystemMessage)
    # The entity-matching trace is retrieved first and surfaced in the injected context.
    assert "user prefers metric units" in str(messages[-1].content)
    assert str(update_results.memories.results[0].content) == "user prefers metric units"


@pytest.mark.asyncio
async def test_entity_temporal_component_round_trip() -> None:
    """The component can be dumped and reloaded, preserving configuration and traces."""
    memory = EntityTemporalMemory(
        name="round_trip",
        traces=[MemoryContent(content="alpha beta", mime_type=MemoryMimeType.TEXT)],
        top_k=3,
        score_threshold=0.1,
    )

    config = memory.dump_component()
    assert isinstance(config, ComponentModel)
    assert config.provider == "autogen_ext.memory.entity_temporal.EntityTemporalMemory"
    assert config.component_type == "memory"

    loaded = Memory.load_component(config)
    assert isinstance(loaded, EntityTemporalMemory)
    assert loaded.name == "round_trip"

    # Traces survived the round-trip and remain queryable with zero LLM calls.
    results = await loaded.query("alpha")
    assert len(results.results) == 1
    assert str(results.results[0].content) == "alpha beta"

"""Tests for the mode-routed memory.

These tests exercise :class:`RoutedMemory` by composing it from existing
:class:`~autogen_core.memory.Memory` backends (``ListMemory``) and driving it through
the public ``Memory`` interface, demonstrating that routing dispatches to the correct
backend and that the class plugs into autogen's declarative component protocol.
"""

import pytest
from autogen_core import ComponentModel
from autogen_core.memory import ListMemory, Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage

from autogen_ext.memory.routed import QueryMode, RoutedMemory, classify_query


def _content(text: str) -> MemoryContent:
    return MemoryContent(content=text, mime_type=MemoryMimeType.TEXT)


def test_classify_query_modes() -> None:
    """The parameter-free classifier maps representative queries to the three modes."""
    assert classify_query("What is the capital of France?") is QueryMode.FACTOID
    assert classify_query("How is Alice related to Bob?") is QueryMode.RELATION
    assert classify_query("Summarize the overall history of the project.") is QueryMode.SYNTHESIS
    assert classify_query("") is QueryMode.FACTOID


@pytest.mark.asyncio
async def test_query_routes_to_mode_specific_backend() -> None:
    """A query only hits the backend bound to its classified mode."""
    factoid = ListMemory(memory_contents=[_content("The capital of France is Paris.")])
    relation = ListMemory(memory_contents=[_content("Alice reports to Bob, who manages Carol.")])
    synthesis = ListMemory(memory_contents=[_content("Project X shipped in three phases.")])

    memory = RoutedMemory(
        routes={QueryMode.FACTOID: factoid, QueryMode.RELATION: relation, QueryMode.SYNTHESIS: synthesis}
    )

    fact = await memory.query("What is the capital of France?")
    assert fact.results
    assert "Paris" in str(fact.results[0].content)
    assert memory.last_mode is QueryMode.FACTOID

    rel = await memory.query("How is Alice related to Bob?")
    assert rel.results
    assert "reports to Bob" in str(rel.results[0].content)
    assert memory.last_mode is QueryMode.RELATION
    # The factoid store must not leak into the relation route.
    assert all("Paris" not in str(item.content) for item in rel.results)


@pytest.mark.asyncio
async def test_unrouted_mode_falls_back_to_default() -> None:
    """A mode with no explicit route is served by the default_mode backend."""
    factoid = ListMemory(memory_contents=[_content("Paris is the capital of France.")])
    memory = RoutedMemory(routes={QueryMode.FACTOID: factoid})

    # A synthesis query has no bound backend; it falls back to FACTOID.
    result = await memory.query("Summarize everything that happened.")
    assert result.results
    assert "Paris" in str(result.results[0].content)
    assert memory.last_mode is QueryMode.SYNTHESIS


@pytest.mark.asyncio
async def test_add_fans_out_to_shared_substrate() -> None:
    """Ingested content reaches every bound backend (shared ingest substrate)."""
    factoid = ListMemory()
    relation = ListMemory()
    memory = RoutedMemory(routes={QueryMode.FACTOID: factoid, QueryMode.RELATION: relation})

    await memory.add(_content("shared ingest item"))

    factoid_hits = await factoid.query("")
    relation_hits = await relation.query("")
    assert any("shared ingest item" in str(item.content) for item in factoid_hits.results)
    assert any("shared ingest item" in str(item.content) for item in relation_hits.results)


@pytest.mark.asyncio
async def test_update_context_routes_latest_user_message() -> None:
    """update_context classifies the latest user message and injects the routed memories."""
    memory = RoutedMemory(
        routes={QueryMode.FACTOID: ListMemory(memory_contents=[_content("Paris is the capital of France.")])}
    )
    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="What is the capital of France?", source="user"))

    result = await memory.update_context(context)

    assert memory.last_mode is QueryMode.FACTOID
    assert result.memories.results
    messages = await context.get_messages()
    assert any("Paris" in str(message.content) for message in messages)


def test_component_roundtrip() -> None:
    """RoutedMemory participates in the declarative Memory component protocol."""
    memory = RoutedMemory(
        name="demo",
        routes={QueryMode.FACTOID: ListMemory(memory_contents=[_content("hello world")])},
    )
    config = memory.dump_component()
    assert isinstance(config, ComponentModel)
    assert config.provider == "autogen_ext.memory.routed.RoutedMemory"

    loaded = Memory.load_component(config)
    assert isinstance(loaded, RoutedMemory)
    assert loaded.name == "demo"
    # The factoid route survives the declarative round-trip (enum serializes to its value).
    dumped_modes = {route["mode"] for route in loaded.dump_component().config["routes"]}
    assert QueryMode.FACTOID in dumped_modes or QueryMode.FACTOID.value in dumped_modes

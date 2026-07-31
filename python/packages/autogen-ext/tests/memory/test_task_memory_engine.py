"""Integration tests for the Task Memory Engine.

These exercise :class:`TaskMemoryEngine` through the existing
:class:`~autogen_core.memory.Memory` ABC and the real model-context surface
from ``autogen_core``, and import the engine via the public
``autogen_ext.memory`` re-export (the integration call site).
"""

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage

# Imported from the (non-new) autogen_ext.memory package re-export -- the call site.
from autogen_ext.memory import TaskMemoryEngine


@pytest.mark.asyncio
async def test_new_creates_node_and_update_records_history() -> None:
    engine = TaskMemoryEngine()
    await engine.add(
        MemoryContent(content="celery", mime_type=MemoryMimeType.TEXT, metadata={"slot": "ingredient:celery"})
    )
    await engine.add(
        MemoryContent(
            content="mushrooms",
            mime_type=MemoryMimeType.TEXT,
            metadata={"slot": "ingredient:celery", "intent": "update", "replaces": "celery"},
        )
    )

    node = next(iter(engine.nodes.values()))
    assert node.value == "mushrooms"
    assert node.history == ["celery"]
    assert node.metadata.get("replaces") == "celery"


@pytest.mark.asyncio
async def test_implicit_update_when_slot_already_exists() -> None:
    # No explicit intent: a repeated slot is classified as UPDATE by the resolver.
    engine = TaskMemoryEngine()
    await engine.add(MemoryContent(content="Boston", mime_type=MemoryMimeType.TEXT, metadata={"slot": "trip:origin"}))
    await engine.add(MemoryContent(content="Seattle", mime_type=MemoryMimeType.TEXT, metadata={"slot": "trip:origin"}))

    node = next(iter(engine.nodes.values()))
    assert node.value == "Seattle"
    assert node.history == ["Boston"]
    assert len(engine.nodes) == 1


@pytest.mark.asyncio
async def test_shared_node_revision_propagates_to_all_parents() -> None:
    # One celery node is depended on by both soup and dumplings; revising it
    # once must flag both dependents stale (global propagation through the DAG).
    engine = TaskMemoryEngine()
    await engine.add(
        MemoryContent(content="celery", mime_type=MemoryMimeType.TEXT, metadata={"slot": "ingredient:celery"})
    )
    celery_id = next(iter(engine.nodes.values())).node_id
    await engine.add(
        MemoryContent(
            content="soup", mime_type=MemoryMimeType.TEXT, metadata={"slot": "dish:soup", "dependencies": [celery_id]}
        )
    )
    await engine.add(
        MemoryContent(
            content="dumplings",
            mime_type=MemoryMimeType.TEXT,
            metadata={"slot": "dish:dumplings", "dependencies": [celery_id]},
        )
    )

    await engine.add(
        MemoryContent(
            content="mushrooms",
            mime_type=MemoryMimeType.TEXT,
            metadata={"slot": "ingredient:celery", "intent": "update"},
        )
    )

    stale = [n for n in engine.nodes.values() if n.metadata.get("stale_after") == "ingredient:celery"]
    assert {n.slot for n in stale} == {"dish:soup", "dish:dumplings"}


@pytest.mark.asyncio
async def test_update_context_surfaces_current_state_not_stale_history() -> None:
    engine = TaskMemoryEngine()
    await engine.add(MemoryContent(content="Boston", mime_type=MemoryMimeType.TEXT, metadata={"slot": "trip:origin"}))
    await engine.add(
        MemoryContent(
            content="Seattle", mime_type=MemoryMimeType.TEXT, metadata={"slot": "trip:origin", "intent": "update"}
        )
    )

    context = BufferedChatCompletionContext(buffer_size=10)
    result = await engine.update_context(context)

    messages = await context.get_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    summary = str(messages[0].content)
    assert "Seattle" in summary
    assert "Boston" not in summary  # superseded value is not surfaced
    assert len(result.memories.results) == 1
    assert result.memories.results[0].content == "Seattle"


@pytest.mark.asyncio
async def test_inactivate_then_check_query_excludes_node() -> None:
    engine = TaskMemoryEngine()
    await engine.add(
        MemoryContent(content="book flight", mime_type=MemoryMimeType.TEXT, metadata={"slot": "task:flight"})
    )
    await engine.add(
        MemoryContent(
            content="", mime_type=MemoryMimeType.TEXT, metadata={"slot": "task:flight", "intent": "inactivate"}
        )
    )

    node = next(iter(engine.nodes.values()))
    assert node.active is False

    checked = await engine.query("flight")  # TRIM "check" intent
    assert checked.results == []


@pytest.mark.asyncio
async def test_roll_back_restores_prior_value() -> None:
    engine = TaskMemoryEngine()
    await engine.add(
        MemoryContent(content="celery", mime_type=MemoryMimeType.TEXT, metadata={"slot": "ingredient:celery"})
    )
    await engine.add(
        MemoryContent(
            content="mushrooms",
            mime_type=MemoryMimeType.TEXT,
            metadata={"slot": "ingredient:celery", "intent": "update"},
        )
    )
    await engine.add(
        MemoryContent(
            content="", mime_type=MemoryMimeType.TEXT, metadata={"slot": "ingredient:celery", "intent": "roll_back"}
        )
    )

    node = next(iter(engine.nodes.values()))
    assert node.value == "celery"


@pytest.mark.asyncio
async def test_clear_resets_dag() -> None:
    engine = TaskMemoryEngine()
    await engine.add(MemoryContent(content="x", mime_type=MemoryMimeType.TEXT, metadata={"slot": "a"}))
    await engine.clear()
    assert engine.nodes == {}


def test_declarative_component_round_trip() -> None:
    engine = TaskMemoryEngine(name="tme")

    dumped = engine.dump_component()
    assert dumped.provider == "autogen_ext.memory.task_memory_engine.TaskMemoryEngine"

    restored = TaskMemoryEngine._from_config(engine._to_config())
    assert isinstance(restored, TaskMemoryEngine)
    assert restored.name == "tme"

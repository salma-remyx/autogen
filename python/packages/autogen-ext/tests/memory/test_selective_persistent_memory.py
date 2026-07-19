import pytest
from autogen_core import ComponentModel
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext

# Exercises the call-site wiring in autogen_ext/memory/__init__.py.
from autogen_ext.memory import SelectivePersistentMemory
from autogen_ext.memory.selective_persistent import MemoryCategory, classify_content


def test_classify_routes_reusable_categories() -> None:
    assert (
        classify_content(MemoryContent(content="Task: build a sales dashboard for Q3", mime_type=MemoryMimeType.TEXT))
        == MemoryCategory.TASK_SPECIFICATION
    )
    assert (
        classify_content(MemoryContent(content="Schema: users(id int, name string)", mime_type=MemoryMimeType.TEXT))
        == MemoryCategory.DATA_SCHEMA
    )
    assert (
        classify_content(
            MemoryContent(content="API endpoint: https://x.example, api_key: sk-abc", mime_type=MemoryMimeType.TEXT)
        )
        == MemoryCategory.TOOL_CONFIGURATION
    )
    assert (
        classify_content(
            MemoryContent(content="Output must be valid JSON with a name field", mime_type=MemoryMimeType.TEXT)
        )
        == MemoryCategory.OUTPUT_CONSTRAINT
    )


def test_classify_discards_reasoning_traces() -> None:
    trace = "Let me think... first I'll parse the file, then I'll sum the rows. So the answer is 42."
    assert (
        classify_content(MemoryContent(content=trace, mime_type=MemoryMimeType.TEXT)) == MemoryCategory.REASONING_TRACE
    )


def test_classify_metadata_override() -> None:
    forced = MemoryContent(
        content="text with no category markers at all",
        mime_type=MemoryMimeType.TEXT,
        metadata={"category": "data_schema"},
    )
    assert classify_content(forced) == MemoryCategory.DATA_SCHEMA

    ephemeral = MemoryContent(
        content="Task: do something important",
        mime_type=MemoryMimeType.TEXT,
        metadata={"persist": False},
    )
    assert classify_content(ephemeral) == MemoryCategory.REASONING_TRACE


@pytest.mark.asyncio
async def test_add_discards_reasoning_and_tags_reusable() -> None:
    # ListMemory is the existing (non-new) backing store; the wrapper integrates with it.
    backend = ListMemory(name="reusable")
    memory = SelectivePersistentMemory(backend=backend)

    await memory.add(MemoryContent(content="Schema: users(id int, name string)", mime_type=MemoryMimeType.TEXT))
    await memory.add(
        MemoryContent(
            content="Let me think... first I'll parse the file, then I'll sum. So the answer is 42.",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    # Only the reusable schema reached the backing store.
    assert len(backend.content) == 1
    assert memory.kept_count == 1
    assert memory.discarded_count == 1
    assert backend.content[0].metadata == {"category": "data_schema"}
    assert memory.category_counts[MemoryCategory.DATA_SCHEMA] == 1
    assert memory.category_counts[MemoryCategory.REASONING_TRACE] == 1


@pytest.mark.asyncio
async def test_update_context_only_injects_persisted_content() -> None:
    backend = ListMemory(name="reusable")
    memory = SelectivePersistentMemory(backend=backend)

    await memory.add(
        MemoryContent(content="API endpoint: https://x.example, api_key: sk-abc", mime_type=MemoryMimeType.TEXT)
    )
    await memory.add(MemoryContent(content="hmm, let me think about this step by step", mime_type=MemoryMimeType.TEXT))

    model_context = BufferedChatCompletionContext(buffer_size=10)
    result = await memory.update_context(model_context)

    # Only the retained tool configuration is surfaced to the agent.
    assert len(result.memories.results) == 1
    messages = await model_context.get_messages()
    assert "api_key" in str(messages[0].content)


@pytest.mark.asyncio
async def test_clear_resets_backend_and_counters() -> None:
    backend = ListMemory(name="reusable")
    memory = SelectivePersistentMemory(backend=backend)
    await memory.add(MemoryContent(content="Task: build a dashboard", mime_type=MemoryMimeType.TEXT))
    assert memory.kept_count == 1

    await memory.clear()
    assert len(backend.content) == 0
    assert memory.kept_count == 0
    assert memory.discarded_count == 0


def test_component_round_trip() -> None:
    backend = ListMemory(name="reusable")
    memory = SelectivePersistentMemory(backend=backend, default_category=MemoryCategory.REASONING_TRACE)

    model: ComponentModel = memory.dump_component()
    assert model.provider == "autogen_ext.memory.selective_persistent.SelectivePersistentMemory"

    loaded = SelectivePersistentMemory.load_component(model)
    assert isinstance(loaded, SelectivePersistentMemory)
    assert isinstance(loaded.backend, ListMemory)
    assert loaded.default_category == MemoryCategory.REASONING_TRACE

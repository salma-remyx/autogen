import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage
from autogen_ext.memory.proactive_memory import ProactiveMemory, ProactiveMemoryConfig


@pytest.mark.asyncio
async def test_intervenes_when_decision_relevant_state_is_buried() -> None:
    """The core selective-intervention behaviour: re-surface buried state."""
    memory = ProactiveMemory()
    await memory.add(
        MemoryContent(
            content="The final answer must be a single word with no punctuation.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "status"},
        )
    )

    context = UnboundedChatCompletionContext()
    # Constraint stated, then buried under many unrelated turns that never echo
    # the distinctive tokens "single"/"word"/"punctuation".
    await context.add_message(UserMessage(content="Summarize the weather in one word.", source="user"))
    for i in range(8):
        await context.add_message(
            AssistantMessage(content=f"Draft {i}: the sky is overcast with light drizzle today.", source="assistant")
        )
    await context.add_message(UserMessage(content="Now produce the final answer.", source="user"))

    result = await memory.update_context(context)

    # Selective intervention: the buried constraint is surfaced.
    assert len(result.memories.results) >= 1
    assert any("single word" in str(r.content) for r in result.memories.results)
    # And the model context now carries a memory-grounded reminder.
    messages = await context.get_messages()
    assert any(isinstance(m, SystemMessage) and "single word" in str(m.content) for m in messages)


@pytest.mark.asyncio
async def test_stays_silent_when_state_is_already_in_recent_context() -> None:
    """No intervention when the relevant state is already reflected recently."""
    memory = ProactiveMemory()
    await memory.add(
        MemoryContent(
            content="The API rate limit is 60 requests per minute.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "procedural"},
        )
    )

    context = UnboundedChatCompletionContext()
    # The latest turn already restates the constraint verbatim -> not buried.
    await context.add_message(
        UserMessage(content="What is the API rate limit again? 60 requests per minute, right?", source="user")
    )

    result = await memory.update_context(context)

    # Remain silent: nothing injected, no context mutation.
    assert len(result.memories.results) == 0
    messages = await context.get_messages()
    assert not any(isinstance(m, SystemMessage) for m in messages)


@pytest.mark.asyncio
async def test_stays_silent_on_empty_bank_or_context() -> None:
    memory = ProactiveMemory()
    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(content="anything", source="user"))

    # Empty bank.
    assert len((await memory.update_context(context)).memories.results) == 0

    await memory.add(MemoryContent(content="a constraint to remember", mime_type=MemoryMimeType.TEXT))
    empty_context = UnboundedChatCompletionContext()
    # Empty context.
    assert len((await memory.update_context(empty_context)).memories.results) == 0


@pytest.mark.asyncio
async def test_query_ranks_by_bm25_relevance() -> None:
    memory = ProactiveMemory()
    await memory.add(MemoryContent(content="PostgreSQL connection string format", mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content="How to bake sourdough bread", mime_type=MemoryMimeType.TEXT))

    results = await memory.query("database connection")

    assert results.results
    assert "PostgreSQL" in str(results.results[0].content)


@pytest.mark.asyncio
async def test_category_falls_back_to_knowledge_for_unknown_metadata() -> None:
    memory = ProactiveMemory()
    await memory.add(MemoryContent(content="Ship behind a feature flag before release.", mime_type=MemoryMimeType.TEXT))
    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(content="How should we ship this?", source="user"))

    await memory.update_context(context)

    messages = await context.get_messages()
    reminder = next(m for m in messages if isinstance(m, SystemMessage))
    assert "[knowledge]" in str(reminder.content)


@pytest.mark.asyncio
async def test_component_round_trip() -> None:
    memory = ProactiveMemory(name="bank", k=5, recency_window=4, score_threshold=0.25)
    config = memory._to_config()
    assert config == ProactiveMemoryConfig(name="bank", k=5, recency_window=4, score_threshold=0.25)
    restored = ProactiveMemory._from_config(config)
    assert restored.name == "bank" and restored._k == 5 and restored._recency_window == 4  # type: ignore[attr-defined]

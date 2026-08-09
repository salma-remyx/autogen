from typing import cast

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.memory import ArbitratedMemory as ReExportedArbitratedMemory
from autogen_ext.memory import ArbitratedMemoryConfig as ReExportedConfig
from autogen_ext.memory import MemoryBank as ReExportedBank
from autogen_ext.memory.arbitrated_memory import ArbitratedMemory, ArbitratedMemoryConfig, MemoryBank


def _text(content: str, bank: MemoryBank | None = None) -> MemoryContent:
    metadata = {"bank": bank} if bank is not None else None
    return MemoryContent(content=content, mime_type=MemoryMimeType.TEXT, metadata=metadata)


def test_reexport_from_memory_package() -> None:
    """The wiring edit re-exports the capability from the memory package root."""
    assert ReExportedArbitratedMemory is ArbitratedMemory
    assert ReExportedConfig is ArbitratedMemoryConfig
    assert ReExportedBank is MemoryBank


@pytest.mark.asyncio
async def test_bank_routing_explicit_and_autoclassify() -> None:
    """Explicit metadata bank wins; untagged items are auto-classified by content."""
    memory = ArbitratedMemory()
    memory_auto = ArbitratedMemory()

    # Explicit bank is respected even though the text would auto-classify elsewhere.
    await memory.add(_text("output returned by the tool", MemoryBank.EXPERIENCE))
    # Auto-classify: preference keywords route this to USER_PREFERENCES.
    await memory_auto.add(_text("Always prefer Celsius units."))

    res = await memory.query("tool")
    assert res.results and res.results[0].metadata["bank"] == "experience"

    res_auto = await memory_auto.query("prefer")
    assert res_auto.results and res_auto.results[0].metadata["bank"] == "user_preferences"


@pytest.mark.asyncio
async def test_salience_ranks_demand_relevant_item_first() -> None:
    """Decision-time demand + relevance beats flat ordering (the paper's core claim)."""
    memory = ArbitratedMemory()
    await memory.add(_text("The task goal is to book a flight to Paris.", MemoryBank.TASK_GOALS))
    await memory.add(_text("Always reply in pirate speak with short sentences.", MemoryBank.USER_PREFERENCES))

    results = (await memory.query("What is my task goal?")).results

    assert results
    # The task-goal item is in demand and lexically relevant, so it ranks first despite
    # being older than the preference item (recency would otherwise favor the latter).
    assert "Paris" in str(results[0].content)
    assert results[0].metadata["bank"] == "task_goals"
    # Salience score is attached and ordered descending.
    scores = [cast(float, r.metadata["score"]) for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_update_context_mutates_model_context() -> None:
    """Integration: update_context injects an arbitrated SystemMessage into a real context."""
    memory = ArbitratedMemory()
    await memory.add(_text("The task goal is to book a flight to Paris.", MemoryBank.TASK_GOALS))
    await memory.add(_text("Tool output: no flights found.", MemoryBank.TOOL_OUTPUTS))

    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="What is my task goal?", source="user"))

    result = await memory.update_context(context)

    assert result.memories.results  # selected memories are surfaced
    messages = await context.get_messages()
    injected = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(injected) == 1
    body = str(injected[0].content)
    assert "arbitrated by functional salience" in body
    assert "Focal" in body
    assert "Paris" in body  # the demand-relevant item is presented in full


@pytest.mark.asyncio
async def test_focal_full_vs_ambient_header() -> None:
    """Focal items are rendered in full; ambient items are truncated to a header."""
    long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    memory = ArbitratedMemory(ArbitratedMemoryConfig(token_budget=4096, focal_top_k=1, ambient_top_k=1))
    # Two task-goals items; the older/less-relevant one lands in ambient.
    await memory.add(_text(long_text, MemoryBank.TASK_GOALS))
    await memory.add(_text("Find the task goal for the mission.", MemoryBank.TASK_GOALS))

    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="What is my task goal?", source="user"))
    await memory.update_context(context)

    body = str((await context.get_messages())[-1].content)
    # The long item is truncated to a header (first 8 words + ellipsis) when ambient.
    assert "alpha beta gamma delta epsilon zeta eta theta..." in body
    assert "kappa lambda mu" not in body  # the tail beyond the header is dropped


@pytest.mark.asyncio
async def test_token_budget_bounds_presentation() -> None:
    """A tighter per-step budget surfaces fewer items than a generous one."""
    items = [f"Task goal number {i} is to complete step i." for i in range(6)]

    big = ArbitratedMemory(ArbitratedMemoryConfig(token_budget=4096, focal_top_k=6, ambient_top_k=6))
    small = ArbitratedMemory(ArbitratedMemoryConfig(token_budget=20, focal_top_k=6, ambient_top_k=6))
    for text in items:
        await big.add(_text(text, MemoryBank.TASK_GOALS))
        await small.add(_text(text, MemoryBank.TASK_GOALS))

    context_big = BufferedChatCompletionContext(buffer_size=10)
    await context_big.add_message(UserMessage(content="task goal", source="user"))
    n_big = len((await big.update_context(context_big)).memories.results)

    context_small = BufferedChatCompletionContext(buffer_size=10)
    await context_small.add_message(UserMessage(content="task goal", source="user"))
    n_small = len((await small.update_context(context_small)).memories.results)

    assert n_big == 6
    assert 1 <= n_small < 6  # budget forces arbitration, but never returns nothing on a hit


def test_component_round_trip() -> None:
    """Declarative Component config survives a dump/load round-trip."""
    memory = ArbitratedMemory(ArbitratedMemoryConfig(name="arb", token_budget=128, focal_top_k=2, ambient_top_k=3))

    dumped = memory.dump_component()
    assert dumped.provider == "autogen_ext.memory.arbitrated_memory.ArbitratedMemory"

    loaded = ArbitratedMemory.load_component(dumped)
    assert isinstance(loaded, ArbitratedMemory)
    assert loaded.name == "arb"
    cfg = loaded._to_config()
    assert cfg.token_budget == 128
    assert cfg.focal_top_k == 2


@pytest.mark.asyncio
async def test_clear_and_empty_update_context() -> None:
    memory = ArbitratedMemory()
    context = BufferedChatCompletionContext(buffer_size=10)

    # No items -> update_context is a no-op that returns no memories.
    result = await memory.update_context(context)
    assert result.memories.results == []

    await memory.add(_text("goal: find the task", MemoryBank.TASK_GOALS))
    await memory.clear()
    assert (await memory.query("goal")).results == []

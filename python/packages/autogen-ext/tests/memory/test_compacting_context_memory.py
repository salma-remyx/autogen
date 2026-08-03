"""Tests for :class:`CompactingContextMemory`.

These exercise the backend through the *existing* ``Memory`` ABC surface (the real
:class:`~autogen_core.model_context.ChatCompletionContext` that ``AssistantAgent`` uses),
not just the new module in isolation.
"""

import pytest
from autogen_core import CancellationToken
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage

from autogen_ext.memory.compacting_context import CompactingContextMemory
from autogen_ext.memory.compacting_context._compacting_context_memory import _estimate_tokens


def _text(content: str, kind: str = "fact") -> MemoryContent:
    return MemoryContent(content=content, mime_type=MemoryMimeType.TEXT, metadata={"kind": kind})


@pytest.mark.asyncio
async def test_empty_memory_injects_nothing() -> None:
    memory = CompactingContextMemory(token_budget=128)
    context = UnboundedChatCompletionContext()
    result = await memory.update_context(context)
    assert result.memories.results == []
    assert await context.get_messages() == []


@pytest.mark.asyncio
async def test_update_context_injects_within_budget() -> None:
    # Ingest far more than the budget can hold.
    memory = CompactingContextMemory(token_budget=120)
    for i in range(40):
        await memory.add(_text(f"Item number {i}: " + "lorem ipsum dolor " * 6))

    context = UnboundedChatCompletionContext()
    await memory.update_context(context)

    messages = await context.get_messages()
    injected = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(injected) == 1
    # Linear-cost guarantee: the injected message never exceeds the token budget.
    assert _estimate_tokens(str(injected[0].content)) <= memory._token_budget + 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_extractive_preserves_salient_old_content_no_accuracy_cliff() -> None:
    # An old, highly-relevant fact plus several newer, irrelevant, bulky entries.
    extractive = CompactingContextMemory(token_budget=120, compaction_strategy="extractive")
    summary = CompactingContextMemory(token_budget=120, compaction_strategy="summary")
    for memory in (extractive, summary):
        await memory.add(_text("The encryption key for the vault is hunter2-alpha.", kind="decision"))
        for i in range(4):
            await memory.add(_text(" ".join(f"chatter{i}-{j}" for j in range(40)), kind="event"))

    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(source="user", content="What is the vault encryption key?"))

    ext_result = await extractive.update_context(context)
    ext_payload = str(ext_result.memories.results[0].content) if ext_result.memories.results else ""

    sum_context = UnboundedChatCompletionContext()
    await sum_context.add_message(UserMessage(source="user", content="What is the vault encryption key?"))
    sum_result = await summary.update_context(sum_context)
    sum_payload = str(sum_result.memories.results[0].content) if sum_result.memories.results else ""

    # Validated compaction survives the old salient fact despite many newer items...
    assert "hunter2-alpha" in ext_payload
    assert "hunter2-alpha" not in sum_payload  # ...while the lossy summary drops it (accuracy cliff).


@pytest.mark.asyncio
async def test_query_scopes_to_relevant_entries() -> None:
    memory = CompactingContextMemory(max_query_results=2)
    await memory.add(_text("The database runs on port 5432."))
    await memory.add(_text("Lemons are yellow citrus fruits."))
    await memory.add(_text("Postgres credentials live in the vault."))

    result = await memory.query("database postgres vault")
    contents = [str(m.content) for m in result.results]
    assert "Lemons are yellow citrus fruits." not in contents
    assert len(result.results) == 2  # capped by max_query_results


@pytest.mark.asyncio
async def test_token_counter_override_is_honored() -> None:
    # Custom counter charges 1 token per whitespace-separated word; budget is 15 words.
    # Under the default char proxy a 15-token budget (~60 chars) is smaller than the
    # header and would inject nothing -- so a non-empty, word-bounded injection proves
    # the custom counter drives compaction.
    memory = CompactingContextMemory(token_budget=15, token_counter=lambda text: len(text.split()))
    for i in range(10):
        await memory.add(_text(f"item number {i}"))

    context = UnboundedChatCompletionContext()
    await memory.update_context(context)
    injected = [m for m in await context.get_messages() if isinstance(m, SystemMessage)]
    assert len(injected) == 1
    # The custom unit (word count) bounds the entire injected message.
    assert len(str(injected[0].content).split()) <= 15


@pytest.mark.asyncio
async def test_add_clear_round_trip() -> None:
    memory = CompactingContextMemory()
    await memory.add(_text("alpha"))
    await memory.add(_text("beta"))
    assert len(memory.content) == 2
    await memory.clear()
    assert memory.content == []


def test_component_dump_and_reconstruct() -> None:
    memory = CompactingContextMemory(
        name="ctx",
        token_budget=512,
        compaction_strategy="summary",
        recency_decay=0.8,
        recent_context_messages=2,
        max_query_results=5,
    )
    model = memory.dump_component()
    assert model.component_type == "memory"
    assert model.provider == "autogen_ext.memory.compacting_context.CompactingContextMemory"

    rebuilt = CompactingContextMemory._from_config(memory._to_config())
    assert rebuilt.name == "ctx"
    assert rebuilt._token_budget == 512  # type: ignore[attr-defined]
    assert rebuilt._strategy == "summary"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancellation_token_accepted_on_add() -> None:
    memory = CompactingContextMemory()
    await memory.add(_text("alpha"), CancellationToken())
    assert len(memory.content) == 1

from typing import Any

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage

# Imported via the public package surface wired in autogen_ext.memory.__init__.
from autogen_ext.memory import PerishableMemory, PerishabilityProfile


def _meta(result_content: MemoryContent) -> dict[str, Any]:
    """Narrow metadata to a dict for assertions (it is Dict[str, Any] | None)."""
    assert result_content.metadata is not None
    return result_content.metadata


@pytest.mark.asyncio
async def test_add_query_decorates_metadata() -> None:
    memory = PerishableMemory(k=5, clock=lambda: 0.0)
    await memory.add(
        MemoryContent(content="The deploy server is online and healthy.", mime_type=MemoryMimeType.TEXT)
    )
    results = await memory.query("server status")
    assert len(results.results) == 1
    meta = _meta(results.results[0])
    assert "validity" in meta
    assert "memory_type" in meta
    assert meta["validity"] == pytest.approx(1.0)  # age 0 at a frozen clock
    await memory.close()


@pytest.mark.asyncio
async def test_fresh_perishable_fact_outranks_stale() -> None:
    """Core paper result: a fresh perishable fact outranks an otherwise-equal stale one."""
    clock = {"now": 0.0}
    memory = PerishableMemory(k=5, clock=lambda: clock["now"])
    await memory.add(MemoryContent(content="server alpha status online", mime_type=MemoryMimeType.TEXT))
    clock["now"] = 6 * 3600.0  # six hours pass; the first memory has aged
    await memory.add(MemoryContent(content="server alpha status offline", mime_type=MemoryMimeType.TEXT))

    results = await memory.query("server alpha status")
    assert len(results.results) == 2
    # The freshly stored (offline) memory must rank first once decay applies.
    assert "offline" in str(results.results[0].content)
    assert _meta(results.results[0])["score"] >= _meta(results.results[1])["score"]
    await memory.close()


@pytest.mark.asyncio
async def test_decay_ablation_flattens_validity() -> None:
    """Reproduces the paper's ablation: with decay disabled, validity is flat."""
    clock = {"now": 0.0}
    memory = PerishableMemory(k=5, decay_enabled=False, clock=lambda: clock["now"])
    await memory.add(MemoryContent(content="server alpha status online", mime_type=MemoryMimeType.TEXT))
    clock["now"] = 6 * 3600.0
    await memory.add(MemoryContent(content="server alpha status offline", mime_type=MemoryMimeType.TEXT))

    results = await memory.query("server alpha status")
    assert len(results.results) == 2
    # With decay off, both memories keep full validity regardless of age.
    assert _meta(results.results[0])["validity"] == pytest.approx(1.0)
    assert _meta(results.results[1])["validity"] == pytest.approx(1.0)
    await memory.close()


@pytest.mark.asyncio
async def test_durable_facts_do_not_decay() -> None:
    """pi=0 facts (paper: decay harms fact tasks) keep full validity over time."""
    clock = {"now": 0.0}
    memory = PerishableMemory(k=5, clock=lambda: clock["now"])
    await memory.add(
        MemoryContent(
            content="Paris is the capital of France",
            mime_type=MemoryMimeType.TEXT,
            metadata={"memory_type": "fact"},
        )
    )
    clock["now"] = 10 * 86400.0  # ten days later
    results = await memory.query("capital of France")
    assert len(results.results) == 1
    assert _meta(results.results[0])["validity"] == pytest.approx(1.0)
    await memory.close()


@pytest.mark.asyncio
async def test_update_context_injects_memory_into_context() -> None:
    """Integration with the existing Memory ABC hook that AssistantAgent calls each step."""
    memory = PerishableMemory(k=5, clock=lambda: 0.0)
    await memory.add(MemoryContent(content="The build is currently green.", mime_type=MemoryMimeType.TEXT))

    context = BufferedChatCompletionContext(buffer_size=5)
    await context.add_message(UserMessage(content="What is the build status?", source="user"))

    result = await memory.update_context(context)
    assert len(result.memories.results) > 0

    # The original user message plus the injected memory SystemMessage.
    messages = await context.get_messages()
    assert len(messages) > 1
    await memory.close()


@pytest.mark.asyncio
async def test_clear_removes_memories() -> None:
    memory = PerishableMemory(k=5, clock=lambda: 0.0)
    await memory.add(MemoryContent(content="ephemeral status note", mime_type=MemoryMimeType.TEXT))
    assert len((await memory.query("status")).results) == 1
    await memory.clear()
    assert len((await memory.query("status")).results) == 0
    await memory.close()


def test_component_roundtrip() -> None:
    memory = PerishableMemory(
        k=7,
        score_threshold=0.1,
        decay_enabled=True,
        profiles={"custom": PerishabilityProfile(pi=0.3, tau=600.0)},
    )
    dumped = memory.dump_component()
    assert dumped.provider == "autogen_ext.memory.PerishableMemory"
    assert dumped.config["k"] == 7
    assert dumped.config["score_threshold"] == pytest.approx(0.1)
    assert dumped.config["decay_enabled"] is True
    assert "custom" in dumped.config["profiles"]

    loaded = PerishableMemory.load_component(dumped)
    assert isinstance(loaded, PerishableMemory)

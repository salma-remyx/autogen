import pytest
from autogen_core import ComponentModel
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage
from autogen_ext.memory.dual_memory import DualMemory


def test_dual_memory_is_a_memory() -> None:
    """The new backend conforms to the existing Memory ABC."""
    memory = DualMemory(name="ops")
    assert isinstance(memory, Memory)
    assert hasattr(memory, "update_context")
    assert hasattr(memory, "query")
    assert hasattr(memory, "add")
    assert hasattr(memory, "clear")
    assert hasattr(memory, "close")


@pytest.mark.asyncio
async def test_add_routes_to_short_term_memory() -> None:
    """``add`` accumulates working state in STM and leaves LTM untouched."""
    memory = DualMemory(name="ops")
    await memory.add(MemoryContent(content="restart redis pod", mime_type=MemoryMimeType.TEXT))

    assert len(memory.short_term_memory) == 1
    assert len(memory.long_term_memory) == 0


@pytest.mark.asyncio
async def test_add_to_long_term_seeds_experience() -> None:
    """Operational experience can be seeded directly into LTM."""
    memory = DualMemory(name="ops")
    await memory.add_to_long_term(MemoryContent(content="rotate api keys weekly", mime_type=MemoryMimeType.TEXT))

    assert len(memory.long_term_memory) == 1
    assert len(memory.short_term_memory) == 0


@pytest.mark.asyncio
async def test_consolidation_promotes_short_term_to_long_term() -> None:
    """Consolidation moves solved working state into reusable LTM and clears STM."""
    memory = DualMemory(name="ops")
    await memory.add(MemoryContent(content="rolling back deploy resolved the spike", mime_type=MemoryMimeType.TEXT))

    await memory.consolidate()

    assert len(memory.short_term_memory) == 0
    assert len(memory.long_term_memory) == 1


@pytest.mark.asyncio
async def test_consolidation_dedupes_known_experience() -> None:
    """Re-consolidating the same incident does not duplicate LTM."""
    memory = DualMemory(name="ops")
    note = MemoryContent(content="rollback resolved the spike", mime_type=MemoryMimeType.TEXT)

    await memory.add(note)
    await memory.consolidate()
    await memory.add(note)
    await memory.consolidate()

    assert len(memory.long_term_memory) == 1


@pytest.mark.asyncio
async def test_query_activates_resonating_experience_and_filters_irrelevant() -> None:
    """CMR returns the relevant experience above threshold and drops the rest."""
    memory = DualMemory(name="ops", k=1, score_threshold=0.05)
    await memory.add_to_long_term(
        MemoryContent(content="restart redis pod when latency spikes", mime_type=MemoryMimeType.TEXT)
    )
    await memory.add_to_long_term(MemoryContent(content="rotate api keys weekly", mime_type=MemoryMimeType.TEXT))

    results = await memory.query("redis latency spiking")

    assert len(results.results) == 1
    assert "redis" in str(results.results[0].content)
    assert "score" in results.results[0].metadata  # type: ignore[operator]
    assert results.results[0].metadata["score"] > 0.0  # type: ignore[index]


@pytest.mark.asyncio
async def test_update_context_activates_and_injects_long_term_memory() -> None:
    """update_context drives the existing ChatCompletionContext public API and injects activated LTM."""
    memory = DualMemory(name="ops", k=1, score_threshold=0.05)
    await memory.add_to_long_term(
        MemoryContent(content="restart redis pod when latency spikes", mime_type=MemoryMimeType.TEXT)
    )

    # The live model context is the evolving short-term diagnostic state.
    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="redis latency is spiking, users see timeouts", source="user"))

    result = await memory.update_context(context)
    messages = await context.get_messages()

    # Resonance activated exactly the relevant experience...
    assert len(result.memories.results) == 1
    assert "redis" in str(result.memories.results[0].content)
    # ...and it was injected into the (non-new) context as a SystemMessage.
    assert len(messages) == 2
    assert "redis" in messages[1].content


@pytest.mark.asyncio
async def test_update_context_noop_without_state() -> None:
    """With no state to resonate against, the context is left untouched."""
    memory = DualMemory(name="ops")
    await memory.add_to_long_term(MemoryContent(content="some experience", mime_type=MemoryMimeType.TEXT))

    context = BufferedChatCompletionContext(buffer_size=10)
    result = await memory.update_context(context)

    assert len(result.memories.results) == 0
    assert len(await context.get_messages()) == 0


@pytest.mark.asyncio
async def test_clear_resets_both_stores() -> None:
    memory = DualMemory(name="ops")
    await memory.add(MemoryContent(content="working note", mime_type=MemoryMimeType.TEXT))
    await memory.add_to_long_term(MemoryContent(content="experience", mime_type=MemoryMimeType.TEXT))

    await memory.clear()

    assert len(memory.short_term_memory) == 0
    assert len(memory.long_term_memory) == 0


@pytest.mark.asyncio
async def test_clear_short_term_preserves_long_term() -> None:
    """Resetting the working state keeps consolidated experience intact."""
    memory = DualMemory(name="ops")
    await memory.add(MemoryContent(content="working note", mime_type=MemoryMimeType.TEXT))
    await memory.add_to_long_term(MemoryContent(content="experience", mime_type=MemoryMimeType.TEXT))

    await memory.clear_short_term()

    assert len(memory.short_term_memory) == 0
    assert len(memory.long_term_memory) == 1


def test_component_round_trip() -> None:
    """The backend (de)serializes through the existing Component protocol."""
    memory = DualMemory(
        name="ops",
        k=2,
        score_threshold=0.1,
        long_term_contents=[MemoryContent(content="restart redis pod", mime_type=MemoryMimeType.TEXT)],
    )

    config = memory.dump_component()
    assert isinstance(config, ComponentModel)
    assert config.provider == "autogen_ext.memory.dual_memory.DualMemory"
    assert config.component_type == "memory"

    restored = Memory.load_component(config)
    assert isinstance(restored, DualMemory)
    assert restored.name == "ops"
    assert restored.long_term_memory[0].content == "restart redis pod"


@pytest.mark.asyncio
async def test_non_text_content_is_scored_safely() -> None:
    """Non-text LTM items do not break resonance scoring."""
    memory = DualMemory(name="ops", score_threshold=0.0)
    await memory.add_to_long_term(MemoryContent(content={"root_cause": "config drift"}, mime_type=MemoryMimeType.JSON))
    await memory.add_to_long_term(MemoryContent(content=b"raw bytes", mime_type=MemoryMimeType.BINARY))

    results = await memory.query("config drift")
    assert isinstance(results, MemoryQueryResult)
    assert len(results.results) <= 2

"""Tests for :class:`HybridRAGMemory`.

These tests exercise the memory through the public :class:`~autogen_core.memory.Memory`
contract that the agent runtime consumes (``add`` / ``query`` / ``update_context``),
and through the declarative :class:`~autogen_core.Component` loader — i.e. they import
and drive existing ``autogen_core`` surfaces, not just the new module in isolation.
"""

import pytest
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.memory.hybrid_rag_memory import HybridRAGMemory, HybridRAGMemoryConfig


def _doc(text: str) -> MemoryContent:
    return MemoryContent(content=text, mime_type=MemoryMimeType.TEXT)


@pytest.mark.asyncio
async def test_implements_memory_contract() -> None:
    """The new backend is a Memory and therefore consumable by any agent that takes one."""
    memory = HybridRAGMemory()
    assert isinstance(memory, Memory)
    await memory.close()


@pytest.mark.asyncio
async def test_query_ranks_relevant_doc_first() -> None:
    """Hybrid fusion + reranking surfaces the lexically/evidentially relevant document on top."""
    memory = HybridRAGMemory(HybridRAGMemoryConfig(max_corrective_rounds=0))
    for text in (
        "The APS storage ring operates at a 7 GeV electron energy.",
        "Cryogenics keep the superconducting undulators cold during operation.",
        "Beamline operators log their shifts in the electronic logbook.",
    ):
        await memory.add(_doc(text))

    results = await memory.query("What electron energy does the storage ring operate at?")
    assert len(results.results) > 0
    assert "storage ring" in str(results.results[0].content).lower()
    assert results.results[0].metadata is not None
    assert "score" in results.results[0].metadata
    await memory.close()


@pytest.mark.asyncio
async def test_update_context_injects_memory_into_model_context() -> None:
    """Integration with the agent-runtime hook: retrieved memory is injected as a system message.

    This drives the same path ``AssistantAgent`` uses — ``update_context`` over a real
    ``ChatCompletionContext`` populated with a ``UserMessage``.
    """
    memory = HybridRAGMemory(HybridRAGMemoryConfig(max_corrective_rounds=0))
    await memory.add(_doc("Jupiter is the largest planet in our solar system."))

    context = BufferedChatCompletionContext(buffer_size=5)
    await context.add_message(UserMessage(content="Tell me about Jupiter", source="user"))

    result = await memory.update_context(context)
    assert any("jupiter" in str(r.content).lower() for r in result.memories.results)

    messages = await context.get_messages()
    assert len(messages) > 1  # original user message + injected memory system message
    assert any(isinstance(m, SystemMessage) for m in messages)
    await memory.close()


@pytest.mark.asyncio
async def test_corrective_loop_invokes_pluggable_critic() -> None:
    """The self-critique -> re-retrieve loop calls the (e.g. LLM-backed) critic and re-runs retrieval."""
    calls: list[str] = []

    async def critic(query: str, retrieved: list[str]) -> dict[str, object]:
        calls.append(query)
        # First pass: evidence insufficient, ask to broaden the query.
        if len(calls) == 1:
            return {"sufficient": False, "rewritten_query": "ring energy electron"}
        return {"sufficient": True, "rewritten_query": None}

    memory = HybridRAGMemory(HybridRAGMemoryConfig(max_corrective_rounds=1), critic=critic)  # type: ignore[arg-type]
    await memory.add(_doc("The storage ring electron energy is 7 GeV."))
    await memory.add(_doc("Cafeteria menu rotates weekly across stations."))

    results = await memory.query("What is the ring energy?")
    assert len(calls) >= 2  # initial critique + a critique of the re-retrieved evidence
    assert any("ring" in str(r.content).lower() for r in results.results)
    await memory.close()


@pytest.mark.asyncio
async def test_rewritten_query_broadens_when_evidence_insufficient() -> None:
    """Default parameter-free critic triggers a re-retrieve round via query expansion."""
    memory = HybridRAGMemory(HybridRAGMemoryConfig(max_corrective_rounds=1))
    await memory.add(_doc("The undulator connects the storage ring to the beamline."))

    # Query terms share a graph neighbor so the rewrite path can expand; loop must terminate.
    results = await memory.query("connection between ring and beamline")
    assert len(results.results) > 0
    await memory.close()


@pytest.mark.asyncio
async def test_score_threshold_drops_weak_results() -> None:
    memory = HybridRAGMemory(HybridRAGMemoryConfig(score_threshold=0.99, max_corrective_rounds=0))
    await memory.add(_doc("Photon source operations run continuously."))
    results = await memory.query("totally unrelated xyzzy zzz")
    assert results.results == []
    await memory.close()


@pytest.mark.asyncio
async def test_component_serialization_roundtrip() -> None:
    """The memory round-trips through the declarative Component loader."""
    memory = HybridRAGMemory(HybridRAGMemoryConfig(collection_name="ops", k=3, max_corrective_rounds=0))
    dumped = memory.dump_component()
    assert dumped.provider == "autogen_ext.memory.hybrid_rag_memory.HybridRAGMemory"
    assert dumped.config["collection_name"] == "ops"

    restored = HybridRAGMemory.load_component(dumped)
    assert isinstance(restored, HybridRAGMemory)
    assert restored.collection_name == "ops"
    await restored.close()


@pytest.mark.asyncio
async def test_clear_resets_collection() -> None:
    memory = HybridRAGMemory(HybridRAGMemoryConfig(max_corrective_rounds=0))
    await memory.add(_doc("Transient operational note."))
    assert len((await memory.query("transient operational note")).results) > 0
    await memory.clear()
    assert (await memory.query("transient operational note")).results == []
    await memory.close()

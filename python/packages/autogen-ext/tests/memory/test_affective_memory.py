"""Tests for AffectiveMemory.

These exercise the memory through the existing ``autogen_core`` Memory ABC and a
real ``BufferedChatCompletionContext`` (non-new modules), importing
``AffectiveMemory`` via the public ``autogen_ext.memory`` re-export so the
wiring edit is covered. They reproduce the PsychoAgent result: the full
affect-sensitive, conflict-aware architecture surfaces conflict-critical traces
that a purely semantic baseline does not.
"""

import pytest
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.memory import AffectiveMemory, AffectiveMemoryConfig

FACTUAL = "deploy the payment service to production"
CONFLICT = "the team argued bitterly about the rollback plan and never resolved it"
QUERY = "deploy the payment service rollback"


def _mem(text: str, affect: float = 0.0, conflict: bool = False) -> MemoryContent:
    return MemoryContent(
        content=text,
        mime_type=MemoryMimeType.TEXT,
        metadata={"affect": affect, "conflict": conflict},
    )


@pytest.mark.asyncio
async def test_conflict_critical_memory_promoted_under_full_config() -> None:
    """Full architecture promotes the unresolved-conflict trace above the topically-fitting factual one."""
    memory = AffectiveMemory(
        AffectiveMemoryConfig(top_k=5, min_similarity=0.0, affect_weight=1.0, conflict_weight=1.0)
    )
    await memory.add(_mem(FACTUAL))
    await memory.add(_mem(CONFLICT, affect=1.0, conflict=True))

    results = (await memory.query(QUERY)).results

    assert results, "expected at least one retrieved memory"
    # The conflict-critical, affective trace is ranked first.
    assert "rollback" in str(results[0].content)
    assert results[0].metadata is not None
    assert results[0].metadata["conflict"] is True
    assert results[0].metadata["rank_score"] >= results[1].metadata["rank_score"]


@pytest.mark.asyncio
async def test_semantic_only_baseline_keeps_factual_first() -> None:
    """With salience/conflict weights zeroed, the topically-fitting factual trace wins (the paper's baseline)."""
    memory = AffectiveMemory(
        AffectiveMemoryConfig(top_k=5, min_similarity=0.0, affect_weight=0.0, conflict_weight=0.0)
    )
    await memory.add(_mem(FACTUAL))
    await memory.add(_mem(CONFLICT, affect=1.0, conflict=True))

    results = (await memory.query(QUERY)).results

    assert results, "expected at least one retrieved memory"
    # Pure-semantic retrieval prefers the factual trace.
    assert "deploy the payment service to production" in str(results[0].content)
    assert results[0].metadata is not None
    assert results[0].metadata["conflict"] is False


@pytest.mark.asyncio
async def test_update_context_injects_conflict_aware_system_message() -> None:
    """update_context mutates a real model context and surfaces the conflict-critical trace first."""
    memory = AffectiveMemory(
        AffectiveMemoryConfig(top_k=5, min_similarity=0.0, affect_weight=1.0, conflict_weight=1.0)
    )
    await memory.add(_mem(FACTUAL))
    await memory.add(_mem(CONFLICT, affect=1.0, conflict=True))

    context = BufferedChatCompletionContext()
    await context.add_message(UserMessage(content=QUERY, source="user"))

    result = await memory.update_context(context)
    messages = await context.get_messages()

    system_messages = [message for message in messages if isinstance(message, SystemMessage)]
    assert system_messages, "expected a SystemMessage to be injected into the model context"

    injected = str(system_messages[-1].content)
    assert "affect-sensitive, conflict-aware retrieval" in injected
    # Conflict-critical trace is listed before the factual one.
    assert injected.index("rollback") < injected.index("payment service to production")
    assert result.memories.results, "expected memories to be returned"


@pytest.mark.asyncio
async def test_declarative_component_roundtrip() -> None:
    """The Memory + Component contract round-trips through dump/load_component (public declarative API)."""
    memory = AffectiveMemory(AffectiveMemoryConfig(top_k=7, affect_weight=0.4, conflict_weight=0.5))
    await memory.add(_mem("hello world"))

    model = memory.dump_component()
    assert model.config["top_k"] == 7
    assert model.config["affect_weight"] == 0.4
    assert model.config["conflict_weight"] == 0.5

    restored = Memory.load_component(model)
    assert isinstance(restored, AffectiveMemory)

    results = (await restored.query("hello world")).results
    assert results
    assert "hello world" in str(results[0].content)


@pytest.mark.asyncio
async def test_clear_and_close() -> None:
    """clear() empties the store; close() is a no-op for this in-memory backend."""
    memory = AffectiveMemory(AffectiveMemoryConfig())
    await memory.add(_mem("a trace"))
    await memory.clear()
    assert not (await memory.query("a trace")).results
    await memory.close()  # should not raise

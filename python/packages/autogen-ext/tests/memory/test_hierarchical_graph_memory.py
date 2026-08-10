"""Tests for :class:`HierarchicalGraphMemory`.

These tests import the *existing* :mod:`autogen_core.memory` ABC and exercise
the new memory through it (add/query/update_context/clear plus the declarative
``Component`` protocol), proving the integration rather than self-testing the new
module in isolation.
"""

import pytest
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import UserMessage

from autogen_ext.memory.hierarchical_graph import (
    HierarchicalGraphMemory,
    HierarchicalGraphMemoryConfig,
)


def _text(content: str, **metadata: object) -> MemoryContent:
    return MemoryContent(content=content, mime_type=MemoryMimeType.TEXT, metadata=dict(metadata) or None)


@pytest.fixture()
def memory() -> HierarchicalGraphMemory:
    return HierarchicalGraphMemory()


# -- Memory ABC conformance + Component protocol ---------------------------------
def test_conforms_to_memory_abc_and_round_trips() -> None:
    mem = HierarchicalGraphMemory(config=HierarchicalGraphMemoryConfig(name="abc"))
    assert isinstance(mem, Memory)

    loaded = HierarchicalGraphMemory.load_component(mem.dump_component())
    assert isinstance(loaded, HierarchicalGraphMemory)
    assert loaded.name == "abc"


# -- add / clear ----------------------------------------------------------------
@pytest.mark.asyncio
async def test_add_creates_cluster_and_unit_then_clear(memory: HierarchicalGraphMemory) -> None:
    await memory.add(_text("The user lives in Paris."))
    assert memory.num_clusters() == 1
    assert memory.num_units() == 1

    await memory.clear()
    assert memory.num_clusters() == 0
    assert memory.num_units() == 0


@pytest.mark.asyncio
async def test_unrelated_facts_form_separate_clusters(memory: HierarchicalGraphMemory) -> None:
    await memory.add(_text("The user lives in Paris."))
    await memory.add(_text("The weather in Tokyo is rainy today."))
    assert memory.num_clusters() == 2
    assert memory.num_units() == 2


# -- path-level localization: query returns the evidence path, not the flat store -
@pytest.mark.asyncio
async def test_query_localizes_to_relevant_subset(memory: HierarchicalGraphMemory) -> None:
    await memory.add(_text("The user lives in Paris."))
    await memory.add(_text("The weather in Tokyo is rainy today."))

    results = (await memory.query("weather in Tokyo")).results

    # Hierarchical localization returns only the support subgraph (Tokyo unit);
    # the flat store holds two units, so the evidence path is strictly smaller.
    assert len(results) == 1
    assert len(results) < memory.num_active_units()
    assert "Tokyo" in str(results[0].content)
    assert not any("Paris" in str(r.content) for r in results)
    # Evidence-path metadata is surfaced for downstream consumers.
    assert results[0].metadata is not None
    assert "cluster" in results[0].metadata
    assert results[0].metadata["score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_query_returns_empty_for_unmatched_query(memory: HierarchicalGraphMemory) -> None:
    await memory.add(_text("The user lives in Paris."))
    assert (await memory.query("quantum entanglement physics")).results == []


# -- coordinated rewriting: intra-unit rewrite + inter-unit dependency propagation
@pytest.mark.asyncio
async def test_conflicting_add_rewrites_unit_and_supersedes_dependent(memory: HierarchicalGraphMemory) -> None:
    # u0: base fact.
    await memory.add(_text("The user lives in Paris."))
    # Related-but-distinct fact -> linked as a dependent of u0 (inter-unit edge),
    # not a conflict (overlap below the conflict threshold).
    await memory.add(_text("The Paris office opens at nine."))
    assert memory.num_units() == 2

    # High overlap with u0 -> coordinated rewrite of u0; its dependent goes stale.
    await memory.add(_text("The user now lives in Paris."))

    # No new unit was created (u0 was revised in place); the dependent was superseded.
    assert memory.num_units() == 2
    assert memory.num_active_units() == 1

    # The surviving unit reflects the rewrite, and stale units are excluded from
    # the evidence path (valid-evidence selection under conflict).
    results = (await memory.query("user lives in Paris")).results
    assert len(results) == 1
    assert results[0].metadata is not None
    assert results[0].metadata["revision"] >= 2


# -- update_context injects only the localized evidence path --------------------
@pytest.mark.asyncio
async def test_update_context_injects_localized_path_only() -> None:
    memory = HierarchicalGraphMemory()
    await memory.add(_text("The user lives in Paris."))
    await memory.add(_text("The weather in Tokyo is rainy today."))

    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(content="weather in Tokyo", source="user"))
    await memory.update_context(context)

    messages = await context.get_messages()
    injected = [str(m.content) for m in messages if "Relevant memory" in str(m.content)]
    assert len(injected) == 1
    assert "Tokyo" in injected[0]
    assert "Paris" not in injected[0]

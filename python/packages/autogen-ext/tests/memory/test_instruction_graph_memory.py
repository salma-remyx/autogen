import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage

from autogen_ext.memory.instruction_graph import (
    EvolveResult,
    InstructionGraphMemory,
    InstructionGraphMemoryConfig,
    InstructionNode,
)


def _instruction(text: str, subject: str, node_type: str = "norm") -> MemoryContent:
    return MemoryContent(
        content=text,
        mime_type=MemoryMimeType.TEXT,
        metadata={"type": node_type, "subject": subject},
    )


@pytest.mark.asyncio
async def test_add_accepts_and_injects_checkpoint() -> None:
    """A clean update is applied and reconstructed into the model context."""
    memory = InstructionGraphMemory()

    result = await memory.evolve(_instruction("Greet the user warmly", "greeting", "identity"))
    assert result.accepted
    assert result.checkpoint_version == 1

    context = BufferedChatCompletionContext(buffer_size=5)
    await memory.update_context(context)
    messages = await context.get_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert "Greet the user warmly" in str(messages[0].content)

    query = await memory.query("anything")
    assert len(query.results) == 1
    assert query.results[0].metadata["subject"] == "greeting"


@pytest.mark.asyncio
async def test_scoped_verification_rejects_conflicting_update() -> None:
    """A conflicting update in the same subject neighborhood is gated out."""
    memory = InstructionGraphMemory()

    first = await memory.evolve(_instruction("Use metric units", "units"))
    assert first.accepted

    second = await memory.evolve(_instruction("Use imperial units", "units"))
    assert not second.accepted
    assert len(second.conflicts) == 1
    assert second.conflicts[0].reason == "antithetical"
    assert second.neighborhood == [first.node.node_id]
    # The checkpoint did not advance and the conflicting node was not stored.
    assert second.checkpoint_version == 1
    query = await memory.query("")
    assert len(query.results) == 1
    assert "metric" in str(query.results[0].content)


@pytest.mark.asyncio
async def test_supersede_consolidates_conflicting_neighbor() -> None:
    """The supersede policy retires the conflicting neighbor (consolidation)."""
    memory = InstructionGraphMemory(config=InstructionGraphMemoryConfig(on_conflict="supersede"))

    first = await memory.evolve(_instruction("Use metric units", "units"))
    second = await memory.evolve(_instruction("Use imperial units", "units"))

    assert second.accepted
    assert second.checkpoint_version == 2
    assert second.superseded == [first.node.node_id]
    # Only the superseding instruction survives in the live checkpoint.
    query = await memory.query("")
    assert len(query.results) == 1
    assert "imperial" in str(query.results[0].content)


@pytest.mark.asyncio
async def test_non_overlapping_same_subject_coexist() -> None:
    """Two non-conflicting instructions scoped to the same subject both persist."""
    memory = InstructionGraphMemory()

    first = await memory.evolve(_instruction("Use metric units", "units"))
    second = await memory.evolve(_instruction("Round numbers to two decimals", "units"))

    assert first.accepted and second.accepted
    assert second.checkpoint_version == 2
    assert second.conflicts == []
    query = await memory.query("")
    assert len(query.results) == 2


@pytest.mark.asyncio
async def test_negation_overlap_flagged_as_conflict() -> None:
    """A direct negation of an existing instruction is detected as a conflict."""
    memory = InstructionGraphMemory()

    await memory.evolve(_instruction("Respond in JSON", "format"))
    result = await memory.evolve(_instruction("Respond in not JSON", "format"))

    assert not result.accepted
    assert len(result.conflicts) == 1
    assert result.conflicts[0].reason == "negation"


@pytest.mark.asyncio
async def test_clear_resets_checkpoint_lineage() -> None:
    """Clearing removes every node and rewinds the checkpoint lineage."""
    memory = InstructionGraphMemory()
    await memory.add(_instruction("Use metric units", "units"))
    assert memory.checkpoint_version == 1

    await memory.clear()
    assert memory.checkpoint_version == 0
    assert (await memory.query("")).results == []


def test_component_roundtrip_preserves_config() -> None:
    """The Memory component (de)serializes through the public package surface."""
    memory = InstructionGraphMemory(config=InstructionGraphMemoryConfig(name="grace", on_conflict="supersede"))
    model = memory.dump_component()

    assert model.provider == "autogen_ext.memory.instruction_graph.InstructionGraphMemory"

    loaded = InstructionGraphMemory.load_component(model)
    assert isinstance(loaded, InstructionGraphMemory)
    assert loaded.name == "grace"


def test_evolve_result_is_the_public_capability_surface() -> None:
    """The named contribution (evolve) is exposed on the public capability module."""
    # EvolveResult is the structured verdict of scoped verification + checkpoint.
    result = EvolveResult(
        accepted=True,
        node=InstructionNode(node_id="x", text="t"),
        conflicts=[],
        neighborhood=[],
        checkpoint_version=1,
        instruction_text="",
    )
    assert result.accepted


@pytest.mark.asyncio
async def test_unknown_node_type_is_rejected_at_proposal_time() -> None:
    """Node types are the paper's closed set (identity/norm/knowledge)."""
    memory = InstructionGraphMemory()

    with pytest.raises(ValueError, match="Invalid instruction node type"):
        await memory.evolve(_instruction("Use metric units", "units", node_type="preference"))

    # The invalid proposal left no trace in the graph.
    assert memory.checkpoint_version == 0
    assert (await memory.query("")).results == []


@pytest.mark.asyncio
async def test_checkpoint_renders_identity_norm_knowledge_order() -> None:
    """The textual checkpoint orders nodes by the paper's type schema."""
    memory = InstructionGraphMemory()

    await memory.evolve(_instruction("Refunds require a receipt", "refunds", "knowledge"))
    await memory.evolve(_instruction("Never promise a refund outright", "refunds", "norm"))
    await memory.evolve(_instruction("You are a telecom support agent", "role", "identity"))

    checkpoint = (await memory.query("")).results
    assert len(checkpoint) == 3
    last = memory.last_evolve_result
    assert last is not None
    text = last.instruction_text
    identity_pos = text.index("telecom support agent")
    norm_pos = text.index("Never promise a refund")
    knowledge_pos = text.index("Refunds require a receipt")
    assert identity_pos < norm_pos < knowledge_pos

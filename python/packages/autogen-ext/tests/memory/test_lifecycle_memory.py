import pytest
from autogen_core import ComponentModel
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext

# Imported through the public re-export added to autogen_ext/memory/__init__.py,
# which is the wiring this module exercises end-to-end.
from autogen_ext.memory import LifecycleMemory
from autogen_ext.memory.lifecycle_memory import AuditKind, LifecycleMemoryConfig


def test_lifecycle_memory_satisfies_memory_protocol() -> None:
    """LifecycleMemory is a valid autogen_core.memory.Memory."""
    memory = LifecycleMemory(name="personal")
    assert isinstance(memory, Memory)
    for attr in ("update_context", "query", "add", "clear", "close"):
        assert hasattr(memory, attr)
    assert memory.name == "personal"


def test_reexport_from_package_root() -> None:
    """The capability is reachable via the public `autogen_ext.memory` API surface."""
    from autogen_ext.memory import LifecycleMemory as ReExported

    assert ReExported is LifecycleMemory


def test_component_roundtrip() -> None:
    """The declarative Component config survives a dump/load round-trip."""
    memory = LifecycleMemory(name="personal", require_evidence=True)
    config = memory.dump_component()
    assert isinstance(config, ComponentModel)
    assert config.provider == "autogen_ext.memory.lifecycle_memory.LifecycleMemory"
    assert config.component_type == "memory"
    assert config.config["require_evidence"] is True

    loaded = LifecycleMemory.load_component(config)
    assert isinstance(loaded, LifecycleMemory)
    assert loaded.name == "personal"
    assert loaded.config.require_evidence is True


@pytest.mark.asyncio
async def test_add_query_correct_forget_restore() -> None:
    """The correction-and-forgetting lifecycle primitive leaves a typed audit trail."""
    memory = LifecycleMemory(name="personal")
    await memory.add(
        MemoryContent(
            content="User lives in Seattle",
            mime_type=MemoryMimeType.TEXT,
            metadata={"source": "onboarding"},
        )
    )
    assert memory.active_count == 1

    # Query surfaces the active memory (substring retrieval proxy).
    results = await memory.query("seattle")
    assert len(results.results) == 1
    assert results.results[0].content == "User lives in Seattle"

    # The audit contract exposes the id of every admitted write.
    memory_id = memory.audit_log[-1].memory_id

    # Correction supersedes the memory in place.
    replacement_id = await memory.correct(
        memory_id,
        MemoryContent(
            content="User lives in Portland", mime_type=MemoryMimeType.TEXT, metadata={"source": "correction"}
        ),
        reason="user moved",
    )
    assert replacement_id is not None
    assert len((await memory.query("seattle")).results) == 0
    portland = await memory.query("portland")
    assert len(portland.results) == 1
    assert portland.results[0].content == "User lives in Portland"
    assert memory.active_count == 1

    # Forgetting is reversible: forget, then roll back via restore().
    assert await memory.forget(replacement_id, reason="erasure") is True
    assert memory.active_count == 0
    assert len((await memory.query("")).results) == 0
    assert await memory.restore(replacement_id) is True
    assert memory.active_count == 1
    assert len((await memory.query("portland")).results) == 1

    kinds = [entry.kind for entry in memory.audit_log]
    assert kinds == [AuditKind.WRITE, AuditKind.CORRECT, AuditKind.FORGET, AuditKind.RESTORE]


@pytest.mark.asyncio
async def test_evidence_gate_rejects_unprovenanced_writes() -> None:
    """With require_evidence, unprovenanced writes are dropped and traced."""
    memory = LifecycleMemory(require_evidence=True)
    await memory.add(MemoryContent(content="floating claim", mime_type=MemoryMimeType.TEXT))
    assert memory.active_count == 0
    assert len((await memory.query("")).results) == 0

    rejects = [entry for entry in memory.audit_log if entry.kind is AuditKind.GATE_REJECT]
    assert len(rejects) == 1
    assert rejects[0].accepted is False

    # A provenanced write is admitted through the gate.
    await memory.add(
        MemoryContent(content="grounded fact", mime_type=MemoryMimeType.TEXT, metadata={"source": "onboarding"})
    )
    assert memory.active_count == 1


@pytest.mark.asyncio
async def test_update_context_injects_active_memories() -> None:
    """Active memories are injected into the model context as a system message."""
    memory = LifecycleMemory()
    await memory.add(MemoryContent(content="User prefers metric units", mime_type=MemoryMimeType.TEXT))
    context = BufferedChatCompletionContext(buffer_size=5)

    result = await memory.update_context(context)
    assert len(result.memories.results) == 1
    messages = await context.get_messages()
    assert len(messages) == 1
    assert "metric units" in str(messages[0].content)

    # An empty memory leaves the context untouched.
    empty_context = BufferedChatCompletionContext(buffer_size=5)
    empty_result = await LifecycleMemory().update_context(empty_context)
    assert len(empty_result.memories.results) == 0
    assert len(await empty_context.get_messages()) == 0


@pytest.mark.asyncio
async def test_clear_is_audited_rollback() -> None:
    """clear() forgets every active memory and records a rollback entry."""
    memory = LifecycleMemory()
    await memory.add(MemoryContent(content="a", mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content="b", mime_type=MemoryMimeType.TEXT))
    await memory.clear()
    assert memory.active_count == 0
    assert [entry.kind for entry in memory.audit_log if entry.kind is AuditKind.CLEAR] == [AuditKind.CLEAR]


def test_config_defaults() -> None:
    config = LifecycleMemoryConfig()
    assert config.require_evidence is False
    assert config.top_k == 10

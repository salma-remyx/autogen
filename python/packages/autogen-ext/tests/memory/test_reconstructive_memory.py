"""Tests for :class:`ReconstructiveMemory`.

These tests drive the new memory through existing (non-new) AutoGen building
blocks — :class:`~autogen_core.memory.ListMemory` as the retrieval store,
:class:`~autogen_core.model_context.BufferedChatCompletionContext` as the live
context, and :class:`~autogen_ext.models.replay.ReplayChatCompletionClient` as a
deterministic stand-in for the reconstruction model — to prove the wiring works
end to end.
"""

from typing import List

import pytest
from autogen_core.memory import ListMemory, MemoryContent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage
from autogen_ext.memory.reconstructive import ReconstructiveMemory
from autogen_ext.models.replay import ReplayChatCompletionClient

# A token that lives only in the raw stored memory. If it ever leaks into the
# model context, the layer replayed instead of reconstructing.
RAW_MEM_TOKEN = "RAW_MEM_TOKEN_42"


@pytest.fixture()
def stored_memory() -> ListMemory:
    store = ListMemory()
    return store


@pytest.mark.asyncio
async def test_update_context_reconstructs_instead_of_replaying(stored_memory: ListMemory) -> None:
    """The reconstruction output is injected; the raw memory is not."""
    await stored_memory.add(
        MemoryContent(content=f"For python tasks, {RAW_MEM_TOKEN}: always pin the version.", mime_type="text/plain")
    )
    guidance = "For the current pip install step, pin the package version explicitly."
    client = ReplayChatCompletionClient(chat_completions=[guidance])
    memory = ReconstructiveMemory(retrieval=stored_memory, model_client=client)

    context = BufferedChatCompletionContext(
        buffer_size=10, initial_messages=[UserMessage(content="Run pip install numpy.", source="user")]
    )

    result = await memory.update_context(context)

    # The reconstruction step actually ran.
    assert len(client.create_calls) == 1
    # Reconstructed guidance is present in the live context...
    messages = await context.get_messages()
    injected = [m for m in messages if "Guidance reconstructed" in getattr(m, "content", "")]
    assert len(injected) == 1
    assert guidance in injected[0].content
    # ...and the raw memory was NOT replayed verbatim.
    assert RAW_MEM_TOKEN not in injected[0].content
    # The caller still observes what was retrieved.
    assert len(result.memories.results) == 1


@pytest.mark.asyncio
async def test_update_context_rejects_to_prevent_negative_transfer(stored_memory: ListMemory) -> None:
    """On rejection nothing is injected, so stale memory cannot cause negative transfer."""
    await stored_memory.add(
        MemoryContent(content=f"Unrelated note: {RAW_MEM_TOKEN} tune the database.", mime_type="text/plain")
    )
    client = ReplayChatCompletionClient(chat_completions=["REJECT"])
    memory = ReconstructiveMemory(retrieval=stored_memory, model_client=client)

    context = BufferedChatCompletionContext(
        buffer_size=10, initial_messages=[UserMessage(content="Draft a haiku about the sea.", source="user")]
    )
    before = await context.get_messages()

    result = await memory.update_context(context)

    after = await context.get_messages()
    # No system message added on rejection.
    assert before == after
    assert not any("Guidance reconstructed" in getattr(m, "content", "") for m in after)
    # Reconstruction was still consulted...
    assert len(client.create_calls) == 1
    # ...and retrieval is still observable.
    assert len(result.memories.results) == 1


@pytest.mark.asyncio
async def test_update_context_no_retrieval_skips_reconstruction(stored_memory: ListMemory) -> None:
    """With nothing retrieved, no model call is made and nothing is injected."""
    client = ReplayChatCompletionClient(chat_completions=["should-not-be-used"])
    memory = ReconstructiveMemory(retrieval=stored_memory, model_client=client)

    context = BufferedChatCompletionContext(
        buffer_size=10, initial_messages=[UserMessage(content="Hello there.", source="user")]
    )

    result = await memory.update_context(context)

    assert len(client.create_calls) == 0
    assert result.memories.results == []
    messages: List = await context.get_messages()
    assert all(not getattr(m, "content", "").startswith("Guidance reconstructed") for m in messages)


@pytest.mark.asyncio
async def test_add_query_delegate_to_backing_store(stored_memory: ListMemory) -> None:
    """Storage/retrieval pass through to the wrapped backend unchanged."""
    client = ReplayChatCompletionClient(chat_completions=[])
    memory = ReconstructiveMemory(retrieval=stored_memory, model_client=client)

    content = MemoryContent(content="remembered fact", mime_type="text/plain")
    await memory.add(content)

    queried = await memory.query("remembered")
    assert len(queried.results) == 1
    assert queried.results[0].content == "remembered fact"

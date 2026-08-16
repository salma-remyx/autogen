"""Tests for memory serving-cost instrumentation.

Exercises the ``attach_serving_cost_recorder`` hook in
:class:`autogen_ext.memory.mem0.Mem0Memory` (the call-site wiring) alongside the
report's token accounting and break-even analysis.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.memory.mem0 import Mem0Memory
from autogen_ext.memory.serving_cost import FullTranscriptBaseline, ServingCostRecorder

mem0 = pytest.importorskip("mem0")


def _make_memory(search_results: List[Dict[str, Any]]) -> Mem0Memory:
    """Build a local Mem0Memory whose client is fully mocked, so no mem0 backend runs."""
    memory = Mem0Memory(user_id="test-user", limit=3, is_cloud=False, config={"path": ":memory:"})
    mock_client = MagicMock()
    mock_client.search.return_value = search_results
    memory._client = mock_client  # type: ignore[assignment]
    return memory


def _token_counter(messages: Any) -> int:
    """Deterministic token estimate so assertions are exact: one token per message."""
    return len(list(messages))


@pytest.mark.asyncio
@patch("autogen_ext.memory.mem0._mem0.Memory0")
async def test_mem0_update_context_records_serving_cost(mock_mem0_class: MagicMock) -> None:
    """The call-site hook prices each Mem0 update_context turn."""
    memory = _make_memory([{"memory": "User lives in Seattle.", "score": 0.9}])
    recorder = ServingCostRecorder(token_counter=_token_counter)
    memory.attach_serving_cost_recorder(recorder)

    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(content="Where do I live?", source="user"))

    assert recorder.report().turns == 0, "nothing recorded before update_context runs"

    result = await memory.update_context(context)

    assert len(result.memories.results) == 1
    report = recorder.report()
    assert report.turns == 1

    sample = report.samples[0]
    # Two messages served (the turn plus the injected memory), one in the transcript.
    assert sample.served_tokens == 2
    assert sample.transcript_tokens == 1
    assert sample.retained_tokens == 1
    assert sample.memory_overhead_tokens == 1
    assert sample.saved_vs_transcript == -1

    # The memory really did land in the context, via the normal update_context path.
    messages = await context.get_messages()
    assert any(isinstance(m, SystemMessage) and "Seattle" in m.content for m in messages)


@pytest.mark.asyncio
@patch("autogen_ext.memory.mem0._mem0.Memory0")
async def test_mem0_without_recorder_is_unchanged(mock_mem0_class: MagicMock) -> None:
    """Attaching a recorder is opt-in: with none attached, update_context is as before."""
    memory = _make_memory([{"memory": "User lives in Seattle.", "score": 0.9}])

    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(content="Where do I live?", source="user"))
    result = await memory.update_context(context)

    assert len(result.memories.results) == 1
    assert len(await context.get_messages()) == 2


@pytest.mark.asyncio
@patch("autogen_ext.memory.mem0._mem0.Memory0")
async def test_mem0_empty_context_records_zero_cost_turn(mock_mem0_class: MagicMock) -> None:
    """An empty context short-circuits, and the open turn is still closed."""
    memory = _make_memory([])
    recorder = ServingCostRecorder(token_counter=_token_counter)
    memory.attach_serving_cost_recorder(recorder)

    result = await memory.update_context(UnboundedChatCompletionContext())

    assert result.memories.results == []
    report = recorder.report()
    assert report.turns == 1
    assert report.samples[0].served_tokens == 0
    assert report.memory_overhead_tokens == 0


@pytest.mark.asyncio
async def test_breakeven_turn_found_when_memory_beats_transcript() -> None:
    """Break-even is the first turn from which memory stays cheaper than the transcript."""
    recorder = ServingCostRecorder(token_counter=_token_counter)
    memory = _make_memory([{"memory": "fact", "score": 0.9}])
    memory.attach_serving_cost_recorder(recorder)
    # Keep only the last two turns, so the served prompt stays flat while the transcript
    # grows. Turn 1 still pays the injection on top of a short history; from turn 2 the
    # memory-backed context serves strictly less than the transcript.
    context = BufferedChatCompletionContext(buffer_size=2)

    for i in range(5):
        await context.add_message(UserMessage(content=f"turn {i}", source="user"))
        await memory.update_context(context)

    report = recorder.report()
    assert report.turns == 5
    assert [s.transcript_tokens for s in report.samples] == [1, 3, 5, 7, 9]
    assert [s.served_tokens for s in report.samples] == [2, 2, 2, 2, 2]
    assert report.breakeven_turn == 2
    assert report.transcript_tokens_total > report.served_tokens_total
    assert "cheaper from turn 2" in report.summary()


@pytest.mark.asyncio
async def test_breakeven_turn_none_when_memory_never_wins() -> None:
    """With an unbounded context the memory injection is pure overhead, so no break-even."""
    recorder = ServingCostRecorder(token_counter=_token_counter)
    memory = _make_memory([{"memory": "fact", "score": 0.9}])
    memory.attach_serving_cost_recorder(recorder)
    context = UnboundedChatCompletionContext()

    for i in range(3):
        await context.add_message(UserMessage(content=f"turn {i}", source="user"))
        await memory.update_context(context)

    report = recorder.report()
    assert report.turns == 3
    assert report.breakeven_turn is None
    assert report.served_tokens_total > report.transcript_tokens_total
    assert "cheaper from never" in report.summary()


@pytest.mark.asyncio
async def test_full_transcript_baseline_prices_turn_at_transcript_cost() -> None:
    """The reference strategy serves exactly the transcript, with zero memory overhead."""
    recorder = ServingCostRecorder(token_counter=_token_counter)
    baseline = FullTranscriptBaseline(recorder)

    context = UnboundedChatCompletionContext()
    for i in range(3):
        await context.add_message(UserMessage(content=f"turn {i}", source="user"))
        result = await baseline.update_context(context)
        assert result.memories.results == [], "the transcript is already in the context"

    report = recorder.report()
    assert report.turns == 3
    assert [s.transcript_tokens for s in report.samples] == [1, 2, 3]
    assert all(s.memory_overhead_tokens == 0 for s in report.samples)
    assert all(s.served_tokens == s.transcript_tokens for s in report.samples)

    await baseline.clear()
    assert recorder.report().turns == 0


@pytest.mark.asyncio
async def test_custom_token_counter_is_used() -> None:
    """A model client's count_tokens can replace the character-based default."""
    calls: List[int] = []

    def counter(messages: Any) -> int:
        calls.append(len(list(messages)))
        return 7

    recorder = ServingCostRecorder(token_counter=counter)
    memory = _make_memory([{"memory": "fact", "score": 0.9}])
    memory.attach_serving_cost_recorder(recorder)
    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(content="hi", source="user"))

    await memory.update_context(context)

    assert calls, "the custom counter was never invoked"
    assert all(sample.served_tokens == 7 for sample in recorder.report().samples)


def test_report_summary_with_no_samples() -> None:
    """An uninstrumented run reports that it recorded nothing."""
    assert ServingCostRecorder().report().summary() == "no turns recorded"
    assert ServingCostRecorder().report().breakeven_turn is None


def test_end_turn_without_begin_turn_raises() -> None:
    """A mismatched end_turn is an error rather than a silently empty sample."""
    recorder = ServingCostRecorder(token_counter=_token_counter)
    with pytest.raises(RuntimeError):
        recorder.end_turn([])


@pytest.mark.asyncio
async def test_recorder_wraps_context_transparently() -> None:
    """instrument() delegates to the wrapped context, so agent behaviour is unchanged."""
    recorder = ServingCostRecorder(token_counter=_token_counter)
    context = recorder.instrument(BufferedChatCompletionContext(buffer_size=1))

    await context.add_message(UserMessage(content="first", source="user"))
    await context.add_message(UserMessage(content="second", source="user"))

    messages = await context.get_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], UserMessage)
    assert messages[0].content == "second"

    await context.clear()
    assert await context.get_messages() == []


@pytest.mark.asyncio
@patch("autogen_ext.memory.mem0._mem0.Memory0")
async def test_mem0_memory_content_roundtrip_with_recorder_attached(mock_mem0_class: MagicMock) -> None:
    """Attaching a recorder does not disturb add/query behaviour."""
    memory = _make_memory([{"memory": "User likes tea.", "score": 0.8}])
    recorder = ServingCostRecorder(token_counter=_token_counter)
    memory.attach_serving_cost_recorder(recorder)

    await memory.add(MemoryContent(content="User likes tea.", mime_type=MemoryMimeType.TEXT))
    results = await memory.query("What does the user like?")

    assert len(results.results) == 1
    assert results.results[0].metadata is not None
    assert results.results[0].metadata.get("score") == 0.8

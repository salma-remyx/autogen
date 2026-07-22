"""Integration tests for the lifecycle-memory diagnostics.

These tests exercise :mod:`autogen_core.memory._lifecycle_memory` against the
existing :class:`~autogen_core.memory.ListMemory` backend — a non-new module —
to confirm the harness reports operation-level failure modes that a single
final-answer score would conceal.
"""

from typing import cast

import pytest
from autogen_core.memory import ListMemory, Memory
from autogen_core.memory._lifecycle_memory import (
    TARGET_METADATA_KEY,
    LifecycleOp,
    MemoryTraceEvent,
    OperationOutcome,
    diagnose,
    expected_state,
)


@pytest.mark.asyncio
async def test_remember_then_reflect_is_satisfied() -> None:
    """A clean remember + reflect trace passes at both the operation and final-answer level."""

    memory = ListMemory()
    gold = [
        MemoryTraceEvent(op=LifecycleOp.REMEMBER, target="user.diet", value="vegan", trigger="I'm vegan"),
        MemoryTraceEvent(op=LifecycleOp.REFLECT, target="user.diet", trigger="what do I eat?"),
    ]

    report = await diagnose(memory, gold)

    assert report.operation_reliability == 1.0
    assert report.final_answer_accuracy is True
    assert not report.inconsistent_but_correct
    assert [d.outcome for d in report.diagnostics] == [OperationOutcome.SATISFIED, OperationOutcome.SATISFIED]


@pytest.mark.asyncio
async def test_update_on_append_only_memory_is_stale_but_final_answer_correct() -> None:
    """The central MemOps finding.

    ``ListMemory`` is append-only, so an UPDATE leaves the superseded value in
    place. Final-answer scoring says "correct" (the new value is queryable); the
    operation-level diagnostic says "stale". The two disagree, which is exactly
    the inconsistency final-answer scoring hides.
    """

    memory = ListMemory()
    gold = [
        MemoryTraceEvent(op=LifecycleOp.REMEMBER, target="user.city", value="Paris", trigger="I live in Paris"),
        MemoryTraceEvent(op=LifecycleOp.UPDATE, target="user.city", value="Lyon", trigger="I moved to Lyon"),
    ]

    report = await diagnose(memory, gold)

    outcomes = [d.outcome for d in report.diagnostics]
    assert outcomes == [OperationOutcome.SATISFIED, OperationOutcome.STALE_VALUE]
    assert report.final_answer_accuracy is True  # 'Lyon' is queryable
    assert report.inconsistent_but_correct is True  # ...despite a stale memory state
    assert report.operation_reliability == 0.5


@pytest.mark.asyncio
async def test_forget_without_handler_is_unforgotten() -> None:
    """The base Memory protocol cannot target-forget, so the entry lingers."""

    memory = ListMemory()
    gold = [
        MemoryTraceEvent(op=LifecycleOp.REMEMBER, target="user.secret", value="s3cr3t", trigger="psst"),
        MemoryTraceEvent(op=LifecycleOp.FORGET, target="user.secret", trigger="forget that"),
    ]

    report = await diagnose(memory, gold)

    assert report.diagnostics[-1].outcome == OperationOutcome.UNFORGOTTEN
    assert report.operation_reliability == 0.5


@pytest.mark.asyncio
async def test_forget_with_managed_handler_is_satisfied() -> None:
    """A managed backend that can target-forget clears the entry.

    The same trace that failed above now passes at the operation level once the
    replay is told how the backend forgets a single target — showing the harness
    discriminates between backends rather than always failing on FORGET.
    """

    memory = ListMemory()

    async def forget(mem: Memory, target: str) -> None:
        listed = cast(ListMemory, mem)
        listed.content = [c for c in listed.content if (c.metadata or {}).get(TARGET_METADATA_KEY) != target]

    gold = [
        MemoryTraceEvent(op=LifecycleOp.REMEMBER, target="user.secret", value="s3cr3t", trigger="psst"),
        MemoryTraceEvent(op=LifecycleOp.FORGET, target="user.secret", trigger="forget that"),
    ]

    report = await diagnose(memory, gold, forget_handler=forget)

    assert report.diagnostics[-1].outcome == OperationOutcome.SATISFIED
    assert report.operation_reliability == 1.0


@pytest.mark.asyncio
async def test_reflect_over_forgotten_target_is_empty() -> None:
    """Reflecting over a cleanly-forgotten target surfaces REFLECTION_EMPTY."""

    memory = ListMemory()

    async def forget(mem: Memory, target: str) -> None:
        listed = cast(ListMemory, mem)
        listed.content = [c for c in listed.content if (c.metadata or {}).get(TARGET_METADATA_KEY) != target]

    gold = [
        MemoryTraceEvent(op=LifecycleOp.REMEMBER, target="tmp.id", value="42", trigger="here"),
        MemoryTraceEvent(op=LifecycleOp.FORGET, target="tmp.id", trigger="clear it"),
        MemoryTraceEvent(op=LifecycleOp.REFLECT, target="tmp.id", trigger="what was it?"),
    ]

    report = await diagnose(memory, gold, forget_handler=forget)

    assert report.diagnostics[-1].outcome == OperationOutcome.REFLECTION_EMPTY


def test_expected_state_reproduces_ordered_trajectory() -> None:
    """A later UPDATE supersedes; a later FORGET clears."""

    gold = [
        MemoryTraceEvent(op=LifecycleOp.REMEMBER, target="t", value="a"),
        MemoryTraceEvent(op=LifecycleOp.UPDATE, target="t", value="b"),
        MemoryTraceEvent(op=LifecycleOp.FORGET, target="t"),
    ]

    assert expected_state(gold) == {"t": None}

    assert expected_state(gold[:2]) == {"t": "b"}

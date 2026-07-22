"""Lifecycle memory operations and operation-level diagnostics.

Adapted from *MemOps: Benchmarking Lifecycle Memory Operations in Long-Horizon
Conversations* (arXiv:2607.12893). MemOps argues that long-term memory is not a
static bag of facts but a *lifecycle* of explicit operations — remembering,
forgetting, updating, and reflecting — and that scoring only the correctness of
a final answer hides the heterogeneous causes of memory failure: missing the
introduction of a fact, binding an operation to the wrong target, or relying on
a stale value after a correction. A correct final answer can therefore rest on
an inconsistent or unsafe memory state.

This module ports MemOps' *core diagnostic mechanism* onto AutoGen's
:class:`~autogen_core.memory.Memory` abstraction. It represents each memory
event as a structured trace (trigger / target / scope / state-transition /
evidence), replays a gold trace against any :class:`Memory`, and emits
operation-level diagnostics instead of a single final-answer score.

Adaptation notes (Mode 2):

* The controllable conversation-generation pipeline and the multi-system
  benchmark/probe suite from the paper are intentionally out of scope — they are
  a dataset concern for a downstream PR, not part of the diagnostic mechanism.
* "Which target does a memory bind to?" — which MemOps answers with a learned
  matcher — is approximated by a parameter-free key written into memory
  metadata by :class:`LifecycleReplay`, so target binding is exact rather than
  inferred.
* UPDATE/FORGET are driven through the base :class:`Memory` protocol, which has
  no targeted update or remove: an append-only backend therefore surfaces as
  ``STALE_VALUE`` / ``UNFORGOTTEN`` — which is precisely the failure mode the
  operation-level view exists to surface.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from ._base_memory import Memory, MemoryContent, MemoryMimeType

#: Metadata key under which :class:`LifecycleReplay` records an item's target.
TARGET_METADATA_KEY = "lifecycle_target"
SCOPE_METADATA_KEY = "lifecycle_scope"
EVIDENCE_METADATA_KEY = "lifecycle_evidence"


class LifecycleOp(str, Enum):
    """The explicit lifecycle operations a memory performs over time."""

    REMEMBER = "remember"
    UPDATE = "update"
    FORGET = "forget"
    REFLECT = "reflect"


class MemoryTraceEvent(BaseModel):
    """A structured memory-event trace, after MemOps.

    Captures the five facets MemOps associates with every memory event: *what*
    happened, *why* (trigger), *what it acts on* (target), *where* (scope),
    *what state change it implies* (value), and the *evidence* supporting it.
    """

    op: LifecycleOp
    target: str
    """Dotted key naming the entity the operation binds to, e.g. ``"user.diet"``."""

    value: Optional[str] = None
    """Payload for REMEMBER/UPDATE. ``None`` for FORGET/REFLECT."""

    trigger: str = ""
    """The conversational cue that prompted the operation."""

    scope: str = "default"
    """Session / partition the operation is scoped to."""

    evidence: str = ""
    """Snippet supporting the operation, e.g. the utterance it was derived from."""


class OperationOutcome(str, Enum):
    """Operation-level diagnostic verdict for a single gold event."""

    SATISFIED = "satisfied"
    """The operation was applied and the resulting state matches the gold."""

    MISSING = "missing"
    """A REMEMBER/UPDATE the gold requires never landed: the target is absent."""

    STALE_VALUE = "stale_value"
    """The target's current value is correct but a superseded value still lingers."""

    UNFORGOTTEN = "unforgotten"
    """A FORGET the gold requires did not take effect: the target is still present."""

    REFLECTION_EMPTY = "reflection_empty"
    """A REFLECT returned no memory for a target the gold state contains."""


class OperationDiagnostic(BaseModel):
    """A single operation-level finding."""

    event: MemoryTraceEvent
    outcome: OperationOutcome
    detail: str


class LifecycleReport(BaseModel):
    """Operation-level evaluation of a memory replayed against a gold trace.

    Carries both the granular diagnostics MemOps argues for *and* the coarse
    final-answer accuracy it argues against, so callers can see where the two
    disagree.
    """

    diagnostics: List[OperationDiagnostic]
    final_answer_accuracy: bool
    """Whether a final query returns the gold value for every still-relevant target."""

    operation_reliability: float
    """Fraction of gold events whose operation-level outcome is SATISFIED."""

    @property
    def inconsistent_but_correct(self) -> bool:
        """True when the final answer is right despite an inconsistent memory state.

        This is the failure mode MemOps exists to surface: final-answer scoring
        credits a correct answer even though the underlying memory holds stale
        or spurious content.
        """

        return self.final_answer_accuracy and self.operation_reliability < 1.0

    def outcome_counts(self) -> Dict[OperationOutcome, int]:
        """Aggregate diagnostics by outcome."""

        counts: Dict[OperationOutcome, int] = defaultdict(int)
        for diag in self.diagnostics:
            counts[diag.outcome] += 1
        return dict(counts)


ForgetHandler = Callable[[Memory, str], Awaitable[None]]
"""How a managed backend forgets a single target. Optional; see :class:`LifecycleReplay`."""


def _tag(value: str, event: MemoryTraceEvent) -> MemoryContent:
    """Build a :class:`MemoryContent` carrying its lifecycle target in metadata."""

    return MemoryContent(
        content=value,
        mime_type=MemoryMimeType.TEXT,
        metadata={
            TARGET_METADATA_KEY: event.target,
            SCOPE_METADATA_KEY: event.scope,
            EVIDENCE_METADATA_KEY: event.evidence,
        },
    )


class LifecycleReplay:
    """Drives a :class:`Memory` through a gold trace using its public protocol.

    The base :class:`Memory` protocol exposes ``add`` / ``query`` / ``clear`` but
    no targeted *update* or *forget*. By default UPDATE therefore appends a new
    value (leaving the old one, which the evaluator flags as stale) and FORGET is
    a no-op (which the evaluator flags as unforgotten). Pass ``forget_handler``
    to teach the replay how a managed backend forgets a single target.
    """

    def __init__(self, memory: Memory, forget_handler: Optional[ForgetHandler] = None) -> None:
        self._memory = memory
        self._forget = forget_handler

    async def apply(self, event: MemoryTraceEvent) -> None:
        """Perform a single lifecycle event on the memory."""

        if event.op in (LifecycleOp.REMEMBER, LifecycleOp.UPDATE):
            assert event.value is not None
            await self._memory.add(_tag(event.value, event))
        elif event.op == LifecycleOp.FORGET:
            if self._forget is not None:
                await self._forget(self._memory, event.target)
            # else: the protocol has no targeted forget -> recorded as a failure.
        # REFLECT does not mutate state; its outcome is diagnosed from queries.

    async def replay(self, gold: Sequence[MemoryTraceEvent]) -> None:
        """Apply an ordered gold trace to the memory in sequence."""

        for event in gold:
            await self.apply(event)


async def observed_state(memory: Memory) -> Dict[str, List[str]]:
    """Read the memory grouped by lifecycle target, preserving insertion order.

    This is the parameter-free proxy for MemOps' learned "bind an operation to
    its target" matcher: every value written via :class:`LifecycleReplay` is
    tagged with its target key, so grouping is exact rather than inferred.
    """

    results = await memory.query("")
    by_target: Dict[str, List[str]] = defaultdict(list)
    for item in results.results:
        meta = item.metadata or {}
        target = meta.get(TARGET_METADATA_KEY)
        if isinstance(target, str):
            by_target[target].append(str(item.content))
    return dict(by_target)


def expected_state(gold: Sequence[MemoryTraceEvent]) -> Dict[str, Optional[str]]:
    """Compute the gold final value per target (``None`` once forgotten).

    Iterating in order reproduces MemOps' ordered memory-state trajectory: a
    later UPDATE supersedes an earlier value, and a later FORGET clears it.
    """

    state: Dict[str, Optional[str]] = {}
    for event in gold:
        if event.op in (LifecycleOp.REMEMBER, LifecycleOp.UPDATE):
            assert event.value is not None
            state[event.target] = event.value
        elif event.op == LifecycleOp.FORGET:
            state[event.target] = None
    return state


def _classify(
    event: MemoryTraceEvent,
    observed: Dict[str, List[str]],
    expected_so_far: Dict[str, Optional[str]],
) -> Tuple[OperationOutcome, str]:
    """Decide the outcome for one gold event against the state right after it."""

    target = event.target
    values = observed.get(target, [])

    if event.op in (LifecycleOp.REMEMBER, LifecycleOp.UPDATE):
        gold_final = expected_so_far.get(target)
        assert gold_final is not None
        if not values:
            return OperationOutcome.MISSING, f"target '{target}' absent after {event.op.value}"
        stale = [v for v in values if v != gold_final]
        if stale and gold_final in values:
            return (
                OperationOutcome.STALE_VALUE,
                f"target '{target}' holds its current value plus {len(stale)} stale value(s)",
            )
        if gold_final not in values:
            return OperationOutcome.MISSING, f"target '{target}' present but not the gold value '{gold_final}'"
        return OperationOutcome.SATISFIED, f"target '{target}' holds '{gold_final}'"

    if event.op == LifecycleOp.FORGET:
        if values:
            return OperationOutcome.UNFORGOTTEN, f"target '{target}' still present after forget"
        return OperationOutcome.SATISFIED, f"target '{target}' forgotten"

    # REFLECT: judged by whether the target is retrievable in the current state.
    if not values:
        return OperationOutcome.REFLECTION_EMPTY, f"reflection over '{target}' returned nothing"
    return OperationOutcome.SATISFIED, f"reflection over '{target}' returned memory"


async def diagnose(
    memory: Memory,
    gold: Sequence[MemoryTraceEvent],
    forget_handler: Optional[ForgetHandler] = None,
) -> LifecycleReport:
    """Replay ``gold`` against ``memory`` and score it operation by operation.

    Emits one diagnostic per gold event and a separate final-answer accuracy so
    the two metrics can disagree — the central MemOps finding. See
    :attr:`LifecycleReport.inconsistent_but_correct`.
    """

    replay = LifecycleReplay(memory, forget_handler=forget_handler)
    diagnostics: List[OperationDiagnostic] = []
    snapshot: Dict[str, List[str]] = {}

    for index, event in enumerate(gold):
        await replay.apply(event)
        snapshot = await observed_state(memory)
        expected_so_far = expected_state(gold[: index + 1])
        outcome, detail = _classify(event, snapshot, expected_so_far)
        diagnostics.append(OperationDiagnostic(event=event, outcome=outcome, detail=detail))

    final_expected = expected_state(gold)
    relevant = {t: v for t, v in final_expected.items() if v is not None}
    final_correct = all(v in snapshot.get(t, []) for t, v in relevant.items()) if relevant else True

    satisfied = sum(1 for d in diagnostics if d.outcome == OperationOutcome.SATISFIED)
    reliability = satisfied / len(diagnostics) if diagnostics else 1.0

    return LifecycleReport(
        diagnostics=diagnostics,
        final_answer_accuracy=final_correct,
        operation_reliability=reliability,
    )

"""Integration tests for EvidenceLedgerMemory.

These exercise the ledger through the public ``autogen_core.memory.Memory`` ABC
and the real model-context objects (non-new modules), proving the new backend
is consumed by the existing memory contract rather than being a standalone stub.
"""

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage

from autogen_ext.memory.evidence_ledger import (
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    EvidenceLedgerMemory,
)


def _text(body: str, **metadata: object) -> MemoryContent:
    return MemoryContent(content=body, mime_type=MemoryMimeType.TEXT, metadata=dict(metadata))


@pytest.mark.asyncio
async def test_tool_observation_and_grounded_claim() -> None:
    ledger = EvidenceLedgerMemory()
    await ledger.add(_text("The weather in Paris is sunny, 22C.", source="get_weather", epistemic_type="observation"))
    await ledger.add(_text("It is a good day for a walk.", epistemic_type="claim", citations=[1]))

    obs, claim = ledger.content
    assert obs.metadata["entry_id"] == 1
    assert obs.metadata["status"] == STATUS_ACTIVE
    assert obs.metadata["grounded"] is True  # tool-grounded observation
    assert claim.metadata["grounded"] is True  # claim cites a resolvable active entry

    # update_context injects only grounded, active evidence with citations.
    ctx = BufferedChatCompletionContext(buffer_size=10)
    result = await ledger.update_context(ctx)
    assert [m.metadata["entry_id"] for m in result.memories.results] == [1, 2]
    messages = await ctx.get_messages()
    assert any(isinstance(m, SystemMessage) and "[#1 observation src=get_weather" in m.content for m in messages)
    assert "cites=[1]" in messages[-1].content


@pytest.mark.asyncio
async def test_phantom_grounding_and_unsupported_claims_are_flagged() -> None:
    ledger = EvidenceLedgerMemory()
    # Claim citing an id that does not exist -> citation-backed hallucination.
    await ledger.add(_text("Revenue grew 30% YoY.", epistemic_type="claim", citations=[99]))
    # Claim with no provenance at all -> unsupported intermediate reasoning.
    await ledger.add(_text("Therefore we should expand.", epistemic_type="claim"))
    # Tool-grounded observation missing its source -> provenance violation.
    await ledger.add(_text("A raw reading.", epistemic_type="observation"))

    assert [e.metadata["grounded"] for e in ledger.content] == [False, False, False]

    reports = await ledger.verify()
    flagged_ids = {r["entry_id"] for r in reports}
    assert flagged_ids == {1, 2, 3}
    assert next(r for r in reports if r["entry_id"] == 1)["unresolved"] == [99]


@pytest.mark.asyncio
async def test_supersede_lifecycle_drops_entries_from_support_set() -> None:
    ledger = EvidenceLedgerMemory()
    await ledger.add(_text("Old forecast: rain.", source="forecast_v1", epistemic_type="observation"))
    await ledger.add(
        _text("Updated forecast: sunny.", source="forecast_v2", epistemic_type="observation", supersedes=[1])
    )
    await ledger.add(_text("Plan a picnic.", epistemic_type="claim", citations=[2]))

    statuses = {e.metadata["entry_id"]: e.metadata["status"] for e in ledger.content}
    assert statuses == {1: STATUS_SUPERSEDED, 2: STATUS_ACTIVE, 3: STATUS_ACTIVE}

    active = (await ledger.query("")).results
    assert {e.metadata["entry_id"] for e in active} == {2, 3}  # superseded entry left the support set


@pytest.mark.asyncio
async def test_supersede_re_flags_dependent_claim_in_verify() -> None:
    ledger = EvidenceLedgerMemory()
    await ledger.add(_text("Baseline reading.", source="tool_a", epistemic_type="observation"))
    await ledger.add(_text("Build on the baseline.", epistemic_type="claim", citations=[1]))
    assert await ledger.verify() == []  # initially everything is grounded

    # Repair retires the support; verify() recomputes and re-flags the claim.
    await ledger.supersede([1])
    reports = await ledger.verify()
    assert [r["entry_id"] for r in reports] == [2]
    assert reports[0]["unresolved"] == [1]


@pytest.mark.asyncio
async def test_query_grounding_filter_and_token_overlap() -> None:
    ledger = EvidenceLedgerMemory(query_score_threshold=0.5)
    await ledger.add(_text("paris is sunny", source="get_weather", epistemic_type="observation"))
    await ledger.add(_text("tokyo is rainy", source="get_weather", epistemic_type="observation"))
    await ledger.add(_text("claim with no citations", epistemic_type="claim"))

    # Threshold filters by token overlap.
    assert {e.content for e in (await ledger.query("paris")).results} == {"paris is sunny"}
    # grounded_only drops the ungrounded claim even with an empty query.
    grounded = await ledger.query("", grounded_only=True)
    assert {e.metadata["entry_id"] for e in grounded.results} == {1, 2}


@pytest.mark.asyncio
async def test_clear_resets_ledger() -> None:
    ledger = EvidenceLedgerMemory()
    await ledger.add(_text("x", source="t", epistemic_type="observation"))
    await ledger.clear()
    assert ledger.content == []
    await ledger.add(_text("y", source="t", epistemic_type="observation"))
    assert ledger.content[0].metadata["entry_id"] == 1  # counter reset


@pytest.mark.asyncio
async def test_component_config_roundtrip() -> None:
    ledger = EvidenceLedgerMemory(name="ledger", query_score_threshold=0.25, inject_citations=False)
    config = ledger.dump_component()
    restored = EvidenceLedgerMemory.load_component(config)
    assert isinstance(restored, EvidenceLedgerMemory)
    assert restored.name == "ledger"
    rconf = restored._to_config()
    assert rconf.query_score_threshold == 0.25
    assert rconf.inject_citations is False

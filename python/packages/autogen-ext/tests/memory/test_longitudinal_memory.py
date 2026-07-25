import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import UnboundedChatCompletionContext

from autogen_ext.memory import (  # imported from the non-new package root (the wiring edit)
    InductionOutcome,
    LongitudinalKind,
    LongitudinalMemory,
    LongitudinalMemoryConfig,
    MemoryTier,
)


def _text(text: str, **metadata: object) -> MemoryContent:
    return MemoryContent(content=text, mime_type=MemoryMimeType.TEXT, metadata=dict(metadata))


@pytest.mark.asyncio
async def test_shared_rule_is_inducted_into_governed_tier() -> None:
    memory = LongitudinalMemory()
    await memory.add(_text("Escalate chest pain to a clinician.", tier="shared", kind="rule"))

    assert memory.last_induction_outcomes == [InductionOutcome.ADD_SHARED]
    assert len(memory.shared) == 1
    assert memory.profile == []  # shared context never leaks into the private profile


@pytest.mark.asyncio
async def test_profile_candidate_is_promoted_after_recurrence() -> None:
    memory = LongitudinalMemory(promotion_threshold=2)
    # First mention: kept as an episodic candidate, not yet a stable fact.
    await memory.add(_text("User wants reports in markdown.", kind="profile", key="report_format"))
    first_outcome = memory.last_induction_outcomes[-1]
    assert first_outcome is InductionOutcome.KEEP_EPISODIC
    assert memory.profile == []

    # Second mention: recurrence promotes it into the profile.
    await memory.add(_text("User wants reports in markdown.", kind="profile", key="report_format"))
    second_outcome = memory.last_induction_outcomes[-1]
    assert second_outcome is InductionOutcome.UPDATE_PROFILE
    assert len(memory.profile) == 1
    # Promoted candidates are removed from the episodic buffer to keep context lean.
    assert memory.episodic == []


@pytest.mark.asyncio
async def test_profile_fact_can_be_confirmed_immediately() -> None:
    memory = LongitudinalMemory(promotion_threshold=99)  # recurrence alone would never promote
    await memory.add(_text("Allergic to penicillin.", kind="profile", key="penicillin_allergy", confirmed=True))
    assert memory.last_induction_outcomes[-1] is InductionOutcome.UPDATE_PROFILE
    assert len(memory.profile) == 1


@pytest.mark.asyncio
async def test_procedure_is_revised_in_place() -> None:
    memory = LongitudinalMemory()
    await memory.add(_text("Step 1: gather vitals.", kind="procedure", key="intake"))
    await memory.add(_text("Step 1: gather vitals. Step 2: log them.", kind="procedure", key="intake"))

    assert memory.last_induction_outcomes == [InductionOutcome.REVISE_PROCEDURE, InductionOutcome.REVISE_PROCEDURE]
    assert len(memory.procedures) == 1  # same key revises, does not duplicate
    assert "Step 2" in memory.procedures[0].content  # type: ignore[operator]


@pytest.mark.asyncio
async def test_excluded_episode_is_never_stored_or_retrievable() -> None:
    memory = LongitudinalMemory()
    await memory.add(_text("Secret: social security number here.", exclude=True))

    assert memory.last_induction_outcomes[-1] is InductionOutcome.EXCLUDE
    assert memory.episodic == [] and memory.profile == []
    result = await memory.query("social security")
    assert result.results == []


@pytest.mark.asyncio
async def test_query_retrieves_across_tiers_by_overlap() -> None:
    memory = LongitudinalMemory()
    await memory.add(_text("Always cite the source dataset.", tier="shared", kind="knowledge"))
    await memory.add(_text("User prefers markdown reports.", kind="profile", key="report_format", confirmed=True))
    await memory.add(_text("Discussed booking a flu shot.", kind="episodic"))

    result = await memory.query("markdown reports")
    contents = [c.content for c in result.results]
    assert "User prefers markdown reports." in contents


@pytest.mark.asyncio
async def test_update_context_injects_governed_layered_summary_not_full_history() -> None:
    memory = LongitudinalMemory(recent_episodic_in_context=1)
    await memory.add(_text("Always flag abnormal blood pressure.", tier="shared", kind="rule"))
    await memory.add(_text("User wants metric units.", kind="profile", key="units", confirmed=True))
    # Many episodic traces — only the configured recent window should surface.
    for i in range(5):
        await memory.add(_text(f"Episode {i}: routine check-in.", kind="episodic"))

    ctx = UnboundedChatCompletionContext()
    result = await memory.update_context(ctx)

    assert len(ctx._messages) == 1  # type: ignore[attr-defined]
    injected = ctx._messages[0].content  # type: ignore[attr-defined]
    assert isinstance(injected, str)
    assert "Shared rules" in injected and "User wants metric units." in injected
    # Only the most recent episodic trace is exposed, not the full history.
    assert "Episode 4" in injected
    assert "Episode 0" not in injected
    # The UpdateContextResult surfaces exactly the governed records that were injected.
    assert len(result.memories.results) == 1 + 1 + 1  # shared + profile + recent episodic


def test_component_config_round_trip() -> None:
    memory = LongitudinalMemory(
        name="patient", promotion_threshold=3, max_episodic=10, recent_episodic_in_context=2, query_k=4
    )
    config = memory._to_config()
    assert config == LongitudinalMemoryConfig(
        name="patient", promotion_threshold=3, max_episodic=10, recent_episodic_in_context=2, query_k=4
    )
    rebuilt = LongitudinalMemory._from_config(config)
    assert (rebuilt.name, rebuilt._promotion_threshold, rebuilt._query_k) == ("patient", 3, 4)  # type: ignore[attr-defined]


def test_kind_normalization_respects_tier() -> None:
    memory = LongitudinalMemory()
    # A "rule" kind in the private tier falls back to episodic (rules belong to shared governance).
    assert memory._kind({"kind": "rule"}, MemoryTier.PRIVATE) is LongitudinalKind.EPISODIC  # type: ignore[attr-defined]
    assert memory._kind({"kind": "profile"}, MemoryTier.PRIVATE) is LongitudinalKind.PROFILE  # type: ignore[attr-defined]

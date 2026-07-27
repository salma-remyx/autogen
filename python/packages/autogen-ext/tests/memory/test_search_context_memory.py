import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import SystemMessage

from autogen_ext.memory import SearchContextMemory


@pytest.fixture()
def socm() -> SearchContextMemory:
    memory = SearchContextMemory(max_retries=2)
    memory.declare_attributes("Ada Lovelace", ["birth_year", "field", "nationality"])
    return memory


def test_failure_budget_suppresses_repeats(socm: SearchContextMemory) -> None:
    # A query that has never failed is still retryable.
    assert socm.should_retry("Ada Lovelace phone number") is True

    # Two failures hit the budget (max_retries=2); the query is now exhausted.
    socm.record_failure("Ada Lovelace phone number")
    socm.record_failure("Ada Lovelace phone number")
    assert socm.should_retry("Ada Lovelace phone number") is False
    assert socm.failures[0].exhausted is True

    # A different query is unaffected, and whitespace/case differences dedupe.
    assert socm.should_retry("Ada Lovelace  email") is True
    socm.record_failure("  ada lovelace EMAIL ")
    socm.record_failure("Ada Lovelace email")
    assert socm.should_retry("Ada Lovelace  EMAIL") is False


def test_evidence_resolves_coverage_gap(socm: SearchContextMemory) -> None:
    assert ("Ada Lovelace", "birth_year") in socm.coverage_gaps()

    socm.add_evidence("Ada Lovelace", "birth_year", "1815", citation="Britannica")

    assert ("Ada Lovelace", "birth_year") not in socm.coverage_gaps()
    assert ("Ada Lovelace", "field") in socm.coverage_gaps()
    assert socm.evidence[0].value == "1815"


@pytest.mark.asyncio
async def test_add_routes_memory_content_by_kind(socm: SearchContextMemory) -> None:
    await socm.add(
        MemoryContent(
            content="mathematics",
            mime_type=MemoryMimeType.TEXT,
            metadata={"kind": "evidence", "entity": "Ada Lovelace", "attribute": "field", "citation": "wiki"},
        )
    )
    await socm.add(
        MemoryContent(
            content="Ada Lovelace home address",
            mime_type=MemoryMimeType.TEXT,
            metadata={"kind": "failure", "reason": "no public record"},
        )
    )
    await socm.add(
        MemoryContent(
            content="declare",
            mime_type=MemoryMimeType.TEXT,
            metadata={"kind": "expect", "entity": "Charles Babbage", "attributes": ["profession"]},
        )
    )

    assert ("Ada Lovelace", "field") not in socm.coverage_gaps()
    assert ("Charles Babbage", "profession") in socm.coverage_gaps()
    assert socm.should_retry("Ada Lovelace home address") is True  # only one failure so far


@pytest.mark.asyncio
async def test_update_context_injects_socm_snapshot(socm: SearchContextMemory) -> None:
    socm.add_evidence("Ada Lovelace", "birth_year", "1815", citation="Britannica")
    socm.record_failure("Ada Lovelace phone number")
    socm.record_failure("Ada Lovelace phone number")

    ctx = UnboundedChatCompletionContext()
    result = await socm.update_context(ctx)

    messages = await ctx.get_messages()
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_messages) == 1
    body = system_messages[0].content
    # Grounded evidence is surfaced ...
    assert "Ada Lovelace.birth_year = 1815" in body
    assert "Britannica" in body
    # ... unresolved gaps are surfaced ...
    assert "Ada Lovelace.field" in body
    # ... and exhausted failures are surfaced so they are not repeated.
    assert "Ada Lovelace phone number" in body
    assert result.memories.results  # the injected snapshot is returned too


@pytest.mark.asyncio
async def test_query_returns_matching_evidence(socm: SearchContextMemory) -> None:
    socm.add_evidence("Ada Lovelace", "birth_year", "1815")

    hit = await socm.query("1815")
    assert len(hit.results) == 1
    assert "Ada Lovelace.birth_year = 1815" in hit.results[0].content

    miss = await socm.query("phone number")
    assert miss.results == []


def test_component_roundtrip() -> None:
    original = SearchContextMemory(
        name="research",
        max_retries=3,
        expected_attributes={"Ada Lovelace": ["birth_year", "field"]},
        global_attributes=["nationality"],
    )
    model = original.dump_component()

    assert model.provider == "autogen_ext.memory.search_context_memory.SearchContextMemory"
    assert model.component_type == "memory"

    restored = SearchContextMemory.load_component(model)
    assert isinstance(restored, SearchContextMemory)
    assert restored.name == "research"
    assert restored.coverage_gaps() == [
        ("Ada Lovelace", "birth_year"),
        ("Ada Lovelace", "field"),
        ("Ada Lovelace", "nationality"),
    ]


@pytest.mark.asyncio
async def test_clear_resets_state(socm: SearchContextMemory) -> None:
    socm.add_evidence("Ada Lovelace", "birth_year", "1815")
    socm.record_failure("missing value")

    await socm.clear()

    # Runtime state is gone ...
    assert socm.evidence == []
    assert socm.failures == []
    # ... but the declared schema survives (it is configuration), so the just-cleared
    # evidence shows up again as an unresolved coverage gap.
    assert ("Ada Lovelace", "birth_year") in socm.coverage_gaps()

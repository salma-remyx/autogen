from pathlib import Path

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage
from autogen_ext.memory.antecedent_link import AntecedentLinkConfig, AntecedentLinkMemory
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig

# Skip all tests if ChromaDB is not available
try:
    import chromadb  # pyright: ignore[reportUnusedImport]
except ImportError:
    pytest.skip("ChromaDB not available", allow_module_level=True)

# An earlier session's decision that a later, differently-phrased event depends on.
ANTECEDENT = "Back in March, Theo quietly settled on the Wrenfield account as his top priority for the year."
# A later session's development sharing the subject but not the phrasing.
DEPENDENT = "The Wrenfield deal collapsed overnight and everyone is asking what Theo will do now."
# Unrelated memories, so the store is large enough that k actually truncates.
FILLER = [
    "The onboarding checklist now lives in the shared drive folder.",
    "Quarterly budget spreadsheets are reviewed by finance every April.",
    "The office plants get watered on alternating Wednesdays.",
    "Someone repainted the bike rack outside the north entrance.",
    "The coffee machine was descheduled for maintenance last week.",
    "New laptops ship with the standard image and two monitors.",
    "The team standup moved to ten fifteen in the east room.",
    "Parking permits renew automatically each January.",
    "The library extended its weekend hours through the summer.",
    "Flu shots are available in the lobby each autumn.",
]


def _host_config(
    tmp_path: Path, collection: str = "antecedent_link", k: int = 3
) -> PersistentChromaDBVectorMemoryConfig:
    return PersistentChromaDBVectorMemoryConfig(
        collection_name=collection,
        allow_reset=True,
        k=k,
        persistence_path=str(tmp_path / collection),
    )


@pytest.mark.asyncio
async def test_host_is_system_of_record(tmp_path: Path) -> None:
    """Every add is forwarded to the wrapped ChromaDBVectorMemory host."""
    host = ChromaDBVectorMemory(config=_host_config(tmp_path))
    memory = AntecedentLinkMemory(host=host)
    await memory.clear()

    await memory.add(MemoryContent(content="Gwen is allergic to walnuts.", mime_type=MemoryMimeType.TEXT))

    host_results = await host.query("walnut allergy")
    assert len(host_results.results) == 1
    assert host_results.results[0].metadata is not None
    assert "antecedent_link_id" in host_results.results[0].metadata

    await memory.close()


@pytest.mark.asyncio
async def test_semantically_distant_antecedent_is_recovered(tmp_path: Path) -> None:
    """The reachability failure CABLE targets: top-k truncation hides the antecedent."""
    # A bounded memory interface: the agent sees only the single most similar memory.
    host = ChromaDBVectorMemory(config=_host_config(tmp_path, k=1))
    memory = AntecedentLinkMemory(host=host, config=AntecedentLinkConfig(direct_neighborhood=1.0))
    await memory.clear()

    await memory.add(MemoryContent(content=ANTECEDENT, mime_type=MemoryMimeType.TEXT))
    for filler in FILLER:
        await memory.add(MemoryContent(content=filler, mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content=DEPENDENT, mime_type=MemoryMimeType.TEXT))

    # The antecedent must be linked to the memory that depends on it.
    assert any("al_0" in targets for targets in memory.links.values())

    # Direct similarity does not surface the antecedent within the host's k...
    direct = await host.query("the deal collapsed overnight what was the plan there anyway")
    assert not any("March" in str(r.content) for r in direct.results)

    # ...but link expansion does.
    expanded = await memory.query("the deal collapsed overnight what was the plan there anyway")
    assert any("March" in str(r.content) for r in expanded.results)
    assert any(
        r.metadata is not None and r.metadata.get("via_antecedent_link") is True for r in expanded.results
    ), "antecedent should be surfaced by link expansion, not by similarity"

    await memory.close()


@pytest.mark.asyncio
async def test_direct_neighborhood_is_not_duplicated(tmp_path: Path) -> None:
    """Candidates the host retriever already recovers are subtracted, not linked."""
    host = ChromaDBVectorMemory(config=_host_config(tmp_path, "antecedent_dup"))
    text = "The quarterly review is scheduled for Tuesday morning."
    # A near-duplicate: the host scores it ~1.0 against its antecedent query,
    # well inside the direct semantic neighborhood.
    near_duplicate = "The quarterly review is scheduled for Tuesday morning, as previously planned."

    # Subtraction disabled: the near-duplicate is verified and linked.
    no_subtraction = AntecedentLinkMemory(host=host, config=AntecedentLinkConfig(direct_neighborhood=1.0))
    await no_subtraction.clear()
    await no_subtraction.add(MemoryContent(content=text, mime_type=MemoryMimeType.TEXT))
    await no_subtraction.add(MemoryContent(content=near_duplicate, mime_type=MemoryMimeType.TEXT))
    assert any(no_subtraction.links.values()), "expected links when subtraction is disabled"

    # Subtraction enabled (the default): the near-duplicate is dropped instead.
    subtracting = AntecedentLinkMemory(host=host, config=AntecedentLinkConfig())
    await subtracting.clear()
    await subtracting.add(MemoryContent(content=text, mime_type=MemoryMimeType.TEXT))
    await subtracting.add(MemoryContent(content=near_duplicate, mime_type=MemoryMimeType.TEXT))
    assert not any(subtracting.links.values()), "near-neighborhood should be subtracted"

    await no_subtraction.close()


@pytest.mark.asyncio
async def test_graph_is_sparse(tmp_path: Path) -> None:
    """At most max_links_per_memory antecedents are linked per new memory."""
    host = ChromaDBVectorMemory(config=_host_config(tmp_path, "antecedent_sparse"))
    memory = AntecedentLinkMemory(host=host, config=AntecedentLinkConfig(max_links_per_memory=1))
    await memory.clear()

    for city in ("Oslo", "Kyoto", "Lima"):
        await memory.add(
            MemoryContent(content=f"Marcus photographed the {city} skyline at dawn.", mime_type=MemoryMimeType.TEXT)
        )
    await memory.add(
        MemoryContent(
            content="Marcus is printing his Osaka and Kyoto skyline photos for the exhibit.",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    assert any(memory.links.values()), "expected at least one link to form"
    assert all(len(targets) <= 1 for targets in memory.links.values())

    await memory.close()


@pytest.mark.asyncio
async def test_expansion_disabled_restores_host_behavior(tmp_path: Path) -> None:
    """expansion_results=0 turns the wrapper back into a pass-through."""
    host = ChromaDBVectorMemory(config=_host_config(tmp_path, "antecedent_off"))
    memory = AntecedentLinkMemory(host=host, config=AntecedentLinkConfig(expansion_results=0))
    await memory.clear()

    await memory.add(MemoryContent(content=ANTECEDENT, mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content=DEPENDENT, mime_type=MemoryMimeType.TEXT))

    results = await memory.query("the deal collapsed overnight")
    assert not any(
        r.metadata is not None and r.metadata.get("via_antecedent_link") is True for r in results.results
    )

    await memory.close()


@pytest.mark.asyncio
async def test_update_context_injects_expanded_memories(tmp_path: Path) -> None:
    """The Memory ABC hook agents call surfaces the antecedent into the model context."""
    host = ChromaDBVectorMemory(config=_host_config(tmp_path, "antecedent_ctx"))
    memory = AntecedentLinkMemory(host=host, config=AntecedentLinkConfig(direct_neighborhood=1.0))
    await memory.clear()

    await memory.add(MemoryContent(content=ANTECEDENT, mime_type=MemoryMimeType.TEXT))
    for filler in FILLER:
        await memory.add(MemoryContent(content=filler, mime_type=MemoryMimeType.TEXT))
    await memory.add(MemoryContent(content=DEPENDENT, mime_type=MemoryMimeType.TEXT))

    context = BufferedChatCompletionContext(buffer_size=5)
    await context.add_message(UserMessage(content="the deal collapsed overnight, what was the plan", source="user"))

    result = await memory.update_context(context)

    assert any("March" in str(r.content) for r in result.memories.results)
    messages = await context.get_messages()
    assert len(messages) > 1
    assert any("March" in str(m.content) for m in messages)

    await memory.close()


@pytest.mark.asyncio
async def test_component_round_trip() -> None:
    """The wrapper serializes through the declarative Component loader."""
    memory = AntecedentLinkMemory(host=ChromaDBVectorMemory(), config=AntecedentLinkConfig(name="linked"))

    dumped = memory.dump_component()
    assert dumped.config["name"] == "linked"
    assert dumped.provider == "autogen_ext.memory.antecedent_link.AntecedentLinkMemory"

    loaded = AntecedentLinkMemory.load_component(dumped)
    assert isinstance(loaded, AntecedentLinkMemory)
    assert loaded.name == "linked"

    await memory.close()
    await loaded.close()

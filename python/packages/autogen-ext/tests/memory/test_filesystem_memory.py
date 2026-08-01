"""Integration tests for :class:`FilesystemMemory`.

These exercise the memory through the existing :mod:`autogen_core.memory`
contract (the non-new modules) -- ``add``/``query``/``update_context`` -- so the
test proves the backend plugs into autogen's live memory-consumption path, not
just its own internals.
"""

from pathlib import Path

import pytest
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage

from autogen_ext.memory import FilesystemMemory, FilesystemMemoryConfig


def _make_item(text: str, category: str) -> MemoryContent:
    return MemoryContent(content=text, mime_type=MemoryMimeType.MARKDOWN, metadata={"category": category})


@pytest.mark.asyncio
async def test_memory_satisfies_memory_protocol(tmp_path: Path) -> None:
    memory = FilesystemMemory(config=FilesystemMemoryConfig(root_path=str(tmp_path / "store")))
    assert isinstance(memory, Memory)
    await memory.close()


@pytest.mark.asyncio
async def test_add_query_roundtrip_persists_to_markdown(tmp_path: Path) -> None:
    memory = FilesystemMemory(config=FilesystemMemoryConfig(root_path=str(tmp_path / "store"), k=5))

    await memory.add(_make_item("The user prefers temperatures reported in Celsius.", "preferences"))
    await memory.add(_make_item("Boil pasta in heavily salted water for nine minutes.", "cooking"))

    # The hierarchy organization filed each item into its own category directory.
    assert (tmp_path / "store" / "preferences").is_dir()
    assert (tmp_path / "store" / "cooking").is_dir()

    results = await memory.query("pasta boiling")
    assert results.results
    assert "pasta" in str(results.results[0].content).lower()
    assert results.results[0].metadata is not None
    assert "score" in results.results[0].metadata
    await memory.close()


@pytest.mark.asyncio
async def test_update_context_mutates_model_context(tmp_path: Path) -> None:
    # Exercises the live forward path: AssistantAgent consumes Memory via
    # update_context, which must inject a SystemMessage into the model context.
    memory = FilesystemMemory(config=FilesystemMemoryConfig(root_path=str(tmp_path / "store"), k=3))
    await memory.add(_make_item("The user's name is Ada and she likes concise answers.", "profile"))

    model_context = BufferedChatCompletionContext(buffer_size=10)
    await model_context.add_message(UserMessage(content="What do you know about the user?", source="user"))
    await memory.update_context(model_context)

    messages = await model_context.get_messages()
    assert any(isinstance(m, SystemMessage) and "Ada" in m.content for m in messages)
    await memory.close()


@pytest.mark.asyncio
async def test_search_economy_hierarchy_reads_fewer_files_than_flat(tmp_path: Path) -> None:
    # The headline result from arXiv:2607.26637v1 -- organization buys search
    # economy. Populate identical material into a hierarchy store and a flat
    # store, then confirm a category-scoped query reads strictly fewer files
    # from the hierarchy while still returning the relevant items.
    categories = {
        "cooking": [
            "Boil pasta in salted water until al dente.",
            "Sear the steak in a cast iron pan with butter.",
            "Whisk eggs and fold into the cake batter slowly.",
        ],
        "travel": [
            "Renew the passport at least six months before it expires.",
            "Book the train ticket to Geneva in advance for a discount.",
            "Exchange currency at the airport for a worse rate.",
        ],
        "coding": [
            "Run the type checker before opening the pull request.",
            "Pin the dependency versions in the lockfile.",
            "Write a regression test for the patched bug.",
        ],
        "health": [
            "Drink water throughout the day to stay hydrated.",
            "Stretch before running to avoid injuring the hamstring.",
            "Sleep seven to nine hours for recovery.",
        ],
    }

    def make(store: Path, organization: str) -> FilesystemMemory:
        return FilesystemMemory(config=FilesystemMemoryConfig(root_path=str(store), organization=organization, k=10))

    hierarchy = make(tmp_path / "hier", "hierarchy")
    flat = make(tmp_path / "flat", "flat")
    for category, items in categories.items():
        for item in items:
            content = _make_item(item, category)
            await hierarchy.add(content)
            await flat.add(content)

    total_items = sum(len(v) for v in categories.values())

    await hierarchy.query("pasta steak eggs cooking recipe")
    await flat.query("pasta steak eggs cooking recipe")

    # Hierarchy prunes to the matched category directory; flat cannot prune.
    assert hierarchy.last_query_files_scanned < flat.last_query_files_scanned
    assert hierarchy.last_query_files_scanned == len(categories["cooking"])
    assert flat.last_query_files_scanned == total_items

    # Recall is preserved: the hierarchy still surfaces the cooking memories.
    results = await hierarchy.query("pasta steak eggs cooking recipe")
    contents = " ".join(str(r.content).lower() for r in results.results)
    assert "pasta" in contents
    await hierarchy.close()
    await flat.close()


@pytest.mark.asyncio
async def test_declarative_component_roundtrip(tmp_path: Path) -> None:
    config = FilesystemMemoryConfig(root_path=str(tmp_path / "store"), name="my_store", k=7)
    memory = FilesystemMemory(config=config)

    assert memory._to_config() == config

    exported = memory.dump_component()
    assert exported.provider == "autogen_ext.memory.FilesystemMemory"

    restored = FilesystemMemory.load_component(exported)
    assert isinstance(restored, FilesystemMemory)
    assert restored.name == "my_store"
    assert restored._to_config().k == 7
    await restored.close()

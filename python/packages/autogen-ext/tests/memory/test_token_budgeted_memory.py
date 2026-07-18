import pytest
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.memory.token_budgeted import TokenBudgetedMemory, TokenBudgetedMemoryConfig
from pydantic import ValidationError


def _text(content: str) -> MemoryContent:
    return MemoryContent(content=content, mime_type=MemoryMimeType.TEXT)


def _query(question: str) -> BufferedChatCompletionContext:
    return BufferedChatCompletionContext(
        buffer_size=10, initial_messages=[UserMessage(content=question, source="user")]
    )


@pytest.mark.asyncio
async def test_curates_fewer_memories_than_raw_list_memory() -> None:
    """The layer injects a curated subset where ListMemory dumps everything raw."""
    base = ListMemory()
    await base.add(_text("The user prefers Python for data science tasks."))
    await base.add(_text("The user prefers Python for data science work."))  # near-duplicate
    await base.add(_text("Bananas are a popular yellow fruit."))  # irrelevant

    question = "What language does the user prefer for data science?"

    # Raw ListMemory injects every stored memory verbatim.
    raw_result = await base.update_context(_query(question))
    assert len(raw_result.memories.results) == 3

    curated = TokenBudgetedMemory(base, relevance_threshold=0.15, dedup_threshold=0.5)
    curated_context = _query(question)
    curated_result = await curated.update_context(curated_context)

    # Fewer than the raw dump: the near-duplicate is collapsed and the irrelevant item dropped.
    assert len(curated_result.memories.results) < 3
    assert any("Python" in str(m.content) for m in curated_result.memories.results)
    assert all("Banana" not in str(m.content) for m in curated_result.memories.results)

    messages = await curated_context.get_messages()
    assert any(isinstance(m, SystemMessage) for m in messages)


@pytest.mark.asyncio
async def test_dedup_collapses_near_duplicates() -> None:
    base = ListMemory()
    await base.add(_text("The user prefers Python for data science tasks."))
    await base.add(_text("The user prefers Python for data science work."))

    memory = TokenBudgetedMemory(base, dedup_threshold=0.5)
    result = await memory.update_context(_query("What language does the user prefer for data science?"))

    assert len(result.memories.results) == 1
    assert "Python" in str(result.memories.results[0].content)


@pytest.mark.asyncio
async def test_relevance_threshold_filters_irrelevant_memories() -> None:
    base = ListMemory()
    await base.add(_text("The user prefers Python."))  # relevant
    await base.add(_text("Bananas are yellow."))  # irrelevant

    memory = TokenBudgetedMemory(base, relevance_threshold=0.2, dedup_threshold=0.99)
    result = await memory.update_context(_query("Which language does the user prefer?"))

    assert len(result.memories.results) == 1
    assert "Python" in str(result.memories.results[0].content)


@pytest.mark.asyncio
async def test_token_budget_caps_injection() -> None:
    base = ListMemory()
    for fruit in (
        "apples are red fruit",
        "bananas are yellow fruit",
        "grapes are purple fruit",
        "oranges are orange fruit",
    ):
        await base.add(_text(fruit))

    memory = TokenBudgetedMemory(base, token_budget=10, dedup_threshold=0.99)
    result = await memory.update_context(_query("tell me about fruit"))

    selected = result.memories.results
    injected_words = sum(len(str(m.content).split()) for m in selected)
    assert injected_words <= 10
    assert len(selected) < 4  # budget stopped the injection before exhausting the store


@pytest.mark.asyncio
async def test_delegates_to_base_store() -> None:
    memory = TokenBudgetedMemory(ListMemory())

    await memory.add(_text("hello world"))
    results = await memory.query("anything")
    assert len(results.results) == 1
    assert results.results[0].content == "hello world"

    await memory.clear()
    assert len((await memory.query("anything")).results) == 0


@pytest.mark.asyncio
async def test_update_context_empty_context_returns_empty() -> None:
    memory = TokenBudgetedMemory(ListMemory(), token_budget=64)
    context = BufferedChatCompletionContext(buffer_size=10)

    result = await memory.update_context(context)

    assert result.memories.results == []
    assert await context.get_messages() == []


def test_component_round_trip() -> None:
    base = ListMemory(name="prefs")
    memory = TokenBudgetedMemory(base, name="wrapper", token_budget=64, dedup_threshold=0.7)

    model = memory.dump_component()

    assert model.provider == "autogen_ext.memory.token_budgeted.TokenBudgetedMemory"
    assert model.config["token_budget"] == 64
    assert model.config["base_memory"]["provider"] == "autogen_core.memory.ListMemory"

    loaded = TokenBudgetedMemory.load_component(model)
    assert isinstance(loaded, TokenBudgetedMemory)
    assert loaded.name == "wrapper"
    assert isinstance(loaded.base_memory, ListMemory)
    assert loaded.base_memory.name == "prefs"


def test_config_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        TokenBudgetedMemoryConfig(token_budget=-1)
    with pytest.raises(ValidationError):
        TokenBudgetedMemoryConfig(dedup_threshold=1.5)

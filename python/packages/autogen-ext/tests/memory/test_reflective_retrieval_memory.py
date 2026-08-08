import time

import pytest
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage

from autogen_ext.memory.reflective_retrieval_memory import ReflectiveRetrievalMemory, ReflectiveRetrievalMemoryConfig


class _RecorderMemory(ListMemory):
    """ListMemory that records the last query string it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.last_query: str | None = None

    async def query(self, query: str | MemoryContent = "", cancellation_token=None, **kwargs):  # type: ignore[override]
        self.last_query = query if isinstance(query, str) else str(query.content)
        return await super().query(query, cancellation_token, **kwargs)


@pytest.mark.asyncio
async def test_delegates_factual_add_and_query_to_underlying() -> None:
    facts = ListMemory()
    memory = ReflectiveRetrievalMemory(ReflectiveRetrievalMemoryConfig(underlying_memory=facts))

    await memory.add(MemoryContent(content="hello world", mime_type=MemoryMimeType.TEXT))

    result = await memory.query("hello")
    assert any(m.content == "hello world" for m in result.results)
    # Factual content reached the wrapped backend, and no reflective lesson was recorded.
    assert len(facts.content) == 1
    assert memory.experiences == []


@pytest.mark.asyncio
async def test_reflective_guidance_augments_query_to_underlying() -> None:
    recorder = _RecorderMemory()
    memory = ReflectiveRetrievalMemory(ReflectiveRetrievalMemoryConfig(underlying_memory=recorder))
    await memory.add(MemoryContent(content="api config lives in settings.json", mime_type=MemoryMimeType.TEXT))

    # Store a lesson learned about retrieving for this kind of query.
    memory.reflect_on_query(
        query="how do I configure api options",
        guidance="search for 'settings' and 'config', not just 'options'",
        helpful=True,
    )

    result = await memory.query("configure api options")

    # The factual evidence is returned ...
    assert any("settings.json" in str(m.content) for m in result.results)
    # ... and the underlying store received the guidance-augmented query.
    assert recorder.last_query is not None
    assert "search for 'settings' and 'config'" in recorder.last_query
    # The consulted lesson accrued usage-frequency credit.
    assert memory.experiences[0].usage_count == 1


@pytest.mark.asyncio
async def test_lesson_stored_via_add_metadata() -> None:
    memory = ReflectiveRetrievalMemory()
    await memory.add(
        MemoryContent(
            content="lesson body",
            mime_type=MemoryMimeType.TEXT,
            metadata={
                "type": "reflective_lesson",
                "query_signature": "deploy the service",
                "guidance": "include 'kubernetes' and 'helm'",
            },
        )
    )
    assert len(memory.experiences) == 1
    assert memory.experiences[0].guidance == "include 'kubernetes' and 'helm'"


@pytest.mark.asyncio
async def test_update_context_injects_only_factual_evidence() -> None:
    facts = ListMemory()
    memory = ReflectiveRetrievalMemory(ReflectiveRetrievalMemoryConfig(underlying_memory=facts))
    await memory.add(MemoryContent(content="the answer is 42", mime_type=MemoryMimeType.TEXT))

    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="what is the answer", source="user"))

    result = await memory.update_context(context)
    assert any("42" in str(m.content) for m in result.memories.results)
    messages = await context.get_messages()
    assert any("42" in str(m.content) for m in messages)


@pytest.mark.asyncio
async def test_temporal_decay_prunes_stale_lessons() -> None:
    memory = ReflectiveRetrievalMemory(
        ReflectiveRetrievalMemoryConfig(min_lifecycle_score=0.5, decay_half_life_days=1.0)
    )
    experience = memory.reflect_on_query("old query thing", guidance="stale guidance")
    experience.created_at = time.time() - 10 * 86400.0  # 10 days old; half-life 1 day

    await memory.query("old query thing")

    assert memory.experiences == []


@pytest.mark.asyncio
async def test_negative_feedback_suppresses_and_prunes_lesson() -> None:
    # A high lifecycle floor makes the effect of negative reuse feedback observable.
    memory = ReflectiveRetrievalMemory(ReflectiveRetrievalMemoryConfig(min_lifecycle_score=0.5))
    experience = memory.reflect_on_query("reset password", guidance="search 'credentials'")

    # Two negative reuse-feedback signals floor the lesson's influence well below the floor.
    memory.record_feedback(experience.id, helpful=False)
    memory.record_feedback(experience.id, helpful=False)

    await memory.query("reset password")

    # The suppressed lesson fell below the lifecycle floor and was pruned.
    assert memory.experiences == []


def test_component_config_round_trip() -> None:
    memory = ReflectiveRetrievalMemory(
        ReflectiveRetrievalMemoryConfig(similarity_threshold=0.25, max_guidance_lessons=5, decay_half_life_days=14.0)
    )
    config = memory.dump_component()
    assert config.provider == "autogen_ext.memory.reflective_retrieval_memory.ReflectiveRetrievalMemory"

    restored = ReflectiveRetrievalMemory.load_component(config)
    assert restored.config.similarity_threshold == 0.25
    assert restored.config.decay_half_life_days == 14.0

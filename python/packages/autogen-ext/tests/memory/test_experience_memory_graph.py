"""Tests for the Experience Memory Graph memory extension.

These tests exercise the new memory through the same public surface that
:class:`~autogen_agentchat.agents.AssistantAgent` uses (the ``Memory`` ABC and
``update_context``), importing from the existing, non-new ``autogen_core``
modules to prove real integration.
"""

import pytest
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage

from autogen_ext.memory.experience_memory_graph import ActionDecisionGraph, ExperienceMemoryGraph

EXPERT_STEPS = [
    {"observation": "a locked door", "action": "find key"},
    {"observation": "a brass key", "action": "pick up key"},
    {"observation": "a locked door", "action": "unlock door"},
    {"observation": "the open door", "action": "open door"},
]


def _expert_memory() -> ExperienceMemoryGraph:
    memory = ExperienceMemoryGraph()
    return memory


@pytest.mark.asyncio
async def test_implements_memory_abc() -> None:
    """The extension is a valid autogen_core Memory component."""
    memory = ExperienceMemoryGraph()
    assert isinstance(memory, Memory)


@pytest.mark.asyncio
async def test_corrects_known_task_failure() -> None:
    """A failed run for a known task yields a one-shot relabel correction from the expert graph."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(
            content={"task": "unlock_door", "kind": "expert", "steps": EXPERT_STEPS},
            mime_type=MemoryMimeType.JSON,
        )
    )
    # Failed run: right first action, wrong middle action, misses the last two.
    result = await memory.query("task: unlock_door\nfind key -> push door -> open door")

    assert len(result.results) == 1
    insight = str(result.results[0].content)
    assert "unlock_door" in insight
    assert "relabel" in insight
    # The correction must point at the missing expert actions.
    assert "pick up key" in insight
    assert "unlock door" in insight
    metadata = result.results[0].metadata
    assert metadata is not None
    assert metadata["edits"][0]["op"] == "relabel"


@pytest.mark.asyncio
async def test_update_context_injects_correction() -> None:
    """update_context (the path AssistantAgent calls) injects the correction as a system message."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(
            content={"task": "unlock_door", "kind": "expert", "steps": EXPERT_STEPS},
            mime_type=MemoryMimeType.JSON,
        )
    )
    context = BufferedChatCompletionContext(buffer_size=5)
    await context.add_message(
        UserMessage(content="task: unlock_door\nfind key -> push door -> open door", source="user")
    )

    update_result = await memory.update_context(context)

    assert len(update_result.memories.results) == 1
    messages = await context.get_messages()
    assert isinstance(messages[-1], SystemMessage)
    assert "relabel" in str(messages[-1].content)


@pytest.mark.asyncio
async def test_no_correction_when_already_on_expert_path() -> None:
    """A run that already follows the expert workflow needs no correction."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(
            content={"task": "unlock_door", "kind": "expert", "steps": EXPERT_STEPS},
            mime_type=MemoryMimeType.JSON,
        )
    )
    result = await memory.query("task: unlock_door\nfind key -> pick up key -> unlock door -> open door")
    assert result.results == []


@pytest.mark.asyncio
async def test_anonymous_query_gated_by_overlap() -> None:
    """Anonymous (task-less) queries only fire when the action overlap clears the threshold."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(
            content={"task": "unlock_door", "kind": "expert", "steps": EXPERT_STEPS},
            mime_type=MemoryMimeType.JSON,
        )
    )
    # Fully divergent, no task -> below threshold -> nothing.
    divergent = await memory.query("push door -> push door")
    assert divergent.results == []
    # Overlapping actions, no task -> above threshold -> correction with an 'add'.
    overlapping = await memory.query("find key -> open door")
    assert len(overlapping.results) == 1
    overlapping_metadata = overlapping.results[0].metadata
    assert overlapping_metadata is not None
    assert any(edit["op"] == "add" for edit in overlapping_metadata["edits"])


@pytest.mark.asyncio
async def test_branching_graph_completes_either_branch() -> None:
    """Two expert workflows sharing a prefix merge into one branching decision graph."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(content="[expert] task: brew\nboil water -> add tea -> steep", mime_type=MemoryMimeType.TEXT)
    )
    await memory.add(
        MemoryContent(content="[expert] task: brew\nboil water -> add coffee -> steep", mime_type=MemoryMimeType.TEXT)
    )

    tea = await memory.query("task: brew\nboil water -> add tea")
    coffee = await memory.query("task: brew\nboil water -> add coffee")

    assert len(tea.results) == 1 and len(coffee.results) == 1
    # Each partial prefix is completed to the shared terminal action via its own branch.
    assert "steep" in str(tea.results[0].content)
    assert "steep" in str(coffee.results[0].content)


def test_action_decision_graph_merges_shared_prefix() -> None:
    """Inserting two paths with a shared prefix yields a single graph with two branches."""
    graph = ActionDecisionGraph(task="brew", kind="expert")
    graph.insert_path([("", "boil water"), ("", "add tea"), ("", "steep")])
    graph.insert_path([("", "boil water"), ("", "add coffee"), ("", "steep")])

    paths = graph.paths()
    assert len(paths) == 2
    # Both branches share the root decision node.
    assert {path[0] for path in paths} == {0}


@pytest.mark.asyncio
async def test_component_roundtrip() -> None:
    """The memory serializes through the declarative Component protocol used by AssistantAgent."""
    memory = ExperienceMemoryGraph()
    dumped = memory.dump_component()
    loaded = ExperienceMemoryGraph.load_component(dumped)
    assert isinstance(loaded, ExperienceMemoryGraph)


@pytest.mark.asyncio
async def test_noop_retry_yields_avoid_edit() -> None:
    """A verbatim retry is parallelized onto the last valid step and surfaces as an 'avoid' edit."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(
            content={"task": "unlock_door", "kind": "expert", "steps": EXPERT_STEPS},
            mime_type=MemoryMimeType.JSON,
        )
    )
    # The run follows the expert workflow exactly but retries "find key" (an invalid no-op).
    result = await memory.query("task: unlock_door\nfind key -> find key -> pick up key -> unlock door -> open door")

    assert len(result.results) == 1
    metadata = result.results[0].metadata
    assert metadata is not None
    avoid = [edit for edit in metadata["edits"] if edit["op"] == "avoid"]
    assert len(avoid) == 1
    assert avoid[0]["actions"] == ["find key"]
    assert avoid[0]["at"] == 1  # anchored to the last valid step
    assert "avoid repeating find key" in str(result.results[0].content)


@pytest.mark.asyncio
async def test_action_tuple_normalization_ignores_phrasing_variants() -> None:
    """Actions match as (verb, object, receptacle) tuples: articles/phrasing create no edits."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(
            content={"task": "unlock_door", "kind": "expert", "steps": EXPERT_STEPS},
            mime_type=MemoryMimeType.JSON,
        )
    )
    # Same decisions as the expert path, phrased with articles.
    result = await memory.query("task: unlock_door\nfind a key -> pick up the key -> unlock the door -> open door")
    assert result.results == []


def test_action_tuple_normalization_merges_phrasing_variants() -> None:
    """Phrasing variants of the same decision merge into one shared graph path."""
    graph = ActionDecisionGraph(task="brew", kind="expert")
    graph.insert_path([("", "boil the water"), ("", "add the tea"), ("", "steep")])
    graph.insert_path([("", "boil water"), ("", "add tea"), ("", "steep")])

    assert len(graph.paths()) == 1


@pytest.mark.asyncio
async def test_ranked_candidates_expose_alternatives() -> None:
    """Top-K retrieval ranks candidate expert paths and exposes runners-up in metadata."""
    memory = _expert_memory()
    await memory.add(
        MemoryContent(
            content={"task": "unlock_door", "kind": "expert", "steps": EXPERT_STEPS},
            mime_type=MemoryMimeType.JSON,
        )
    )
    await memory.add(
        MemoryContent(
            content={
                "task": "fetch_key",
                "kind": "expert",
                "steps": [
                    {"observation": "", "action": "find key"},
                    {"observation": "", "action": "pick up key"},
                    {"observation": "", "action": "return key"},
                ],
            },
            mime_type=MemoryMimeType.JSON,
        )
    )
    result = await memory.query("task: unlock_door\nfind key -> push door -> open door")

    assert len(result.results) == 1
    metadata = result.results[0].metadata
    assert metadata is not None
    assert metadata["task"] == "unlock_door"
    assert any(alt["task"] == "fetch_key" for alt in metadata["alternatives"])

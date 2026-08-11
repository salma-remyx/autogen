from typing import List, cast

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_core import FunctionCall
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    SystemMessage,
    UserMessage,
)

from autogen_ext.memory.distilled_memory import (
    DistilledFunctionMemory,
    DistilledMemory,
    DistilledMemoryConfig,
    DistilledSubtaskMemory,
    DistilledWorkflowMemory,
)
from autogen_ext.models.replay import ReplayChatCompletionClient


def _build_memory() -> DistilledMemory:
    return DistilledMemory(
        DistilledMemoryConfig(
            workflow_memories=[
                DistilledWorkflowMemory(
                    task="book a flight",
                    strategy="Search by IATA airport code, then filter by date.",
                )
            ],
            subtask_memories=[
                DistilledSubtaskMemory(
                    task="book a flight",
                    example="search_flights(origin='JFK', dest='SFO', date='2026-09-01').",
                )
            ],
            function_memories=[
                DistilledFunctionMemory(
                    tool_name="search_flights",
                    conventions="origin/dest are IATA codes; date is ISO 8601.",
                    pitfalls="City names silently return empty results.",
                )
            ],
        )
    )


@pytest.mark.asyncio
async def test_proactive_injection_into_model_context() -> None:
    """Workflow + subtask memory are injected proactively as a SystemMessage."""
    memory = _build_memory()
    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="book me a flight to SFO", source="user"))

    result = await memory.update_context(context)

    messages = await context.get_messages()
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
    combined = system_msgs[0].content
    assert "Search by IATA airport code" in combined  # workflow strategy
    assert "search_flights(origin='JFK'" in combined  # subtask example
    # Proactive tiers are reported in the update result.
    assert any(mc.metadata and mc.metadata.get("tier") == "workflow+subtask" for mc in result.memories.results)


@pytest.mark.asyncio
async def test_reactive_function_memory_on_tool_error() -> None:
    """Function memory is retrieved reactively when a tool call errors."""
    memory = _build_memory()
    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="book a flight", source="user"))
    await context.add_message(
        AssistantMessage(content=[FunctionCall(id="1", arguments="{}", name="search_flights")], source="student")
    )
    await context.add_message(
        FunctionExecutionResultMessage(
            content=[
                FunctionExecutionResult(
                    content="Error: 'New York' is not a valid airport code",
                    name="search_flights",
                    call_id="1",
                    is_error=True,
                )
            ]
        )
    )

    result = await memory.update_context(context)

    messages = await context.get_messages()
    function_msgs = [m for m in messages if isinstance(m, SystemMessage) and "function memory for tool" in m.content]
    assert len(function_msgs) == 1
    guidance = function_msgs[0].content
    assert "IATA codes" in guidance  # conventions
    assert "City names silently return empty results" in guidance  # pitfalls
    assert any(mc.metadata and mc.metadata.get("tier") == "function" for mc in result.memories.results)


@pytest.mark.asyncio
async def test_no_reactive_injection_without_error() -> None:
    """A successful tool call must not trigger function-memory injection."""
    memory = _build_memory()
    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="book a flight", source="user"))
    await context.add_message(
        AssistantMessage(content=[FunctionCall(id="1", arguments="{}", name="search_flights")], source="student")
    )
    await context.add_message(
        FunctionExecutionResultMessage(
            content=[FunctionExecutionResult(content="OK", name="search_flights", call_id="1", is_error=False)]
        )
    )

    await memory.update_context(context)

    messages = await context.get_messages()
    assert not any(isinstance(m, SystemMessage) and "function memory for tool" in m.content for m in messages)


@pytest.mark.asyncio
async def test_query_retrieves_function_memory_by_tool_name() -> None:
    """query() is the reactive retrieval surface for function-tier memory."""
    memory = _build_memory()
    result = await memory.query("search_flights")
    assert len(result.results) == 1
    content = str(result.results[0].content)
    assert "IATA codes" in content
    assert result.results[0].metadata is not None
    assert result.results[0].metadata["tool_name"] == "search_flights"


@pytest.mark.asyncio
async def test_add_routes_by_tier() -> None:
    """add() incrementally populates tiers, enabling offline teacher distillation."""
    memory = DistilledMemory()
    await memory.add(
        MemoryContent(
            content="Always confirm currency before quoting prices.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"tier": "workflow", "task": "handle payment"},
        )
    )
    await memory.add(
        MemoryContent(
            content="Do not pass city names.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"tier": "function", "tool_name": "search_flights", "pitfalls": "returns empty"},
        )
    )

    context = BufferedChatCompletionContext(buffer_size=10)
    await context.add_message(UserMessage(content="hi", source="user"))
    await memory.update_context(context)
    messages = await context.get_messages()
    assert any(isinstance(m, SystemMessage) and "confirm currency before quoting prices" in m.content for m in messages)


@pytest.mark.asyncio
async def test_proactive_injection_through_assistant_agent() -> None:
    """The existing AssistantAgent inference loop invokes DistilledMemory.

    AssistantAgent calls Memory.update_context before each inference step; this
    asserts the distilled workflow strategy actually reaches the model prompt,
    proving the new memory is wired into the existing call site.
    """
    memory = _build_memory()
    mock_client = ReplayChatCompletionClient(["Booked."])
    agent = AssistantAgent("student", model_client=mock_client, memory=[memory])

    await agent.run(task="book me a flight to San Francisco")

    assert len(mock_client.create_calls) >= 1
    sent_messages = cast(List[object], mock_client.create_calls[0]["messages"])
    sent_text = " ".join(str(getattr(m, "content", "")) for m in sent_messages)
    assert "Search by IATA airport code" in sent_text
    await memory.close()

import logging
from typing import Any, List, Set

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import FunctionExecutionResultMessage, SystemMessage
from typing_extensions import Self

from ._distilled_configs import (
    DistilledFunctionMemory,
    DistilledMemoryConfig,
    DistilledSubtaskMemory,
    DistilledWorkflowMemory,
)

logger = logging.getLogger(__name__)


class DistilledMemory(Memory, Component[DistilledMemoryConfig]):
    """Hierarchical teacher->student memory (Agent Memory Distillation, AMD).

    A training-free memory that consumes knowledge *distilled* from a (larger)
    teacher agent's successful trajectories and exposes it to a (smaller) student
    agent through three complementary tiers:

    * **Workflow memory** -- task-level strategy. Injected *proactively* into the
      model context at the start of a task.
    * **Subtask memory** -- concrete behavioural examples at intermediate
      granularity. Injected *proactively* alongside workflow memory.
    * **Function memory** -- per-tool calling conventions and common pitfalls.
      Retrieved *reactively* when a tool call returns an error.

    This class implements the AMD runtime mechanism (hierarchical storage +
    proactive injection of workflow/subtask memory + reactive retrieval of
    function memory on tool-calling errors). The teacher-distillation pipeline
    that *produces* the entries is intentionally out of scope: tiers are populated
    from the :class:`DistilledMemoryConfig` (i.e. the output of an offline teacher
    run) or incrementally via :meth:`add`. This is a Mode 2 (adapted port)
    integration: the paper's core mechanism is kept at full fidelity while its
    learned teacher / trajectory-extraction pipeline is substituted with
    config-driven population.

    The class plugs into any agent that consumes the
    :class:`~autogen_core.memory.Memory` ABC -- e.g.
    ``AssistantAgent(memory=[DistilledMemory(...)])`` -- whose inference loop
    calls :meth:`update_context` before each inference step, invoking this memory
    automatically.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.memory.distilled_memory import (
                DistilledFunctionMemory,
                DistilledMemory,
                DistilledMemoryConfig,
                DistilledSubtaskMemory,
                DistilledWorkflowMemory,
            )


            async def main() -> None:
                memory = DistilledMemory(
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
                assistant = AssistantAgent(name="student", model_client=..., memory=[memory])
                await assistant.run(task="book me a flight to San Francisco")


            asyncio.run(main())

    Args:
        config: Distilled tiers. When ``None`` an empty memory is created and the
            tiers can be populated later via :meth:`add`.
    """

    component_config_schema = DistilledMemoryConfig
    component_provider_override = "autogen_ext.memory.distilled_memory.DistilledMemory"

    def __init__(self, config: DistilledMemoryConfig | None = None) -> None:
        self._config = config or DistilledMemoryConfig()
        self._name = self._config.name or "distilled_memory"
        self._proactively_injected = False
        self._reactive_tools_injected: Set[str] = set()

    @property
    def name(self) -> str:
        """Memory instance identifier."""
        return self._name

    # ---- formatting helpers ------------------------------------------------

    def _format_proactive(self) -> str | None:
        workflows = self._config.workflow_memories
        subtasks = self._config.subtask_memories[: self._config.max_subtasks]
        if not workflows and not subtasks:
            return None
        lines: List[str] = ["Distilled agent memory (transferred from a teacher):"]
        if workflows:
            lines.append("\nWorkflow strategies:")
            for wf in workflows:
                lines.append(f"- [{wf.task}] {wf.strategy}")
        if subtasks:
            lines.append("\nBehavioural subtask examples:")
            for st in subtasks:
                lines.append(f"- [{st.task}] {st.example}")
        return "\n".join(lines)

    def _match_function_memories(self, tool_name: str) -> List[DistilledFunctionMemory]:
        needle = tool_name.lower()
        exact = [fm for fm in self._config.function_memories if fm.tool_name.lower() == needle]
        if exact:
            return exact
        return [fm for fm in self._config.function_memories if needle and needle in fm.tool_name.lower()]

    def _format_function(self, tool_name: str) -> str | None:
        matches = self._match_function_memories(tool_name)
        if not matches:
            return None
        lines = [f"Distilled function memory for tool '{tool_name}' (retrieved on tool-calling error):"]
        for fm in matches:
            lines.append(f"- Calling convention: {fm.conventions}")
            lines.append(f"- Common pitfalls: {fm.pitfalls}")
        return "\n".join(lines)

    @staticmethod
    def _errored_tool_names(messages: List[Any]) -> Set[str]:
        errored: Set[str] = set()
        for msg in messages:
            if isinstance(msg, FunctionExecutionResultMessage):
                for result in msg.content:
                    if result.is_error and result.name:
                        errored.add(result.name)
        return errored

    # ---- Memory ABC --------------------------------------------------------

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject workflow/subtask memory proactively and function memory reactively.

        Proactive tiers (Workflow + Subtask) are injected once at task start.
        Function memory is injected reactively when a preceding tool call returned
        an error (``FunctionExecutionResult`` with ``is_error=True``); each tool's
        guidance is injected at most once to avoid prompt bloat.
        """
        injected: List[MemoryContent] = []

        # Proactive: inject workflow + subtask memory once, at task start.
        if not self._proactively_injected:
            proactive = self._format_proactive()
            if proactive is not None:
                await model_context.add_message(SystemMessage(content=proactive))
                injected.append(
                    MemoryContent(
                        content=proactive,
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"tier": "workflow+subtask"},
                    )
                )
            self._proactively_injected = True

        # Reactive: retrieve function memory on tool-calling errors.
        messages = await model_context.get_messages()
        errored_tools = self._errored_tool_names(messages)
        for tool_name in sorted(errored_tools - self._reactive_tools_injected):
            guidance = self._format_function(tool_name)
            if guidance is not None:
                await model_context.add_message(SystemMessage(content=guidance))
                injected.append(
                    MemoryContent(
                        content=guidance,
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"tier": "function", "tool_name": tool_name},
                    )
                )
            self._reactive_tools_injected.add(tool_name)

        return UpdateContextResult(memories=MemoryQueryResult(results=injected))

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve function-tier memory by tool name.

        With no query, the full function-memory catalogue is returned. This is the
        reactive retrieval surface: callers (or an agent's error-handling path) can
        look up calling conventions / pitfalls for a tool that just failed.
        """
        _ = cancellation_token, kwargs
        needle = query if isinstance(query, str) else str(query.content)
        needle = needle.strip().lower()
        if not needle:
            return MemoryQueryResult(
                results=[
                    MemoryContent(
                        content=f"{fm.tool_name}: {fm.conventions} (pitfalls: {fm.pitfalls})",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"tier": "function", "tool_name": fm.tool_name},
                    )
                    for fm in self._config.function_memories
                ]
            )
        matches = self._match_function_memories(needle)
        return MemoryQueryResult(
            results=[
                MemoryContent(
                    content=f"Calling convention: {fm.conventions}\nCommon pitfalls: {fm.pitfalls}",
                    mime_type=MemoryMimeType.TEXT,
                    metadata={"tier": "function", "tool_name": fm.tool_name},
                )
                for fm in matches
            ]
        )

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a distilled entry, routed by ``metadata['tier']``.

        Supports incremental population of the tiers produced by an offline
        teacher-distillation run. ``tier`` is one of ``"workflow"``, ``"subtask"``
        (default), or ``"function"``; the remaining fields are read from
        ``metadata`` (``task`` for workflow/subtask, ``tool_name`` / ``pitfalls``
        for function).
        """
        _ = cancellation_token
        metadata = content.metadata or {}
        tier = str(metadata.get("tier", "subtask")).lower()
        text = str(content.content)
        if tier == "workflow":
            self._config.workflow_memories.append(
                DistilledWorkflowMemory(task=str(metadata.get("task", "")), strategy=text)
            )
        elif tier == "function":
            self._config.function_memories.append(
                DistilledFunctionMemory(
                    tool_name=str(metadata.get("tool_name", "")),
                    conventions=text,
                    pitfalls=str(metadata.get("pitfalls", "")),
                )
            )
        else:  # default subtask
            self._config.subtask_memories.append(
                DistilledSubtaskMemory(task=str(metadata.get("task", "")), example=text)
            )
        # Newly added proactive tiers have not been injected into this task yet.
        if tier in ("workflow", "subtask"):
            self._proactively_injected = False

    async def clear(self) -> None:
        """Clear all distilled tiers and reset injection state."""
        self._config = DistilledMemoryConfig(name=self._name)
        self._proactively_injected = False
        self._reactive_tools_injected.clear()

    async def close(self) -> None:
        """No external resources to release."""
        pass

    # ---- Component ---------------------------------------------------------

    def _to_config(self) -> DistilledMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: DistilledMemoryConfig) -> Self:
        return cls(config=config)

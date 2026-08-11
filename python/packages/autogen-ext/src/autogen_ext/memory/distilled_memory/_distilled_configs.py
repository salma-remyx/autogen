from typing import List

from pydantic import BaseModel, Field


class DistilledWorkflowMemory(BaseModel):
    """Workflow memory: a task-level strategy distilled from a teacher trajectory.

    The highest-granularity tier in Agent Memory Distillation (AMD) — encodes the
    overall strategy for a task and is injected *proactively* at task start.
    """

    task: str
    """Task or intent this strategy applies to (used as a matching key)."""

    strategy: str
    """High-level workflow / strategy text distilled from a successful teacher run."""


class DistilledSubtaskMemory(BaseModel):
    """Subtask memory: a concrete behavioural example at intermediate granularity.

    The tier AMD found contributes the largest accuracy gains; injected *proactively*
    at task start alongside workflow memory.
    """

    task: str
    """Task or intent this example relates to."""

    example: str
    """A concrete behavioural example (e.g. a sub-trajectory the teacher executed)."""


class DistilledFunctionMemory(BaseModel):
    """Function memory: per-tool calling conventions and common pitfalls.

    The finest-granularity tier — retrieved *reactively* upon a tool-calling error
    rather than injected up front.
    """

    tool_name: str
    """Name of the tool / function this guidance applies to."""

    conventions: str
    """How to call the function correctly (signature, required args, types)."""

    pitfalls: str
    """Common mistakes and how to avoid them."""


class DistilledMemoryConfig(BaseModel):
    """Configuration for :class:`DistilledMemory`.

    The three memory tiers are populated from the *output* of an offline
    teacher-distillation run (in AMD a large teacher generates successful
    trajectories, from which Workflow / Subtask / Function entries are extracted).
    This runtime class consumes those pre-distilled entries and implements the
    proactive-injection + reactive-retrieval mechanism; it does not run the
    teacher itself.
    """

    name: str | None = None
    """Optional identifier for this memory instance."""

    workflow_memories: List[DistilledWorkflowMemory] = Field(default_factory=list)
    """Workflow-tier entries (task-level strategy), injected proactively."""

    subtask_memories: List[DistilledSubtaskMemory] = Field(default_factory=list)
    """Subtask-tier entries (behavioural examples), injected proactively."""

    function_memories: List[DistilledFunctionMemory] = Field(default_factory=list)
    """Function-tier entries (per-tool conventions + pitfalls), retrieved reactively."""

    max_subtasks: int = 3
    """Maximum number of subtask examples to inject proactively (caps prompt growth)."""

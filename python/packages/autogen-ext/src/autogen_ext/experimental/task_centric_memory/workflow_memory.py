"""Agent Workflow Memory (AWM) for task-centric memory.

Adapted from "Agent Workflow Memory" (Wang et al., arXiv:2409.07429). AWM learns
by inducing reusable routines (workflows) from solved tasks, storing them, and
retrieving the relevant ones to inject as guidance the next time a similar task is
attempted -- so the agent reuses hard-won procedure instead of re-deriving it.

This module implements the paper's induce -> store -> retrieve -> inject loop and
wires it into the existing :class:`MemoryController` contract as an opt-in
:class:`WorkflowMemoryController` subclass (a drop-in replacement that *adds*
workflow memory on top of the controller's usual insight/exemplar storage).

Adapted port (Mode 2) -- what was kept at fidelity and what was substituted:

  * **Kept:** the core induce -> store -> retrieve-by-task-similarity -> inject
    loop, and the notion of a *reusable routine* (an ordered list of steps)
    distinct from a raw task-solution exemplar.
  * **Substituted:** the paper induces routines from web-navigation action
    trajectories with a dedicated LLM pipeline. Here solutions are free-form
    text, so workflow induction defaults to a parameter-free step extractor and
    optionally accepts an LLM-backed ``extractor`` callable.
  * **Substituted:** the paper's learned/embedding-based retrieval is replaced by
    a token-overlap (Jaccard) similarity store -- the same parameter-free-proxy
    pattern used elsewhere in this package. The retrieve-by-task-similarity
    signal is preserved.
  * **Out of scope:** the paper's WebArena/Mind2Web benchmark harness; evaluation
    belongs in a downstream PR.
"""

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Tuple, TypedDict

from autogen_core.models import ChatCompletionClient

from .memory_controller import MemoryController, MemoryControllerConfig
from .utils.page_logger import PageLogger

__all__ = ["Workflow", "WorkflowMemory", "WorkflowMemoryController", "WorkflowMemoryControllerConfig"]


@dataclass
class Workflow:
    """
    A reusable routine induced from a solved task.

    Attributes:
        task: The task description this workflow was induced from. Used as the
            retrieval key when later searching for a relevant workflow.
        steps: The ordered, reusable action steps that compose the routine.
    """

    task: str
    steps: List[str]


def _normalize_step(step: str) -> str:
    """Strips a leading bullet/enumerator (e.g. ``"1."``, ``"- "``) and trailing sentence punctuation."""
    step = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", step)
    return step.strip().rstrip(".;,").strip()


def extract_workflow(task: str, solution: str) -> Workflow:
    """
    Parameter-free workflow induction: distills a solution into an ordered list of
    reusable routine steps.

    Splits on line boundaries first (the common shape of a step-by-step solution),
    falling back to sentence / clause boundaries for single-block solutions.
    Bullets, enumerators, and trailing punctuation are stripped so the resulting
    routine is reusable.
    """
    task = task.strip()
    solution = solution.strip()

    # Prefer line-oriented steps, since worked solutions are usually listed line by line.
    steps = [_normalize_step(line) for line in re.split(r"\r?\n+", solution)]
    steps = [step for step in steps if step]

    if len(steps) <= 1:
        # Single block: split on sentence/clause boundaries so we still expose the routine.
        pieces = re.split(r"(?<=[.;])\s+|;\s*|\bthen\b", solution)
        steps = [_normalize_step(piece) for piece in pieces]
        steps = [step for step in steps if step]

    if not steps:
        # Nothing to induce from; record the raw solution as a single step so it is still reusable.
        steps = [solution] if solution else []

    return Workflow(task=task, steps=steps)


class WorkflowMemory:
    """
    Stores induced workflows and retrieves them by task similarity.

    Implements AWM's store + retrieve-by-similarity using a parameter-free
    token-overlap (Jaccard) similarity proxy in place of the paper's learned
    embeddings.

    Args:
        similarity_threshold: The minimum Jaccard overlap (in [0, 1]) between a
            query task and a stored workflow's task for the workflow to qualify
            as relevant.
        max_workflows: The default maximum number of workflows to retrieve.
    """

    def __init__(self, similarity_threshold: float = 0.05, max_workflows: int = 5) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_workflows = max_workflows
        self._workflows: List[Workflow] = []

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Lowercases and tokenizes text into significant tokens (length > 2)."""
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        """Returns the Jaccard overlap between two token sets, or 0.0 if either is empty."""
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        union = len(left | right)
        return intersection / union if union else 0.0

    def add_workflow(self, workflow: Workflow) -> None:
        """Adds one induced workflow to the store."""
        self._workflows.append(workflow)

    async def add_task_solution_pair(
        self,
        task: str,
        solution: str,
        extractor: Callable[[str, str], Awaitable[Workflow]] | None = None,
    ) -> Workflow:
        """
        Induces a reusable workflow from a task-solution pair, stores it, and returns it.

        Args:
            task: The task the solution solves.
            solution: The solution to distill into a reusable routine.
            extractor: An optional LLM-backed induction callable. When ``None``,
                the parameter-free :func:`extract_workflow` is used.
        """
        if extractor is not None:
            workflow = await extractor(task, solution)
        else:
            workflow = extract_workflow(task, solution)
        self.add_workflow(workflow)
        return workflow

    def retrieve_relevant_workflows(self, task: str, max_results: int | None = None) -> List[Workflow]:
        """
        Returns the stored workflows most similar to ``task``, most similar first.

        Workflows whose task overlaps ``task`` below ``similarity_threshold`` are filtered out.
        """
        if not self._workflows:
            return []
        limit = max_results if max_results is not None else self.max_workflows
        query_tokens = self._tokens(task)
        scored: List[Tuple[float, Workflow]] = [
            (self._jaccard(query_tokens, self._tokens(workflow.task)), workflow) for workflow in self._workflows
        ]
        scored = [(score, workflow) for score, workflow in scored if score >= self.similarity_threshold]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [workflow for _, workflow in scored[:limit]]

    @staticmethod
    def format_workflow_section(workflows: List[Workflow]) -> str:
        """
        Formats retrieved workflows as a guidance section suitable for appending to a task
        description, mirroring :meth:`MemoryController._format_memory_section`.
        """
        if not workflows:
            return ""
        section = "## Suggested reusable workflow that may help solve tasks like this\n"
        for workflow in workflows:
            for index, step in enumerate(workflow.steps, start=1):
                section += "{}. {}\n".format(index, step)
        return section


# Following the nested-config pattern used by MemoryControllerConfig.
class WorkflowMemoryControllerConfig(TypedDict, total=False):
    similarity_threshold: float
    max_workflows: int


class WorkflowMemoryController(MemoryController):
    """
    (EXPERIMENTAL, RESEARCH IN PROGRESS)

    A :class:`MemoryController` that *also* induces, stores, and injects reusable
    workflows -- Agent Workflow Memory (arXiv:2409.07429).

    Use this as a drop-in replacement for :class:`MemoryController`. In addition to
    the controller's usual insight/exemplar storage and retrieval, every
    task-solution pair is distilled into a reusable routine, and the most relevant
    routine is injected as guidance when a similar task is later assigned.

    Args:
        reset: True to empty the memory bank before starting.
        client: The model client to use internally.
        task_assignment_callback: An optional callback used to assign a task to any agent managed by the caller.
        config: An optional dict forwarded to :class:`MemoryController`.
        logger: An optional logger. If None, a default logger will be created.
        workflow_config: An optional dict to override the workflow-memory settings:

            - similarity_threshold: Minimum Jaccard overlap for a workflow to qualify as relevant.
            - max_workflows: The maximum number of workflows to inject per task.

    Example:

        .. code-block:: python

            from autogen_ext.experimental.task_centric_memory.workflow_memory import WorkflowMemoryController

            controller = WorkflowMemoryController(reset=True, client=client, logger=logger)

            # Learn a reusable routine from a solved task.
            await controller.add_task_solution_pair_to_memory(task, solution)

            # A later, similar task is automatically guided by the induced workflow.
            response = await controller.assign_task(similar_task, use_memory=True)
    """

    def __init__(
        self,
        reset: bool,
        client: ChatCompletionClient,
        task_assignment_callback: Callable[[str], Awaitable[Tuple[str, str]]] | None = None,
        config: MemoryControllerConfig | None = None,
        logger: PageLogger | None = None,
        workflow_config: WorkflowMemoryControllerConfig | None = None,
    ) -> None:
        super().__init__(reset, client, task_assignment_callback, config, logger)
        similarity_threshold = 0.05
        max_workflows = 5
        if workflow_config is not None:
            similarity_threshold = workflow_config.get("similarity_threshold", similarity_threshold)
            max_workflows = workflow_config.get("max_workflows", max_workflows)
        self.workflow_memory = WorkflowMemory(
            similarity_threshold=similarity_threshold,
            max_workflows=max_workflows,
        )

    async def add_task_solution_pair_to_memory(self, task: str, solution: str) -> None:
        """
        Induces a reusable workflow from the task-solution pair, stores it, then stores the
        raw exemplar via the parent controller (preserving existing behavior).
        """
        self.logger.enter_function()
        workflow = await self.workflow_memory.add_task_solution_pair(task, solution)
        self.logger.info("\nINDUCED WORKFLOW ({} steps):\n{}".format(len(workflow.steps), "\n".join(workflow.steps)))
        self.logger.leave_function()
        await super().add_task_solution_pair_to_memory(task, solution)

    def retrieve_relevant_workflows(self, task: str, max_results: int | None = None) -> List[Workflow]:
        """
        Returns any induced workflows that seem relevant to the task.
        """
        return self.workflow_memory.retrieve_relevant_workflows(task, max_results)

    async def _append_any_relevant_memories(self, task: str) -> str:
        """
        Appends any relevant memories to the task description, including any relevant
        reusable workflow induced from past task-solution pairs.
        """
        task = await super()._append_any_relevant_memories(task)
        workflows = self.workflow_memory.retrieve_relevant_workflows(task)
        workflow_section = WorkflowMemory.format_workflow_section(workflows)
        if len(workflow_section) > 0:
            self.logger.info("Relevant reusable workflow(s) were retrieved from workflow memory.\n")
            task = task + "\n\n" + workflow_section
        return task

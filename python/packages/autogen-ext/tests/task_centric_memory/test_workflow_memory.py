"""Tests for Agent Workflow Memory (AWM) wired into task-centric memory.

The dependency-free core (workflow induction, store, retrieve, inject) is
exercised directly, and the :class:`WorkflowMemoryController` integration is
exercised through the existing :class:`MemoryController` contract it subclasses.
"""

import pytest

# Optional extra: the task-centric-memory package backs its MemoryBank with chromadb.
pytest.importorskip("chromadb")

from autogen_ext.experimental.task_centric_memory.memory_controller import MemoryController  # noqa: E402
from autogen_ext.experimental.task_centric_memory.workflow_memory import (  # noqa: E402
    Workflow,
    WorkflowMemory,
    WorkflowMemoryController,
    extract_workflow,
)
from autogen_ext.models.replay import ReplayChatCompletionClient  # noqa: E402

# Disable the controller's LLM-driven stages so the workflow path is exercised deterministically.
_NO_LLM_CONFIG = {
    "generalize_task": False,
    "revise_generalized_task": False,
    "generate_topics": False,
    "validate_memos": False,
}


def test_extract_workflow_induces_ordered_steps_from_lines() -> None:
    workflow = extract_workflow(
        task="Book a one-way flight from Seattle to Boston.",
        solution="1. Open the airline website.\n2. Search for the Seattle to Boston route.\n3. Pick a flight and pay.",
    )
    assert workflow.task == "Book a one-way flight from Seattle to Boston."
    # Bullets/enumerators are stripped, and the routine is preserved as ordered, reusable steps.
    assert workflow.steps == [
        "Open the airline website",
        "Search for the Seattle to Boston route",
        "Pick a flight and pay",
    ]


def test_extract_workflow_splits_single_block_on_sentence_boundaries() -> None:
    workflow = extract_workflow(
        task="Cancel a subscription.",
        solution="Open settings; then click billing then choose cancel.",
    )
    assert workflow.steps == ["Open settings", "click billing", "choose cancel"]


def test_workflow_memory_retrieves_by_task_similarity() -> None:
    memory = WorkflowMemory()
    memory.add_workflow(extract_workflow("Book a flight to Boston", "Search flights; pick one; pay."))
    memory.add_workflow(extract_workflow("Reset the account password", "Open settings; click security; reset."))

    # A similar task retrieves the flight workflow, not the password workflow.
    retrieved = memory.retrieve_relevant_workflows("I need to book a flight to Boston for tomorrow")
    assert len(retrieved) == 1
    assert retrieved[0].task == "Book a flight to Boston"

    # An unrelated task retrieves nothing.
    assert memory.retrieve_relevant_workflows("Translate this paragraph into French") == []


def test_format_workflow_section_is_empty_without_workflows() -> None:
    assert WorkflowMemory.format_workflow_section([]) == ""
    section = WorkflowMemory.format_workflow_section([Workflow(task="t", steps=["do x", "do y"])])
    assert "Suggested reusable workflow" in section
    assert "1. do x" in section
    assert "2. do y" in section


@pytest.mark.asyncio
async def test_controller_induces_and_injects_workflow(tmp_path) -> None:
    # The integration: WorkflowMemoryController subclasses the existing MemoryController and
    # adds induce -> store -> retrieve -> inject of reusable workflows.
    controller = WorkflowMemoryController(
        reset=True,
        client=ReplayChatCompletionClient(["not used"]),
        config={**_NO_LLM_CONFIG, "MemoryBank": {"path": str(tmp_path / "bank")}},
    )

    # Sanity: the workflow controller IS-A memory controller (the existing call-site class).
    assert isinstance(controller, MemoryController)

    # Learn a reusable routine from a solved task.
    await controller.add_task_solution_pair_to_memory(
        task="Book a one-way flight from Seattle to Boston.",
        solution="1. Open the airline website.\n2. Search for the Seattle to Boston route.\n3. Pick a flight and pay.",
    )

    # The induced workflow is retrievable for a similar task.
    workflows = controller.retrieve_relevant_workflows("Help me book a flight from Seattle to Boston.")
    assert len(workflows) == 1
    assert workflows[0].steps[0] == "Open the airline website"

    # AWM injection: the relevant workflow is appended as guidance to a similar task description.
    augmented = await controller._append_any_relevant_memories("Help me book a flight from Seattle to Boston.")
    assert "Suggested reusable workflow" in augmented
    assert "Pick a flight and pay" in augmented


@pytest.mark.asyncio
async def test_controller_workflow_memory_is_independent_of_exemplar_store(tmp_path) -> None:
    # Workflow memory is populated even when the raw exemplar store is empty (no prior memos),
    # so the workflow injection path does not depend on the ChromaDB-backed memo retrieval.
    controller = WorkflowMemoryController(
        reset=True,
        client=ReplayChatCompletionClient(["not used"]),
        config={**_NO_LLM_CONFIG, "MemoryBank": {"path": str(tmp_path / "bank")}},
    )
    controller.workflow_memory.add_workflow(Workflow(task="book a flight to boston", steps=["search", "pay"]))

    augmented = await controller._append_any_relevant_memories("book a flight to boston today")
    assert "Suggested reusable workflow" in augmented
    assert "search" in augmented

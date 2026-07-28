"""Experience Memory Graph — one-shot error correction for agents.

Adapted (Mode 2) port of *Experience Memory Graph: One-Shot Error Correction
for Agents* (arXiv:2607.13884). The paper reformulates agent failure recovery as
a graph-matching problem: trajectories are turned into directed *action-decision
graphs*, a failed run is matched against stored *expert* graphs, and the
divergence is returned as an explicit *graph edit path* (which actions to add,
delete, or relabel under a given observation). At inference time the agent
receives the correction once and executes loop-free — no trial-and-error.

What is kept at fidelity (the paper's core mechanism):

* trajectories become action-decision graphs with branching and shared prefixes
  (common successful subgraphs);
* corrections are derived from a deterministic edit path between a failed run and
  an expert path (add / delete / relabel under an observation);
* experiences live in a memory with intra-task nodes and cross-task edges linking
  tasks that share a decision node.

What is intentionally substituted for a target-native fit (Mode 2):

* the paper's *learned* trajectory matcher is replaced by a parameter-free
  ``difflib`` aligner over action labels plus lexical overlap — deterministic,
  zero ML dependencies, matching this fork's stdlib-only convention;
* the ALFWorld / ScienceWorld benchmark and offline graph-mining pipeline are cut
  — the module plugs into AutoGen's :class:`~autogen_core.memory.Memory` ABC, so
  an :class:`~autogen_agentchat.agents.AssistantAgent` consumes corrections
  through ``update_context`` during inference. Evaluation belongs downstream.
"""

import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, cast

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel
from typing_extensions import Self

logger = logging.getLogger(__name__)

# A trajectory step is an (observation, action) pair. Either may be empty.
Step = Tuple[str, str]


def _normalize(text: str) -> str:
    """Normalize text for stable, case-insensitive matching."""
    return " ".join(text.lower().split())


@dataclass
class ActionNode:
    """One decision point in an action-decision graph: an observation and the action taken."""

    action: str
    observation: str = ""

    @property
    def action_norm(self) -> str:
        return _normalize(self.action)

    @property
    def observation_norm(self) -> str:
        return _normalize(self.observation)

    def matches(self, step: Step) -> bool:
        """Two steps are the same decision when their normalized actions agree.

        Observations are secondary metadata (the first non-empty one wins), so a
        shared *action* prefix across runs is recognized as a common subgraph.
        """
        return self.action_norm == _normalize(step[1])


class ActionDecisionGraph:
    """A directed action-decision graph built from one or more trajectories.

    Inserting several trajectories for the same task merges their shared action
    prefixes into shared nodes and creates branches where they diverge — the
    paper's "common subgraphs (successful workflows)" realized structurally.
    """

    def __init__(self, task: str, kind: str) -> None:
        self.task = task
        self.kind = kind
        self.nodes: List[ActionNode] = []
        self.outgoing: Dict[int, List[int]] = {}
        self.roots: List[int] = []

    def _new_node(self, step: Step) -> int:
        action = step[1].strip()
        observation = step[0].strip()
        index = len(self.nodes)
        self.nodes.append(ActionNode(action=action, observation=observation))
        return index

    def insert_path(self, steps: Sequence[Step]) -> None:
        """Merge a trajectory into the graph, branching where it diverges."""
        if not steps:
            return
        current = self._find_child(self.roots, steps[0])
        if current is None:
            current = self._new_node(steps[0])
            self.roots.append(current)
        for step in steps[1:]:
            child = self._find_child(self.outgoing.get(current, []), step)
            if child is None:
                child = self._new_node(step)
                self.outgoing.setdefault(current, []).append(child)
            current = child

    def _find_child(self, candidates: Sequence[int], step: Step) -> int | None:
        """Return the index of a candidate node matching ``step``, if any."""
        for index in candidates:
            if self.nodes[index].matches(step):
                return index
        return None

    def paths(self, max_paths: int = 16) -> List[List[int]]:
        """Enumerate root-to-leaf node paths (capped) through the decision graph."""
        results: List[List[int]] = []

        def dfs(index: int, prefix: List[int]) -> None:
            if len(results) >= max_paths:
                return
            path = prefix + [index]
            children = self.outgoing.get(index, [])
            if not children:
                results.append(path)
                return
            for child in children:
                dfs(child, path)

        for root in self.roots:
            if len(results) >= max_paths:
                break
            dfs(root, [])
        return results

    def node_signatures(self) -> set[Tuple[str, str]]:
        """Set of (observation_norm, action_norm) decision nodes — used for cross-task edges."""
        return {(node.observation_norm, node.action_norm) for node in self.nodes}


class _ExperienceMemoryGraphConfig(BaseModel):
    """Declarative configuration for :class:`ExperienceMemoryGraph`."""

    name: str | None = None
    """Optional name for this memory store."""

    match_threshold: float = 0.2
    """Minimum action-overlap ratio (0-1) for an expert path to be considered relevant."""

    max_paths: int = 16
    """Cap on the number of expert graph paths explored per query (bounds branching graphs)."""


@dataclass
class _StoredGraph:
    graph: ActionDecisionGraph
    related_tasks: set[str] = field(default_factory=set[str])


class ExperienceMemoryGraph(Memory, Component[_ExperienceMemoryGraphConfig]):
    """A memory that recovers from agent failures with one-shot graph-edit corrections.

    Store *expert* (successful) and *failure* trajectories as action-decision
    graphs. On query, the (failed) run is aligned against the stored expert
    graphs and the divergence is returned as an explicit correction — which
    actions to add, delete, or relabel — so the agent can recover in a single,
    loop-free pass instead of iterating by trial-and-error.

    Plugs into any :class:`~autogen_agentchat.agents.AssistantAgent` via the
    standard ``memory=[...]`` argument; the agent calls
    :meth:`update_context` before inference and receives the correction as a
    system message.

    Trajectories can be added as JSON content:

    .. code-block:: python

        await memory.add(
            MemoryContent(
                content={
                    "task": "unlock_door",
                    "kind": "expert",
                    "steps": [
                        {"observation": "locked door", "action": "find key"},
                        {"observation": "brass key", "action": "pick up key"},
                        {"observation": "locked door", "action": "unlock door"},
                    ],
                },
                mime_type=MemoryMimeType.JSON,
            )
        )

    or as a compact text path (``->``-separated actions, optional ``[kind]`` and
    ``task:`` headers):

    .. code-block:: python

        await memory.add(
            MemoryContent(
                content="[expert] task: unlock_door\\nfind key -> pick up key -> unlock door",
                mime_type=MemoryMimeType.TEXT,
            )
        )

    Querying with a failed run returns the correction:

    .. code-block:: python

        result = await memory.query("push door -> push door")
        print(result.results[0].content)  # the one-shot edit path
    """

    component_type = "memory"
    component_config_schema = _ExperienceMemoryGraphConfig
    component_provider_override = "autogen_ext.memory.experience_memory_graph.ExperienceMemoryGraph"

    def __init__(self, config: _ExperienceMemoryGraphConfig | None = None) -> None:
        self._config = config or _ExperienceMemoryGraphConfig()
        self._stored: List[_StoredGraph] = []

    @property
    def name(self) -> str | None:
        return self._config.name

    # ------------------------------------------------------------------ Memory

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Store a trajectory (expert or failure) as an action-decision graph."""
        task, kind, steps = self._parse_trajectory(content)
        if not steps:
            logger.warning("ExperienceMemoryGraph: skipping empty trajectory for task '%s'", task)
            return
        stored = next((s for s in self._stored if s.graph.task == task and s.graph.kind == kind), None)
        if stored is None:
            graph = ActionDecisionGraph(task=task, kind=kind)
            graph.insert_path(steps)
            stored = _StoredGraph(graph=graph)
            self._stored.append(stored)
        else:
            stored.graph.insert_path(steps)
        self._refresh_cross_task_edges()

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Match a (failed) run against expert graphs and return a one-shot correction."""
        query_task, query_steps = self._parse_query(query)
        if not query_steps:
            return MemoryQueryResult(results=[])
        query_actions = [_normalize(step[1]) for step in query_steps]

        best = self._best_expert_path(query_actions, query_task)
        if best is None:
            return MemoryQueryResult(results=[])

        ratio, expert_graph, path_indices, _ = best
        query_task_known = bool(query_task) and query_task != "default"
        matched_same_task = query_task_known and expert_graph.task == query_task
        # A known task with its own expert is always relevant (the point is to correct
        # failures, including fully divergent ones). Anonymous or cross-task matches
        # must clear the overlap threshold to count as relevant.
        if not matched_same_task and ratio < self._config.match_threshold:
            return MemoryQueryResult(results=[])

        expert_actions = [expert_graph.nodes[i].action for i in path_indices]
        expert_observations = [expert_graph.nodes[i].observation for i in path_indices]
        edits = self._edit_path(query_actions, [step[0] for step in query_steps], expert_actions, expert_observations)
        if not edits:
            # The run already matches a successful workflow — no correction needed.
            return MemoryQueryResult(results=[])

        cross_task = query_task_known and not matched_same_task
        insight = self._render_correction(expert_graph.task, edits, cross_task=cross_task)
        metadata: Dict[str, Any] = {
            "task": expert_graph.task,
            "overlap": round(ratio, 4),
            "edits": edits,
            "cross_task": cross_task,
        }
        return MemoryQueryResult(
            results=[MemoryContent(content=insight, mime_type=MemoryMimeType.TEXT, metadata=metadata)]
        )

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject a one-shot correction into the context before inference (loop-free guidance)."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)
        results = await self.query(query_text)
        if results.results:
            await model_context.add_message(SystemMessage(content=str(results.results[0].content)))
        return UpdateContextResult(memories=results)

    async def clear(self) -> None:
        self._stored.clear()

    async def close(self) -> None:
        # Everything is in-memory; nothing to release.
        return None

    # ------------------------------------------------------------- Matching

    def _candidate_experts(self, query_task: str) -> List[_StoredGraph]:
        """Experts to search: same task, then cross-task-linked tasks, then all (fallback)."""
        experts = [s for s in self._stored if s.graph.kind == "expert"]
        if not query_task or query_task == "default":
            return experts
        same = [s for s in experts if s.graph.task == query_task]
        if not same:
            return experts
        related_ids: set[str] = set()
        for stored in same:
            related_ids |= stored.related_tasks
        related = [s for s in experts if s.graph.task in related_ids and s.graph.task != query_task]
        return same + related if related else experts

    def _best_expert_path(
        self, query_actions: Sequence[str], query_task: str
    ) -> Tuple[float, ActionDecisionGraph, List[int], bool] | None:
        """Find the expert path best aligned to the query by action overlap.

        Ranking prefers higher action overlap, then same-task experts, then longer
        expert paths.
        """
        best_ratio = -1.0
        best: Tuple[ActionDecisionGraph, List[int], bool] | None = None
        for stored in self._candidate_experts(query_task):
            same_task = (not query_task or query_task == "default") or stored.graph.task == query_task
            for path_indices in stored.graph.paths(max_paths=self._config.max_paths):
                path_actions = [stored.graph.nodes[i].action_norm for i in path_indices]
                ratio = difflib.SequenceMatcher(a=query_actions, b=path_actions, autojunk=False).ratio()
                current = (ratio, 1 if same_task else 0, len(path_indices))
                previous = (
                    best_ratio,
                    1 if best is not None and best[2] else 0,
                    len(best[1]) if best is not None else 0,
                )
                if best is None or current > previous:
                    best_ratio = ratio
                    best = (stored.graph, path_indices, same_task)
        if best is None:
            return None
        expert_graph, path_indices, same_task = best
        return best_ratio, expert_graph, path_indices, same_task

    @staticmethod
    def _edit_path(
        query_actions: Sequence[str],
        query_observations: Sequence[str],
        expert_actions: Sequence[str],
        expert_observations: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Compute add/delete/relabel edits between the failed run and the expert path."""
        matcher = difflib.SequenceMatcher(a=list(query_actions), b=list(expert_actions), autojunk=False)
        edits: List[Dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                edits.append(
                    {
                        "op": "relabel",
                        "at": i1 + 1,
                        "observation": _first_observation(query_observations, i1, expert_observations, j1),
                        "from": list(query_actions[i1:i2]),
                        "to": list(expert_actions[j1:j2]),
                    }
                )
            elif tag == "delete":
                edits.append(
                    {
                        "op": "remove",
                        "at": i1 + 1,
                        "observation": _first_observation(query_observations, i1, expert_observations, j1),
                        "actions": list(query_actions[i1:i2]),
                    }
                )
            elif tag == "insert":
                edits.append(
                    {
                        "op": "add",
                        "at": i1 + 1,
                        "observation": _first_observation(expert_observations, j1, query_observations, i1),
                        "actions": list(expert_actions[j1:j2]),
                    }
                )
        return edits

    @staticmethod
    def _render_correction(task: str, edits: Sequence[Dict[str, Any]], cross_task: bool) -> str:
        """Render the graph edit path as a one-shot correction prompt."""
        lines: List[str] = []
        scope = " (recovered from a related task)" if cross_task else ""
        lines.append(f"One-shot correction for task '{task}'{scope}, recovered from an expert workflow:")
        for edit in edits:
            obs = f" (observation: {edit['observation']})" if edit.get("observation") else ""
            if edit["op"] == "relabel":
                lines.append(
                    f"  step {edit['at']}{obs}: relabel {' -> '.join(edit['from'])} as {' -> '.join(edit['to'])}"
                )
            elif edit["op"] == "add":
                lines.append(f"  before step {edit['at']}{obs}: add {' -> '.join(edit['actions'])}")
            else:  # remove
                lines.append(f"  step {edit['at']}{obs}: remove {' -> '.join(edit['actions'])}")
        lines.append("Apply these edits to recover in a single, loop-free pass — no trial-and-error.")
        return "\n".join(lines)

    # ------------------------------------------------------------- Parsing

    def _parse_trajectory(self, content: MemoryContent) -> Tuple[str, str, List[Step]]:
        metadata = content.metadata or {}
        raw = content.content
        if isinstance(raw, dict):
            task = str(raw.get("task") or metadata.get("task") or "default")
            kind = str(raw.get("kind") or metadata.get("kind") or "failure").lower()
            raw_steps: Any = raw.get("steps") or raw.get("trajectory") or []
            steps = [_step_from_item(item) for item in raw_steps]
            return task, kind, steps
        if isinstance(raw, str):
            return _parse_text_trajectory(raw, metadata)
        raise ValueError(f"Unsupported trajectory content type: {type(raw)}")

    @staticmethod
    def _parse_query(query: str | MemoryContent) -> Tuple[str, List[Step]]:
        if isinstance(query, str):
            task, _, steps = _parse_text_trajectory(query, {})
            return task, steps
        if isinstance(query, MemoryContent):
            if isinstance(query.content, str):
                task, _, steps = _parse_text_trajectory(query.content, query.metadata or {})
                return task, steps
            if isinstance(query.content, dict):
                task = str(query.content.get("task") or (query.metadata or {}).get("task") or "default")
                raw_steps: Any = query.content.get("steps") or query.content.get("trajectory") or []
                return task, [_step_from_item(item) for item in raw_steps]
        return "", []

    # ------------------------------------------------------------- Cross-task

    def _refresh_cross_task_edges(self) -> None:
        """Link tasks that share an action-decision node (the paper's cross-task edges)."""
        by_task: Dict[str, set[Tuple[str, str]]] = {}
        for stored in self._stored:
            by_task.setdefault(stored.graph.task, set()).update(stored.graph.node_signatures())
        tasks = list(by_task)
        for stored in self._stored:
            related: set[str] = set()
            for other_task in tasks:
                if other_task == stored.graph.task:
                    continue
                if by_task[stored.graph.task] & by_task[other_task]:
                    related.add(other_task)
            stored.related_tasks = related

    # ------------------------------------------------------------- Component

    def _to_config(self) -> _ExperienceMemoryGraphConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: _ExperienceMemoryGraphConfig) -> Self:
        return cls(config=config)


def _step_from_item(item: Any) -> Step:
    """Coerce a raw step (str or dict) into an (observation, action) pair."""
    if isinstance(item, str):
        return ("", item.strip())
    if isinstance(item, dict):
        data = cast(Dict[str, Any], item)
        action = str(data.get("action") or data.get("a") or "").strip()
        observation = str(data.get("observation") or data.get("obs") or "").strip()
        return (observation, action)
    raise ValueError(f"Unsupported step type: {type(item)}")


def _parse_text_trajectory(text: str, metadata: Dict[str, Any]) -> Tuple[str, str, List[Step]]:
    """Parse a compact text trajectory: optional ``[kind]``/``task:`` headers and ``->`` paths."""
    lowered = text.lower()
    task = str(metadata.get("task") or "default")
    kind = str(metadata.get("kind") or "").lower()
    steps: List[Step] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("task:"):
            task = line.split(":", 1)[1].strip()
        elif low.startswith("kind:"):
            kind = line.split(":", 1)[1].strip().lower()
        elif low.startswith("[") and "]" in line:
            kind = low[1 : low.index("]")].strip()
            remainder = line[line.index("]") + 1 :].strip()
            if remainder:
                steps.extend(_split_action_path(remainder))
        else:
            steps.extend(_split_action_path(line))
    if not kind:
        if "expert" in lowered:
            kind = "expert"
        elif "fail" in lowered:
            kind = "failure"
        else:
            kind = "failure"
    return task, kind, steps


def _split_action_path(segment: str) -> List[Step]:
    """Split a ``->`` / ``=>`` separated action path into steps (obs optional via ``|`` or ``:``)."""
    tokens = [tok.strip() for tok in segment.replace("=>", "->").split("->") if tok.strip()]
    steps: List[Step] = []
    for token in tokens:
        if "|" in token:
            observation, action = token.split("|", 1)
            steps.append((observation.strip(), action.strip()))
        elif ":" in token:
            observation, action = token.split(":", 1)
            steps.append((observation.strip(), action.strip()))
        else:
            steps.append(("", token))
    return steps


def _first_observation(primary: Sequence[str], primary_index: int, fallback: Sequence[str], fallback_index: int) -> str:
    """Return the most relevant observation for an edit, preferring the primary run."""
    if 0 <= primary_index < len(primary) and primary[primary_index]:
        return primary[primary_index]
    if 0 <= fallback_index < len(fallback) and fallback[fallback_index]:
        return fallback[fallback_index]
    return ""

"""Task Memory Engine — revision-aware spatial task memory (TMS-DAG).

Implements the core mechanism of "Task Memory Engine: Spatial Memory for
Robust Multi-Step LLM Agents" (arXiv:2505.19436): a Task-Memory-State DAG
(TMS-DAG) whose nodes track task *slots* and their *revision history*, plus a
TRIM (Task Representation and Intent Management) layer that classifies each
incoming memory as ``new`` / ``update`` / ``inactivate`` / ``roll_back`` and
propagates dependency-tracked revisions across the DAG.

Adapted port (Mode 2). The parts kept at full fidelity are the paper's core:
the DAG data structure (shared nodes with ``parent``/``dependencies``), the
``TaskNode`` revision ``history``, the TRIM intent set, and the
``PropagateDeps`` revision propagation (a revised node flags every downstream
dependent stale, and because shared DAG nodes have a single source of truth
the change is visible to all parents). The auxiliary component that is
*substituted* is the paper's learned intent classifier ``f_TRIM(s, G)`` (an
LLM few-shot classifier): it is replaced by a parameter-free
:class:`MetadataIntentResolver` that derives the intent from
``MemoryContent.metadata`` plus slot matching, behind the pluggable
:class:`IntentResolver` protocol so an LLM-backed resolver can be supplied at
construction time. The paper's separate hallucination-rate benchmark suite is
intentionally out of scope (evaluation belongs downstream).
"""

import json
import logging
from enum import Enum
from typing import Any, Protocol

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

logger = logging.getLogger(__name__)


class TaskIntent(str, Enum):
    """TRIM intent categories (see arXiv:2505.19436, TRIM component)."""

    NEW = "new"
    UPDATE = "update"
    INACTIVATE = "inactivate"
    ROLL_BACK = "roll_back"


class TaskNode(BaseModel):
    """A node in the TMS-DAG.

    A node is addressed by its semantic ``slot`` (e.g. ``ingredient:celery``)
    rather than insertion order. Several parent tasks may depend on the same
    node (a DAG, not a tree); updating that shared node therefore propagates
    globally to every parent.
    """

    node_id: str
    slot: str
    value: str
    parent: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentResolution(BaseModel):
    """A resolved TRIM intent for an incoming memory."""

    intent: TaskIntent
    slot: str
    value: str
    parent: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    replaces: str | None = None


def _coerce_value(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, sort_keys=True)
    return str(content)


class IntentResolver(Protocol):
    """Pluggable TRIM classifier ``z = f_TRIM(content, graph)``.

    The paper learns this with an LLM; the default
    :class:`MetadataIntentResolver` is parameter-free and derives the intent
    from ``MemoryContent.metadata`` plus slot matching against the current DAG.
    """

    def resolve(self, content: MemoryContent, nodes: dict[str, TaskNode]) -> IntentResolution: ...


class MetadataIntentResolver:
    """Parameter-free default TRIM intent resolver.

    The intent is taken from ``metadata["intent"]`` when present. Otherwise a
    memory whose ``slot`` (or, falling back, coerced content) already exists
    as an active node is classified ``UPDATE``; any other memory is ``NEW``.
    """

    def resolve(self, content: MemoryContent, nodes: dict[str, TaskNode]) -> IntentResolution:
        meta = content.metadata or {}
        slot = str(meta["slot"]) if meta.get("slot") is not None else _coerce_value(content.content)
        value = _coerce_value(content.content)

        raw_intent = meta.get("intent")
        existing = next((n for n in nodes.values() if n.slot == slot and n.active), None)
        if isinstance(raw_intent, str):
            intent = TaskIntent(raw_intent)
        elif existing is not None:
            intent = TaskIntent.UPDATE
        else:
            intent = TaskIntent.NEW

        raw_parent = meta.get("parent")
        raw_deps = meta.get("dependencies")
        raw_replaces = meta.get("replaces")
        return IntentResolution(
            intent=intent,
            slot=slot,
            value=value,
            parent=str(raw_parent) if isinstance(raw_parent, str) else None,
            dependencies=[str(d) for d in raw_deps] if isinstance(raw_deps, list) else [],
            replaces=str(raw_replaces) if isinstance(raw_replaces, str) else None,
        )


class TaskMemoryEngineConfig(BaseModel):
    """Configuration for the :class:`TaskMemoryEngine` declarative component."""

    name: str | None = None
    """Optional identifier for this memory instance."""


class TaskMemoryEngine(Memory, Component[TaskMemoryEngineConfig]):
    """Revision-aware task memory backed by a Task-Memory-State DAG.

    Unlike linear memory (e.g. :class:`~autogen_core.memory.ListMemory`), each
    entry is keyed by a semantic *slot* and tracked across revisions: a
    correction to a slot pushes the prior value onto the slot's history and
    propagates to every downstream dependent, so the context surfaced to the
    model always reflects the *current* task state rather than a stale,
    contradicted history.

    The TRIM intent of each added memory is resolved by an
    :class:`IntentResolver` (default: :class:`MetadataIntentResolver`, driven
    by ``MemoryContent.metadata``). Recognised metadata keys:

    - ``slot``: semantic key (defaults to the coerced content).
    - ``intent``: one of ``new`` / ``update`` / ``inactivate`` / ``roll_back``
      (inferred from slot existence when omitted).
    - ``parent`` / ``dependencies``: node ids defining DAG edges.
    - ``replaces``: value superseded by an ``update`` (recorded on the node).

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory import TaskMemoryEngine


            async def main() -> None:
                memory = TaskMemoryEngine()
                await memory.add(
                    MemoryContent(content="celery", mime_type=MemoryMimeType.TEXT, metadata={"slot": "ingredient:celery"})
                )
                # A correction revises the slot instead of appending a duplicate.
                await memory.add(
                    MemoryContent(
                        content="mushrooms",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"slot": "ingredient:celery", "intent": "update", "replaces": "celery"},
                    )
                )


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance.
        intent_resolver: Optional TRIM classifier. Defaults to
            :class:`MetadataIntentResolver`.

    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.task_memory_engine.TaskMemoryEngine"
    component_config_schema = TaskMemoryEngineConfig

    def __init__(self, name: str | None = None, intent_resolver: IntentResolver | None = None) -> None:
        self._name = name or "task_memory_engine"
        self._resolver: IntentResolver = intent_resolver or MetadataIntentResolver()
        self._nodes: dict[str, TaskNode] = {}
        self._counter = 0

    @property
    def name(self) -> str:
        """Get the memory instance identifier."""
        return self._name

    @property
    def nodes(self) -> dict[str, TaskNode]:
        """Get the current TMS-DAG nodes keyed by node id."""
        return self._nodes

    def _next_node_id(self) -> str:
        self._counter += 1
        return f"task_{self._counter}"

    def _find_node(self, slot: str) -> TaskNode | None:
        return next((n for n in self._nodes.values() if n.slot == slot), None)

    # --- TRIM operations ---

    def _apply_new(self, res: IntentResolution) -> TaskNode:
        node_id = self._next_node_id()
        node = TaskNode(
            node_id=node_id,
            slot=res.slot,
            value=res.value,
            parent=res.parent,
            dependencies=list(res.dependencies),
            metadata={"replaces": res.replaces} if res.replaces else {},
        )
        self._nodes[node_id] = node
        return node

    def _apply_update(self, res: IntentResolution) -> TaskNode:
        node = self._find_node(res.slot)
        if node is None:
            # No prior node for this slot — create one rather than dropping the revision.
            return self._apply_new(res)
        node.history.append(node.value)
        node.value = res.value
        if res.replaces:
            node.metadata["replaces"] = res.replaces
        self._propagate(node)
        return node

    def _apply_inactivate(self, res: IntentResolution) -> TaskNode | None:
        node = self._find_node(res.slot)
        if node is None:
            return None
        node.active = False
        self._propagate(node)
        return node

    def _apply_roll_back(self, res: IntentResolution) -> TaskNode | None:
        node = self._find_node(res.slot)
        if node is None or not node.history:
            return node
        superseded = node.value
        node.value = node.history.pop()
        node.history.append(superseded)
        self._propagate(node)
        return node

    def _propagate(self, node: TaskNode) -> None:
        """Propagate a revision to dependent downstream nodes.

        Mirrors ``PropagateDeps(n_k, G)`` from the paper: every node that
        depends on the revised node is flagged ``stale`` (annotated with the
        triggering slot) so context retrieval reflects the updated state.
        Because shared DAG nodes are stored once and referenced by id, the
        revision is visible to all parents without extra work.
        """
        for other in self._nodes.values():
            if node.node_id in other.dependencies:
                other.metadata["stale"] = True
                other.metadata["stale_after"] = node.slot

    # --- Memory ABC ---

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Surface the current (active, revised) task state into the model context.

        Adds a single :class:`~autogen_core.models.SystemMessage` summarising
        every active node's slot/value, omitting superseded values so the model
        is not presented with contradicted history.
        """
        active = [n for n in self._nodes.values() if n.active]
        if not active:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        lines = [f"- {n.slot}: {n.value}" + (f" (parent: {n.parent})" if n.parent else "") for n in active]
        summary = "Current task state (dependency-tracked; revisions propagated):\n" + "\n".join(lines)
        await model_context.add_message(SystemMessage(content=summary))

        memories = [
            MemoryContent(
                content=n.value, mime_type=MemoryMimeType.TEXT, metadata={"slot": n.slot, "node_id": n.node_id}
            )
            for n in active
        ]
        return UpdateContextResult(memories=MemoryQueryResult(results=memories))

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return active nodes matching the query (the TRIM ``check`` intent).

        Matching is by substring against the slot or value (case-insensitive).
        An empty query returns every active node. The graph is not modified.
        """
        _ = cancellation_token, kwargs
        needle = query if isinstance(query, str) else _coerce_value(getattr(query, "content", ""))
        needle_lc = needle.lower()
        matches = [
            MemoryContent(
                content=n.value, mime_type=MemoryMimeType.TEXT, metadata={"slot": n.slot, "node_id": n.node_id}
            )
            for n in self._nodes.values()
            if n.active and (not needle_lc or needle_lc in n.slot.lower() or needle_lc in n.value.lower())
        ]
        return MemoryQueryResult(results=matches)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Interpret ``content`` as a TRIM task operation and mutate the DAG."""
        _ = cancellation_token
        res = self._resolver.resolve(content, self._nodes)
        if res.intent is TaskIntent.NEW:
            self._apply_new(res)
        elif res.intent is TaskIntent.UPDATE:
            self._apply_update(res)
        elif res.intent is TaskIntent.INACTIVATE:
            self._apply_inactivate(res)
        elif res.intent is TaskIntent.ROLL_BACK:
            self._apply_roll_back(res)

    async def clear(self) -> None:
        """Clear every task node from the DAG."""
        self._nodes.clear()
        self._counter = 0

    async def close(self) -> None:
        """Release resources (none held)."""

    @classmethod
    def _from_config(cls, config: TaskMemoryEngineConfig) -> Self:
        return cls(name=config.name)

    def _to_config(self) -> TaskMemoryEngineConfig:
        return TaskMemoryEngineConfig(name=self._name)

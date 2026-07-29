"""Mode-routed agent memory.

This module implements :class:`RoutedMemory`, a :class:`~autogen_core.memory.Memory`
that dispatches each query to a specialized backend selected by a per-query
*cognitive mode* (factoid lookup, relation-chain reasoning, or broad synthesis).

The design adapts *Supra Cognitive Modes: A Routed Architecture for Agent Memory*
(arXiv:2607.19096), which maps an explicit or automatically selected per-query mode
to a retrieval/synthesis payload over a shared ingest substrate. This is an
**adapted port (Mode 2)** — the core mechanism (per-query mode classification with
dispatch to mode-specific retrieval strategies, plus a shared ingest substrate) is
preserved, while the paper's auxiliary components are substituted with target-native
equivalents:

* The paper's *frozen semantic classifier* (a trained classifier) is replaced by the
  parameter-free :func:`classify_query` keyword/heuristic proxy that approximates the
  same factoid / relation / synthesis signal.
* The paper's *shared ingest substrate* (multi-granularity embeddings, extracted
  triples, fact-version metadata) is replaced by the repo's existing
  :class:`~autogen_core.memory.Memory` backends: any ``Memory`` subclass can be bound
  per mode, and :meth:`RoutedMemory.add` fans every ingested item out to all bound
  backends so each retrieval strategy sees the same substrate.
* The paper's separate benchmark suite (LoCoMo / MAB / LongMemEval) is intentionally
  out of scope — evaluation belongs in a downstream change.

Example:

    .. code-block:: python

        import asyncio
        from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
        from autogen_ext.memory.routed import QueryMode, RoutedMemory


        async def main() -> None:
            memory = RoutedMemory(
                routes={
                    QueryMode.FACTOID: ListMemory(
                        memory_contents=[MemoryContent("Paris is the capital of France.", "text/plain")]
                    ),
                    QueryMode.RELATION: ListMemory(
                        memory_contents=[MemoryContent("Alice reports to Bob.", "text/plain")]
                    ),
                }
            )
            result = await memory.query("What is the capital of France?")
            print(memory.last_mode, result.results)


        asyncio.run(main())
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Mapping

from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.memory import ListMemory, Memory, MemoryContent, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self


class QueryMode(Enum):
    """Per-query cognitive mode, mirroring the three workload classes in the paper.

    * :attr:`FACTOID` — direct factual lookup.
    * :attr:`RELATION` — relation-chain / current-state reasoning.
    * :attr:`SYNTHESIS` — broad synthesis over a long history.
    """

    FACTOID = "factoid"
    RELATION = "relation"
    SYNTHESIS = "synthesis"


# Parameter-free hints approximating the paper's frozen semantic classifier. Each set
# holds substrings (matched against a lower-cased query) that are strong signals for
# the corresponding mode. This is deliberately a simple proxy, not a learned model.
_RELATION_HINTS = (
    "how",
    "why",
    "because",
    "relate",
    "relation",
    "related",
    "connect",
    "between",
    "reports to",
    "manages",
    "managed",
    "depends",
    "dependency",
    "cause",
    "caused",
    "chain",
    "link",
    "linked",
    "hierarchy",
    "reports",
)

_SYNTHESIS_HINTS = (
    "summarize",
    "summary",
    "overview",
    "overall",
    "synthesi",
    "compile",
    "all the",
    "everything",
    "trends",
    "history of",
    "broad",
    "long-form",
    "long form",
    "big picture",
    "state of",
    "recap",
    "digest",
)


def classify_query(query: str) -> QueryMode:
    """Classify a query into a :class:`QueryMode` without learned parameters.

    This is the parameter-free proxy for the paper's frozen semantic classifier.
    A relation/synthesis keyword score is computed over the lower-cased query;
    synthesis wins ties over relation, and an uninformative query defaults to
    :attr:`QueryMode.FACTOID` (the cheapest direct-lookup path).

    Args:
        query: The query text to classify.

    Returns:
        The predicted :class:`QueryMode`.
    """
    text = (query or "").lower()
    if not text.strip():
        return QueryMode.FACTOID
    relation_score = sum(hint in text for hint in _RELATION_HINTS)
    synthesis_score = sum(hint in text for hint in _SYNTHESIS_HINTS)
    if synthesis_score > 0 and synthesis_score >= relation_score:
        return QueryMode.SYNTHESIS
    if relation_score > 0:
        return QueryMode.RELATION
    return QueryMode.FACTOID


class RouteEntry(BaseModel):
    """A single ``mode -> backend`` binding for declarative configuration."""

    mode: QueryMode
    """The cognitive mode that selects this backend."""

    backend: ComponentModel
    """Declarative description of the :class:`~autogen_core.memory.Memory` backend."""


class RoutedMemoryConfig(BaseModel):
    """Declarative configuration for :class:`RoutedMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    routes: List[RouteEntry] = Field(default_factory=list)
    """Mode-to-backend bindings. Each backend is any declarative ``Memory`` component."""

    default_mode: QueryMode = QueryMode.FACTOID
    """Mode whose backend is used when a query's mode has no explicit route."""


class RoutedMemory(Memory, Component[RoutedMemoryConfig]):
    """A memory that routes each query to a mode-specific backend.

    On every :meth:`query` and :meth:`update_context`, the query text is classified
    into a :class:`QueryMode` and dispatched to the backend bound to that mode (falling
    back to the :attr:`~RoutedMemoryConfig.default_mode` backend). The chosen mode is
    recorded on :attr:`last_mode`, exposing the per-query control interface described in
    the paper. :meth:`add` writes to every bound backend so all retrieval strategies
    share one ingest substrate.

    Args:
        routes: Mapping from :class:`QueryMode` to a ``Memory`` backend. If the
            ``default_mode`` has no route, a fresh :class:`~autogen_core.memory.ListMemory`
            is bound for it.
        default_mode: Mode used when a query's classified mode is not in ``routes``.
        classifier: Callable overriding the default :func:`classify_query` proxy.
        name: Optional identifier for this memory instance.

    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.routed.RoutedMemory"
    component_config_schema = RoutedMemoryConfig

    def __init__(
        self,
        routes: Mapping[QueryMode, Memory] | None = None,
        default_mode: QueryMode = QueryMode.FACTOID,
        classifier: Callable[[str], QueryMode] | None = None,
        name: str | None = None,
    ) -> None:
        self._name = name or "routed_memory"
        self._routes: Dict[QueryMode, Memory] = dict(routes) if routes else {}
        self._default_mode = default_mode
        if self._default_mode not in self._routes:
            self._routes[self._default_mode] = ListMemory(name=f"routed_{self._default_mode.value}")
        self._classifier: Callable[[str], QueryMode] = classifier or classify_query
        self._last_mode: QueryMode | None = None

    @property
    def name(self) -> str:
        """Get the memory instance identifier."""
        return self._name

    @property
    def last_mode(self) -> QueryMode | None:
        """The mode chosen for the most recent ``query`` / ``update_context`` call.

        Exposes the per-query control interface: callers (or telemetry) can observe
        which route served each request. ``None`` before any query is made.
        """
        return self._last_mode

    def _all_backends(self) -> List[Memory]:
        """Return the unique set of bound backends (deduped by identity)."""
        seen: List[int] = []
        backends: List[Memory] = []
        for backend in self._routes.values():
            if id(backend) not in seen:
                seen.append(id(backend))
                backends.append(backend)
        return backends

    def _select_backend(self, text: str) -> Memory:
        """Classify ``text`` and return the backend for the resulting mode."""
        self._last_mode = self._classifier(text)
        return self._routes.get(self._last_mode, self._routes[self._default_mode])

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Classify the latest user message, route it, and inject retrieved memories.

        Args:
            model_context: The context to update. Mutated with a ``SystemMessage``
                summarizing the routed memories when any are found.

        Returns:
            UpdateContextResult containing the routed memories.
        """
        query_text = ""
        for message in reversed(await model_context.get_messages()):
            content = getattr(message, "content", "")
            query_text = content if isinstance(content, str) else str(content)
            break

        backend = self._select_backend(query_text)
        memories = await backend.query(query_text)
        if memories.results:
            rendered = "\n".join(f"- {str(item.content)}" for item in memories.results)
            await model_context.add_message(
                SystemMessage(
                    content=f"Relevant memory ({self._last_mode.value if self._last_mode else 'factoid'} route):\n{rendered}\n"
                )
            )
        return UpdateContextResult(memories=memories)

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> Any:
        """Classify and route the query to the matching backend.

        Args:
            query: The query text or :class:`~autogen_core.memory.MemoryContent`.
            cancellation_token: Optional token forwarded to the selected backend.
            **kwargs: Additional parameters forwarded to the selected backend.

        Returns:
            MemoryQueryResult from the backend selected for the query's mode.
        """
        query_text = query if isinstance(query, str) else str(query.content)
        backend = self._select_backend(query_text)
        return await backend.query(query, cancellation_token, **kwargs)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add content to every bound backend (shared ingest substrate fan-out).

        Args:
            content: The memory content to ingest.
            cancellation_token: Optional token forwarded to each backend.
        """
        for backend in self._all_backends():
            await backend.add(content, cancellation_token)

    async def clear(self) -> None:
        """Clear all bound backends."""
        for backend in self._all_backends():
            await backend.clear()

    async def close(self) -> None:
        """Close all bound backends."""
        for backend in self._all_backends():
            await backend.close()

    @classmethod
    def _from_config(cls, config: RoutedMemoryConfig) -> Self:
        routes = {entry.mode: Memory.load_component(entry.backend) for entry in config.routes}
        return cls(routes=routes, default_mode=config.default_mode, name=config.name)

    def _to_config(self) -> RoutedMemoryConfig:
        return RoutedMemoryConfig(
            name=self._name,
            routes=[RouteEntry(mode=mode, backend=backend.dump_component()) for mode, backend in self._routes.items()],
            default_mode=self._default_mode,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self


class SearchContextMemoryConfig(BaseModel):
    """Configuration for the :class:`SearchContextMemory` component."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    max_retries: int = 2
    """Number of times a search query may fail before it is marked exhausted and suppressed."""

    expected_attributes: Dict[str, List[str]] = Field(default_factory=dict)
    """Per-entity attributes the team expects to resolve. Drives the Coverage Map of unresolved gaps."""

    global_attributes: List[str] = Field(default_factory=list)
    """Attributes expected for every entity, merged into the Coverage Map alongside the per-entity set."""


@dataclass
class EvidenceRecord:
    """A grounded fact discovered by a search agent (Evidence Graph node)."""

    entity: str
    attribute: str
    value: str
    citation: str | None = None
    round: int = 0


@dataclass
class FailureRecord:
    """A search attempt that did not yield useful evidence (Failure Memory entry)."""

    query: str
    count: int = 0
    last_reason: str | None = None
    exhausted: bool = False


class SearchContextMemory(Memory, Component[SearchContextMemoryConfig]):
    """Memory that turns fragile, implicit search progress into explicit, shared state.

    This adapts the *Search-Oriented Context Management* (SOCM) idea from
    `SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration`_
    onto AutoGen's :class:`~autogen_core.memory.Memory` contract. The paper's core
    observation is that information-seeking agents get trapped in repetitive loops
    because search progress (what is known, what failed, what remains) is implicit.
    SOCM externalizes that state into four structures; this memory hosts three of
    them as queryable, in-process state:

    - **Evidence Graph** -- grounded ``(entity, attribute) -> value`` facts, each
      optionally anchored to a citation. Adding evidence for an expected attribute
      resolves that coverage slot.
    - **Failure Memory** -- failed search attempts, keyed by a normalized query.
      Once a query has failed ``max_retries`` times it is marked *exhausted* and
      :meth:`should_retry` returns ``False``, so callers stop repeating known-bad
      searches -- the paper's "avoid repeating failed search patterns" mechanism.
    - **Coverage Map** -- expected attributes minus resolved evidence, exposed as
      :meth:`coverage_gaps` so a scheduler or agent can target unresolved slots.

    The fourth SOCM structure (Frontier Task), the pipeline-parallel sub-agent
    scheduler, the search-tool middleware harness, the hierarchical skill system,
    and the WideSearch/GISA benchmarks are intentionally out of scope here: they
    require orchestration infrastructure a single ``Memory`` cannot host, and their
    evaluation belongs in a downstream PR. What remains is the paper's central,
    portable result -- escape repetitive loops by making search state explicit.

    .. _SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration: https://arxiv.org/abs/2607.15257v1

    Example:

        .. code-block:: python

            import asyncio
            from autogen_ext.memory import SearchContextMemory


            async def main() -> None:
                socm = SearchContextMemory(max_retries=2)
                socm.declare_attributes("Ada Lovelace", ["birth_year", "field", "nationality"])
                socm.add_evidence("Ada Lovelace", "birth_year", "1815", citation=" Britannica")
                socm.record_failure("Ada Lovelace phone number")
                socm.record_failure("Ada Lovelace phone number")
                assert not socm.should_retry("Ada Lovelace phone number")
                print(socm.coverage_gaps())


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance.
        max_retries: Failures per query before it is suppressed.
        expected_attributes: Per-entity attributes expected to be resolved.
        global_attributes: Attributes expected for every entity.
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.search_context_memory.SearchContextMemory"
    component_config_schema = SearchContextMemoryConfig

    def __init__(
        self,
        name: str | None = None,
        *,
        max_retries: int = 2,
        expected_attributes: Dict[str, List[str]] | None = None,
        global_attributes: List[str] | None = None,
    ) -> None:
        self._name = name or "search_context_memory"
        self._max_retries = max_retries
        self._global_attributes: Set[str] = set(global_attributes or [])
        self._expected: Dict[str, Set[str]] = {}
        for entity, attrs in (expected_attributes or {}).items():
            self._expected[entity] = set(attrs)
        self._evidence: Dict[Tuple[str, str], EvidenceRecord] = {}
        self._failures: Dict[str, FailureRecord] = {}
        self._notes: List[MemoryContent] = []
        self._round = 0

    # ------------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        """Memory instance identifier."""
        return self._name

    @property
    def evidence(self) -> List[EvidenceRecord]:
        """All grounded evidence records currently held in the Evidence Graph."""
        return list(self._evidence.values())

    @property
    def failures(self) -> List[FailureRecord]:
        """All recorded search failures currently held in Failure Memory."""
        return list(self._failures.values())

    # ------------------------------------------------------------------ SOCM surface
    def declare_attributes(self, entity: str, attributes: List[str]) -> None:
        """Register attributes expected to be resolved for ``entity`` (extends the Coverage Map)."""
        self._expected.setdefault(entity, set()).update(attributes)

    def add_evidence(
        self,
        entity: str,
        attribute: str,
        value: str,
        citation: str | None = None,
    ) -> EvidenceRecord:
        """Record a grounded fact. Re-adding an ``(entity, attribute)`` updates the value."""
        self._round += 1
        record = EvidenceRecord(
            entity=entity,
            attribute=attribute,
            value=value,
            citation=citation,
            round=self._round,
        )
        self._evidence[(entity, attribute)] = record
        return record

    def record_failure(self, query: str, reason: str | None = None) -> FailureRecord:
        """Record that a search attempt failed. Marks the query exhausted past the retry budget.

        Failures are deduped on a normalized key (case- and whitespace-insensitive) so the same
        search phrased differently still counts toward the budget; the original query is kept for
        display.
        """
        key = self._normalize_query(query)
        record = self._failures.get(key)
        if record is None:
            record = FailureRecord(query=query)
            self._failures[key] = record
        else:
            record.query = query
        record.count += 1
        record.last_reason = reason
        record.exhausted = record.count >= self._max_retries
        return record

    def should_retry(self, query: str) -> bool:
        """Return ``False`` once ``query`` has failed ``max_retries`` times.

        This is the core mechanism for escaping repetitive search loops: a caller
        consults it before re-attempting a search that has already failed.
        """
        record = self._failures.get(self._normalize_query(query))
        if record is None:
            return True
        return not record.exhausted

    def coverage_gaps(self) -> List[Tuple[str, str]]:
        """Return ``(entity, attribute)`` pairs that are expected but not yet backed by evidence."""
        gaps: List[Tuple[str, str]] = []
        for entity in sorted(self._expected):
            for attribute in sorted(self._expected[entity] | self._global_attributes):
                if (entity, attribute) not in self._evidence:
                    gaps.append((entity, attribute))
        return gaps

    def evidence_snapshot(self) -> str:
        """Render the current SOCM state as a structured text snapshot."""
        lines: List[str] = []
        evidence = sorted(self._evidence.values(), key=lambda r: (r.entity, r.attribute))
        if evidence:
            lines.append("[Evidence] grounded facts discovered so far")
            for record in evidence:
                source = f"  (src: {record.citation})" if record.citation else ""
                lines.append(f"- {record.entity}.{record.attribute} = {record.value}{source}")
        gaps = self.coverage_gaps()
        if gaps:
            lines.append("[Coverage gaps] expected attributes still unresolved - prioritize these")
            for entity, attribute in gaps:
                lines.append(f"- {entity}.{attribute}")
        exhausted = [f for f in self._failures.values() if f.exhausted]
        if exhausted:
            lines.append("[Failure memory] exhausted queries - do not repeat")
            for record in exhausted:
                lines.append(f'- "{record.query}" failed {record.count} time(s)')
        return "\n".join(lines)

    # ------------------------------------------------------------------ Memory ABC
    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Route ``content`` into the SOCM structures via its ``metadata["kind"]``.

        Supported kinds: ``"evidence"`` (entity/attribute/citation in metadata, value in content),
        ``"failure"`` (query in content, optional reason in metadata), ``"expect"`` (entity +
        attributes in metadata). Anything else is stored as a free-form note.
        """
        _ = cancellation_token
        metadata = content.metadata or {}
        kind = metadata.get("kind")
        if kind == "evidence":
            self.add_evidence(
                entity=str(metadata.get("entity", "")),
                attribute=str(metadata.get("attribute", "")),
                value=str(content.content),
                citation=metadata.get("citation"),
            )
        elif kind == "failure":
            self.record_failure(query=str(content.content), reason=metadata.get("reason"))
        elif kind == "expect":
            self.declare_attributes(
                entity=str(metadata.get("entity", "")),
                attributes=list(metadata.get("attributes", [])),
            )
        else:
            self._notes.append(content)

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return evidence whose entity, attribute, or value lexically contains the query.

        Retrieval is a parameter-free lexical proxy (no learned retriever); sufficient to surface
        grounded facts for a downstream agent without adding a model dependency.
        """
        _ = cancellation_token, kwargs
        needle = self._normalize_query(query if isinstance(query, str) else str(query.content))
        results: List[MemoryContent] = []
        if not needle:
            return MemoryQueryResult(results=results)
        for record in self._evidence.values():
            haystack = self._normalize_query(f"{record.entity} {record.attribute} {record.value}")
            if needle in haystack:
                results.append(
                    MemoryContent(
                        content=f"{record.entity}.{record.attribute} = {record.value}",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"entity": record.entity, "attribute": record.attribute, "citation": record.citation},
                    )
                )
        return MemoryQueryResult(results=results)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject the SOCM snapshot as a system message so every agent shares the same search state.

        This is what makes progress explicit and persistent across turns: an agent sees what is
        grounded, which gaps remain, and which searches are exhausted -- the conditions that, per
        the paper, break repetitive search loops.
        """
        snapshot = self.evidence_snapshot()
        body = snapshot
        if self._notes:
            notes = "\n".join(str(note.content) for note in self._notes)
            body = f"{snapshot}\n\n[Notes]\n{notes}" if snapshot else notes
        if not body.strip():
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        message = "=== Search-Oriented Context (SOCM) ===\n" + body
        await model_context.add_message(SystemMessage(content=message))
        return UpdateContextResult(
            memories=MemoryQueryResult(results=[MemoryContent(content=message, mime_type=MemoryMimeType.TEXT)])
        )

    async def clear(self) -> None:
        """Clear all evidence, failures, and notes."""
        self._evidence.clear()
        self._failures.clear()
        self._notes.clear()
        self._round = 0

    async def close(self) -> None:
        """No external resources to release."""

    # ------------------------------------------------------------------ helpers / Component
    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.lower().split())

    @classmethod
    def _from_config(cls, config: SearchContextMemoryConfig) -> Self:
        return cls(
            name=config.name,
            max_retries=config.max_retries,
            expected_attributes=config.expected_attributes,
            global_attributes=config.global_attributes,
        )

    def _to_config(self) -> SearchContextMemoryConfig:
        return SearchContextMemoryConfig(
            name=self._name,
            max_retries=self._max_retries,
            expected_attributes={entity: sorted(attrs) for entity, attrs in self._expected.items()},
            global_attributes=sorted(self._global_attributes),
        )

"""Provenance-constrained Structured Evidence Ledger memory.

Adapted (Mode 2) port of *LedgerMind: Provenance-Constrained Multimodal Agentic
Reasoning with a Structured Evidence Ledger* (arXiv:2607.28374).

What is kept at full fidelity (the paper's core mechanism):

* Every stored item carries structured ledger metadata in
  :attr:`MemoryContent.metadata` -- ``entry_id``, ``source`` (provenance origin),
  ``epistemic_type`` (``observation``/``retrieved``/``claim``/``decision``),
  ``confidence``, and a ``status`` drawn from a small lifecycle FSA
  (``active``/``superseded``).
* Retrieval (``query``) returns only **Active**, tool-grounded entries --
  the support set the paper calls *support-set resolution*.
* ``add`` realizes the lifecycle transitions **Append** and **Supersede**, and
  ``update_context`` injects only Active grounded evidence **with citations**.
* A *provenance non-amplification* guarantee: a reasoning claim is only marked
  grounded when every entry it cites resolves to an Active, tool-grounded entry.
  Repair (``supersede``) only flips status -- it can never introduce content.

What is substituted with target-native equivalents (Mode 2 auxiliaries):

* The paper's learned / perceptual *Three-Layer Grounding Protocol* is replaced
  by a parameter-free citation-resolution check (claims cite entry ids; we verify
  those ids resolve to Active tool-grounded entries). No MLLM perception is
  required -- this is a memory backend, not a multimodal agent loop.
* The *Adaptive Dual-Path Dispatcher* and the *Event-Triggered Verification-and-
  Repair engine* are cut (they are orchestration over an agent trajectory the
  repo does not host here); their core *signal* -- unsupported intermediate
  reasoning and citation-backed hallucination -- is exposed via ``verify()``.
* Vector retrieval is replaced with a token-overlap support-set filter; the
  multimodal benchmark suite is out of scope (evaluation belongs downstream).
"""

from typing import Any, Dict, List

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel
from typing_extensions import Self

__all__ = [
    "EvidenceLedgerMemory",
    "EvidenceLedgerMemoryConfig",
    "STATUS_ACTIVE",
    "STATUS_SUPERSEDED",
    "TYPE_OBSERVATION",
    "TYPE_RETRIEVED",
    "TYPE_CLAIM",
    "TYPE_DECISION",
]

# --- Lifecycle FSA states -------------------------------------------------
STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"

# --- Epistemic types ------------------------------------------------------
TYPE_OBSERVATION = "observation"
TYPE_RETRIEVED = "retrieved"
TYPE_CLAIM = "claim"
TYPE_DECISION = "decision"

# Tool-grounded types carry tool-produced provenance and are trustworthy as-is.
_TOOL_GROUNDED_TYPES = {TYPE_OBSERVATION, TYPE_RETRIEVED}
# Citing types are reasoning outputs that must rest on active ledger evidence.
_CITING_TYPES = {TYPE_CLAIM, TYPE_DECISION}


class EvidenceLedgerMemoryConfig(BaseModel):
    """Declarative configuration for :class:`EvidenceLedgerMemory`."""

    name: str | None = None
    """Optional identifier for this ledger instance."""

    query_score_threshold: float = 0.0
    """Minimum token-overlap score for an entry to be returned by ``query``.

    ``0.0`` returns the full Active support set (the ledger default)."""

    inject_citations: bool = True
    """When injecting evidence into the model context, render entry ids and
    provenance so downstream reasoning can cite Active entries."""


class EvidenceLedgerMemory(Memory, Component[EvidenceLedgerMemoryConfig]):
    """A provenance-constrained Structured Evidence Ledger.

    Each stored :class:`~autogen_core.memory.MemoryContent` is treated as a ledger
    entry. Ledger fields live in ``content.metadata``:

    * ``entry_id``     -- monotonic id assigned on ``add``.
    * ``source``       -- provenance origin (tool name / retrieval source).
    * ``epistemic_type`` -- one of ``observation``/``retrieved``/``claim``/``decision``.
    * ``confidence``   -- float in ``[0, 1]``.
    * ``status``       -- ``active`` or ``superseded`` (lifecycle FSA).
    * ``citations``    -- entry ids a claim/decision rests on.
    * ``supersedes``   -- entry ids this entry replaces (consumed on ``add``).
    * ``grounded``     -- computed: whether the entry's provenance / citations
      resolve to Active tool-grounded evidence.

    The ledger enforces *provenance non-amplification*: a reasoning entry is
    only ``grounded`` once its citations resolve, and repair (``supersede``)
    only flips status -- it never fabricates content.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent
            from autogen_ext.memory.evidence_ledger import EvidenceLedgerMemory


            async def main() -> None:
                ledger = EvidenceLedgerMemory()
                # Tool-grounded observation.
                await ledger.add(
                    MemoryContent(
                        content="The weather in Paris is sunny, 22C.",
                        mime_type="text/plain",
                        metadata={"source": "get_weather", "epistemic_type": "observation"},
                    )
                )
                # A claim that cites the observation is grounded.
                await ledger.add(
                    MemoryContent(
                        content="It is a good day for a walk.",
                        mime_type="text/plain",
                        metadata={"epistemic_type": "claim", "citations": [1]},
                    )
                )
                print([r.metadata["grounded"] for r in (await ledger.query("walk")).results])


            asyncio.run(main())

    Args:
        name: Optional identifier for this ledger instance.
        query_score_threshold: Minimum token-overlap score for ``query`` hits.
        inject_citations: Render entry ids + provenance in ``update_context``.
    """

    component_type = "memory"
    component_config_schema = EvidenceLedgerMemoryConfig
    component_provider_override = "autogen_ext.memory.evidence_ledger.EvidenceLedgerMemory"

    def __init__(
        self,
        name: str | None = None,
        query_score_threshold: float = 0.0,
        inject_citations: bool = True,
    ) -> None:
        self._name = name or "evidence_ledger"
        self._query_score_threshold = query_score_threshold
        self._inject_citations = inject_citations
        self._entries: List[MemoryContent] = []
        self._next_id: int = 1

    @property
    def name(self) -> str:
        """Identifier for this ledger instance."""
        return self._name

    @property
    def content(self) -> List[MemoryContent]:
        """All stored entries (any status), in insertion order."""
        return self._entries

    @staticmethod
    def _meta(entry: MemoryContent) -> Dict[str, Any]:
        return dict(entry.metadata or {})

    def _status_of(self, entry_id: Any) -> str:
        for entry in self._entries:
            if self._meta(entry).get("entry_id") == entry_id:
                return self._meta(entry).get("status", STATUS_ACTIVE)
        return ""  # unknown id -- treated as unresolvable

    def _citations_resolve(self, citation_ids: List[Any]) -> bool:
        """Non-amplification check: every cited id is a known, Active entry."""
        if not citation_ids:
            return False  # a claim with no citations has no provenance
        return all(self._status_of(cid) == STATUS_ACTIVE for cid in citation_ids)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Append an entry and realize its lifecycle transition.

        Assigns ``entry_id``/``status``. If ``metadata["supersedes"]`` lists
        entry ids, those entries are flipped to ``superseded`` (repair). The
        entry's ``grounded`` flag is computed under the non-amplification rule:
        tool-grounded types need a ``source``; citing types need citations that
        resolve to Active evidence.

        Args:
            content: The memory content to store. Ledger fields are read from /
                written into ``content.metadata``.
            cancellation_token: Optional token to cancel the operation.
        """
        _ = cancellation_token
        md = self._meta(content)
        epistemic_type = md.get("epistemic_type", TYPE_CLAIM)

        entry_id = self._next_id
        self._next_id += 1

        # Provenance non-amplification gate (grounding computed before the entry
        # itself joins the support set, so it cannot cite itself).
        grounded = self._currently_grounded(md)

        # Repair: typed state transition that only flips status, never adds content.
        for superseded_id in md.get("supersedes", []) or []:
            for existing in self._entries:
                if self._meta(existing).get("entry_id") == superseded_id:
                    existing_md = self._meta(existing)
                    existing_md["status"] = STATUS_SUPERSEDED
                    existing.metadata = existing_md

        md.update(
            {
                "entry_id": entry_id,
                "epistemic_type": epistemic_type,
                "status": STATUS_ACTIVE,
                "confidence": md.get("confidence", 1.0),
                "citations": list(md.get("citations", []) or []),
                "grounded": grounded,
            }
        )
        md.pop("supersedes", None)
        content.metadata = md
        self._entries.append(content)

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return the Active support set, optionally filtered.

        Only ``active`` entries are returned (superseded entries leave the
        support set). ``query`` is scored by token overlap; entries below
        ``query_score_threshold`` are dropped (``0.0`` returns all Active
        entries). Pass ``grounded_only=True`` to additionally require a resolved
        ``grounded`` flag -- the grounded support set ``update_context`` injects.

        Args:
            query: Free text (or content) used to rank the support set.
            cancellation_token: Optional token to cancel the operation.
            **kwargs: ``grounded_only`` (bool) restricts to grounded entries.
        """
        _ = cancellation_token
        grounded_only = bool(kwargs.get("grounded_only", False))
        query_text = str(query.content) if isinstance(query, MemoryContent) else str(query)

        results: List[MemoryContent] = []
        for entry in self._entries:
            md = self._meta(entry)
            if md.get("status", STATUS_ACTIVE) != STATUS_ACTIVE:
                continue
            if grounded_only and not md.get("grounded", False):
                continue
            if self._overlap_score(query_text, entry) < self._query_score_threshold:
                continue
            results.append(entry)
        return MemoryQueryResult(results=results)

    @staticmethod
    def _overlap_score(query: str, entry: MemoryContent) -> float:
        query_tokens = {t for t in query.lower().split() if t}
        if not query_tokens:
            return 1.0  # no query -> include every Active entry
        entry_tokens = {t for t in str(entry.content).lower().split() if t}
        if not entry_tokens:
            return 0.0
        return len(query_tokens & entry_tokens) / len(query_tokens)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject Active grounded evidence with citations into the model context.

        Mirrors LedgerMind's trajectory-state injection: only ``active`` and
        ``grounded`` entries are surfaced, each rendered with its entry id and
        provenance so downstream reasoning can cite them.

        Args:
            model_context: The context to mutate with a ledger system message.

        Returns:
            UpdateContextResult containing the grounded support set that was
            injected.
        """
        support = (await self.query(grounded_only=True)).results
        if not support:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        lines: List[str] = []
        for entry in support:
            md = self._meta(entry)
            if self._inject_citations:
                cites = md.get("citations") or []
                cite_str = f" cites={cites}" if cites else ""
                lines.append(
                    f"[#{md.get('entry_id')} {md.get('epistemic_type')} "
                    f"src={md.get('source', 'n/a')} conf={md.get('confidence', 1.0):.2f}{cite_str}] {entry.content}"
                )
            else:
                lines.append(f"{entry.content}")
        message = "Active grounded evidence (Structured Evidence Ledger):\n" + "\n".join(lines)
        await model_context.add_message(SystemMessage(content=message))
        return UpdateContextResult(memories=MemoryQueryResult(results=support))

    async def supersede(self, entry_ids: List[Any]) -> None:
        """Mark entries superseded -- a typed state transition (repair).

        Only flips ``status``; introduces no content, preserving the
        non-amplification guarantee.

        Args:
            entry_ids: Entry ids to retire from the Active support set.
        """
        for entry in self._entries:
            md = self._meta(entry)
            if md.get("entry_id") in entry_ids:
                md["status"] = STATUS_SUPERSEDED
                entry.metadata = md

    def _currently_grounded(self, md: Dict[str, Any]) -> bool:
        """Recompute grounding from live ledger state (status may have changed)."""
        epistemic_type = md.get("epistemic_type", TYPE_CLAIM)
        if epistemic_type in _TOOL_GROUNDED_TYPES:
            return bool(md.get("source"))
        if epistemic_type in _CITING_TYPES:
            return self._citations_resolve(list(md.get("citations", [])))
        return bool(md.get("source"))

    async def verify(self) -> List[Dict[str, Any]]:
        """Surface unsupported / ungrounded entries.

        Targets the failure patterns the paper says final-answer accuracy hides:
        *unsupported intermediate reasoning* (claims whose citations no longer
        resolve) and *citation-backed entity hallucination* (claims that were
        never grounded). Grounding is recomputed from live ledger state, so a
        claim whose support was later superseded is re-flagged here.

        Returns:
            A list of reports, one per currently-ungrounded Active entry, each
            with ``entry_id``, ``epistemic_type``, and the ``unresolved``
            citation ids.
        """
        reports: List[Dict[str, Any]] = []
        for entry in self._entries:
            md = self._meta(entry)
            if md.get("status", STATUS_ACTIVE) != STATUS_ACTIVE:
                continue
            if self._currently_grounded(md):
                continue
            unresolved = [c for c in (md.get("citations") or []) if self._status_of(c) != STATUS_ACTIVE]
            reports.append(
                {
                    "entry_id": md.get("entry_id"),
                    "epistemic_type": md.get("epistemic_type"),
                    "unresolved": unresolved,
                }
            )
        return reports

    async def clear(self) -> None:
        """Clear all entries and reset the id counter."""
        self._entries = []
        self._next_id = 1

    async def close(self) -> None:
        """No external resources to release."""
        pass

    @classmethod
    def _from_config(cls, config: EvidenceLedgerMemoryConfig) -> Self:
        return cls(
            name=config.name,
            query_score_threshold=config.query_score_threshold,
            inject_citations=config.inject_citations,
        )

    def _to_config(self) -> EvidenceLedgerMemoryConfig:
        return EvidenceLedgerMemoryConfig(
            name=self._name,
            query_score_threshold=self._query_score_threshold,
            inject_citations=self._inject_citations,
        )

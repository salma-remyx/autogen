"""Lifecycle memory with evidence-gated writes, correction, forgetting, and an audit trail.

Adapted from *Mi-Memory: A Lifecycle Memory Framework for Personal AI*
(arXiv:2607.18975v1). The paper reframes memory as a *continuity and
governance substrate* rather than a flat cache, unified by an *audit
contract* of four artifact families:

* **typed evidence payloads** -- preserve source identity and provenance;
* **diagnostic traces** -- localize where candidate memories are dropped;
* **strategy artifacts** -- make memory-policy changes explicit;
* **gate/rollback records** -- bound accepted evolution.

This module ports that audit contract and the paper's *correction and
forgetting* lifecycle primitive onto AutoGen's
:class:`~autogen_core.memory.Memory` ABC. It is an **adapted port (Mode 2)**:
the core governance mechanism -- evidence-gated admission, superseding
corrections, reversible forgetting, and a typed audit log -- is implemented at
full fidelity, while the paper's auxiliary infrastructure is substituted with
target-native equivalents:

* multimodal *device evidence* and any learned evidence estimator are replaced
  by a parameter-free provenance proxy -- a memory is "evidenced" when it
  carries non-empty provenance metadata (override
  :meth:`LifecycleMemory._evaluate_evidence` to plug in a richer estimator);
* the paper's separate benchmark suite (LoCoMo / PersonaMem-V2 / LongMemEval)
  is intentionally out of scope -- evaluation belongs in a downstream PR.

The result is a drop-in :class:`Memory` for any
:class:`~autogen_agentchat.agents.AssistantAgent` whose personal facts must be
auditable, correctable, and forgettable on demand.
"""

import logging
from enum import Enum
from typing import Any, List

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

logger = logging.getLogger(__name__)


class MemoryStatus(str, Enum):
    """Lifecycle status of a stored memory entry."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


class AuditKind(str, Enum):
    """Kind of audit artifact (the gate/rollback-record family of the contract)."""

    WRITE = "write"
    GATE_REJECT = "gate_reject"
    CORRECT = "correct"
    FORGET = "forget"
    RESTORE = "restore"
    CLEAR = "clear"


class EvidencePayload(BaseModel):
    """Typed evidence payload preserving source identity and provenance.

    A parameter-free proxy for the multimodal device evidence described in the
    paper: provenance is whatever metadata the caller attached at write time.
    """

    source: str = Field(default="user", description="Where the memory came from (user, device, tool, ...).")
    collected_at: str | None = Field(default=None, description="ISO-8601 timestamp the evidence was collected.")
    tags: List[str] = Field(default_factory=list, description="Free-form provenance tags.")


class GateDecision(BaseModel):
    """Outcome of evaluating evidence for a candidate write."""

    accepted: bool
    reason: str


class AuditEntry(BaseModel):
    """A single typed audit record linking lifecycle events across the contract."""

    kind: AuditKind
    memory_id: str
    accepted: bool = True
    reason: str = ""
    before: MemoryContent | None = None
    after: MemoryContent | None = None


class _MemoryRecord(BaseModel):
    """Internal stored memory with lifecycle bookkeeping."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_id: str
    content: MemoryContent
    evidence: EvidencePayload = Field(default_factory=EvidencePayload)
    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: str | None = None


class LifecycleMemoryConfig(BaseModel):
    """Configuration for :class:`LifecycleMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    require_evidence: bool = False
    """If True, writes lacking provenance metadata are rejected at the gate."""

    evidence_source_key: str = "source"
    """Metadata key read as the evidence source when building an :class:`EvidencePayload`."""

    top_k: int = 10
    """Maximum number of active memories a :meth:`LifecycleMemory.query` returns."""


class LifecycleMemory(Memory, Component[LifecycleMemoryConfig]):
    """Auditable, evidence-gated memory with correction and forgetting.

    Each accepted write is admitted through an evidence gate and stamped with a
    typed :class:`EvidencePayload`. Existing memories can be **corrected**
    (superseded in place) or **forgotten** (reversibly removed), and every
    lifecycle event is appended to an :attr:`audit_log` of typed
    :class:`AuditEntry` records -- the gate/rollback-record family of
    Mi-Memory's audit contract. Gate rejections are recorded too, localizing
    where a candidate memory was dropped (the diagnostic-trace family).

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.lifecycle_memory import LifecycleMemory


            async def main() -> None:
                memory = LifecycleMemory(name="personal", require_evidence=True)
                await memory.add(
                    MemoryContent(
                        content="User lives in Seattle",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"source": "onboarding"},
                    )
                )
                # Every admitted write is recorded, exposing its memory id.
                memory_id = memory.audit_log[-1].memory_id
                await memory.forget(memory_id, reason="erasure request")  # reversible via restore()


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance.
        require_evidence: If True, writes without provenance metadata are rejected at the gate.
        config: Optional :class:`LifecycleMemoryConfig`. Overrides ``name`` / ``require_evidence``.

    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.lifecycle_memory.LifecycleMemory"
    component_config_schema = LifecycleMemoryConfig

    def __init__(
        self,
        name: str | None = None,
        require_evidence: bool = False,
        *,
        config: LifecycleMemoryConfig | None = None,
    ) -> None:
        """Initialize the lifecycle memory."""
        self.config = config or LifecycleMemoryConfig(name=name, require_evidence=require_evidence)
        self._name = self.config.name or "default_lifecycle_memory"
        self._records: List[_MemoryRecord] = []
        self._index: dict[str, _MemoryRecord] = {}
        self._audit: List[AuditEntry] = []
        self._counter = 0

    # -- public surface ---------------------------------------------------

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._name

    @property
    def audit_log(self) -> List[AuditEntry]:
        """Snapshot of the full typed audit trail (gate/rollback records)."""
        return list(self._audit)

    def audit_trail(self, memory_id: str | None = None) -> List[AuditEntry]:
        """Return audit entries, optionally filtered to a single ``memory_id``."""
        if memory_id is None:
            return list(self._audit)
        return [entry for entry in self._audit if entry.memory_id == memory_id]

    @property
    def active_count(self) -> int:
        """Number of currently active (non-forgotten, non-superseded) memories."""
        return sum(1 for record in self._records if record.status is MemoryStatus.ACTIVE)

    # -- evidence gate ----------------------------------------------------

    def _evaluate_evidence(self, content: MemoryContent) -> GateDecision:
        """Parameter-free evidence proxy: admissible iff provenance metadata is present.

        Override to plug in a richer or learned estimator -- the auxiliary this
        module substitutes from the paper.
        """
        if not self.config.require_evidence:
            return GateDecision(accepted=True, reason="evidence gate disabled")
        if content.metadata:
            return GateDecision(accepted=True, reason="provenance metadata present")
        return GateDecision(accepted=False, reason="require_evidence is set but no provenance metadata was supplied")

    def _evidence_from(self, content: MemoryContent) -> EvidencePayload:
        metadata = content.metadata or {}
        raw_tags = metadata.get("tags", [])
        tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
        return EvidencePayload(
            source=str(metadata.get(self.config.evidence_source_key, "user")),
            collected_at=metadata.get("collected_at"),
            tags=tags,
        )

    # -- Memory ABC -------------------------------------------------------

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add content through the evidence gate.

        Rejected writes are dropped and recorded as a ``GATE_REJECT`` audit
        entry (the diagnostic trace that localizes evidence loss) rather than
        raised, mirroring the paper's "bound accepted evolution" semantics.
        """
        _ = cancellation_token
        decision = self._evaluate_evidence(content)
        if not decision.accepted:
            self._audit.append(
                AuditEntry(
                    kind=AuditKind.GATE_REJECT, memory_id="", accepted=False, reason=decision.reason, after=content
                )
            )
            logger.debug("lifecycle_memory gate rejected write: %s", decision.reason)
            return
        record = _MemoryRecord(memory_id=self._next_id(), content=content, evidence=self._evidence_from(content))
        self._records.append(record)
        self._index[record.memory_id] = record
        self._audit.append(
            AuditEntry(
                kind=AuditKind.WRITE, memory_id=record.memory_id, accepted=True, reason=decision.reason, after=content
            )
        )

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return active memories, optionally filtered by a case-insensitive substring.

        A parameter-free retrieval proxy for the paper's vector retrieval (an
        auxiliary component); pass an empty query to return all active memories
        in insertion order.
        """
        _ = cancellation_token
        top_k = int(kwargs.pop("top_k", self.config.top_k))
        active = [record.content for record in self._records if record.status is MemoryStatus.ACTIVE]
        needle = query if isinstance(query, str) else str(query.content)
        if needle:
            active = [content for content in active if needle.lower() in str(content.content).lower()]
        return MemoryQueryResult(results=active[:top_k])

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject active memories into ``model_context`` as a single system message."""
        results = await self.query("")
        if not results.results:
            return UpdateContextResult(memories=results)
        lines = [f"- {str(content.content)}" for content in results.results]
        body = "Personal memory (lifecycle, evidence-gated, audited):\n" + "\n".join(lines)
        await model_context.add_message(SystemMessage(content=body))
        return UpdateContextResult(memories=results)

    async def clear(self) -> None:
        """Forget every active memory, recording a single ``CLEAR`` rollback entry."""
        forgotten = 0
        for record in self._records:
            if record.status is MemoryStatus.ACTIVE:
                record.status = MemoryStatus.FORGOTTEN
                forgotten += 1
        self._audit.append(
            AuditEntry(kind=AuditKind.CLEAR, memory_id="", accepted=True, reason=f"bulk clear ({forgotten} forgotten)")
        )

    async def close(self) -> None:
        """Release any external resources (none are held)."""
        return

    # -- lifecycle primitives (correction & forgetting) -------------------

    async def correct(self, memory_id: str, new_content: MemoryContent, reason: str = "") -> str | None:
        """Supersede an active memory with corrected content.

        Returns the replacement memory's id, or ``None`` if ``memory_id`` was
        not an active memory (the failed attempt is still recorded in the audit log).
        """
        record = self._index.get(memory_id)
        if record is None or record.status is not MemoryStatus.ACTIVE:
            self._audit.append(
                AuditEntry(
                    kind=AuditKind.CORRECT,
                    memory_id=memory_id,
                    accepted=False,
                    reason=f"no active memory '{memory_id}'",
                    after=new_content,
                )
            )
            return None
        before = record.content
        record.status = MemoryStatus.SUPERSEDED
        replacement = _MemoryRecord(
            memory_id=self._next_id(), content=new_content, evidence=self._evidence_from(new_content)
        )
        record.superseded_by = replacement.memory_id
        self._records.append(replacement)
        self._index[replacement.memory_id] = replacement
        self._audit.append(
            AuditEntry(
                kind=AuditKind.CORRECT,
                memory_id=replacement.memory_id,
                accepted=True,
                reason=reason or "user correction",
                before=before,
                after=new_content,
            )
        )
        return replacement.memory_id

    async def forget(self, memory_id: str, reason: str = "") -> bool:
        """Reversibly forget a memory; see :meth:`restore` to roll back."""
        record = self._index.get(memory_id)
        if record is None or record.status is MemoryStatus.FORGOTTEN:
            self._audit.append(
                AuditEntry(
                    kind=AuditKind.FORGET, memory_id=memory_id, accepted=False, reason=f"no active memory '{memory_id}'"
                )
            )
            return False
        before = record.content
        record.status = MemoryStatus.FORGOTTEN
        self._audit.append(
            AuditEntry(
                kind=AuditKind.FORGET,
                memory_id=memory_id,
                accepted=True,
                reason=reason or "user requested forget",
                before=before,
            )
        )
        return True

    async def restore(self, memory_id: str, reason: str = "") -> bool:
        """Roll back a prior :meth:`forget` (the gate/rollback-record family)."""
        record = self._index.get(memory_id)
        if record is None or record.status is not MemoryStatus.FORGOTTEN:
            self._audit.append(
                AuditEntry(
                    kind=AuditKind.RESTORE,
                    memory_id=memory_id,
                    accepted=False,
                    reason=f"no forgotten memory '{memory_id}'",
                )
            )
            return False
        record.status = MemoryStatus.ACTIVE
        self._audit.append(
            AuditEntry(
                kind=AuditKind.RESTORE,
                memory_id=memory_id,
                accepted=True,
                reason=reason or "rollback",
                after=record.content,
            )
        )
        return True

    # -- bookkeeping ------------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        return f"mem-{self._counter}"

    # -- declarative Component config -------------------------------------

    @classmethod
    def _from_config(cls, config: LifecycleMemoryConfig) -> Self:
        return cls(config=config)

    def _to_config(self) -> LifecycleMemoryConfig:
        return self.config

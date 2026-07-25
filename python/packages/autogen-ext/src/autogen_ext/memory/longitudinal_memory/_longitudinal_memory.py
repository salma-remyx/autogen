"""Governed, self-evolving longitudinal memory with per-episode induction.

Adapted from *HealthClaw: A Self-Evolving Agent for Longitudinal Personal Health
Management* (https://arxiv.org/abs/2607.13940v1). The paper's core mechanism is
preserved at full fidelity and mapped onto the :class:`~autogen_core.memory.Memory`
protocol:

* a **two-tier store** separating governed *shared* context (safety rules and
  reusable knowledge) from *private* longitudinal memory (profile facts, reusable
  procedures, episodic traces), and
* a per-episode **induction policy** that, after each episode, decides one of the
  four HealthClaw outcomes: update the profile, revise a procedure, keep it as an
  episodic trace, or exclude it. Repeated episodic observations are *promoted* into
  profile facts, so the memory self-evolves and the prompt-side context shrinks.

Mode-2 substitutions (so the core fits the target without external infrastructure):

* The paper's LLM-driven induction is replaced by a deterministic, parameter-free
  policy keyed on record identity and recurrence (a proxy for the same signal).
* The paper's learned retrieval is replaced by token-overlap scoring.
* The synthetic year-long benchmark, biomedical tasks and privacy probes are cut;
  evaluation belongs in a downstream PR.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Set

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = [
    "InductionOutcome",
    "LongitudinalKind",
    "LongitudinalMemory",
    "LongitudinalMemoryConfig",
    "MemoryTier",
]


class MemoryTier(str, Enum):
    """Tier separating governed shared context from private longitudinal memory."""

    SHARED = "shared"
    PRIVATE = "private"


class LongitudinalKind(str, Enum):
    """Kind of a longitudinal record, which governs its induction outcome."""

    RULE = "rule"
    """Shared: a safety / policy rule that must always govern responses."""

    KNOWLEDGE = "knowledge"
    """Shared: reusable domain knowledge applicable across episodes."""

    PROFILE = "profile"
    """Private: a stable fact about the user (preference, attribute, baseline)."""

    PROCEDURE = "procedure"
    """Private: a reusable step-by-step procedure keyed by a task name."""

    EPISODIC = "episodic"
    """Private: a one-off encounter trace, retained in a bounded buffer."""


class InductionOutcome(str, Enum):
    """Outcome of running induction over an episode (the four HealthClaw outcomes)."""

    UPDATE_PROFILE = "update_profile"
    REVISE_PROCEDURE = "revise_procedure"
    KEEP_EPISODIC = "keep_episodic"
    EXCLUDE = "exclude"
    ADD_SHARED = "add_shared"
    """Episode was placed in the governed shared tier (rules / knowledge)."""


class LongitudinalMemoryConfig(BaseModel):
    """Declarative configuration for :class:`LongitudinalMemory`."""

    name: str | None = None
    promotion_threshold: int = Field(
        default=2, ge=1, description="Recurrences that promote an episodic candidate into the profile."
    )
    max_episodic: int = Field(default=64, ge=1, description="Bound on retained episodic traces (FIFO).")
    recent_episodic_in_context: int = Field(
        default=3, ge=0, description="How many recent episodic traces update_context injects."
    )
    query_k: int = Field(default=5, ge=1, description="Maximum records query returns.")


_SHARED_KINDS: Set[LongitudinalKind] = {LongitudinalKind.RULE, LongitudinalKind.KNOWLEDGE}


def _tokenize(text: str) -> Set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok}


class LongitudinalMemory(Memory, Component[LongitudinalMemoryConfig]):
    """Governed, self-evolving longitudinal memory with per-episode induction.

    Implements the HealthClaw two-tier store and induction policy on top of the
    :class:`~autogen_core.memory.Memory` protocol. ``add`` ingests an episode and runs
    induction; ``query`` retrieves across the shared, profile, procedure and episodic
    layers; ``update_context`` injects a compact, governed summary so the prompt only
    carries consolidated profile/procedure facts plus a bounded episodic window rather
    than the full history.

    An episode is a :class:`~autogen_core.memory.MemoryContent` whose ``metadata``
    may carry:

    * ``tier`` (``"shared"`` | ``"private"``; default ``"private"``),
    * ``kind`` (a :class:`LongitudinalKind` value; default ``"episodic"``),
    * ``key`` (identity for profile/procedure records, used for promotion and revision),
    * ``exclude`` (truthy to drop the episode outright — the privacy-governance kernel),
    * ``confirmed`` (truthy to promote a profile fact immediately, bypassing recurrence).

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory import LongitudinalMemory


            async def main() -> None:
                memory = LongitudinalMemory()
                # Shared, always-on safety rule.
                await memory.add(
                    MemoryContent(
                        content="Escalate chest pain to a clinician.",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"tier": "shared", "kind": "rule"},
                    )
                )
                # A preference stated across two episodes -> promoted into the profile.
                for _ in range(2):
                    await memory.add(
                        MemoryContent(
                            content="User wants reports in markdown.",
                            mime_type=MemoryMimeType.TEXT,
                            metadata={"kind": "profile", "key": "report_format"},
                        )
                    )


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance.
        promotion_threshold: Recurrences after which a keyed episodic candidate is
            promoted into the profile as a stable fact.
        max_episodic: Bound on retained episodic traces (oldest dropped first).
        recent_episodic_in_context: Number of recent episodic traces ``update_context``
            injects into the model context.
        query_k: Maximum number of records ``query`` returns.
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.LongitudinalMemory"
    component_config_schema = LongitudinalMemoryConfig

    def __init__(
        self,
        name: str | None = None,
        promotion_threshold: int = 2,
        max_episodic: int = 64,
        recent_episodic_in_context: int = 3,
        query_k: int = 5,
    ) -> None:
        self._name = name or "default_longitudinal_memory"
        self._promotion_threshold = promotion_threshold
        self._max_episodic = max_episodic
        self._recent_episodic_in_context = recent_episodic_in_context
        self._query_k = query_k

        # Governed shared tier: safety rules + reusable knowledge.
        self._shared: List[MemoryContent] = []
        # Private longitudinal tiers.
        self._profile: Dict[str, MemoryContent] = {}
        self._procedures: Dict[str, MemoryContent] = {}
        self._episodic: List[MemoryContent] = []
        # Recurrence counts for keyed candidates awaiting promotion.
        self._candidates: Dict[str, int] = {}
        # Most recent induction outcomes, in insertion order (for inspection / tests).
        self.last_induction_outcomes: List[InductionOutcome] = []

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._name

    @property
    def shared(self) -> List[MemoryContent]:
        """Governed shared records (rules and knowledge)."""
        return list(self._shared)

    @property
    def profile(self) -> List[MemoryContent]:
        """Consolidated private profile facts."""
        return list(self._profile.values())

    @property
    def procedures(self) -> List[MemoryContent]:
        """Reusable private procedures."""
        return list(self._procedures.values())

    @property
    def episodic(self) -> List[MemoryContent]:
        """Retained private episodic traces (most-recent last)."""
        return list(self._episodic)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Ingest an episode and run induction over it.

        The induction outcome (one of :class:`InductionOutcome`) is appended to
        :attr:`last_induction_outcomes`. Excluded episodes are never stored.
        """
        _ = cancellation_token
        outcome = self._induct(content)
        self.last_induction_outcomes.append(outcome)

    def _induct(self, content: MemoryContent) -> InductionOutcome:
        meta: Dict[str, Any] = content.metadata or {}
        if meta.get("exclude"):
            return InductionOutcome.EXCLUDE

        tier = self._tier(meta)
        kind = self._kind(meta, tier)

        if tier is MemoryTier.SHARED:
            self._shared.append(content)
            return InductionOutcome.ADD_SHARED

        if kind is LongitudinalKind.PROCEDURE:
            key = str(meta.get("key") or self._text(content))
            # Revise an existing procedure in place, otherwise create it.
            self._procedures[key] = content
            return InductionOutcome.REVISE_PROCEDURE

        if kind is LongitudinalKind.PROFILE:
            key = str(meta.get("key") or self._text(content))
            if meta.get("confirmed"):
                self._promote(key, content)
                return InductionOutcome.UPDATE_PROFILE
            self._candidates[key] = self._candidates.get(key, 0) + 1
            if self._candidates[key] >= self._promotion_threshold:
                self._promote(key, content)
                return InductionOutcome.UPDATE_PROFILE
            self._push_episodic(content)
            return InductionOutcome.KEEP_EPISODIC

        # Default: a one-off episodic trace.
        self._push_episodic(content)
        return InductionOutcome.KEEP_EPISODIC

    def _promote(self, key: str, content: MemoryContent) -> None:
        # The latest formulation of a stable fact wins; the candidate is consumed.
        self._profile[key] = content
        self._candidates.pop(key, None)
        # Drop promoted candidates from the episodic buffer to keep context lean.
        self._episodic = [c for c in self._episodic if str((c.metadata or {}).get("key")) != key]

    def _push_episodic(self, content: MemoryContent) -> None:
        self._episodic.append(content)
        if len(self._episodic) > self._max_episodic:
            self._episodic = self._episodic[-self._max_episodic :]

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve the top records across all tiers by token-overlap with the query."""
        _ = cancellation_token, kwargs
        terms = _tokenize(self._as_text(query))
        if not terms:
            return MemoryQueryResult(results=[])
        scored = sorted(self._all_retrievable(), key=lambda c: self._score(terms, c), reverse=True)
        return MemoryQueryResult(results=scored[: self._query_k])

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject a compact, governed, layered summary into the model context.

        Only consolidated profile facts, procedures, the shared rules and a bounded
        recent episodic window are surfaced — not the full history — which is the
        reduced prompt-side context exposure HealthClaw targets.
        """
        records: List[MemoryContent] = []
        sections: List[str] = []

        if self._shared:
            sections.append("Shared rules / knowledge (always govern responses):\n" + self._bullets(self._shared))
            records.extend(self._shared)
        if self._profile:
            profile = list(self._profile.values())
            sections.append("Consolidated profile facts:\n" + self._bullets(profile))
            records.extend(profile)
        if self._procedures:
            procedures = list(self._procedures.values())
            sections.append("Reusable procedures:\n" + self._bullets(procedures))
            records.extend(procedures)
        recent = self._episodic[-self._recent_episodic_in_context :] if self._recent_episodic_in_context else []
        if recent:
            sections.append("Recent episodic traces:\n" + self._bullets(recent))
            records.extend(recent)

        if not sections:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        await model_context.add_message(SystemMessage(content="\n\n".join(sections) + "\n"))
        return UpdateContextResult(memories=MemoryQueryResult(results=records))

    async def clear(self) -> None:
        """Clear every tier and induction state."""
        self._shared.clear()
        self._profile.clear()
        self._procedures.clear()
        self._episodic.clear()
        self._candidates.clear()
        self.last_induction_outcomes.clear()

    async def close(self) -> None:
        """No external resources to release."""
        pass

    @classmethod
    def _from_config(cls, config: LongitudinalMemoryConfig) -> Self:
        return cls(
            name=config.name,
            promotion_threshold=config.promotion_threshold,
            max_episodic=config.max_episodic,
            recent_episodic_in_context=config.recent_episodic_in_context,
            query_k=config.query_k,
        )

    def _to_config(self) -> LongitudinalMemoryConfig:
        return LongitudinalMemoryConfig(
            name=self._name,
            promotion_threshold=self._promotion_threshold,
            max_episodic=self._max_episodic,
            recent_episodic_in_context=self._recent_episodic_in_context,
            query_k=self._query_k,
        )

    # -- helpers -----------------------------------------------------------------

    def _all_retrievable(self) -> List[MemoryContent]:
        return [*self._shared, *self._profile.values(), *self._procedures.values(), *self._episodic]

    def _score(self, query_terms: Set[str], content: MemoryContent) -> float:
        text_terms = _tokenize(self._text(content))
        if not text_terms or not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)

    def _tier(self, meta: Dict[str, Any]) -> MemoryTier:
        raw = str(meta.get("tier", "private")).lower()
        return MemoryTier.SHARED if raw == "shared" else MemoryTier.PRIVATE

    def _kind(self, meta: Dict[str, Any], tier: MemoryTier) -> LongitudinalKind:
        raw = str(meta.get("kind", "")).lower()
        for kind in LongitudinalKind:
            if kind.value == raw:
                # Shared kinds only valid in the shared tier; private kinds only in private.
                if (kind in _SHARED_KINDS) == (tier is MemoryTier.SHARED):
                    return kind
        return LongitudinalKind.RULE if tier is MemoryTier.SHARED else LongitudinalKind.EPISODIC

    def _text(self, content: MemoryContent) -> str:
        return content.content if isinstance(content.content, str) else str(content.content)

    def _as_text(self, query: str | MemoryContent) -> str:
        return query if isinstance(query, str) else self._text(query)

    def _bullets(self, items: List[MemoryContent]) -> str:
        return "\n".join(f"- {self._text(item)}" for item in items)

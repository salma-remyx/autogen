"""Type-conditioned temporal-decay memory backend.

Adapted from "Caching for the Future: Scrub Jay Episodic Memory Principles
for Agent Memory Systems" (arXiv:2608.04746). The paper operationalizes a
property of western scrub-jay episodic memory -- per-memory, type-conditioned
temporal decay -- as a per-memory perishability coefficient in an LLM-agent
memory store, so that perishable facts decay out of retrieved context while
durable facts persist.

This is a **Mode 2 (adapted port)**: the core mechanism -- every memory bound
to a timestamp and weighted at retrieval by a type-conditioned exponential
decay -- is retained at full fidelity. The paper's auxiliary components are
substituted with target-native equivalents:

* The paper estimates each memory's perishability with an auto-classified
  coefficient produced by ``O(1)`` LLM calls. We replace that learned estimator
  with a **parameter-free, rule-based perishability classifier** (content
  keyword overlap plus an explicit ``memory_type`` metadata field), so the
  backend has no model dependency and runs fully offline.
* The paper's query-adaptive retrieval uses an embedding model. autogen-ext
  ships no default embedding dependency, so we use a **parameter-free
  token-overlap relevance score**, combined multiplicatively with the temporal
  validity weight -- the same ``relevance * decay`` score shape.
* The Temporal Generalization Test (TGT) benchmark / Generalization Gap
  metric and the retroactive ``O(1)``-LLM-call revision pass are out of scope:
  benchmark evaluation and revision belong in downstream work.

The ``decay_enabled`` toggle reproduces the paper's central ablation, which
collapses the temporal-reasoning gain when type-conditioned decay is removed.
"""

import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self


class PerishabilityProfile(BaseModel):
    """Per-type perishability for a memory class.

    The validity weight of a memory that has aged ``a`` seconds is
    ``exp(-pi * a / tau)``.

    * ``pi`` in ``[0, 1]`` is the perishability / decay rate. Higher means
      validity drops faster as the memory ages. ``pi = 0`` makes a memory
      durable (facts), matching the paper's finding that decay harms
      fact-consolidation tasks.
    * ``tau`` > 0 is the utility horizon in seconds: the wall-clock timescale
      over which age is measured. At ``age == tau`` the validity weight is
      ``exp(-pi)``.
    """

    pi: float = Field(default=0.5, ge=0.0, le=1.0)
    tau: float = Field(default=3600.0, gt=0.0)


# Default per-type perishability taxonomy. Higher pi / lower tau == more
# perishable. "fact"/"identity" use pi=0 so durable knowledge is never decayed.
DEFAULT_PROFILES: Dict[str, PerishabilityProfile] = {
    "current_status": PerishabilityProfile(pi=1.0, tau=1800.0),
    "status": PerishabilityProfile(pi=1.0, tau=1800.0),
    "state": PerishabilityProfile(pi=0.9, tau=3600.0),
    "observation": PerishabilityProfile(pi=0.7, tau=7200.0),
    "event": PerishabilityProfile(pi=0.6, tau=21600.0),
    "plan": PerishabilityProfile(pi=0.4, tau=86400.0),
    "task": PerishabilityProfile(pi=0.4, tau=86400.0),
    "preference": PerishabilityProfile(pi=0.1, tau=2_592_000.0),
    "identity": PerishabilityProfile(pi=0.0, tau=31_536_000.0),
    "fact": PerishabilityProfile(pi=0.0, tau=31_536_000.0),
}

# Parameter-free perishability classifier (proxy for the paper's LLM estimator).
# Order matters only for tie-breaking; the most specific hints win by hit count.
_KEYWORD_HINTS: Dict[str, tuple[str, ...]] = {
    "current_status": ("current status", "currently", "right now", "is now", "status:", "in progress", "pending"),
    "status": ("status update", "standup", "scrum", "as of"),
    "state": ("state is", "the system is", "service is", "is online", "is offline", "deployed"),
    "observation": ("observed", "detected", "found that", "noticed", "measured"),
    "event": ("happened", "occurred", "took place", "met with", "attended"),
    "plan": ("plan:", "planned", "milestone", "roadmap", "next sprint", "aim to"),
    "task": ("task:", "todo", "to-do", "need to", "assignee", "ticket"),
    "preference": ("prefers", "likes", "dislikes", "wants", "favorite", "always"),
    "identity": ("name is", "my name", "user is", "located in", "lives in", "works at"),
    "fact": ("the capital", "is the largest", "is defined as", "stands for", "is a"),
}

_STOPWORDS = frozenset(
    {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in", "on", "for", "with", "that", "this", "it", "as", "by", "at", "be", "has", "have", "i", "you", "we", "they"}
)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1 and tok not in _STOPWORDS}


def _classify_type(text: str) -> str:
    """Map free text to a memory type by keyword hit count (no model call)."""
    lowered = text.lower()
    best_type = ""
    best_hits = 0
    for mem_type, hints in _KEYWORD_HINTS.items():
        hits = sum(1 for hint in hints if hint in lowered)
        if hits > best_hits:
            best_hits = hits
            best_type = mem_type
    return best_type


def _validity(profile: PerishabilityProfile, age_seconds: float) -> float:
    """Type-conditioned temporal-decay weight in (0.0, 1.0]."""
    if profile.pi <= 0.0 or age_seconds <= 0.0:
        return 1.0
    return math.exp(-(profile.pi * age_seconds) / profile.tau)


def _relevance(query_tokens: set[str], content_tokens: set[str]) -> float:
    """Token-overlap relevance (Jaccard) between a query and a memory."""
    if not query_tokens or not content_tokens:
        return 0.0
    intersection = len(query_tokens & content_tokens)
    if intersection == 0:
        return 0.0
    return intersection / len(query_tokens | content_tokens)


@dataclass(slots=True)
class _Record:
    content: MemoryContent
    stored_at: float
    profile: PerishabilityProfile
    memory_type: str


class PerishableMemoryConfig(BaseModel):
    """Declarative configuration for :class:`PerishableMemory`."""

    name: str | None = None
    k: int = Field(default=3, ge=1)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    decay_enabled: bool = True
    profiles: Dict[str, PerishabilityProfile] = Field(default_factory=dict)
    default_profile: PerishabilityProfile = Field(default_factory=lambda: PerishabilityProfile(pi=0.5, tau=21600.0))


class PerishableMemory(Memory, Component[PerishableMemoryConfig]):
    """In-memory backend that retrieves by query-adaptive, type-conditioned temporal decay.

    Each added memory is bound to a timestamp (the "when" of the paper's
    What-Where-When tuple) and a per-type perishability profile. At retrieval,
    the score is ``relevance * validity`` where ``relevance`` is token overlap
    with the query and ``validity = exp(-pi * age / tau)`` decays with age.
    Durable memories (``pi = 0``, e.g. facts) never decay; perishable ones
    (e.g. current status) drop out of context as they age.

    Set ``decay_enabled=False`` to reproduce the paper's decay ablation, which
    flattens the temporal ordering back to relevance-only retrieval.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory import PerishableMemory


            async def main() -> None:
                memory = PerishableMemory(k=3)
                await memory.add(
                    MemoryContent(content="The deploy is currently green.", mime_type=MemoryMimeType.TEXT)
                )
                results = await memory.query("deploy status")
                print(results.results[0].metadata["validity"])


            asyncio.run(main())

    Args:
        name: Optional identifier for this memory instance.
        k: Maximum number of memories to return per query.
        score_threshold: Minimum ``relevance * validity`` score to keep.
        decay_enabled: If ``False``, disable type-conditioned decay (ablation).
        profiles: Per-type perishability overrides, merged over the defaults.
        default_profile: Profile used when a memory's type cannot be inferred.
        clock: Callable returning the current time in seconds (testable).
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.PerishableMemory"
    component_config_schema = PerishableMemoryConfig

    def __init__(
        self,
        name: str | None = None,
        *,
        k: int = 3,
        score_threshold: float = 0.0,
        decay_enabled: bool = True,
        profiles: Mapping[str, PerishabilityProfile] | None = None,
        default_profile: PerishabilityProfile | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._name = name or "perishable_memory"
        self._k = k
        self._score_threshold = score_threshold
        self._decay_enabled = decay_enabled
        self._user_profiles: Dict[str, PerishabilityProfile] = dict(profiles) if profiles else {}
        self._profiles: Dict[str, PerishabilityProfile] = {**DEFAULT_PROFILES, **self._user_profiles}
        self._default_profile = default_profile or PerishableMemoryConfig().default_profile
        self._clock = clock or time.time
        self._records: List[_Record] = []

    @property
    def name(self) -> str:
        """Get the memory instance identifier."""
        return self._name

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, MemoryContent):
            inner = content.content
            if isinstance(inner, str):
                return inner
            if isinstance(inner, dict):
                return json.dumps(inner, sort_keys=True)
            return str(inner)
        if isinstance(content, str):
            return content
        return str(content)

    def _resolve_profile(self, content: MemoryContent, text: str) -> tuple[PerishabilityProfile, str]:
        """Resolve the perishability profile for a memory (proxy for the paper's LLM estimator)."""
        metadata = content.metadata or {}
        # Explicit per-memory pi/tau override wins over type-based lookup.
        if "pi" in metadata or "tau" in metadata:
            pi = min(1.0, max(0.0, float(metadata.get("pi", self._default_profile.pi))))
            tau = max(1e-6, float(metadata.get("tau", self._default_profile.tau)))
            return PerishabilityProfile(pi=pi, tau=tau), str(metadata.get("memory_type") or "custom")
        memory_type = metadata.get("memory_type")
        if isinstance(memory_type, str) and memory_type in self._profiles:
            return self._profiles[memory_type], memory_type
        classified = _classify_type(text)
        if classified:
            return self._profiles[classified], classified
        return self._default_profile, "default"

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add a memory, binding it to a timestamp and a perishability profile."""
        _ = cancellation_token
        text = self._extract_text(content)
        profile, memory_type = self._resolve_profile(content, text)
        metadata = content.metadata or {}
        stored_at = float(metadata["stored_at"]) if isinstance(metadata.get("stored_at"), (int, float)) else self._clock()
        self._records.append(_Record(content=content, stored_at=stored_at, profile=profile, memory_type=memory_type))

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return memories ranked by query-adaptive, recency-adjusted score."""
        _ = cancellation_token, kwargs
        query_tokens = _tokenize(self._extract_text(query))
        now = self._clock()
        scored: List[tuple[float, _Record, float]] = []
        for record in self._records:
            relevance = _relevance(query_tokens, _tokenize(self._extract_text(record.content)))
            if relevance <= 0.0:
                continue
            age = max(0.0, now - record.stored_at)
            validity = _validity(record.profile, age) if self._decay_enabled else 1.0
            score = relevance * validity
            if score < self._score_threshold:
                continue
            scored.append((score, record, validity))
        scored.sort(key=lambda item: item[0], reverse=True)

        results: List[MemoryContent] = []
        for score, record, validity in scored[: self._k]:
            metadata = dict(record.content.metadata or {})
            metadata["score"] = score
            metadata["validity"] = validity
            metadata["memory_type"] = record.memory_type
            metadata["stored_at"] = record.stored_at
            metadata["age_seconds"] = max(0.0, now - record.stored_at)
            results.append(MemoryContent(content=record.content.content, mime_type=record.content.mime_type, metadata=metadata))
        return MemoryQueryResult(results=results)

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Inject the most relevant, freshest memories into the model context."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)
        results = await self.query(query_text)
        if results.results:
            lines = [f"{i}. {self._extract_text(memory)}" for i, memory in enumerate(results.results, 1)]
            memory_context = (
                "\nRelevant memory content (ranked by recency-adjusted relevance):\n" + "\n".join(lines) + "\n"
            )
            await model_context.add_message(SystemMessage(content=memory_context))
        return UpdateContextResult(memories=results)

    async def clear(self) -> None:
        """Remove all stored memories."""
        self._records = []

    async def close(self) -> None:
        """No external resources to release."""

    def _to_config(self) -> PerishableMemoryConfig:
        return PerishableMemoryConfig(
            name=self._name,
            k=self._k,
            score_threshold=self._score_threshold,
            decay_enabled=self._decay_enabled,
            profiles=self._user_profiles,
            default_profile=self._default_profile,
        )

    @classmethod
    def _from_config(cls, config: PerishableMemoryConfig) -> Self:
        return cls(
            name=config.name,
            k=config.k,
            score_threshold=config.score_threshold,
            decay_enabled=config.decay_enabled,
            profiles=config.profiles,
            default_profile=config.default_profile,
        )

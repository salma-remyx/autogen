"""Hierarchical graph memory with path-level localization and coordinated rewriting.

This module is an *adapted port* (Mode 2) of HiGram -- *Hierarchical Graph Memory
for LLM Agents with Path-level Localization and Rewrite* (arxiv:2608.05095).

Kept at full fidelity (the paper's core mechanism):
  * A **hierarchical graph memory** -- coarse-to-fine, with upper-level *clusters*
    grouping fine-grained *MemoryUnits*, plus directed inter-unit dependency edges.
  * **MicroGraph path-level localization** -- both :meth:`add` and :meth:`query`
    first localize to a support subgraph (best cluster(s) -> best unit(s)) and
    operate only on that evidence path, rather than over the whole flat store.
  * **Coordinated rewriting** -- on a conflicting add, the localized unit is
    rewritten *and* its dependent units are propagated, so intra-unit memory and
    inter-unit dependencies are revised jointly (the conflict-aware update).

Substituted with target-native equivalents (the auxiliary components):
  * The paper's learned query/update-conditioned estimator is replaced by a
    **parameter-free lexical-overlap similarity** (tokenized overlap coefficient).
    The localization *signal* (rank-then-narrow) is preserved; only the scorer is.
  * The paper's LLM-driven unit rewriter is replaced by a **deterministic merge
    policy** (augment-on-superset, overwrite-otherwise). The coordinated
    *structure* update is preserved; only the text-rewrite oracle is substituted.
  * The paper's separate benchmark/eval framework is intentionally cut --
    evaluation belongs in a downstream PR.

The class conforms to :class:`autogen_core.memory.Memory`, so any
:class:`~autogen_agentchat.agents.AssistantAgent` that accepts a ``Memory``
consumes it through ``add`` / ``query`` / ``update_context`` without further
wiring.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from typing_extensions import Self

from ._configs import HierarchicalGraphMemoryConfig

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Grammatical function words only (articles, prepositions, conjunctions, pronouns,
# auxiliaries, determiners) -- removed so the overlap signal rides on content words.
_STOPWORDS = frozenset(
    (
        "a an the and or but if so than of to in on at by with from into over under about "
        "is are was were be been being do does did has have had will would can could should "
        "shall may might must it its he she they them their there here we us our you your i me "
        "my his her this that these those what which who whom whose when where why how all any "
        "both each few more most other some such only own same not no very too as"
    ).split()
)


def _tokenize(text: str) -> Set[str]:
    """Lowercase alphanumeric tokens with function words removed (overlap-proxy vocabulary)."""
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) > 1 and tok not in _STOPWORDS}


def _overlap(a: Set[str], b: Set[str]) -> float:
    """Overlap-coefficient similarity in ``[0, 1]``.

    Favors subset matches, which is what localization needs: a query whose
    vocabulary is contained in a cluster's representative keywords scores high.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class _Unit:
    """A fine-grained MemoryUnit (a leaf of the hierarchical graph)."""

    unit_id: str
    cluster_id: str
    text: str
    tokens: Set[str]
    revision: int = 1
    superseded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Cluster:
    """An upper-level node grouping related MemoryUnits."""

    cluster_id: str
    name: str
    keywords: Set[str]
    unit_ids: List[str] = field(default_factory=list)


class HierarchicalGraphMemory(Memory, Component[HierarchicalGraphMemoryConfig]):
    """In-process hierarchical graph memory with path-level localization and coordinated rewriting.

    ``HierarchicalGraphMemory`` organizes content into a coarse-to-fine graph:
    upper-level :class:`_Cluster` nodes group fine-grained memory units, and units
    carry directed dependency edges. Both ingestion (:meth:`add`) and retrieval
    (:meth:`query`) first localize to a support subgraph and operate only on that
    evidence path, so retrieval returns the relevant slice instead of the whole
    flat store. When an added fact conflicts with an existing unit, the unit is
    rewritten *and* its dependents are propagated -- the conflict-aware
    coordinated update.

    This is an adapted port of HiGram (arxiv:2608.05095); see the module docstring
    for the exact fidelity / substitution boundaries.

    Args:
        config: Optional :class:`HierarchicalGraphMemoryConfig`. Defaults to a
            sensible configuration when ``None``.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.hierarchical_graph import HierarchicalGraphMemory


            async def main() -> None:
                memory = HierarchicalGraphMemory()
                await memory.add(MemoryContent(content="The user lives in Paris.", mime_type=MemoryMimeType.TEXT))
                results = await memory.query("Where does the user live?")
                print(results.results)


            asyncio.run(main())

    """

    component_config_schema = HierarchicalGraphMemoryConfig
    component_provider_override = "autogen_ext.memory.hierarchical_graph.HierarchicalGraphMemory"

    def __init__(self, config: HierarchicalGraphMemoryConfig | None = None) -> None:
        self._config = config or HierarchicalGraphMemoryConfig()
        self._name = self._config.name or "hierarchical_graph_memory"
        self._clusters: Dict[str, _Cluster] = {}
        self._units: Dict[str, _Unit] = {}
        # unit_id -> set of unit_ids that depend on it (children). When a unit is
        # rewritten, its dependents are propagated (marked superseded).
        self._dependents: Dict[str, Set[str]] = {}

    # -- text extraction -------------------------------------------------
    def _extract_text(self, content_item: str | MemoryContent) -> str:
        """Extract searchable text from a query or memory content item."""
        if isinstance(content_item, str):
            return content_item
        mime_type = content_item.mime_type
        if mime_type == MemoryMimeType.IMAGE:
            raise ValueError("Image content cannot be localized by the lexical-overlap proxy")
        content = content_item.content
        if mime_type in (MemoryMimeType.TEXT, MemoryMimeType.MARKDOWN):
            return str(content)
        if mime_type == MemoryMimeType.JSON and isinstance(content, dict):
            return str(content).lower()
        return str(content)

    # -- localization (MicroGraph support subgraph) ----------------------
    def _rank_clusters(self, tokens: Set[str]) -> List[Tuple[str, float]]:
        """Return clusters ranked by overlap with the query vocabulary."""
        scored = [(cid, _overlap(tokens, cluster.keywords)) for cid, cluster in self._clusters.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _best_unit_in(self, cluster: _Cluster, tokens: Set[str]) -> Tuple[str | None, float]:
        """Return the highest-overlap unit id in ``cluster`` (and its score), if any."""
        best_id: str | None = None
        best_score = 0.0
        for uid in cluster.unit_ids:
            unit = self._units.get(uid)
            if unit is None:
                continue
            score = _overlap(tokens, unit.tokens)
            if score > best_score:
                best_score = score
                best_id = uid
        return best_id, best_score

    # -- coordinated rewrite (add) ---------------------------------------
    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add content via path-level localization + coordinated rewriting.

        Localizes to a support cluster (creating one if none supports the content),
        then either rewrites a conflicting unit in place (propagating to its
        dependents) or appends a new unit -- linking it as a dependent of a related
        unit when one exists.
        """
        text = self._extract_text(content)
        tokens = _tokenize(text)
        meta = dict(content.metadata or {})

        # 1) Localize to a support cluster (path-level localization).
        ranked = self._rank_clusters(tokens)
        best_cid: str | None = None
        best_cscore = 0.0
        if ranked:
            best_cid, best_cscore = ranked[0]
        if best_cid is None or best_cscore < self._config.cluster_score_threshold:
            best_cid = self._new_cluster(meta.get("cluster"), tokens)

        cluster = self._clusters[best_cid]

        # 2) Localize to a candidate unit within the cluster.
        cand_id, cand_score = self._best_unit_in(cluster, tokens)

        if cand_id is not None and cand_score >= self._config.conflict_score_threshold:
            # 3a) Coordinated rewrite: revise the localized unit + propagate to dependents.
            logger.debug("Coordinated rewrite of unit %s (overlap=%.3f)", cand_id, cand_score)
            self._rewrite_unit(cand_id, text, tokens, meta)
        else:
            # 3b) Append a new unit. If a related (non-conflicting) unit exists,
            #     record an inter-unit dependency edge (the new unit depends on it).
            new_id = self._append_unit(best_cid, text, tokens, meta)
            if cand_id is not None and cand_score > 0.0:
                self._link_dependency(parent_id=cand_id, child_id=new_id)

    def _new_cluster(self, name: Any, keywords: Set[str]) -> str:
        cluster_id = f"c_{uuid.uuid4().hex[:8]}"
        self._clusters[cluster_id] = _Cluster(
            cluster_id=cluster_id,
            name=str(name) if name else f"cluster_{len(self._clusters)}",
            keywords=set(keywords),
        )
        return cluster_id

    def _append_unit(self, cluster_id: str, text: str, tokens: Set[str], meta: Dict[str, Any]) -> str:
        unit_id = f"u_{uuid.uuid4().hex[:10]}"
        self._units[unit_id] = _Unit(
            unit_id=unit_id,
            cluster_id=cluster_id,
            text=text,
            tokens=set(tokens),
            metadata=meta,
        )
        self._clusters[cluster_id].unit_ids.append(unit_id)
        # Absorb the unit's vocabulary into the cluster's representative keywords.
        self._clusters[cluster_id].keywords |= tokens
        return unit_id

    def _rewrite_unit(self, unit_id: str, text: str, tokens: Set[str], meta: Dict[str, Any]) -> None:
        """Jointly revise intra-unit memory and inter-unit dependencies (deterministic policy)."""
        unit = self._units[unit_id]
        # Substitute for the paper's LLM rewriter:
        #   incoming vocabulary is a strict superset -> augment (new fact adds information)
        #   otherwise                               -> overwrite (new fact corrects/refines)
        if tokens > unit.tokens:
            unit.text = f"{unit.text} ;; {text}"
            unit.tokens |= tokens
        else:
            unit.text = text
            unit.tokens = set(tokens)
        unit.revision += 1
        unit.metadata.update(meta)
        # Inter-unit dependency propagation: dependents of a rewritten unit go stale.
        for dep_id in self._dependents.get(unit_id, set()):
            dependent = self._units.get(dep_id)
            if dependent is not None:
                dependent.superseded = True
                dependent.metadata["superseded_by_parent_revision"] = unit.revision

    def _link_dependency(self, parent_id: str, child_id: str) -> None:
        """Record that ``child_id`` depends on ``parent_id`` (builds the inter-unit graph)."""
        self._dependents.setdefault(parent_id, set()).add(child_id)

    # -- path-level retrieval (query) ------------------------------------
    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Retrieve the localized evidence path for ``query``.

        Narrows to the top overlapping clusters, then the top overlapping units
        within each -- returning only the support subgraph rather than the whole
        store. Stale (superseded) units are dropped for valid-evidence selection.
        """
        tokens = _tokenize(self._extract_text(query))
        results: List[MemoryContent] = []
        if not tokens:
            return MemoryQueryResult(results=results)

        ranked = self._rank_clusters(tokens)
        top_clusters = [
            (cid, cscore)
            for cid, cscore in ranked[: self._config.top_k_clusters]
            if cscore >= self._config.cluster_score_threshold
        ]
        unit_threshold = self._config.unit_score_threshold
        for cid, cscore in top_clusters:
            cluster = self._clusters[cid]
            scored_units: List[Tuple[str, float]] = []
            for uid in cluster.unit_ids:
                unit = self._units[uid]
                if unit.superseded:  # valid-evidence selection: drop stale units
                    continue
                score = _overlap(tokens, unit.tokens)
                if score >= unit_threshold:
                    scored_units.append((uid, score))
            scored_units.sort(key=lambda item: item[1], reverse=True)
            for uid, uscore in scored_units[: self._config.top_k_units]:
                results.append(self._unit_to_content(uid, cluster.name, cscore, uscore))
        return MemoryQueryResult(results=results)

    def _unit_to_content(
        self, unit_id: str, cluster_name: str, cluster_score: float, unit_score: float
    ) -> MemoryContent:
        unit = self._units[unit_id]
        meta = dict(unit.metadata)
        meta.update(
            {
                "cluster": cluster_name,
                "cluster_id": unit.cluster_id,
                "unit_id": unit.unit_id,
                "revision": unit.revision,
                "score": unit_score,
                "cluster_score": cluster_score,
            }
        )
        return MemoryContent(content=unit.text, mime_type=MemoryMimeType.TEXT, metadata=meta)

    # -- context injection (update_context) ------------------------------
    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject the localized evidence path for the last user turn into ``model_context``."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        last = messages[-1]
        query_text = last.content if isinstance(last.content, str) else str(last)
        query_results = await self.query(query_text)
        if query_results.results:
            lines = [f"{i}. {str(memory.content)}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memory (hierarchical evidence path):\n" + "\n".join(lines)
            await model_context.add_message(SystemMessage(content=memory_context))
        return UpdateContextResult(memories=query_results)

    # -- lifecycle -------------------------------------------------------
    async def clear(self) -> None:
        """Clear all clusters, units, and dependency edges."""
        self._clusters.clear()
        self._units.clear()
        self._dependents.clear()

    async def close(self) -> None:
        """No external resources are held; nothing to clean up."""
        pass

    # -- introspection ---------------------------------------------------
    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._name

    def num_clusters(self) -> int:
        """Number of upper-level clusters."""
        return len(self._clusters)

    def num_units(self) -> int:
        """Number of MemoryUnits (including superseded ones)."""
        return len(self._units)

    def num_active_units(self) -> int:
        """Number of MemoryUnits that are not superseded (the queryable set)."""
        return sum(1 for unit in self._units.values() if not unit.superseded)

    # -- Component protocol ----------------------------------------------
    def _to_config(self) -> HierarchicalGraphMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: HierarchicalGraphMemoryConfig) -> Self:
        return cls(config=config)

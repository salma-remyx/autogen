"""Hybrid retrieval-augmented memory with a corrective agentic loop.

This module implements :class:`HybridRAGMemory`, a :class:`~autogen_core.memory.Memory`
backend that fuses three retrieval channels — dense, sparse (BM25), and a
knowledge-graph (co-occurrence) channel — with query-type-adaptive reciprocal rank
fusion (RRF), a cross-encoder/graph-proximity reranker, and a self-critique ->
re-retrieve corrective loop. Any agent that accepts a ``Memory``
(e.g. ``AssistantAgent(memory=[...])``) consumes it with no plumbing change.

Adapted (Mode 2 port) from:

    APS-RAG: "A corrective agentic hybrid RAG and an operations-grounded evaluation
    for a scientific facility" (arXiv:2607.24663).

The paper's *core mechanism* is preserved at full fidelity: multi-channel
adaptive-RRF fusion, a dedicated rerank stage, and a corrective re-retrieval loop.
The paper's *auxiliary* components are replaced with parameter-free, dependency-light
equivalents so the memory runs in-process with no external services:

* learned dense embedder (OpenAI / sentence-transformers) -> feature-hashing
  bag-of-words vectors with cosine similarity (the "hashing trick"),
* Elasticsearch sparse index -> an in-process BM25 ranker,
* Neo4j knowledge graph -> an in-process term co-occurrence graph whose
  proximity score rewards documents that connect multiple query entities,
* learned cross-encoder reranker -> a lexical-precision + graph-proximity
  relevance scorer (the paper reports this stage is the single largest
  contributor to answer quality),
* LLM self-critique -> a parameter-free sufficiency heuristic, with an optional
  pluggable async ``critic`` callable so a real LLM can drive the corrective
  loop (the paper's agentic self-correction).
"""

import hashlib
import logging
import math
import re
from collections import defaultdict
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self, TypedDict

logger = logging.getLogger(__name__)

__all__ = ["HybridRAGMemory", "HybridRAGMemoryConfig"]

# A ranked channel result is a list of (doc_index, raw_score) ordered best-first.
RankedChannel = List[Tuple[int, float]]
# An async critic inspects the current query + retrieved contents and decides
# whether the evidence is sufficient, optionally suggesting a rewritten query.
class CriticDecision(TypedDict):
    """Verdict returned by an optional async critic driving the corrective loop."""

    sufficient: bool
    rewritten_query: Optional[str]


CriticFn = Callable[[str, List[str]], Awaitable[CriticDecision]]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
# Common English stopwords used for query expansion / rewriting only.
_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "been", "being", "with", "as", "at", "by", "this", "that", "it", "from", "which",
    "what", "who", "when", "where", "why", "how", "do", "does", "did", "can", "could",
}


class QueryType(str, Enum):
    """Coarse query intent used to adapt retrieval-channel weights (paper's adaptive RRF)."""

    FACTUAL = "factual"  # lookup / wh-questions -> BM25-dominant
    RELATIONAL = "relational"  # entity-connecting -> KG-dominant
    PROCEDURAL = "procedural"  # how-to / steps -> dense-friendly
    UNKNOWN = "unknown"  # fall back to equal weights


# Per-query-type channel weights for adaptive RRF. These encode the paper's
# "query-type-adaptive reciprocal-rank fusion": different intents trust different channels.
DEFAULT_CHANNEL_WEIGHTS: Dict[QueryType, Dict[str, float]] = {
    QueryType.FACTUAL: {"dense": 1.0, "sparse": 1.2, "kg": 0.4},
    QueryType.RELATIONAL: {"dense": 0.6, "sparse": 0.8, "kg": 1.4},
    QueryType.PROCEDURAL: {"dense": 1.1, "sparse": 1.0, "kg": 0.5},
    QueryType.UNKNOWN: {"dense": 1.0, "sparse": 1.0, "kg": 1.0},
}

_RELATIONAL_CUES = {"between", "versus", "vs", "relation", "relate", "related", "connect", "link", "compare"}
_PROCEDURAL_CUES = {"how", "steps", "step", "install", "run", "configure", "setup", "procedure", "deploy", "build"}


class HybridRAGMemoryConfig(BaseModel):
    """Configuration for :class:`HybridRAGMemory`."""

    collection_name: str = "hybrid_rag"
    """Logical name for this memory collection."""

    k: int = 5
    """Number of documents to return after fusion + reranking."""

    rerank_top_n: int = 20
    """Number of fused candidates the reranker re-scores before truncating to ``k``."""

    rrf_k: int = 60
    """RRF smoothing constant (the ``k`` in ``1 / (rrf_k + rank)``)."""

    max_corrective_rounds: int = 1
    """Max number of self-critique -> re-retrieve iterations (0 disables the corrective loop)."""

    sufficiency_score: float = 0.35
    """Minimum rerank score for the top result to be deemed sufficient by the default critic."""

    sufficiency_coverage: float = 0.6
    """Minimum fraction of salient query terms covered by the top result for sufficiency."""

    score_threshold: float | None = None
    """Drop results whose rerank score falls below this; ``None`` keeps all."""

    allow_reset: bool = False
    """Whether :meth:`HybridRAGMemory.reset` is permitted."""

    dense_dim: int = 256
    """Dimensionality of the feature-hashed dense vectors."""

    bm25_k1: float = 1.5
    """BM25 term-frequency saturation parameter."""

    bm25_b: float = 0.75
    """BM25 length-normalization parameter."""

    channel_weights: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    """Optional overrides keyed by query-type name (e.g. ``{"factual": {"sparse": 1.5}}``)."""

    enabled_channels: Set[str] = Field(default_factory=lambda: {"dense", "sparse", "kg"})
    """Which channels to run. Defaults to all three."""


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t]


def _set_terms(tokens: List[str]) -> Set[str]:
    return {t for t in tokens if t not in _STOPWORDS}


class _DocIndices:
    """In-process indices backing the three retrieval channels."""

    def __init__(self, dense_dim: int, k1: float, b: float) -> None:
        self._dense_dim = dense_dim
        self._k1 = k1
        self._b = b
        self.texts: List[str] = []
        self.tokens: List[List[str]] = []
        self.term_sets: List[Set[str]] = []
        # BM25 state
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._doc_len: List[int] = []
        # Dense state (feature-hashed bag-of-words vectors)
        self._dense: List[Dict[int, int]] = []
        # KG state (term co-occurrence graph): term -> set of co-occurring terms
        self._graph: Dict[str, Set[str]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def avgdl(self) -> float:
        return (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

    def _hash_vec(self, terms: Set[str]) -> Dict[int, int]:
        vec: Dict[int, int] = defaultdict(int)
        for term in terms:
            h = int(hashlib.blake2b(term.encode("utf-8"), digest_size=4).hexdigest(), 16)
            vec[abs(h) % self._dense_dim] += 1
        return vec

    def add(self, text: str) -> int:
        idx = len(self.texts)
        toks = _tokenize(text)
        terms = _set_terms(toks)
        self.texts.append(text)
        self.tokens.append(toks)
        self.term_sets.append(terms)
        # BM25
        tf: Dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        self._tf.append(tf)
        for t in tf:
            self._df[t] += 1
        self._doc_len.append(len(toks))
        # Dense
        self._dense.append(self._hash_vec(terms))
        # KG: add co-occurrence edges among salient terms in this document
        salient = list(terms)
        for i, a in enumerate(salient):
            for bj in salient[i + 1 :]:
                self._graph[a].add(bj)
                self._graph[bj].add(a)
        return idx

    def clear(self) -> None:
        self.texts.clear()
        self.tokens.clear()
        self.term_sets.clear()
        self._tf.clear()
        self._df.clear()
        self._doc_len.clear()
        self._dense.clear()
        self._graph.clear()

    # --- channel scorers ---

    def dense_rank(self, query_terms: Set[str]) -> RankedChannel:
        qv = self._hash_vec(query_terms)
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        scored: RankedChannel = []
        for idx, dv in enumerate(self._dense):
            dot = sum(count * dv.get(slot, 0) for slot, count in qv.items())
            if dot <= 0:
                continue
            dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
            scored.append((idx, dot / (qn * dn)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def sparse_rank(self, query_terms: Set[str]) -> RankedChannel:
        n = len(self.texts)
        if n == 0:
            return []
        avgdl = self.avgdl or 1.0
        scored: RankedChannel = []
        for idx in range(n):
            s = 0.0
            tf = self._tf[idx]
            dl = self._doc_len[idx] or 1
            for t in query_terms:
                if t not in tf:
                    continue
                df = self._df.get(t, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                f = tf[t]
                s += idf * (f * (self._k1 + 1.0)) / (f + self._k1 * (1.0 - self._b + self._b * dl / avgdl))
            if s > 0:
                scored.append((idx, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def kg_rank(self, query_terms: Set[str]) -> RankedChannel:
        """Graph-proximity: reward documents that contain and connect multiple query entities."""
        qset = set(query_terms)
        scored: RankedChannel = []
        for idx, terms in enumerate(self.term_sets):
            present = terms & qset
            if not present:
                continue
            # connected query-entity pairs: co-occur in the global co-occurrence graph
            present_list = list(present)
            connected = 0
            for i, a in enumerate(present_list):
                neighbors = self._graph.get(a, set())
                for bj in present_list[i + 1 :]:
                    if bj in neighbors:
                        connected += 1
            # single-entity docs still get partial credit proportional to entity degree
            degree = sum(len(self._graph.get(a, set())) for a in present)
            score = connected + 0.15 * (degree / max(len(present), 1)) + 0.05 * len(present)
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def graph_neighbors(self, term: str) -> Set[str]:
        return self._graph.get(term, set())

    def doc_frequency(self, term: str) -> int:
        """Number of indexed documents containing ``term``."""
        return self._df.get(term, 0)


class HybridRAGMemory(Memory, Component[HybridRAGMemoryConfig]):
    """Hybrid dense+sparse+KG retrieval memory with a corrective re-retrieval loop.

    See the module docstring for the paper attribution and the Mode 2 component
    substitutions. The class implements the :class:`~autogen_core.memory.Memory`
    contract, so it is consumed by any agent that takes a ``Memory`` (for example
    ``AssistantAgent(memory=[HybridRAGMemory(...)])``) with no other wiring.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.hybrid_rag_memory import HybridRAGMemory

            async def main() -> None:
                memory = HybridRAGMemory()
                await memory.add(
                    MemoryContent(content="The APS uses a 7-GeV storage ring.", mime_type=MemoryMimeType.TEXT)
                )
                results = await memory.query("What storage ring does the APS use?")
                print([str(r.content) for r in results.results])
                await memory.close()

            asyncio.run(main())

    Args:
        config: Optional :class:`HybridRAGMemoryConfig`. Defaults are used when ``None``.
        critic: Optional async callable ``(query, retrieved_texts) -> CriticDecision`` used
            in place of the default sufficiency heuristic to drive the corrective loop.
            Supply an LLM-backed critic to reproduce the paper's agentic self-correction.
    """

    component_config_schema = HybridRAGMemoryConfig
    component_provider_override = "autogen_ext.memory.hybrid_rag_memory.HybridRAGMemory"

    def __init__(
        self,
        config: HybridRAGMemoryConfig | None = None,
        critic: CriticFn | None = None,
    ) -> None:
        self._config = config or HybridRAGMemoryConfig()
        self._critic = critic
        self._indices = _DocIndices(self._config.dense_dim, self._config.bm25_k1, self._config.bm25_b)
        self._contents: Dict[int, MemoryContent] = {}

    @property
    def collection_name(self) -> str:
        """Name of this memory collection."""
        return self._config.collection_name

    # --- query-type adaptation (paper's adaptive RRF) ---

    def _classify(self, query_tokens: List[str]) -> QueryType:
        token_set = set(query_tokens)
        if token_set & _PROCEDURAL_CUES:
            return QueryType.PROCEDURAL
        if token_set & _RELATIONAL_CUES:
            return QueryType.RELATIONAL
        # relational if multiple salient entities are mentioned
        salient = [t for t in query_tokens if t not in _STOPWORDS and len(t) > 3]
        if len(set(salient)) >= 2 and any(c in token_set for c in ("and", "between", "vs")):
            return QueryType.RELATIONAL
        if token_set & {"what", "who", "when", "where", "which"}:
            return QueryType.FACTUAL
        return QueryType.UNKNOWN

    def _weights_for(self, qtype: QueryType) -> Dict[str, float]:
        base = dict(DEFAULT_CHANNEL_WEIGHTS[qtype])
        override = self._config.channel_weights.get(qtype.value, {})
        base.update(override)
        return base

    # --- fusion + reranking ---

    @staticmethod
    def _rrf(channel_ranks: List[Tuple[RankedChannel, float]], rrf_k: int) -> Dict[int, float]:
        fused: Dict[int, float] = defaultdict(float)
        for ranks, weight in channel_ranks:
            for rank_pos, (idx, _score) in enumerate(ranks):
                fused[idx] += weight / (rrf_k + rank_pos + 1)
        return fused

    def _rerank_score(self, query_terms: Set[str], idx: int) -> float:
        """Parameter-free cross-encoder proxy: lexical precision + graph-proximity boost."""
        terms = self._indices.term_sets[idx]
        if not query_terms:
            return 0.0
        overlap = len(query_terms & terms)
        precision = overlap / len(query_terms)
        # boost for documents that connect query entities through the KG
        ql = list(query_terms)
        connect = 0
        for i, a in enumerate(ql):
            if a not in terms:
                continue
            for bj in ql[i + 1 :]:
                if bj in terms and bj in self._indices.graph_neighbors(a):
                    connect += 1
        connect_boost = 0.1 * connect / max(len(query_terms), 1)
        return precision + connect_boost

    def _rewrite_query(self, query_terms: List[str]) -> Set[str]:
        """Pseudo-relevance feedback: drop the weakest term, expand via top graph neighbor."""
        if not query_terms:
            return set()
        ranked = sorted(set(query_terms), key=lambda t: self._indices.doc_frequency(t))
        kept = set(query_terms)
        if len(ranked) > 1:
            kept.discard(ranked[0])  # drop rarest (likely noisy) term
        # expand strongest term with a graph neighbor
        strongest = max(ranked, key=lambda t: self._indices.doc_frequency(t))
        neighbors = sorted(self._indices.graph_neighbors(strongest))
        if neighbors:
            kept.add(neighbors[0])
        return kept

    def _sufficient(self, query_terms: Set[str], top_idx: int | None, top_score: float) -> bool:
        """Default parameter-free critic: is the top result relevant and covering the query?"""
        if top_idx is None:
            return False
        terms = self._indices.term_sets[top_idx]
        coverage = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
        return top_score >= self._config.sufficiency_score or coverage >= self._config.sufficiency_coverage

    async def _retrieve_round(self, query_terms: Set[str], qtype: QueryType) -> Dict[int, float]:
        weights = self._weights_for(qtype)
        channels: List[Tuple[RankedChannel, float]] = []
        enabled = self._config.enabled_channels
        if "dense" in enabled:
            channels.append((self._indices.dense_rank(query_terms), weights.get("dense", 1.0)))
        if "sparse" in enabled:
            channels.append((self._indices.sparse_rank(query_terms), weights.get("sparse", 1.0)))
        if "kg" in enabled:
            channels.append((self._indices.kg_rank(query_terms), weights.get("kg", 1.0)))
        return self._rrf(channels, self._config.rrf_k)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Run the hybrid + corrective retrieval pipeline and return ranked memories."""
        query_text = query if isinstance(query, str) else str(query.content)
        query_tokens = _tokenize(query_text)
        qtype = self._classify(query_tokens)
        query_terms = _set_terms(query_tokens) or set(query_tokens)
        original_terms = query_terms  # final rerank always scores against the original query

        fused = await self._retrieve_round(query_terms, qtype)
        rounds = 0

        while True:
            # rerank fused candidates
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            top_n = ranked[: self._config.rerank_top_n]
            reranked = sorted(
                ((idx, self._rerank_score(query_terms, idx), rrf_score) for idx, rrf_score in top_n),
                key=lambda x: x[1],
                reverse=True,
            )

            top_idx = reranked[0][0] if reranked else None
            top_score = reranked[0][1] if reranked else 0.0

            # corrective loop: self-critique sufficiency, then re-retrieve if insufficient
            should_continue = False
            if self._critic is not None:
                contents = [
                    self._contents[i].content if i in self._contents else self._indices.texts[i] for i, _, _ in reranked[: self._config.k]
                ]
                decision = await self._critic(query_text, [str(c) for c in contents])
                rewritten = decision.get("rewritten_query")
                if not decision["sufficient"] and rewritten:
                    new_tokens = _tokenize(rewritten)
                    query_terms = _set_terms(new_tokens) or set(new_tokens)
                    should_continue = True
            elif not self._sufficient(query_terms, top_idx, top_score):
                should_continue = True
                query_terms = self._rewrite_query(list(query_terms))

            if should_continue and rounds < self._config.max_corrective_rounds and query_terms:
                next_fused = await self._retrieve_round(query_terms, qtype)
                # merge rounds via RRF (dedup by doc index, keep accumulated weight)
                for idx, score in next_fused.items():
                    fused[idx] = fused.get(idx, 0.0) + score
                rounds += 1
                continue
            break

        # build final results
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        top_n = ranked[: self._config.rerank_top_n]
        reranked = sorted(
            ((idx, self._rerank_score(original_terms, idx)) for idx, _ in top_n),
            key=lambda x: x[1],
            reverse=True,
        )
        results: List[MemoryContent] = []
        for idx, score in reranked[: self._config.k]:
            if self._config.score_threshold is not None and score < self._config.score_threshold:
                continue
            base = self._contents.get(idx)
            metadata = dict(base.metadata) if base and base.metadata else {}
            metadata["score"] = score
            metadata.setdefault("id", str(idx))
            results.append(
                MemoryContent(
                    content=base.content if base else self._indices.texts[idx],
                    mime_type=base.mime_type if base else MemoryMimeType.TEXT,
                    metadata=metadata,
                )
            )
        return MemoryQueryResult(results=results)

    def _extract_text(self, content: str | MemoryContent) -> str:
        if isinstance(content, str):
            return content
        c = content.content
        if isinstance(c, (dict, list)):
            return str(c)
        return str(c)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject fused+reranked memories as a system message, mirroring the agent-runtime hook."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        last = messages[-1]
        query_text = last.content if isinstance(last.content, str) else str(last)
        query_results = await self.query(query_text)
        if query_results.results:
            lines = [f"{i}. {str(m.content)}" for i, m in enumerate(query_results.results, 1)]
            await model_context.add_message(SystemMessage(content="Relevant memory content:\n" + "\n".join(lines)))
        return UpdateContextResult(memories=query_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        text = self._extract_text(content)
        idx = self._indices.add(text)
        self._contents[idx] = content

    async def clear(self) -> None:
        self._indices.clear()
        self._contents.clear()

    async def close(self) -> None:
        """Release in-process resources."""
        self._contents.clear()

    async def reset(self) -> None:
        if not self._config.allow_reset:
            raise RuntimeError("Reset not allowed. Set allow_reset=True in config to enable.")
        await self.clear()

    def _to_config(self) -> HybridRAGMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: HybridRAGMemoryConfig) -> Self:
        return cls(config=config)

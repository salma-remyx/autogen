"""Retriever-complementary antecedent linking for an existing vector memory.

Semantic similarity recall is strong for topical recall but weak for evidence
that is *semantically distant* from the query it later helps explain: an earlier
motivation, plan, or preference rarely shares vocabulary with the event that
depends on it. This module implements the memory-reachability mechanism of
*CABLE: Extending the Reach of Memory Retrieval via Complementary Antecedent-Based
Linking and Expansion* (arXiv:2608.17911) on top of any
:class:`~autogen_core.memory.Memory` host:

- **Write time** (``add``): for each new memory, generate antecedent-oriented
  queries, retrieve prior memories with them, subtract the candidates that the
  host retriever would already recover directly, verify the remainder, and store
  the accepted links in a sparse directed graph.
- **Query time** (``query`` / ``update_context``): retrieve seeds from the host
  retriever, then expand them along the stored links to surface implicit
  supporting evidence that direct similarity misses.

Adaptations from the paper (Mode 2): the LLM-based antecedent-query generator and
the LLM link verifier are replaced with a parameter-free lexical heuristic --
entity- and proper-noun phrase overlap -- so the augmentation stays training-free
and offline. The paper's benchmark suites (LoCoMo, MA-LongMemEval) are
intentionally out of scope; evaluation belongs in a downstream change.
"""

import re
from typing import Any, Dict, List, Set

from autogen_core import CancellationToken, Component, ComponentBase
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

from .chromadb._chroma_configs import PersistentChromaDBVectorMemoryConfig

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below between
    both but by can did do does doing down during each few for from further had has have having he her here
    hers him his how i if in into is it its just me more most my no nor not now of off on once only or other
    our out over own same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why will with you
    your""".split()
)


def _significant_terms(text: str) -> Set[str]:
    """Lowercased content words for a piece of memory text."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _normalize_entity(token: str) -> str:
    """Lowercase an entity token and strip a possessive suffix."""
    lowered = token.lower()
    return lowered[:-2] if lowered.endswith("'s") else lowered


def _entity_terms(text: str) -> Set[str]:
    """Proper nouns from a piece of memory text, lowercased.

    Proper nouns carry most of the antecedent signal the paper's LLM generator
    targets (people, places, projects); extracting them lexically keeps the
    augmentation parameter-free. Sentence-initial capitals that are just
    stopwords ("The ...") are dropped.
    """
    spans: Set[str] = set()
    for match in re.finditer(r"\b(?:[A-Z][a-zA-Z0-9'-]+(?:\s+[A-Z][a-zA-Z0-9'-]+)+)\b", text):
        spans.update(_normalize_entity(part) for part in match.group(0).split())
    for word in _WORD_RE.findall(text):
        if word[0].isupper():
            spans.add(_normalize_entity(word))
    return spans - _STOPWORDS


def _score_of(content: MemoryContent) -> float:
    """Read the host-reported similarity score off a query result, if present."""
    metadata = content.metadata or {}
    score = metadata.get("score")
    return float(score) if isinstance(score, (int, float)) else 0.0


class AntecedentLinkConfig(BaseModel):
    """Configuration for :class:`AntecedentLinkMemory`."""

    name: str = "antecedent_link_memory"
    """Identifier for this memory instance."""

    expansion_depth: int = Field(default=1, ge=1, le=3, description="How many link hops to follow from each seed.")
    max_links_per_memory: int = Field(default=3, ge=0, description="Upper bound on accepted links per new memory.")
    min_link_score: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum verified link strength.")
    direct_neighborhood: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Host-similarity score at or above which a candidate is treated as "
        "already recovered by the host retriever and subtracted. Set to 1.0 to disable "
        "subtraction and link every verified candidate.",
    )
    expansion_results: int = Field(default=2, ge=0, description="How many linked memories to admit per query.")
    candidate_k: int = Field(
        default=10, ge=1, description="Candidate pool size for antecedent retrieval, independent of the host's k."
    )


class AntecedentLinkMemory(Memory, ComponentBase[AntecedentLinkConfig], Component[AntecedentLinkConfig]):
    """Wrap a host memory with a sparse graph of retriever-complementary links.

    The host memory (e.g. :class:`~autogen_ext.memory.chromadb.ChromaDBVectorMemory`)
    remains the system of record: every ``add`` is forwarded to it and every
    ``query`` still starts from its similarity search. This wrapper only adds the
    links the host cannot express -- associations that are relevant but *not*
    semantically similar -- and expands retrieved seeds along them.

    Links point from a newer memory back to the older memories it implicitly
    depends on, so expansion surfaces antecedents: earlier plans, motivations, or
    preferences that a later query never mentions in similar words.

    Example:

        .. code-block:: python

            from autogen_ext.memory.antecedent_link import AntecedentLinkMemory
            from autogen_ext.memory.chromadb import (
                ChromaDBVectorMemory,
                PersistentChromaDBVectorMemoryConfig,
            )

            host = ChromaDBVectorMemory(
                config=PersistentChromaDBVectorMemoryConfig(collection_name="long_term", k=3)
            )
            memory = AntecedentLinkMemory(host=host)

    Args:
        host: The memory to augment. Its ``query`` must populate
            ``metadata['score']`` on results for neighborhood subtraction to work;
            :class:`~autogen_ext.memory.chromadb.ChromaDBVectorMemory` does.
        config: Link-construction and expansion parameters.

    """

    component_config_schema = AntecedentLinkConfig
    component_provider_override = "autogen_ext.memory.antecedent_link.AntecedentLinkMemory"

    def __init__(self, host: Memory, config: AntecedentLinkConfig | None = None) -> None:
        self._host = host
        self._config = config or AntecedentLinkConfig()
        # Directed adjacency: newer memory id -> linked antecedent ids.
        self._links: Dict[str, Set[str]] = {}
        self._entries: Dict[str, MemoryContent] = {}

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._config.name

    @property
    def host(self) -> Memory:
        """The memory being augmented."""
        return self._host

    @property
    def links(self) -> Dict[str, Set[str]]:
        """The sparse antecedent graph, as ``{memory_id: {antecedent_id, ...}}``."""
        return {node: set(targets) for node, targets in self._links.items()}

    def _next_id(self) -> str:
        return f"al_{len(self._entries)}"

    def _antecedent_queries(self, text: str) -> List[str]:
        """Derive antecedent-oriented queries from a new memory.

        The paper prompts an LLM for queries about what *preceded* this memory;
        the lexical stand-in asks for its entities and its distinctive terms,
        which is where such references surface.
        """
        entities = sorted(_entity_terms(text))
        terms = _significant_terms(text)
        queries = [" ".join(entities)] if entities else []
        distinctive = sorted(terms - _entity_terms(" ".join(entities)))
        if distinctive:
            queries.append(" ".join(distinctive))
        return queries

    def _link_score(self, new_text: str, prior: MemoryContent) -> float:
        """Verify a candidate link: entity overlap weighted by term overlap.

        Entity matches are the evidence that two memories refer to the same
        underlying subject; term overlap alone is what the host retriever already
        captures, so it only modulates the score.
        """
        new_entities = _entity_terms(new_text)
        prior_entities = _entity_terms(str(prior.content))
        if not new_entities:
            return 0.0
        overlap = new_entities & prior_entities
        if not overlap:
            return 0.0
        new_terms = _significant_terms(new_text)
        prior_terms = _significant_terms(str(prior.content))
        denominator = len(new_terms | prior_terms) or 1
        return (len(overlap) + 0.5 * len(new_terms & prior_terms)) / denominator

    async def _candidate_search(self, query: str) -> List[MemoryContent]:
        """Retrieve antecedent candidates without the host's relevance cutoff.

        Link construction must see the memories the host would *discard* -- a
        candidate that fails the host's ``score_threshold`` is exactly the
        semantically distant evidence this wrapper exists to link. Subtraction of
        the direct neighborhood happens afterwards, against the reported scores,
        rather than by inheriting the cutoff. ``n_results`` is widened so the
        candidate pool is not truncated to the host's serving-time ``k``.

        Hosts forward ``**kwargs`` to their backing store, so an override the
        store does not recognize raises; that is caught here and retried as a
        verbatim host query rather than failing the write.
        """
        try:
            results = await self._host.query(query, n_results=self._config.candidate_k, score_threshold=0.0)
        except (TypeError, ValueError):
            results = await self._host.query(query, n_results=self._config.candidate_k)
        return list(results.results)

    async def _build_links(self, content: MemoryContent, memory_id: str) -> None:
        """Construct this memory's links: retrieve, subtract, verify, accept."""
        text = str(content.content)
        if self._config.max_links_per_memory == 0 or not self._entries:
            return

        candidates: Dict[str, MemoryContent] = {}
        # Antecedent-oriented retrieval against the host, using the derived queries.
        for query in self._antecedent_queries(text):
            for result in await self._candidate_search(query):
                candidate_id = self._entry_id(result)
                if candidate_id is None or candidate_id == memory_id:
                    continue  # Not one of ours, or the new memory retrieving itself.
                prior_result = candidates.get(candidate_id)
                # Keep whichever retrieval scored the candidate higher, so the
                # neighborhood subtraction below judges it at its best.
                if prior_result is None or _score_of(result) > _score_of(prior_result):
                    candidates[candidate_id] = result

        accepted: List[str] = []
        for candidate_id, result in candidates.items():
            # Subtract the direct semantic neighborhood: anything the host
            # retriever already recovers on its own would only duplicate it.
            if _score_of(result) >= self._config.direct_neighborhood:
                continue
            score = self._link_score(text, self._entries[candidate_id])
            if score >= self._config.min_link_score:
                accepted.append(candidate_id)

        # Keep only the strongest few links -- the graph stays sparse by design.
        accepted.sort(key=lambda cid: self._link_score(text, self._entries[cid]), reverse=True)
        self._links[memory_id] = set(accepted[: self._config.max_links_per_memory])

    def _entry_id(self, content: MemoryContent) -> str | None:
        """Invert host metadata back to a wrapper-assigned memory id."""
        metadata = content.metadata or {}
        raw = metadata.get("antecedent_link_id")
        return str(raw) if raw is not None else None

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)
        query_results = await self.query(query_text)

        if query_results.results:
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(query_results.results, 1)]
            memory_context = "\nRelevant memory content:\n" + "\n".join(memory_strings)
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=query_results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        # Tag the content so host results can be mapped back to graph nodes.
        metadata = dict(content.metadata or {})
        memory_id = self._next_id()
        metadata["antecedent_link_id"] = memory_id
        tagged = MemoryContent(content=content.content, mime_type=content.mime_type, metadata=metadata)

        await self._host.add(tagged, cancellation_token)
        self._entries[memory_id] = MemoryContent(
            content=content.content, mime_type=content.mime_type, metadata=metadata
        )

        # Links are built after the write so the host retrieves the new memory's
        # antecedents, not the new memory itself.
        await self._build_links(self._entries[memory_id], memory_id)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        # Seeds: exactly what the host retriever recovers on its own, including
        # its score_threshold -- only the links are allowed to bypass it.
        seeds = await self._host.query(query, cancellation_token, **kwargs)
        results = list(seeds.results)

        if self._config.expansion_results == 0:
            return MemoryQueryResult(results=results)

        # Expansion: follow links out of the seeds, breadth-first, without
        # revisiting. Antecedents the seeds implicitly depend on enter the result
        # set even though they are semantically distant from the query.
        frontier = [node for node in (self._entry_id(seed) for seed in seeds.results) if node is not None]
        seen: Set[str] = set(frontier)
        budget = self._config.expansion_results
        for _ in range(self._config.expansion_depth):
            next_frontier: List[str] = []
            for node in frontier:
                for neighbor in self._links.get(node, ()):
                    if neighbor in seen or neighbor not in self._entries:
                        continue
                    seen.add(neighbor)
                    expanded = self._entries[neighbor]
                    expanded_metadata = dict(expanded.metadata or {})
                    expanded_metadata["via_antecedent_link"] = True
                    results.append(
                        MemoryContent(
                            content=expanded.content,
                            mime_type=expanded.mime_type,
                            metadata=expanded_metadata,
                        )
                    )
                    budget -= 1
                    if budget <= 0:
                        return MemoryQueryResult(results=results)
                    next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return MemoryQueryResult(results=results)

    async def clear(self) -> None:
        self._links.clear()
        self._entries.clear()
        await self._host.clear()

    async def close(self) -> None:
        await self._host.close()

    def _to_config(self) -> AntecedentLinkConfig:
        """Serialize the link configuration."""
        return self._config

    @classmethod
    def _from_config(cls, config: AntecedentLinkConfig) -> Self:
        """Deserialize into a wrapper around a default persistent ChromaDB host."""
        from autogen_ext.memory.chromadb import ChromaDBVectorMemory

        return cls(
            host=ChromaDBVectorMemory(config=PersistentChromaDBVectorMemoryConfig(collection_name=config.name)),
            config=config,
        )

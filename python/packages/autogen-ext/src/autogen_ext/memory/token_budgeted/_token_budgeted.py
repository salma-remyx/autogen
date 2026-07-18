from typing import Any

from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.memory import ListMemory, Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self


class TokenBudgetedMemoryConfig(BaseModel):
    """Configuration for :class:`TokenBudgetedMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    token_budget: int = Field(default=512, ge=0)
    """Soft cap on the approximate number of tokens a single ``update_context`` call may inject. ``0`` disables the cap."""

    max_entries: int = Field(default=20, ge=1)
    """Hard cap on the number of memories injected into the context per turn."""

    dedup_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    """Word-overlap (Jaccard) similarity at or above which two candidate memories are treated as duplicates."""

    relevance_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    """Minimum query relevance (word-overlap Jaccard) for a memory to be eligible for injection. ``0.0`` keeps ranking only."""

    base_memory: ComponentModel | None = None
    """Serialized wrapped memory store. When ``None`` the layer wraps a fresh :class:`~autogen_core.memory.ListMemory`."""


class TokenBudgetedMemory(Memory, Component[TokenBudgetedMemoryConfig]):
    """Memory layer that injects a curated, token-budgeted slice of context.

    Adapted from *Memori: A Persistent Memory Layer for Efficient, Context-Aware LLM
    Agents* (arXiv:2603.19935), which treats agent memory as a data structuring problem:
    rather than injecting large volumes of raw conversation into the prompt, an
    "augmentation pipeline" selects, de-duplicates and compacts the memories that are
    relevant to the current turn.

    This component wraps any :class:`~autogen_core.memory.Memory` store and applies that
    augmentation at the :meth:`update_context` boundary -- the exact point memories are
    written into the model context. On each turn it:

    1. Derives a query from the latest message and asks the wrapped store for candidate
       memories.
    2. Scores each candidate by lexical relevance to the query.
    3. De-duplicates near-redundant candidates (the "data structuring" step).
    4. Injects only the most relevant, de-duplicated memories that fit within a
       configurable token budget -- never the full raw history.

    This directly addresses :class:`~autogen_core.memory.ListMemory`'s behaviour of
    appending every stored memory verbatim, which is the "inject raw conversation" pattern
    Memori is designed to avoid.

    Adapted port (Mode 2) -- substituted components:

    - Memori's server-side Advanced Augmentation pipeline and its learned / LLM relevance
      estimator are replaced by a parameter-free lexical-overlap (word-Jaccard) proxy.
    - Memori's tokenizer-based token counting is replaced by a whitespace word-count
      approximation (``_approx_token_count``).
    - Persistence and retrieval are delegated to the wrapped ``Memory`` store (Memori's
      storage contract maps onto autogen's ``Memory`` ABC), so no external Memori SDK or
      service is required.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
            from autogen_ext.memory.token_budgeted import TokenBudgetedMemory


            async def main() -> None:
                base = ListMemory()
                memory = TokenBudgetedMemory(base, token_budget=128, dedup_threshold=0.6)

                await memory.add(MemoryContent(content="User prefers Python.", mime_type=MemoryMimeType.TEXT))
                await memory.close()


            asyncio.run(main())

    Args:
        base_memory: The persistent store to wrap. Defaults to a fresh ``ListMemory``.
        name: Optional identifier for this memory instance.
        token_budget: Soft cap on the approximate tokens injected per turn (``0`` disables).
        max_entries: Hard cap on the number of memories injected per turn.
        dedup_threshold: Word-overlap (Jaccard) similarity at or above which two memories are duplicates.
        relevance_threshold: Minimum query relevance for a memory to be injected.
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.token_budgeted.TokenBudgetedMemory"
    component_config_schema = TokenBudgetedMemoryConfig

    def __init__(
        self,
        base_memory: Memory | None = None,
        *,
        name: str | None = None,
        token_budget: int = 512,
        max_entries: int = 20,
        dedup_threshold: float = 0.8,
        relevance_threshold: float = 0.0,
    ) -> None:
        self._base_memory: Memory = base_memory if base_memory is not None else ListMemory()
        self._name = name or "token_budgeted_memory"
        self._token_budget = token_budget
        self._max_entries = max_entries
        self._dedup_threshold = dedup_threshold
        self._relevance_threshold = relevance_threshold

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._name

    @property
    def base_memory(self) -> Memory:
        """The wrapped persistent memory store."""
        return self._base_memory

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject a curated, token-budgeted slice of relevant memories into ``model_context``.

        Args:
            model_context: The context to update. Mutated with a ``SystemMessage`` when
                relevant memories are selected.

        Returns:
            UpdateContextResult containing the memories that were injected.
        """
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message)

        candidates = (await self._base_memory.query(query_text)).results
        selected = self._augment(query_text, candidates)

        if selected:
            memory_strings = [f"{i}. {str(memory.content)}" for i, memory in enumerate(selected, 1)]
            memory_context = "\nRelevant memory content (curated):\n" + "\n".join(memory_strings) + "\n"
            await model_context.add_message(SystemMessage(content=memory_context))

        return UpdateContextResult(memories=MemoryQueryResult(results=selected))

    def _augment(self, query_text: str, candidates: list[MemoryContent]) -> list[MemoryContent]:
        """Run Memori's augmentation pipeline: rank, de-duplicate, then budget.

        Candidates are scored by lexical relevance to the query, filtered by the
        relevance threshold, de-duplicated against already-selected memories, and
        finally capped by both ``max_entries`` and ``token_budget``.
        """
        query_tokens = _tokenize(query_text)

        scored: list[tuple[float, MemoryContent, set[str]]] = []
        for memory in candidates:
            memory_tokens = _tokenize(_memory_to_text(memory))
            relevance = _jaccard(query_tokens, memory_tokens)
            if relevance >= self._relevance_threshold:
                scored.append((relevance, memory, memory_tokens))

        # Stable sort by descending relevance; ties preserve insertion order.
        scored.sort(key=lambda item: -item[0])

        selected: list[tuple[MemoryContent, set[str]]] = []
        used_tokens = 0
        for _relevance, memory, memory_tokens in scored:
            if len(selected) >= self._max_entries:
                break
            if any(_jaccard(memory_tokens, chosen_tokens) >= self._dedup_threshold for _, chosen_tokens in selected):
                continue
            memory_size = _approx_token_count(_memory_to_text(memory))
            if self._token_budget > 0 and used_tokens + memory_size > self._token_budget:
                # Greedy by relevance: stop once the next memory would overflow the budget.
                break
            selected.append((memory, memory_tokens))
            used_tokens += memory_size

        return [memory for memory, _ in selected]

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add content to the wrapped memory store."""
        await self._base_memory.add(content, cancellation_token)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query the wrapped memory store."""
        return await self._base_memory.query(query, cancellation_token, **kwargs)

    async def clear(self) -> None:
        """Clear the wrapped memory store."""
        await self._base_memory.clear()

    async def close(self) -> None:
        """Close the wrapped memory store."""
        await self._base_memory.close()

    def _to_config(self) -> TokenBudgetedMemoryConfig:
        base_model = self._base_memory.dump_component() if isinstance(self._base_memory, Component) else None
        return TokenBudgetedMemoryConfig(
            name=self._name,
            token_budget=self._token_budget,
            max_entries=self._max_entries,
            dedup_threshold=self._dedup_threshold,
            relevance_threshold=self._relevance_threshold,
            base_memory=base_model,
        )

    @classmethod
    def _from_config(cls, config: TokenBudgetedMemoryConfig) -> Self:
        base_memory = Memory.load_component(config.base_memory) if config.base_memory is not None else None
        return cls(
            base_memory,
            name=config.name,
            token_budget=config.token_budget,
            max_entries=config.max_entries,
            dedup_threshold=config.dedup_threshold,
            relevance_threshold=config.relevance_threshold,
        )


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _memory_to_text(memory: MemoryContent) -> str:
    content = memory.content
    return content if isinstance(content, str) else str(content)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def _approx_token_count(text: str) -> int:
    """Dependency-free proxy for a model tokenizer's token count (whitespace word count)."""
    return len(text.split())

"""Selective persistent memory backend.

Adapted from "Shared Selective Persistent Memory for Agentic LLM Systems"
(arXiv:2607.09493). The paper's contribution is *selective* persistence:
retain four categories of reusable context (task specifications, data schemas,
tool configurations, output constraints) and discard session-specific
reasoning traces instead of naively persisting the whole conversation history
-- which it shows actively degrades completion by biasing the agent with stale
traces.

This backend wraps any other :class:`~autogen_core.memory.Memory` store and
applies that selection on the write path: :meth:`add` classifies incoming
:class:`~autogen_core.memory.MemoryContent`, tags reusable items with their
category, forwards them to the backing store, and silently drops reasoning
traces. Reads (:meth:`query` / :meth:`update_context`) delegate to the backing
store, so only reusable context ever reaches the agent.

Substitutions (Mode 2 adapted port):

* The paper's learned LLM classifier is replaced by the parameter-free
  :mod:`~autogen_ext.memory.selective_persistent._classifier` heuristic.

Intentionally out of scope:

* The shared-workspace transfer / role-based-access-control layer and the
  zero-token data-refresh mechanism are application/platform concerns, not
  library memory-backend behavior, and belong in downstream work.

Example:

    .. code-block:: python

        import asyncio
        from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
        from autogen_ext.memory.selective_persistent import SelectivePersistentMemory


        async def main() -> None:
            memory = SelectivePersistentMemory(backend=ListMemory(name="reusable"))
            await memory.add(MemoryContent(content="Schema: users(id int, name string)", mime_type="text/plain"))
            await memory.add(MemoryContent(content="Let me think... first I'll parse the file.", mime_type="text/plain"))
            assert memory.kept_count == 1
            assert memory.discarded_count == 1


        asyncio.run(main())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from autogen_core import CancellationToken, Component, ComponentBase, ComponentModel
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from pydantic import BaseModel, Field
from typing_extensions import Self

from ._classifier import REUSABLE_CATEGORIES, MemoryCategory, classify_content


class SelectivePersistentMemoryConfig(BaseModel):
    """Configuration for :class:`SelectivePersistentMemory` component."""

    backend: Optional[ComponentModel] = Field(
        default=None,
        description="The backing memory store that retains the selected reusable content.",
    )
    default_category: MemoryCategory = Field(
        default=MemoryCategory.REASONING_TRACE,
        description="Category assigned to content that matches no rule (discarded by default).",
    )
    discard_categories: List[MemoryCategory] = Field(
        default_factory=lambda: [MemoryCategory.REASONING_TRACE],
        description="Categories dropped on add rather than forwarded to the backend.",
    )
    name: Optional[str] = Field(default=None, description="Optional identifier for this memory instance.")


# pyright: reportGeneralTypeIssues=false
class SelectivePersistentMemory(
    Memory, Component[SelectivePersistentMemoryConfig], ComponentBase[SelectivePersistentMemoryConfig]
):
    """Selective persistent memory that retains reusable context and discards reasoning traces.

    Wraps a backing :class:`~autogen_core.memory.Memory` store and applies the
    paper's selective-persistence policy on the write path: each
    :meth:`add` is classified into one of four reusable categories
    (task specification, data schema, tool configuration, output constraint)
    or flagged as a reasoning trace. Reusable items are tagged with their
    category and forwarded to the backend; reasoning traces are dropped so
    they never bias future sessions. Reads delegate to the backend, so the
    agent only ever sees the curated reusable context.

    Args:
        backend: The backing memory store that persists the selected content.
        default_category: Category assigned when content matches no rule.
            Defaults to ``REASONING_TRACE`` so ambiguous content is discarded.
        discard_categories: Categories dropped on :meth:`add` instead of
            forwarded. Defaults to ``(REASONING_TRACE,)``.
        name: Optional identifier for this memory instance.
    """

    component_type = "memory"
    component_provider_override = "autogen_ext.memory.selective_persistent.SelectivePersistentMemory"
    component_config_schema = SelectivePersistentMemoryConfig

    def __init__(
        self,
        backend: Memory,
        default_category: MemoryCategory = MemoryCategory.REASONING_TRACE,
        discard_categories: Tuple[MemoryCategory, ...] = (MemoryCategory.REASONING_TRACE,),
        name: Optional[str] = None,
    ) -> None:
        if backend is None:
            raise ValueError("backend memory is required; wrap a Memory store such as ListMemory")
        self._backend = backend
        self._default_category = default_category
        self._discard_categories = tuple(discard_categories)
        self._name = name or "selective_persistent_memory"
        self._kept = 0
        self._discarded = 0
        self._category_counts: Dict[MemoryCategory, int] = {category: 0 for category in MemoryCategory}

    @property
    def backend(self) -> Memory:
        """The backing memory store that retains selected reusable content."""
        return self._backend

    @property
    def name(self) -> str:
        """Get the memory instance identifier."""
        return self._name

    @property
    def default_category(self) -> MemoryCategory:
        """Category assigned to content that matches no rule."""
        return self._default_category

    @property
    def discard_categories(self) -> Tuple[MemoryCategory, ...]:
        """Categories dropped on :meth:`add` rather than forwarded."""
        return self._discard_categories

    @property
    def kept_count(self) -> int:
        """Number of items forwarded to the backing store."""
        return self._kept

    @property
    def discarded_count(self) -> int:
        """Number of items dropped before persistence."""
        return self._discarded

    @property
    def category_counts(self) -> Dict[MemoryCategory, int]:
        """How many added items were classified into each category."""
        return dict(self._category_counts)

    @property
    def reusable_categories(self) -> Tuple[MemoryCategory, ...]:
        """The four reusable categories the paper retains."""
        return REUSABLE_CATEGORIES

    def classify(self, content: MemoryContent) -> MemoryCategory:
        """Classify ``content`` using this instance's configured default category."""
        return classify_content(content, self._default_category)

    async def add(self, content: MemoryContent, cancellation_token: Optional[CancellationToken] = None) -> None:
        """Classify ``content`` and selectively persist it.

        Reusable categories are tagged with ``metadata["category"]`` and
        forwarded to the backing store; discard categories are dropped.

        Args:
            content: The memory content to selectively persist.
            cancellation_token: Optional token to cancel operation.
        """
        if cancellation_token is not None and cancellation_token.is_cancelled():
            return

        category = self.classify(content)
        self._category_counts[category] += 1

        if category in self._discard_categories:
            self._discarded += 1
            return

        tagged = content.model_copy(deep=True)
        tagged.metadata = {**(content.metadata or {}), "category": category.value}
        self._kept += 1
        await self._backend.add(tagged, cancellation_token)

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query the backing store for retained reusable content.

        Args:
            query: Query content item.
            cancellation_token: Optional token to cancel operation.
            **kwargs: Additional implementation-specific parameters.

        Returns:
            ``MemoryQueryResult`` from the backing store.
        """
        return await self._backend.query(query, cancellation_token=cancellation_token, **kwargs)

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Update the model context with reusable content from the backing store.

        Args:
            model_context: The context to update.

        Returns:
            ``UpdateContextResult`` from the backing store.
        """
        return await self._backend.update_context(model_context)

    async def clear(self) -> None:
        """Clear the backing store and reset the selection counters."""
        await self._backend.clear()
        self._kept = 0
        self._discarded = 0
        self._category_counts = {category: 0 for category in MemoryCategory}

    async def close(self) -> None:
        """Clean up the backing store's resources."""
        await self._backend.close()

    @classmethod
    def _from_config(cls, config: SelectivePersistentMemoryConfig) -> Self:
        if config.backend is None:
            raise ValueError("backend memory is required")
        backend = Memory.load_component(config.backend)
        return cls(
            backend=backend,
            default_category=config.default_category,
            discard_categories=tuple(config.discard_categories),
            name=config.name,
        )

    def _to_config(self) -> SelectivePersistentMemoryConfig:
        return SelectivePersistentMemoryConfig(
            backend=self._backend.dump_component(),
            default_category=self._default_category,
            discard_categories=list(self._discard_categories),
            name=self._name,
        )

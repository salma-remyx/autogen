"""Filesystem-based memory backend.

Stores each memory entry as a Markdown file inside a directory tree on disk,
mirroring the "filesystem-as-memory" medium studied in:

    Filesystem-Based Memory for LLM Agents: Organization, Evolution, and
    Sustainability (arXiv:2607.26637v1).

The paper's central empirical finding is that *organization buys search
economy*: a store arranged into a named directory hierarchy can skip
irrelevant directories at retrieval time, roughly halving the number of files
that must be read when the store is large. :class:`FilesystemMemory` realizes
that mechanism inside autogen's :class:`~autogen_core.memory.Memory` contract.

This is a Mode 2 (adapted) port. The paper's *management agent* (which decides
how incoming content is filed) is replaced by a parameter-free, deterministic
filing policy that routes each item into a ``<category>/`` subdirectory based
on its ``metadata['category']``; the paper's *search agent* is replaced by the
Memory ABC's ``query``/``update_context`` using a lexical term-overlap scorer;
and the paper's separate benchmark / growth-study harness is intentionally cut
(it belongs in a downstream evaluation PR). What is preserved at full fidelity
is the core mechanism the paper isolates: persistence as a Markdown filesystem
whose directory organization is precisely what lowers retrieval cost.
"""

import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel
from typing_extensions import Self

logger = logging.getLogger(__name__)

# Tiny, dependency-free English stopword list so that term-overlap scoring and
# directory routing key on content words rather than "the/a/and".
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on that the to was were will with".split()
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Leading single-line frontmatter: ``<!-- autogen-memory {json} -->``. Keeps the
# header on one line so files written by hand (no frontmatter) still load.
_FRONTMATTER_RE = re.compile(r"^<!-- autogen-memory (?P<json>[^\n]*) -->\s*\n")

_DEFAULT_ROOT = ".autogen_memory"


class FilesystemMemoryConfig(BaseModel):
    """Declarative configuration for :class:`FilesystemMemory`."""

    name: str | None = None
    """Optional identifier for this memory instance."""

    root_path: str = _DEFAULT_ROOT
    """Directory that backs the memory store; created if it does not exist."""

    organization: Literal["hierarchy", "flat"] = "hierarchy"
    """``hierarchy`` files items into ``<category>/`` subdirectories (the paper's
    organized shape that buys search economy); ``flat`` is a verbatim dump."""

    k: int = 3
    """Maximum number of results returned per query (``<= 0`` returns all)."""

    score_threshold: float = 0.0
    """Minimum term-overlap score for a result to be returned."""

    clear_on_start: bool = False
    """If ``True``, wipe ``root_path`` when the memory is constructed."""


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on word boundaries, and drop stopwords."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return slug or fallback


def _content_to_text(content: MemoryContent) -> str:
    """Render a memory's content as storable Markdown text."""
    body = content.content
    if isinstance(body, str):
        return body
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    if isinstance(body, dict):
        return json.dumps(body, ensure_ascii=False, indent=2, default=str)
    raise ValueError(f"FilesystemMemory cannot persist content of type {type(body).__name__} as Markdown")


def _serialize(content: MemoryContent) -> str:
    """Serialize a memory to a Markdown file body (frontmatter + content)."""
    header = json.dumps(
        {"mime_type": str(content.mime_type), "metadata": content.metadata or {}},
        ensure_ascii=False,
    )
    return f"<!-- autogen-memory {header} -->\n" + _content_to_text(content)


def _deserialize(text: str) -> MemoryContent:
    """Parse a Markdown file body back into a :class:`MemoryContent`."""
    metadata: Dict[str, Any] = {}
    mime_type: str | MemoryMimeType = MemoryMimeType.TEXT
    body = text
    match = _FRONTMATTER_RE.match(text)
    if match:
        try:
            payload = json.loads(match.group("json"))
            mime_type = payload.get("mime_type", MemoryMimeType.TEXT.value)
            metadata = dict(payload.get("metadata") or {})
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not parse memory frontmatter: %s", exc)
        body = text[match.end() :]
    if isinstance(mime_type, str):
        try:
            mime_type = MemoryMimeType(mime_type)
        except ValueError:
            pass  # keep the raw mime type string
    return MemoryContent(content=body, mime_type=mime_type, metadata=metadata)


class FilesystemMemory(Memory, Component[FilesystemMemoryConfig]):
    """A memory backend that persists entries as a Markdown directory tree.

    Each :class:`~autogen_core.memory.MemoryContent` is written to its own
    ``.md`` file under :attr:`root_path`. In the default ``hierarchy``
    organization, items are filed into ``<category>/`` subdirectories derived
    from ``metadata['category']``; at query time, retrieval only reads the
    directories whose category overlaps the query, so an organized store reads
    strictly fewer files than a flat dump for the same recall -- the search
    economy result from arXiv:2607.26637v1.

    The number of files read by the most recent :meth:`query` is exposed via
    :attr:`last_query_files_scanned` so the cost/economy trade-off is observable
    at the integration boundary.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory import FilesystemMemory


            async def main() -> None:
                memory = FilesystemMemory(config=FilesystemMemoryConfig(root_path="./my_agent_memory"))
                await memory.add(
                    MemoryContent(
                        content="User prefers replies in Celsius.",
                        mime_type=MemoryMimeType.MARKDOWN,
                        metadata={"category": "preferences"},
                    )
                )
                results = await memory.query("temperature preference")
                print(memory.last_query_files_scanned, results)
                await memory.close()


            asyncio.run(main())

    Args:
        config: Optional :class:`FilesystemMemoryConfig`. Defaults to a
            hierarchy store under ``./.autogen_memory``.
    """

    component_config_schema = FilesystemMemoryConfig
    component_provider_override = "autogen_ext.memory.FilesystemMemory"

    def __init__(self, config: FilesystemMemoryConfig | None = None) -> None:
        self._config = config or FilesystemMemoryConfig()
        self._name = self._config.name or "filesystem_memory"
        self._root = Path(self._config.root_path)
        if self._config.clear_on_start:
            self._wipe()
        self._root.mkdir(parents=True, exist_ok=True)
        self._last_query_files_scanned = 0

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._name

    @property
    def root_path(self) -> str:
        """Absolute path of the backing directory."""
        return str(self._root)

    @property
    def last_query_files_scanned(self) -> int:
        """Number of Markdown files read by the most recent :meth:`query`.

        This is the retrieval-cost signal: a ``hierarchy`` store reads fewer
        files than a ``flat`` store for a category-scoped query, which is the
        search-economy finding from arXiv:2607.26637v1.
        """
        return self._last_query_files_scanned

    def _wipe(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)

    def _category_of(self, content: MemoryContent) -> str:
        return _slug(str((content.metadata or {}).get("category", "general")), fallback="general")

    def _candidate_dirs(self, query_text: str, explicit_categories: set[str]) -> List[Path]:
        """Directories to scan for this query.

        For ``flat`` organization there are no subdirectories to prune, so the
        whole root is always scanned (the verbatim-dump baseline). For
        ``hierarchy`` organization, only directories whose category name shares
        a term with the query are read -- this is the parameter-free routing
        that realizes search economy, falling back to all directories when no
        category matches so recall is preserved.
        """
        if not self._root.exists():
            return []
        if self._config.organization == "flat":
            return [self._root]
        existing = {child.name: child for child in self._root.iterdir() if child.is_dir()}
        if not existing:
            return []
        if explicit_categories:
            matched = [existing[c] for c in explicit_categories if c in existing]
            if matched:
                return matched
        query_tokens = set(_tokenize(query_text))
        matched = [d for name, d in existing.items() if set(_tokenize(name)) & query_tokens]
        return matched or list(existing.values())

    def _read_files(self, dirs: List[Path]) -> List[Tuple[MemoryContent, str]]:
        scanned: List[Tuple[MemoryContent, str]] = []
        for directory in dirs:
            for path in sorted(directory.glob("*.md")):
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError as exc:
                    logger.warning("Skipping unreadable memory file %s: %s", path, exc)
                    continue
                scanned.append((_deserialize(text), text))
        return scanned

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject category-scoped memories into the model context as a system message."""
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        last_message = messages[-1]
        query_text = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
        results = await self.query(query_text)
        if results.results:
            lines = [f"{i}. {str(memory.content)}" for i, memory in enumerate(results.results, 1)]
            await model_context.add_message(
                SystemMessage(content="Relevant memory content (filesystem):\n" + "\n".join(lines))
            )
        return UpdateContextResult(memories=results)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Persist a memory entry as a Markdown file in the store."""
        _ = cancellation_token
        category = self._category_of(content)
        target_dir = self._root if self._config.organization == "flat" else self._root / category
        target_dir.mkdir(parents=True, exist_ok=True)

        slug = _slug("_".join(_tokenize(_content_to_text(content))[:6]), fallback="memory")
        path = target_dir / f"{slug}.md"
        if path.exists():  # distinct memories can share a slug; disambiguate without clobbering
            path = target_dir / f"{slug}_{uuid.uuid4().hex[:8]}.md"

        metadata = {**(content.metadata or {}), "id": path.stem, "category": category}
        path.write_text(_serialize(content.model_copy(update={"metadata": metadata})), encoding="utf-8")

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return memories ranked by term overlap, reading only candidate directories."""
        _ = cancellation_token
        query_text = query if isinstance(query, str) else _content_to_text(query)

        explicit_categories: set[str] = set()
        if isinstance(query, MemoryContent) and query.metadata and "category" in query.metadata:
            explicit_categories.add(_slug(str(query.metadata["category"]), fallback="general"))
        if kwargs.get("category"):
            explicit_categories.add(_slug(str(kwargs["category"]), fallback="general"))

        scanned = self._read_files(self._candidate_dirs(query_text, explicit_categories))
        self._last_query_files_scanned = len(scanned)

        query_tokens = set(_tokenize(query_text))
        scored: List[Tuple[float, MemoryContent]] = []
        for content, text in scanned:
            body_tokens = set(_tokenize(text))
            overlap = query_tokens & body_tokens
            score = len(overlap) / len(query_tokens) if query_tokens else 0.0
            if self._config.score_threshold and score < self._config.score_threshold:
                continue
            metadata = {**(content.metadata or {}), "score": score}
            scored.append((score, content.model_copy(update={"metadata": metadata})))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: self._config.k] if self._config.k > 0 else scored
        return MemoryQueryResult(results=[content for _, content in top])

    async def clear(self) -> None:
        """Remove every stored Markdown file and category directory."""
        self._wipe()
        self._root.mkdir(parents=True, exist_ok=True)
        self._last_query_files_scanned = 0

    async def close(self) -> None:
        """Nothing to release; files are flushed on every :meth:`add`."""
        pass

    def _to_config(self) -> FilesystemMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: FilesystemMemoryConfig) -> Self:
        return cls(config=config)

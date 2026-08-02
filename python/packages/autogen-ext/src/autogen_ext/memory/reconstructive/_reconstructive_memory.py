"""Reconstructive memory — adapt retrieved experience to the live context.

Adapted from *MemHarness: Memory Is Reconstructed, Not Replayed*
(arXiv:2607.28272). MemHarness observes that most memory-augmented agents
*replay* retrieved experience verbatim, which causes negative transfer when the
abstract stored experience does not match the concrete decision-time state. Its
core mechanism is instead: at each step, *retrieve* past experience, then
*critique and reconstruct* it conditioned on the current state into
context-grounded guidance — or *reject* it when it does not apply, so stale or
irrelevant memories are never injected.

This module ports that retrieve -> critique -> reconstruct-or-reject loop onto
AutoGen's :class:`~autogen_core.memory.Memory` contract (a Mode 2 adapted port):

* The paper's GRPO-trained unified policy is replaced by a prompt-driven
  reconstruction step over the repo's own
  :class:`~autogen_core.models.ChatCompletionClient`.
* Retrieval, storage and ``add`` / ``query`` / ``clear`` / ``close`` are
  delegated to any existing AutoGen memory backend, so vector search and storage
  are reused rather than rebuilt.
* Only the *injection* behaviour changes: where a replay backend would dump raw
  memories into the context, this layer injects the reconstructed guidance — or
  nothing, on rejection.

The GRPO training procedure, the vLLM/Flash-Attn/Milvus serving stack, and the
ALFWorld / WebShop benchmark suite are intentionally out of scope; they belong
in a downstream training/evaluation PR.
"""

from __future__ import annotations

from typing import Any

from autogen_core import CancellationToken
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, SystemMessage, UserMessage

DEFAULT_REJECT_SENTINEL = "REJECT"
"""Token the model emits to signal that the retrieved experience does not apply and must not be injected."""

DEFAULT_MAX_CONTEXT_MESSAGES = 8
"""Default number of recent live messages shown to the reconstruction step as the current state."""

_DEFAULT_PROMPT = """\
You are a memory-reconstruction module for an autonomous agent. Retrieved past \
experience is abstract and general; the agent's current state is concrete and \
changing. Decide whether the retrieved experience actually applies here and, if \
so, rewrite it as concise, action-oriented guidance grounded in the current \
state. Never replay raw memory verbatim.

If the retrieved experience does NOT apply to the current state — applying it \
would mislead the agent and cause negative transfer — respond with exactly this \
token and nothing else:

{sentinel}

Otherwise, respond with ONLY the reconstructed guidance: a few short directives \
the agent can act on right now. No preamble, no labels, no quotes.\
"""


def _content_text(content: str | list[Any]) -> str:
    """Best-effort flattening of a model completion or message body to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


def _render_memory(item: MemoryContent) -> str:
    """Render a stored :class:`MemoryContent` item as prompt text."""
    content = item.content
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def _strip_label(text: str) -> str:
    """Drop a leading ``GUIDANCE:``-style label some models prepend."""
    first_line, sep, rest = text.partition("\n")
    head = first_line.strip()
    if ":" in head and head.split(":", 1)[0].strip().upper().isalpha():
        head = head.split(":", 1)[1].strip()
    rebuilt = head + sep + rest
    return rebuilt.strip()


class ReconstructiveMemory(Memory):
    """A memory that reconstructs retrieved experience against the live context.

    Wraps a retrieval ``Memory`` (e.g. :class:`~autogen_core.memory.ListMemory`
    or a vector backend such as ``ChromaDBVectorMemory``) together with a
    :class:`~autogen_core.models.ChatCompletionClient`. On every
    :meth:`update_context` it retrieves past experience via the wrapped backend,
    then asks the model to reconstruct that experience into state-aligned
    guidance conditioned on the current conversation — or to reject it when it
    does not apply. Only the reconstructed guidance (or nothing, on rejection)
    is written into the model context, so irrelevant memories never cause
    negative transfer.

    Storage and retrieval are delegated to the wrapped backend; this layer only
    changes how retrieved memories are injected.

    Example:

        .. code-block:: python

            import asyncio

            from autogen_core.memory import ListMemory, MemoryContent
            from autogen_ext.memory.reconstructive import ReconstructiveMemory
            from autogen_ext.models.replay import ReplayChatCompletionClient


            async def main() -> None:
                store = ListMemory()
                await store.add(MemoryContent(content="Always confirm the file path before writing.", mime_type="text/plain"))
                memory = ReconstructiveMemory(
                    retrieval=store,
                    model_client=ReplayChatCompletionClient(
                        chat_completions=["Verify the destination path exists before calling write()."]
                    ),
                )


            asyncio.run(main())

    Args:
        retrieval: The backing :class:`~autogen_core.memory.Memory` used to store
            and retrieve experience.
        model_client: The chat client used to reconstruct retrieved experience
            against the current state.
        name: Optional identifier for this memory instance.
        reject_sentinel: The exact token the model emits to reject the retrieved
            experience (defaults to ``"REJECT"``).
        max_context_messages: How many of the most recent live messages to show
            the reconstruction step as the current state.
    """

    def __init__(
        self,
        retrieval: Memory,
        model_client: ChatCompletionClient,
        *,
        name: str | None = None,
        reject_sentinel: str = DEFAULT_REJECT_SENTINEL,
        max_context_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
    ) -> None:
        self._retrieval = retrieval
        self._model_client = model_client
        self._name = name or "reconstructive_memory"
        self._reject_sentinel = reject_sentinel
        self._max_context_messages = max_context_messages

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""
        return self._name

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Reconstruct retrieved experience into state-aligned guidance before injecting it.

        Retrieves experience relevant to the latest message, then either injects
        the model's reconstructed guidance as a :class:`~autogen_core.models.SystemMessage`
        or — on rejection — injects nothing so the irrelevant memory cannot cause
        negative transfer.
        """
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        retrieved = await self._retrieval.query(_content_text(_message_body(messages[-1])))
        if not retrieved.results:
            return UpdateContextResult(memories=retrieved)

        guidance = await self._reconstruct(messages, retrieved.results)
        if guidance is not None:
            await model_context.add_message(
                SystemMessage(content="Guidance reconstructed from prior experience:\n" + guidance)
            )

        # ``memories`` reports what was retrieved, regardless of whether it was
        # reconstructed-and-injected or rejected, so callers can still observe it.
        return UpdateContextResult(memories=retrieved)

    async def _reconstruct(self, messages: list[LLMMessage], retrieved: list[MemoryContent]) -> str | None:
        experience = "\n".join(f"- {_render_memory(item)}" for item in retrieved)
        recent = messages[-self._max_context_messages :]
        current_state = "\n".join(
            f"{type(message).__name__}: {_content_text(_message_body(message))}" for message in recent
        )
        user_blob = (
            "<retrieved_experience>\n"
            f"{experience}\n"
            "</retrieved_experience>\n\n"
            "<current_state>\n"
            f"{current_state}\n"
            "</current_state>"
        )
        result: CreateResult = await self._model_client.create(
            [
                SystemMessage(content=_DEFAULT_PROMPT.replace("{sentinel}", self._reject_sentinel)),
                UserMessage(content=user_blob, source=self._name),
            ]
        )
        return self._parse(result)

    def _parse(self, result: CreateResult) -> str | None:
        text = _content_text(result.content).strip()
        if not text or text.upper() == self._reject_sentinel.upper():
            # Rejection: retrieved experience does not apply -> inject nothing.
            return None
        return _strip_label(text)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Query the backing retrieval store (delegated)."""
        return await self._retrieval.query(query, cancellation_token=cancellation_token, **kwargs)

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add content to the backing retrieval store (delegated)."""
        await self._retrieval.add(content, cancellation_token=cancellation_token)

    async def clear(self) -> None:
        """Clear the backing retrieval store (delegated)."""
        await self._retrieval.clear()

    async def close(self) -> None:
        """Close the backing retrieval store (delegated)."""
        await self._retrieval.close()


def _message_body(message: LLMMessage) -> str | list[Any]:
    """Return the ``content`` field of any :class:`~autogen_core.models.LLMMessage` variant."""
    body = getattr(message, "content", "")
    if isinstance(body, (str, list)):
        return body
    return str(body)

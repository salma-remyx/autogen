"""Serving-cost instrumentation for memory-backed model contexts.

Long-running agents increasingly rely on a memory system (Mem0, ChromaDB, Redis, ...)
instead of resending the whole conversation each turn, but AutoGen currently exposes no
way to observe what that choice costs: how many prompt tokens the memory system injects
per turn, versus simply resubmitting the transcript. Without that number the choice
between, say, :class:`~autogen_ext.memory.mem0.Mem0Memory` and a
:class:`~autogen_core.model_context.BufferedChatCompletionContext` rolling window is a
guess, and it cannot be revisited once a conversation grows.

This module supplies the measurement half of that comparison:

- :class:`ServingCostRecorder.instrument` wraps any ``ChatCompletionContext`` so that
  every :meth:`~autogen_core.memory.Memory.update_context` call is priced: the tokens
  the wrapped context retains on its own (the rolling-window view) versus the tokens it
  actually serves once the memory system has injected its retrieved content. The
  difference is the memory system's per-turn serving cost in prompt tokens, measured on
  the live agent loop rather than estimated.
- :class:`FullTranscriptBaseline` prices the reference strategy -- resend everything --
  so the two numbers can be put side by side.
- :meth:`ServingCostRecorder.report` folds the samples into a
  :class:`MemoryServingCostReport`, whose :attr:`~MemoryServingCostReport.breakeven_turn`
  is the first turn from which the memory-backed context stays cheaper to serve than the
  full transcript (``None`` when it never is within the observed run).

Adapted from *Total Recall at What Cost? Benchmarking the Serving Cost of Agentic Memory
Systems* (arXiv:2608.11879). The paper's findings that a memory system's serving cost is
driven by internal memory behavior rather than by conversation length, and that the
break-even point against the full transcript is system-dependent, are what the recorder
makes observable inside AutoGen. The paper's benchmark suite (LoCoMo accuracy, two
backbones, 400-turn runs) is intentionally out of scope: accuracy measurement belongs in
a benchmark harness, not a library extension. The per-turn token accounting and the
break-even analysis are the parts an AutoGen user needs at runtime.

Example:

    .. code-block:: python

        import asyncio
        from autogen_core.memory import MemoryContent
        from autogen_core.model_context import UnboundedChatCompletionContext
        from autogen_core.models import UserMessage
        from autogen_ext.memory.mem0 import Mem0Memory
        from autogen_ext.memory.serving_cost import ServingCostRecorder


        async def main() -> None:
            memory = Mem0Memory(is_cloud=False, config={"path": ":memory:"})
            await memory.add(MemoryContent(content="User lives in Seattle.", mime_type="text/plain"))

            recorder = ServingCostRecorder()
            context = recorder.instrument(UnboundedChatCompletionContext())
            agent_context = context  # pass to AssistantAgent(model_context=...)

            await agent_context.add_message(UserMessage(content="Where do I live?", source="user"))
            await recorder.record_update_context(memory, agent_context)

            print(recorder.report().summary())


        asyncio.run(main())

"""

from typing import Any, Callable, List, Mapping, Sequence

from autogen_core import CancellationToken
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import AssistantMessage, FunctionExecutionResultMessage, LLMMessage, SystemMessage, UserMessage
from pydantic import BaseModel, Field

__all__ = [
    "FullTranscriptBaseline",
    "InstrumentedModelContext",
    "MemoryServingCostReport",
    "ServingCostRecorder",
    "ServingCostSample",
]

# Characters per token used when no model client is supplied for real tokenization.
# Only affects the unit of measurement: memory and transcript costs share the same
# estimator, so their comparison is unaffected by the constant.
_CHARS_PER_TOKEN = 4.0


def _content_chars(content: Any) -> int:
    """Characters contributed by message content of any supported shape."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, Sequence):
        return sum(len(part) for part in content if isinstance(part, str))
    return len(str(content))


def _message_tokens(messages: Sequence[LLMMessage]) -> int:
    """Estimate prompt tokens for ``messages``.

    Counts the rendered text of each message plus a per-message role envelope -- the
    quantity a provider bills for the prompt.
    """
    chars = 0
    for message in messages:
        if isinstance(message, SystemMessage):
            chars += len(message.content)
        elif isinstance(message, UserMessage):
            chars += _content_chars(message.content)
        elif isinstance(message, AssistantMessage):
            chars += _content_chars(message.content)
            if message.thought:
                chars += len(message.thought)
        elif isinstance(message, FunctionExecutionResultMessage):
            chars += sum(len(result.content) for result in message.content)
        else:  # pragma: no cover - the discriminator exhausts the union
            chars += len(str(message))
        chars += 4
    return round(chars / _CHARS_PER_TOKEN)


def _all_messages(context: ChatCompletionContext) -> List[LLMMessage]:
    """Every message the context holds, bypassing its retention policy."""
    return list(getattr(context, "_messages", []))


class ServingCostSample(BaseModel):
    """Per-turn cost measurements for one ``update_context`` call."""

    turn: int
    """1-based index of the turn within the recorded run."""

    transcript_tokens: int
    """Tokens in the entire conversation so far, including the current turn."""

    retained_tokens: int
    """Tokens the wrapped context retains on its own, before memory injection."""

    served_tokens: int
    """Tokens actually served once the memory system has injected its content."""

    @property
    def memory_overhead_tokens(self) -> int:
        """Prompt tokens the memory system added to this turn's request."""
        return self.served_tokens - self.retained_tokens

    @property
    def saved_vs_transcript(self) -> int:
        """Prompt tokens saved this turn relative to resubmitting the full transcript."""
        return self.transcript_tokens - self.served_tokens


class MemoryServingCostReport(BaseModel):
    """Aggregate serving-cost comparison for one recorded run."""

    samples: List[ServingCostSample] = Field(default_factory=list)
    """Per-turn measurements, in turn order."""

    @property
    def turns(self) -> int:
        """Number of turns recorded."""
        return len(self.samples)

    @property
    def memory_overhead_tokens(self) -> int:
        """Mean prompt tokens the memory system adds per turn."""
        if not self.samples:
            return 0
        return round(sum(sample.memory_overhead_tokens for sample in self.samples) / len(self.samples))

    @property
    def served_tokens_total(self) -> int:
        """Total prompt tokens served across the run by the memory-backed context."""
        return sum(sample.served_tokens for sample in self.samples)

    @property
    def transcript_tokens_total(self) -> int:
        """Total prompt tokens the full-transcript reference strategy would serve."""
        return sum(sample.transcript_tokens for sample in self.samples)

    @property
    def breakeven_turn(self) -> int | None:
        """First turn from which the memory-backed context stays cheaper than the transcript.

        ``None`` when it is more expensive on every observed turn, or when no turns were
        recorded.
        """
        for index in range(len(self.samples)):
            if all(sample.served_tokens < sample.transcript_tokens for sample in self.samples[index:]):
                return self.samples[index].turn
        return None

    def summary(self) -> str:
        """One-line comparison, in the paper's cost-versus-transcript framing."""
        if not self.samples:
            return "no turns recorded"
        breakeven = "never" if self.breakeven_turn is None else f"turn {self.breakeven_turn}"
        return (
            f"{self.turns} turns: memory {self.served_tokens_total} vs transcript "
            f"{self.transcript_tokens_total} prompt tokens "
            f"(+{self.memory_overhead_tokens}/turn from memory, cheaper from {breakeven})"
        )


class InstrumentedModelContext(ChatCompletionContext):
    """A ``ChatCompletionContext`` that records what memory injection costs per turn.

    Every operation delegates to the wrapped context, so an agent configured with this
    wrapper behaves exactly as before; the wrapper exists so the recorder can observe
    the context the memory system actually mutates.

    Args:
        inner: The context the agent would have used.
        recorder: Recorder the turn samples are appended to.
    """

    def __init__(self, inner: ChatCompletionContext, recorder: "ServingCostRecorder") -> None:
        super().__init__()
        self._inner = inner
        self._recorder = recorder

    async def add_message(self, message: LLMMessage) -> None:
        await self._inner.add_message(message)

    async def get_messages(self) -> List[LLMMessage]:
        return await self._inner.get_messages()

    async def clear(self) -> None:
        await self._inner.clear()

    async def save_state(self) -> Mapping[str, Any]:
        return await self._inner.save_state()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        await self._inner.load_state(state)


class FullTranscriptBaseline(Memory):
    """Reference memory that resubmits the whole conversation every turn.

    The strategy a memory system is meant to beat on serving cost: no summarization, no
    retrieval -- the full history is the context. Recording a run that uses it alongside
    a run that uses a real memory backend produces the two sides of the paper's
    comparison for the same conversation.

    Args:
        recorder: Recorder the baseline turns are appended to.
    """

    component_type = "memory"

    def __init__(self, recorder: "ServingCostRecorder") -> None:
        self._recorder = recorder

    @property
    def name(self) -> str:
        return "full_transcript_baseline"

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        # The transcript is already in the context, so there is nothing to inject; the
        # point is to price the turn at full-transcript cost.
        self._recorder.record_baseline(await model_context.get_messages())
        return UpdateContextResult(memories=MemoryQueryResult(results=[]))

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        _ = query, cancellation_token, kwargs
        return MemoryQueryResult(results=[])

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        _ = content, cancellation_token

    async def clear(self) -> None:
        self._recorder.reset()

    async def close(self) -> None:
        pass


class ServingCostRecorder:
    """Collects per-turn serving-cost samples for one conversation.

    Args:
        token_counter: Optional token estimator override, e.g. a model client's
            ``count_tokens`` bound to the backbone actually in use. Defaults to a
            character-based estimate, which is sufficient for comparing two strategies
            measured with the same estimator.
    """

    def __init__(self, token_counter: Callable[[Sequence[LLMMessage]], int] | None = None) -> None:
        self._count = token_counter or _message_tokens
        self._samples: List[ServingCostSample] = []
        self._turn = 0
        self._pending_retained: int | None = None
        self._pending_transcript: int | None = None

    def instrument(self, context: ChatCompletionContext) -> InstrumentedModelContext:
        """Wrap ``context`` so memory injections into it are priced per turn."""
        return InstrumentedModelContext(context, self)

    async def begin_turn(self, context: ChatCompletionContext) -> None:
        """Snapshot a turn before the memory system injects into ``context``.

        Called by an instrumented memory backend at the top of its ``update_context``;
        paired with :meth:`end_turn`. Measuring from inside the backend keeps the
        pre-injection view exact even when the backend queries the context itself.
        """
        self._turn += 1
        # The reference strategy resubmits the whole conversation, so it is priced
        # against the unfiltered history rather than the window the context serves.
        self._pending_transcript = self._count(_all_messages(context))
        # The context's own retention policy is the rolling-window view the agent would
        # have served with no memory attached.
        self._pending_retained = self._count(await context.get_messages())

    def end_turn(self, served_messages: Sequence[LLMMessage]) -> None:
        """Close the turn opened by :meth:`begin_turn` and append its sample.

        Args:
            served_messages: The context's messages after the memory system injected,
                i.e. what this turn actually serves to the model.

        Raises:
            RuntimeError: If called without a matching :meth:`begin_turn`.
        """
        if self._pending_retained is None or self._pending_transcript is None:
            raise RuntimeError("end_turn() called without a matching begin_turn().")
        retained, transcript = self._pending_retained, self._pending_transcript
        self._pending_retained = self._pending_transcript = None
        self._samples.append(
            ServingCostSample(
                turn=self._turn,
                transcript_tokens=transcript,
                retained_tokens=retained,
                served_tokens=self._count(list(served_messages)),
            )
        )

    async def record_update_context(self, memory: Memory, context: ChatCompletionContext) -> None:
        """Price one memory-driven turn without instrumenting the backend.

        Convenience for callers using a memory backend that has no instrumentation hook:
        prices the turn around a plain ``update_context`` call. Backends that call
        :meth:`begin_turn`/:meth:`end_turn` themselves (e.g.
        :class:`~autogen_ext.memory.mem0.Mem0Memory` via
        ``attach_serving_cost_recorder``) are priced more precisely and should not use
        this.

        Args:
            memory: The memory backend attached to the agent.
            context: The context the memory system updates.
        """
        await self.begin_turn(context)
        try:
            await memory.update_context(context)
            served = await context.get_messages()
        finally:
            self.end_turn(served)

    def record_baseline(self, messages: Sequence[LLMMessage]) -> ServingCostSample:
        """Append a sample describing a full-transcript turn.

        The transcript is both what is retained and what is served, so memory overhead
        is zero and the sample marks the reference cost for its turn.
        """
        tokens = self._count(messages)
        self._turn += 1
        sample = ServingCostSample(
            turn=self._turn, transcript_tokens=tokens, retained_tokens=tokens, served_tokens=tokens
        )
        self._samples.append(sample)
        return sample

    def report(self) -> MemoryServingCostReport:
        """Return the samples recorded so far as an aggregate report."""
        return MemoryServingCostReport(samples=list(self._samples))

    def reset(self) -> None:
        """Drop all samples and restart turn numbering."""
        self._samples = []
        self._turn = 0

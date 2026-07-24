import logging
import re
from typing import Any, List, Literal, cast

from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import Self

logger = logging.getLogger(__name__)

# Parameter-free proxy for the paper's semantic verifier. Antithetical token
# pairs flag mutually-exclusive instruction values within a local neighborhood.
_ANTITHETICAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("metric", "imperial"),
    ("celsius", "fahrenheit"),
    ("kilograms", "pounds"),
    ("always", "never"),
    ("required", "forbidden"),
    ("allow", "disallow"),
    ("allow", "prohibit"),
    ("enabled", "disabled"),
    ("enable", "disable"),
    ("include", "exclude"),
    ("json", "xml"),
    ("formal", "casual"),
    ("concise", "verbose"),
    ("uppercase", "lowercase"),
)

_NEGATION_MARKERS: tuple[str, ...] = ("not ", "n't", "never", "cannot", "without", "except", "avoid")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "you",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "will",
        "would",
        "should",
        "must",
        "shall",
        "when",
        "then",
        "than",
        "them",
        "they",
        "their",
        "what",
        "which",
        "very",
        "much",
        "some",
        "such",
        "also",
        "each",
        "both",
        "either",
        "neither",
        "make",
        "made",
        "using",
        "used",
        "based",
        "given",
        "please",
        "output",
        "respond",
        "report",
        "always",
        "never",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")

# The paper's typed semantic graph uses three node types: identity (who the
# agent is), norm (behavioral rules/constraints), and knowledge (domain facts
# and procedures). Rendering and verification treat the type as a closed set.
InstructionNodeType = Literal["identity", "norm", "knowledge"]

_NODE_TYPES: tuple[InstructionNodeType, ...] = ("identity", "norm", "knowledge")

# Checkpoint rendering order: identity statements first, then behavioral
# norms, then domain knowledge.
_TYPE_ORDER: dict[InstructionNodeType, int] = {"identity": 0, "norm": 1, "knowledge": 2}


class InstructionNode(BaseModel):
    """A persistent instruction represented as a typed, subject-scoped graph node."""

    node_id: str
    type: InstructionNodeType = "norm"
    subject: str = "general"
    text: str
    superseded_by: str | None = None
    """Identifier of the node that retired this one during consolidation, or ``None`` while live."""


class Conflict(BaseModel):
    """A conflict surfaced by scoped verification against one neighborhood node."""

    neighbor_id: str
    reason: str
    pair: str | None = None


class EvolveResult(BaseModel):
    """Outcome of one scoped-verification evolution step (GRACE's ``evolve``)."""

    accepted: bool
    node: InstructionNode
    conflicts: List[Conflict]
    neighborhood: List[str]
    checkpoint_version: int
    superseded: List[str] = Field(default_factory=list)
    instruction_text: str


class InstructionGraphMemoryConfig(BaseModel):
    """Configuration for :class:`InstructionGraphMemory`."""

    name: str | None = None
    on_conflict: Literal["reject", "supersede"] = "reject"
    nodes: List[InstructionNode] = Field(default_factory=list)


class InstructionGraphMemory(Memory, Component[InstructionGraphMemoryConfig]):
    """Memory that evolves persistent instructions on a typed semantic graph.

    Adapted from "Scoped Verification for Reliable Long-Horizon Agentic Context
    Evolution under Distribution Shift" (GRACE). The persistent instruction
    component is maintained as a graph of typed, subject-scoped nodes rather
    than flat text. Node types follow the paper's schema: ``identity`` (who
    the agent is), ``norm`` (behavioral rules), and ``knowledge`` (domain
    facts). Each proposed update is verified only against the local
    neighborhood of nodes sharing its subject (scoped verification), and
    accepted updates are reconstructed into a single textual instruction
    checkpoint that :meth:`update_context` injects as a ``SystemMessage``.

    The paper verifies proposed updates with an LLM; this implementation
    substitutes a parameter-free verifier (antithetical-token and
    negation-overlap heuristics) that approximates the same "does this conflict
    locally?" signal. The ``on_conflict="supersede"`` policy implements the
    paper's consolidation mechanism: a conflicting neighbor is retired (its
    lineage recorded) rather than the whole instruction set re-verified.

    Args:
        config: Optional configuration. Defaults to a reject-policy graph.

    Example:

        .. code-block:: python

            import asyncio
            from autogen_core.memory import MemoryContent, MemoryMimeType
            from autogen_ext.memory.instruction_graph import InstructionGraphMemory


            async def main() -> None:
                memory = InstructionGraphMemory()
                await memory.add(
                    MemoryContent(
                        content="Report temperatures in metric units",
                        mime_type=MemoryMimeType.TEXT,
                        metadata={"type": "norm", "subject": "units"},
                    )
                )


            asyncio.run(main())
    """

    component_config_schema = InstructionGraphMemoryConfig
    component_provider_override = "autogen_ext.memory.instruction_graph.InstructionGraphMemory"

    def __init__(self, config: InstructionGraphMemoryConfig | None = None) -> None:
        """Initialize the instruction graph memory."""
        self.config = config or InstructionGraphMemoryConfig()
        self._name = self.config.name or "instruction_graph_memory"
        self._on_conflict = self.config.on_conflict
        self._nodes: dict[str, InstructionNode] = {n.node_id: n for n in self.config.nodes}
        self._seq = len(self._nodes)
        self._checkpoint_version = 0
        self._checkpoints: list[str] = [self._render()]
        self._last_evolve: EvolveResult | None = None

    @property
    def name(self) -> str:
        """Return the memory instance identifier."""
        return self._name

    @property
    def checkpoint_version(self) -> int:
        """Return the current instruction checkpoint version."""
        return self._checkpoint_version

    @property
    def last_evolve_result(self) -> EvolveResult | None:
        """Return the result of the most recent :meth:`evolve` call, if any."""
        return self._last_evolve

    async def evolve(self, content: MemoryContent) -> EvolveResult:
        """Propose, scoped-verify, and (on accept) apply an instruction update.

        Args:
            content: The proposed instruction. ``metadata`` may carry ``type``
                (one of ``"identity"``, ``"norm"``, ``"knowledge"``; default
                ``"norm"``), ``subject`` and ``node_id``; ``content`` is the
                instruction text.

        Returns:
            EvolveResult describing the neighborhood examined, any conflicts,
            whether the node was applied, and the resulting checkpoint.

        Raises:
            ValueError: If ``metadata["type"]`` is not a valid node type.
        """
        node = self._node_from_content(content)
        conflicts, neighbors = self._verify(node)
        superseded: List[str] = []
        accepted = not conflicts
        if conflicts and self._on_conflict == "supersede":
            accepted = True
            for conflict in conflicts:
                neighbor = self._nodes.get(conflict.neighbor_id)
                if neighbor is not None and neighbor.superseded_by is None:
                    neighbor.superseded_by = node.node_id
                    superseded.append(conflict.neighbor_id)

        if accepted:
            self._nodes[node.node_id] = node
            self._checkpoint_version += 1
            self._checkpoints.append(self._render())
            logger.debug("Accepted instruction node %s at checkpoint %d", node.node_id, self._checkpoint_version)
        else:
            logger.warning(
                "Rejected instruction node %s: %d conflict(s) in neighborhood of %d",
                node.node_id,
                len(conflicts),
                len(neighbors),
            )

        result = EvolveResult(
            accepted=accepted,
            node=node,
            conflicts=conflicts,
            neighborhood=[n.node_id for n in neighbors],
            checkpoint_version=self._checkpoint_version,
            superseded=superseded,
            instruction_text=self._checkpoints[-1],
        )
        self._last_evolve = result
        return result

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        """Inject the reconstructed instruction checkpoint as a system message.

        Args:
            model_context: The context to update. Mutated only if instructions exist.

        Returns:
            UpdateContextResult containing the live instruction nodes.
        """
        text = self._checkpoints[-1]
        if text:
            await model_context.add_message(SystemMessage(content=text))
        return UpdateContextResult(memories=MemoryQueryResult(results=self._to_contents(self._live_nodes())))

    async def add(self, content: MemoryContent, cancellation_token: CancellationToken | None = None) -> None:
        """Add an instruction through scoped verification.

        Conflicting updates are gated by ``on_conflict``: rejected under the
        default ``"reject"`` policy, consolidated under ``"supersede"``. Inspect
        :attr:`last_evolve_result` for the verdict.

        Args:
            content: The proposed instruction content.
            cancellation_token: Optional token to cancel operation. Not used.
        """
        _ = cancellation_token
        await self.evolve(content)

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Return the live instruction nodes.

        Args:
            query: Ignored in this implementation.
            cancellation_token: Optional token to cancel operation. Not used.
            **kwargs: Additional parameters (ignored).

        Returns:
            MemoryQueryResult containing all non-superseded instruction nodes.
        """
        _ = query, cancellation_token, kwargs
        return MemoryQueryResult(results=self._to_contents(self._live_nodes()))

    async def clear(self) -> None:
        """Clear all instruction nodes and reset the checkpoint lineage."""
        self._nodes = {}
        self._seq = 0
        self._checkpoint_version = 0
        self._checkpoints = [self._render()]
        self._last_evolve = None

    async def close(self) -> None:
        """Clean up resources. No external resources are held."""
        pass

    @classmethod
    def _from_config(cls, config: InstructionGraphMemoryConfig) -> Self:
        return cls(config=config)

    def _to_config(self) -> InstructionGraphMemoryConfig:
        return InstructionGraphMemoryConfig(
            name=self._name, on_conflict=self._on_conflict, nodes=list(self._nodes.values())
        )

    # -- internal helpers -------------------------------------------------

    def _live_nodes(self) -> List[InstructionNode]:
        return [n for n in self._nodes.values() if n.superseded_by is None]

    def _node_from_content(self, content: MemoryContent) -> InstructionNode:
        metadata = content.metadata or {}
        raw_type = str(metadata.get("type", "norm"))
        if raw_type not in _NODE_TYPES:
            raise ValueError(f"Invalid instruction node type {raw_type!r}; expected one of {list(_NODE_TYPES)}.")
        node_type = cast(InstructionNodeType, raw_type)
        subject = str(metadata.get("subject", "general"))
        node_id = str(metadata.get("node_id") or metadata.get("id") or self._next_id(node_type, subject))
        return InstructionNode(node_id=node_id, type=node_type, subject=subject, text=str(content.content))

    def _next_id(self, node_type: str, subject: str) -> str:
        self._seq += 1
        return f"{node_type}:{subject}:{self._seq}"

    def _verify(self, node: InstructionNode) -> tuple[List[Conflict], List[InstructionNode]]:
        neighbors = [
            n
            for n in self._nodes.values()
            if n.superseded_by is None and n.node_id != node.node_id and n.subject == node.subject
        ]
        conflicts: List[Conflict] = []
        for neighbor in neighbors:
            conflict = self._detect_conflict(node, neighbor)
            if conflict is not None:
                conflicts.append(conflict)
        return conflicts, neighbors

    def _detect_conflict(self, proposed: InstructionNode, neighbor: InstructionNode) -> Conflict | None:
        proposed_tokens = self._tokenize(proposed.text)
        neighbor_tokens = self._tokenize(neighbor.text)
        for left, right in _ANTITHETICAL_PAIRS:
            if (left in proposed_tokens and right in neighbor_tokens) or (
                right in proposed_tokens and left in neighbor_tokens
            ):
                return Conflict(neighbor_id=neighbor.node_id, reason="antithetical", pair=f"{left}/{right}")

        proposed_negated = self._is_negated(proposed.text)
        neighbor_negated = self._is_negated(neighbor.text)
        if proposed_negated != neighbor_negated:
            shared = self._content_tokens(proposed_tokens) & self._content_tokens(neighbor_tokens)
            if shared:
                return Conflict(neighbor_id=neighbor.node_id, reason="negation", pair=sorted(shared)[0])
        return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_WORD_RE.findall(text.lower()))

    @staticmethod
    def _content_tokens(tokens: set[str]) -> set[str]:
        return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}

    @staticmethod
    def _is_negated(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in _NEGATION_MARKERS)

    def _render(self) -> str:
        live = self._live_nodes()
        if not live:
            return ""
        lines = ["Persistent instructions (graph checkpoint):"]
        for node in sorted(live, key=lambda n: (_TYPE_ORDER[n.type], n.subject, n.node_id)):
            lines.append(f"- [{node.type}:{node.subject}] {node.text}")
        return "\n".join(lines) + "\n"

    def _to_contents(self, nodes: List[InstructionNode]) -> List[MemoryContent]:
        return [
            MemoryContent(
                content=node.text,
                mime_type=MemoryMimeType.TEXT,
                metadata={"node_id": node.node_id, "type": node.type, "subject": node.subject},
            )
            for node in nodes
        ]

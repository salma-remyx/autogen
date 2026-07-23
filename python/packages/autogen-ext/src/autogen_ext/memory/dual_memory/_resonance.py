"""Cross-memory resonance scoring.

Adapted from *OpsMem: Dual-Memory Reasoning with Cross-Memory Resonance for
Failure Diagnosis* (arXiv:2607.11357).

OpsMem activates long-term memories that *resonate* with the current
short-term diagnostic state using a learned resonance estimator. As a
Mode 2 adapted port, the learned estimator is substituted here by a
parameter-free token-overlap proxy (see the module docstring of
``_dual_memory``). The proxy approximates the same signal — "how relevant
is this stored experience to the current working state?" — without any
learned parameters or external model dependency, and a pluggable
:class:`ResonanceScorer` protocol lets a richer (e.g. embedding-based)
scorer be dropped in without touching the memory backend.
"""

import math
import re
from collections import Counter
from typing import Protocol

# A small, dependency-free English stop-word set. Kept intentionally short so
# that domain terms (the signal OpsMem cares about) are not discarded.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "when",
        "will",
        "with",
        "we",
        "you",
        "your",
        "i",
        "he",
        "she",
        "they",
        "but",
        "not",
        "can",
        "do",
        "does",
        "if",
        "then",
        "than",
        "so",
        "into",
        "our",
        "their",
        "them",
        "his",
        "her",
        "us",
        "me",
        "my",
        "no",
        "yes",
        "see",
        "seen",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, remove_stop_words: bool = True) -> list[str]:
    """Lower-case and split *text* into alphanumeric tokens.

    Args:
        text: The text to tokenize.
        remove_stop_words: If True, drop common stop words so that domain
            terms dominate the resonance score.

    Returns:
        The list of tokens (order preserved, duplicates kept for counting).
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stop_words:
        tokens = [t for t in tokens if t not in _STOP_WORDS]
    return tokens


class ResonanceScorer(Protocol):
    """Scores how strongly a stored memory resonates with a query state."""

    def score(self, query_text: str, candidate_text: str) -> float:
        """Return a resonance score in ``[0.0, 1.0]`` (higher = more relevant)."""
        ...


class TokenOverlapResonanceScorer:
    """Parameter-free resonance via cosine similarity of token-count vectors.

    This is the default Mode 2 substitution for OpsMem's learned resonance
    estimator: it carries the same "lexical relevance to the current state"
    signal with no learned weights and no external model.
    """

    def __init__(self, remove_stop_words: bool = True) -> None:
        self._remove_stop_words = remove_stop_words

    def score(self, query_text: str, candidate_text: str) -> float:
        query_tokens = tokenize(query_text, self._remove_stop_words)
        candidate_tokens = tokenize(candidate_text, self._remove_stop_words)
        if not query_tokens or not candidate_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        candidate_counts = Counter(candidate_tokens)

        shared = query_counts.keys() & candidate_counts.keys()
        dot = sum(query_counts[t] * candidate_counts[t] for t in shared)
        query_norm = math.sqrt(sum(v * v for v in query_counts.values()))
        candidate_norm = math.sqrt(sum(v * v for v in candidate_counts.values()))
        if query_norm == 0.0 or candidate_norm == 0.0:
            return 0.0
        return dot / (query_norm * candidate_norm)


def get_scorer(name: str, remove_stop_words: bool = True) -> ResonanceScorer:
    """Build a resonance scorer by name.

    Args:
        name: Scorer identifier. Currently supports ``"token_overlap"``.
        remove_stop_words: Forwarded to the scorer.

    Returns:
        A :class:`ResonanceScorer` instance.

    Raises:
        ValueError: If *name* does not name a known scorer.
    """
    if name == "token_overlap":
        return TokenOverlapResonanceScorer(remove_stop_words=remove_stop_words)
    raise ValueError(f"Unknown resonance scorer: {name!r}")

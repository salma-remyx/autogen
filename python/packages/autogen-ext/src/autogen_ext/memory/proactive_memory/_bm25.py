"""Okapi BM25 ranker for the proactive memory bank.

Dependency-free, parameter-free retrieval. The proactive-memory design pairs its
*intervention policy* with BM25 retrieval over the structured memory bank; this
module keeps that retrieval primitive. The intervention decision itself lives in
:mod:`_proactive_memory`.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# A small stopword set keeps decision-relevant content tokens dominant. Kept
# inline (rather than depending on nltk/sklearn) so the memory backend has no
# extra dependencies beyond autogen-core.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "it",
        "as",
        "from",
        "you",
        "your",
        "we",
        "our",
        "i",
        "do",
        "does",
        "did",
        "not",
        "no",
        "so",
        "can",
        "will",
        "just",
        "into",
    }
)


def tokenize_text(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords and single characters."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


class BM25:
    """Okapi BM25 ranker over a fixed corpus of documents.

    Args:
        documents: The corpus to rank against.
        k1: Term-frequency saturation (typical: 1.2–2.0).
        b: Length-normalization strength in ``[0, 1]`` (typical: 0.75).
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs: List[List[str]] = [tokenize_text(d) for d in documents]
        self._n = len(self._docs)
        self._avgdl = (sum(len(d) for d in self._docs) / self._n) if self._n else 0.0

        # Document frequency per term -> smoothed inverse document frequency.
        df: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (self._n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

        # Per-document term frequencies.
        self._tf: List[dict[str, int]] = []
        for doc in self._docs:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self._tf.append(tf)

    def score(self, query: str) -> List[float]:
        """Return one BM25 score per document, in original corpus order."""
        q_terms = set(tokenize_text(query))
        avgdl = self._avgdl or 1.0
        scores: List[float] = []
        for i, doc in enumerate(self._docs):
            tf = self._tf[i]
            dl = len(doc) or 1
            denom_base = self._k1 * (1.0 - self._b + self._b * (dl / avgdl))
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                s += self._idf.get(term, 0.0) * (f * (self._k1 + 1.0)) / (f + denom_base)
            scores.append(s)
        return scores

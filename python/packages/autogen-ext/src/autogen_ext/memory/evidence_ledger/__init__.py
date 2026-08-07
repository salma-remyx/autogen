"""Provenance-constrained Structured Evidence Ledger memory backend.

Adapted from *LedgerMind: Provenance-Constrained Multimodal Agentic Reasoning
with a Structured Evidence Ledger* (arXiv:2607.28374).
"""

from ._evidence_ledger import (
    STATUS_ACTIVE,
    STATUS_SUPERSEDED,
    TYPE_CLAIM,
    TYPE_DECISION,
    TYPE_OBSERVATION,
    TYPE_RETRIEVED,
    EvidenceLedgerMemory,
    EvidenceLedgerMemoryConfig,
)

__all__ = [
    "EvidenceLedgerMemory",
    "EvidenceLedgerMemoryConfig",
    "STATUS_ACTIVE",
    "STATUS_SUPERSEDED",
    "TYPE_OBSERVATION",
    "TYPE_RETRIEVED",
    "TYPE_CLAIM",
    "TYPE_DECISION",
]

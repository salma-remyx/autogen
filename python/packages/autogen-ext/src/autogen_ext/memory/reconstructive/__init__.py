"""Reconstructive memory extension.

Adapted from *MemHarness: Memory Is Reconstructed, Not Replayed* (arXiv:2607.28272).
See :mod:`autogen_ext.memory.reconstructive._reconstructive_memory` for details.
"""

from ._reconstructive_memory import (
    DEFAULT_MAX_CONTEXT_MESSAGES,
    DEFAULT_REJECT_SENTINEL,
    ReconstructiveMemory,
)

__all__ = [
    "ReconstructiveMemory",
    "DEFAULT_REJECT_SENTINEL",
    "DEFAULT_MAX_CONTEXT_MESSAGES",
]

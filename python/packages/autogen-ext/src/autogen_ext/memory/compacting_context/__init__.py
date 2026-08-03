"""Agentic context-lifecycle memory with validated compaction.

Public surface::

    from autogen_ext.memory.compacting_context import CompactingContextMemory

The backend plugs into the existing :class:`~autogen_core.memory.Memory` ABC and is
consumed on the agent forward path exactly like ``canvas`` / ``chromadb`` / ``mem0`` /
``redis`` (e.g. ``AssistantAgent(memory=[CompactingContextMemory(...)])``).
"""

from ._compacting_context_memory import CompactingContextMemory, CompactingContextMemoryConfig

__all__ = ["CompactingContextMemory", "CompactingContextMemoryConfig"]

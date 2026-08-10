"""Configuration for :class:`HierarchicalGraphMemory`.

Adapted from HiGram -- *Hierarchical Graph Memory for LLM Agents with Path-level
Localization and Rewrite* (arxiv:2608.05095). The knobs below correspond to the
localization and rewrite thresholds of the adapted port: every threshold operates
on a parameter-free lexical-overlap similarity in ``[0.0, 1.0]``, which is the
target-native substitute for the paper's learned query/update-conditioned
estimator.
"""

from pydantic import BaseModel, Field


class HierarchicalGraphMemoryConfig(BaseModel):
    """Configuration for :class:`~autogen_ext.memory.hierarchical_graph.HierarchicalGraphMemory`.

    All thresholds operate on a parameter-free lexical-overlap similarity in
    ``[0.0, 1.0]`` (the adapted-port substitute for the paper's learned
    query-conditioned estimator).
    """

    name: str | None = None
    """Optional identifier for this memory instance."""

    cluster_score_threshold: float = Field(
        default=0.1,
        description=(
            "Minimum overlap for an upper-level cluster to be reused on add or "
            "included in the localized evidence path on query. Below it a new "
            "cluster is created (add) / the cluster is excluded (query)."
        ),
    )
    unit_score_threshold: float = Field(
        default=0.1,
        description="Minimum overlap for a MemoryUnit to be retained in the localized evidence path on query.",
    )
    conflict_score_threshold: float = Field(
        default=0.5,
        description=(
            "Overlap above which an incoming add is treated as a revision/conflict "
            "of an existing MemoryUnit and triggers a coordinated rewrite "
            "(intra-unit rewrite plus inter-unit dependency propagation) instead "
            "of appending a new unit."
        ),
    )
    top_k_clusters: int = Field(
        default=3, ge=1, description="Maximum upper-level clusters retained in a query evidence path."
    )
    top_k_units: int = Field(
        default=5, ge=1, description="Maximum MemoryUnits retained per cluster in a query evidence path."
    )

"""Action normalization for experience-memory decision graphs.

Two pieces of the node schema from *Experience Memory Graph: One-Shot Error
Correction for Agents* (arXiv:2607.13884), realized deterministically and
without dependencies (Mode 2 adaptation):

* actions are normalized into ``(verb, object, receptacle)`` tuples — the
  paper's decision-node key — so superficially different phrasings
  ("unlock door" vs "unlock the door") resolve to the same node and align
  as equal during edit-path computation;
* invalid no-op actions — a verbatim retry of the immediately preceding
  action, which cannot have changed the environment state — are
  *parallelized* onto the last valid step instead of advancing the
  trajectory. This is the paper's key trick for meaningful corrections: a
  retry surfaces as an explicit "avoid X under observation o" edit rather
  than silently extending the failed path.

The paper learns nothing here either: verbs are first tokens, articles are
dropped, and receptacles are split off only for ``put``/``place``-style
verbs ("put mug in cabinet" -> verb "put", object "mug", receptacle
"cabinet"). Everything else is a plain ``(verb, object)`` pair.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

# A trajectory step is an (observation, action) pair. Mirrors the Step alias
# in _experience_memory_graph; redeclared to keep this module import-free.
Step = Tuple[str, str]

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_RECEPTACLE_PREPS = re.compile(r"\s+(?:in|on|into|onto|inside|under)\s+")
_RECEPTACLE_VERBS = frozenset({"put", "place", "set", "insert", "drop"})


@dataclass(frozen=True)
class NormalizedAction:
    """The paper's ``(type, object, receptacle)`` action tuple, parsed deterministically."""

    verb: str
    obj: str
    receptacle: str = ""

    def signature(self) -> Tuple[str, str, str]:
        """Hashable matching key for this action."""
        return (self.verb, self.obj, self.receptacle)


def normalize_action(action: str) -> NormalizedAction:
    """Parse a free-form action string into a ``(verb, object, receptacle)`` tuple."""
    tokens = _ARTICLES.sub(" ", " ".join(action.lower().split())).split()
    if not tokens:
        return NormalizedAction(verb="", obj="")
    verb = tokens[0]
    obj = " ".join(tokens[1:])
    receptacle = ""
    if verb in _RECEPTACLE_VERBS and obj:
        parts = _RECEPTACLE_PREPS.split(obj, maxsplit=1)
        if len(parts) == 2:
            obj, receptacle = parts
    return NormalizedAction(verb=verb, obj=obj, receptacle=receptacle)


def action_signature(action: str) -> Tuple[str, str, str]:
    """Stable tuple matching key for an action string."""
    return normalize_action(action).signature()


def canonical_action(action: str) -> str:
    """Canonical string form of an action's tuple, for sequence alignment.

    Two actions are equal under alignment iff their tuples are equal.
    """
    return " ".join(part for part in normalize_action(action).signature() if part)


def parallelize_noop_retries(steps: Sequence[Step]) -> Tuple[List[Step], List[Tuple[int, Step]]]:
    """Split a trajectory into its valid backbone and parallelized no-op retries.

    A step whose action tuple repeats the immediately preceding step's is an
    invalid no-op (the environment state cannot have changed); per the paper
    it is attached to the last valid node instead of advancing the
    trajectory. Returns ``(valid_steps, retries)`` where each retry is
    ``(backbone_index, (observation, action))``.
    """
    valid: List[Step] = []
    retries: List[Tuple[int, Step]] = []
    for observation, action in steps:
        if valid and action_signature(action) == action_signature(valid[-1][1]):
            retries.append((len(valid) - 1, (observation, action)))
            continue
        valid.append((observation, action))
    return valid, retries


def avoid_edits(retries: Sequence[Tuple[int, Step]], backbone_observations: Sequence[str]) -> List[Dict[str, Any]]:
    """Render parallelized no-op retries as explicit ``avoid`` graph edits.

    Each edit carries the observation the retry happened under (falling back
    to the anchor step's observation) so the correction reads "avoid X under
    observation o", matching the paper's delete-edit semantics.
    """
    edits: List[Dict[str, Any]] = []
    for backbone_index, (observation, action) in retries:
        anchor = observation
        if not anchor and 0 <= backbone_index < len(backbone_observations):
            anchor = backbone_observations[backbone_index]
        edits.append(
            {
                "op": "avoid",
                "at": backbone_index + 1,
                "observation": anchor,
                "actions": [action],
            }
        )
    return edits

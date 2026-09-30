from dataclasses import dataclass
from typing import Any

# Add more if necessary
MetadataType = Any

type Score = float
type Metadata = dict[str, MetadataType]
type ContinueTask = bool


@dataclass(frozen=True)
class Scoring:
    score: Score
    metadata: Metadata
    continue_task: ContinueTask

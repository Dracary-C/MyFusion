from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class StepRecord:
    round_index: int
    restored: Any
    assessment: Any = None
    stop: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class RestoreAssessRepeat:
    def __init__(
        self,
        restorer: Callable[[Any], Any],
        assessor: Callable[[Any], Any],
        stopper: Optional[Callable[[Any, Any, int], bool]] = None,
        max_rounds: int = 3,
    ) -> None:
        self.restorer = restorer
        self.assessor = assessor
        self.stopper = stopper
        self.max_rounds = max_rounds

    def run(self, image: Any) -> tuple[Any, List[StepRecord]]:
        history: List[StepRecord] = []
        current = image
        for round_index in range(self.max_rounds):
            restored = self.restorer(current)
            assessment = self.assessor(restored)
            stop = bool(self.stopper(restored, assessment, round_index)) if self.stopper else False
            history.append(
                StepRecord(
                    round_index=round_index,
                    restored=restored,
                    assessment=assessment,
                    stop=stop,
                )
            )
            current = restored
            if stop:
                break
        return current, history


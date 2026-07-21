"""
Data models for the three-task Visual Cognition / Spatial Reasoning leaderboard.

Each task (do_you_see_me, minds_eye, spatial) is submitted independently and
produces a :class:`TaskScore`. Stable model IDs connect those task results into
one leaderboard identity.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class GroupResult:
    """Accuracy for one sub-group of a task (a capability or a dataset)."""
    name: str
    total_samples: int
    correct_samples: int
    accuracy: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Diagnostics:
    """Robustness diagnostics from the paper's optional evaluation conditions."""
    conditions_present: List[str]
    standard_accuracy: Optional[float] = None
    cot_accuracy: Optional[float] = None
    cot_delta: Optional[float] = None                  # acc(cot) - acc(standard)
    shortcut_score: Optional[float] = None             # acc(no_image); lower better
    shortcut_score_cot: Optional[float] = None         # acc(no_image, cot); lower better
    hallucination_resistance: Optional[float] = None   # acc(no_image_plus); higher better
    hallucination_resistance_cot: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class TaskScore:
    """The result of scoring a single task submission."""
    task_id: str
    submission_id: str
    model_name: str
    submitted_at: datetime
    accuracy: float
    total_samples: int
    correct_samples: int
    model_id: Optional[str] = None
    groups: Dict[str, GroupResult] = field(default_factory=dict)
    analysis: Dict[str, Dict[str, GroupResult]] = field(default_factory=dict)
    diagnostics: Optional[Diagnostics] = None
    model_meta: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Paper-aligned reporting:
    #   macro_accuracy - benchmark headline score using its documented
    #                    task/dataset aggregation contract.
    #   task_spread    - descriptive standard deviation across heterogeneous
    #                    task/dataset scores. It is not repeated-run uncertainty.
    #   accuracy_std   - backward-compatible alias for task_spread.
    #   random_baseline- chance-level accuracy (Mind's Eye reports this).
    #   score_method   - stable identifier for the headline aggregation contract.
    #   grading        - grading pipeline and pinned judge metadata.
    macro_accuracy: Optional[float] = None
    task_spread: Optional[float] = None
    accuracy_std: Optional[float] = None
    random_baseline: Optional[float] = None
    score_method: Optional[str] = None
    grading: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "submission_id": self.submission_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "submitted_at": self.submitted_at.isoformat()
            if isinstance(self.submitted_at, datetime) else self.submitted_at,
            "accuracy": round(self.accuracy, 4),
            "micro_accuracy": round(self.accuracy, 4),
            "macro_accuracy": round(self.macro_accuracy, 4)
            if self.macro_accuracy is not None else None,
            "task_spread": round(self.task_spread, 4)
            if self.task_spread is not None else None,
            "accuracy_std": round(self.accuracy_std, 4)
            if self.accuracy_std is not None else None,
            "random_baseline": round(self.random_baseline, 4)
            if self.random_baseline is not None else None,
            "score_method": self.score_method,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "groups": {k: asdict(v) for k, v in self.groups.items()},
            "analysis": {
                axis: {key: asdict(value) for key, value in values.items()}
                for axis, values in self.analysis.items()
            },
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "grading": self.grading,
            "model_meta": self.model_meta,
            "metadata": self.metadata,
        }

"""
Data models for the three-task Visual Cognition / Spatial Reasoning leaderboard.

Each task (do_you_see_me, minds_eye, spatial) is submitted independently and
produces a :class:`TaskScore`. The leaderboard store keys these by model name
and combines do_you_see_me + minds_eye into the Visual Cognition Index (VCI).
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
    hallucination_resistance: Optional[float] = None   # acc(no_image_plus); higher better

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
    groups: Dict[str, GroupResult] = field(default_factory=dict)
    diagnostics: Optional[Diagnostics] = None
    model_meta: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "submission_id": self.submission_id,
            "model_name": self.model_name,
            "submitted_at": self.submitted_at.isoformat()
            if isinstance(self.submitted_at, datetime) else self.submitted_at,
            "accuracy": round(self.accuracy, 4),
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "groups": {k: asdict(v) for k, v in self.groups.items()},
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "model_meta": self.model_meta,
            "metadata": self.metadata,
        }

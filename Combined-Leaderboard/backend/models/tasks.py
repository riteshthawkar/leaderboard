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
    # Paper-faithful reporting:
    #   macro_accuracy - mean of per-group accuracies (matches the papers' "Avg"
    #                    column, which averages across sub-tasks / datasets).
    #   accuracy_std   - std. dev. of per-group accuracies (papers report
    #                    "mean accuracy +/- std. dev.").
    #   random_baseline- chance-level accuracy (Mind's-Eye reports this).
    #   grading        - which grading pipeline was used (LLM judge/extractor or
    #                    deterministic fallback) and the judge model.
    macro_accuracy: Optional[float] = None
    accuracy_std: Optional[float] = None
    random_baseline: Optional[float] = None
    grading: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "submission_id": self.submission_id,
            "model_name": self.model_name,
            "submitted_at": self.submitted_at.isoformat()
            if isinstance(self.submitted_at, datetime) else self.submitted_at,
            "accuracy": round(self.accuracy, 4),
            "macro_accuracy": round(self.macro_accuracy, 4)
            if self.macro_accuracy is not None else None,
            "accuracy_std": round(self.accuracy_std, 4)
            if self.accuracy_std is not None else None,
            "random_baseline": round(self.random_baseline, 4)
            if self.random_baseline is not None else None,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "groups": {k: asdict(v) for k, v in self.groups.items()},
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "grading": self.grading,
            "model_meta": self.model_meta,
            "metadata": self.metadata,
        }

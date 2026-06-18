"""
Data models for the unified Visual Cognition golden-set evaluation.

Kept separate from models/submission.py so the existing per-benchmark pipeline
continues to work unchanged.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class EvalCondition(Enum):
    """Evaluation conditions following the CoT-Spatial paper design."""
    STANDARD = "standard"          # normal image + question
    COT = "cot"                    # chain-of-thought prompting
    NO_IMAGE = "no_image"          # blank image (shortcut probe)
    NO_IMAGE_PLUS = "no_image_plus"  # blank image + "cannot determine" is correct


@dataclass
class GoldenTaskResult:
    task: str
    benchmark: str
    capability: str
    layer: str
    total_samples: int
    correct_samples: int
    accuracy: float


@dataclass
class GoldenCapabilityResult:
    capability: str
    layer: str
    total_samples: int
    correct_samples: int
    accuracy: float


@dataclass
class GoldenLayerResult:
    layer: str
    label: str
    total_samples: int
    correct_samples: int
    accuracy: float


@dataclass
class GoldenDiagnostics:
    """Robustness diagnostics from optional conditions (the paper's contribution)."""
    conditions_present: List[str]
    cot_delta: Optional[float] = None              # acc(cot) - acc(standard)
    shortcut_score: Optional[float] = None         # acc(no_image); lower is better
    hallucination_resistance: Optional[float] = None  # acc(no_image_plus); higher better
    standard_accuracy: Optional[float] = None
    cot_accuracy: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class GoldenSubmissionScore:
    submission_id: str
    model_name: str
    submitted_at: datetime
    vci: float
    overall_accuracy: float
    total_samples: int
    correct_samples: int
    task_results: Dict[str, GoldenTaskResult] = field(default_factory=dict)
    capability_results: Dict[str, GoldenCapabilityResult] = field(default_factory=dict)
    layer_results: Dict[str, GoldenLayerResult] = field(default_factory=dict)
    diagnostics: Optional[GoldenDiagnostics] = None
    model_meta: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "submission_id": self.submission_id,
            "model_name": self.model_name,
            "submitted_at": self.submitted_at.isoformat()
            if isinstance(self.submitted_at, datetime) else self.submitted_at,
            "vci": round(self.vci, 4),
            "overall_accuracy": round(self.overall_accuracy, 4),
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "task_results": {k: asdict(v) for k, v in self.task_results.items()},
            "capability_results": {k: asdict(v) for k, v in self.capability_results.items()},
            "layer_results": {k: asdict(v) for k, v in self.layer_results.items()},
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "model_meta": self.model_meta,
            "metadata": self.metadata,
        }

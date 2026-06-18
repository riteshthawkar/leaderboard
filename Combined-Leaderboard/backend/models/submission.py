"""
Data models for leaderboard submissions and scoring.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class BenchmarkType(Enum):
    """Supported benchmark types."""
    DO_YOU_SEE_ME = "do_you_see_me"
    MINDS_EYE = "minds_eye"


class TaskDifficulty(Enum):
    """Difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class PredictionScore:
    """Individual prediction scoring result."""
    image_name: str
    question: str
    ground_truth: str
    prediction: str
    score: float  # 0.0 or 1.0
    reasoning: Optional[str] = None
    task_name: Optional[str] = None
    difficulty: Optional[str] = None


@dataclass
class TaskResults:
    """Aggregated results for a single task."""
    task_name: str
    total_samples: int
    correct_samples: int
    accuracy: float
    std_dev: float = 0.0
    predictions: List[PredictionScore] = field(default_factory=list)


@dataclass
class SubmissionScore:
    """Complete submission score results."""
    submission_id: str
    model_name: str
    benchmark: BenchmarkType
    submitted_at: datetime
    task_results: Dict[str, TaskResults]
    overall_accuracy: float
    total_samples: int
    correct_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "submission_id": self.submission_id,
            "model_name": self.model_name,
            "benchmark": self.benchmark.value,
            "submitted_at": self.submitted_at.isoformat(),
            "overall_accuracy": self.overall_accuracy,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "task_results": {
                task_name: {
                    "task_name": result.task_name,
                    "total_samples": result.total_samples,
                    "correct_samples": result.correct_samples,
                    "accuracy": result.accuracy,
                    "std_dev": result.std_dev,
                }
                for task_name, result in self.task_results.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""
    rank: int
    submission_id: str
    model_name: str
    benchmark: str
    overall_accuracy: float
    total_samples: int
    correct_samples: int
    submitted_at: str
    task_accuracy: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

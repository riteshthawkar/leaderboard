"""
Leaderboard management and ranking system.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from models.submission import LeaderboardEntry, SubmissionScore, BenchmarkType
from config import RESULTS_DIR


class LeaderboardManager:
    """Manages leaderboard storage and retrieval."""

    def __init__(self, leaderboard_file: Path = None):
        """Initialize leaderboard manager.
        
        Args:
            leaderboard_file: Path to leaderboard JSON file
        """
        if leaderboard_file is None:
            self.leaderboard_file = RESULTS_DIR / "leaderboard.json"
        else:
            self.leaderboard_file = Path(leaderboard_file)

        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure leaderboard file exists."""
        if not self.leaderboard_file.exists():
            self.leaderboard_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_leaderboard({})

    def _load_leaderboard(self) -> Dict:
        """Load leaderboard data."""
        try:
            with open(self.leaderboard_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_leaderboard(self, data: Dict):
        """Save leaderboard data."""
        self.leaderboard_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.leaderboard_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_submission(self, submission_score: SubmissionScore):
        """Add submission to leaderboard.
        
        Args:
            submission_score: Submission to add
        """
        data = self._load_leaderboard()
        benchmark_key = submission_score.benchmark.value

        if benchmark_key not in data:
            data[benchmark_key] = []

        entry = {
            "submission_id": submission_score.submission_id,
            "model_name": submission_score.model_name,
            "overall_accuracy": submission_score.overall_accuracy,
            "total_samples": submission_score.total_samples,
            "correct_samples": submission_score.correct_samples,
            "submitted_at": submission_score.submitted_at.isoformat(),
            "task_accuracy": {
                task: result.accuracy
                for task, result in submission_score.task_results.items()
            },
        }

        data[benchmark_key].append(entry)
        self._save_leaderboard(data)

    def get_leaderboard(
        self,
        benchmark: Optional[BenchmarkType] = None,
        limit: Optional[int] = None,
        sort_by: str = "overall_accuracy",
    ) -> List[LeaderboardEntry]:
        """Get leaderboard rankings.
        
        Args:
            benchmark: Filter by benchmark (None = all)
            limit: Limit number of results
            sort_by: Sort key (default: overall_accuracy)
            
        Returns:
            Sorted list of leaderboard entries
        """
        data = self._load_leaderboard()
        entries = []

        for bench_key, bench_data in data.items():
            if benchmark and bench_key != benchmark.value:
                continue

            for item in bench_data:
                entry = LeaderboardEntry(
                    rank=0,  # Will be set below
                    submission_id=item["submission_id"],
                    model_name=item["model_name"],
                    benchmark=bench_key,
                    overall_accuracy=item["overall_accuracy"],
                    total_samples=item["total_samples"],
                    correct_samples=item["correct_samples"],
                    submitted_at=item["submitted_at"],
                    task_accuracy=item.get("task_accuracy", {}),
                )
                entries.append(entry)

        # Sort by specified key
        entries.sort(
            key=lambda x: getattr(x, sort_by),
            reverse=(sort_by == "overall_accuracy"),
        )

        # Assign ranks
        for rank, entry in enumerate(entries, 1):
            entry.rank = rank

        if limit:
            entries = entries[:limit]

        return entries

    def get_leaderboard_by_task(
        self,
        task_name: str,
        benchmark: Optional[BenchmarkType] = None,
        limit: Optional[int] = None,
    ) -> List[LeaderboardEntry]:
        """Get leaderboard filtered by task.
        
        Args:
            task_name: Task to filter by
            benchmark: Filter by benchmark (None = all)
            limit: Limit number of results
            
        Returns:
            Sorted list of leaderboard entries
        """
        data = self._load_leaderboard()
        entries = []

        for bench_key, bench_data in data.items():
            if benchmark and bench_key != benchmark.value:
                continue

            for item in bench_data:
                task_accuracy = item.get("task_accuracy", {})
                if task_name not in task_accuracy:
                    continue

                entry = LeaderboardEntry(
                    rank=0,
                    submission_id=item["submission_id"],
                    model_name=item["model_name"],
                    benchmark=bench_key,
                    overall_accuracy=task_accuracy[task_name],
                    total_samples=item["total_samples"],
                    correct_samples=item["correct_samples"],
                    submitted_at=item["submitted_at"],
                    task_accuracy={task_name: task_accuracy[task_name]},
                )
                entries.append(entry)

        # Sort by task accuracy
        entries.sort(
            key=lambda x: x.overall_accuracy,
            reverse=True,
        )

        # Assign ranks
        for rank, entry in enumerate(entries, 1):
            entry.rank = rank

        if limit:
            entries = entries[:limit]

        return entries

    def get_submission_details(self, submission_id: str) -> Optional[Dict]:
        """Get detailed information about a submission.
        
        Args:
            submission_id: Submission ID
            
        Returns:
            Submission details or None if not found
        """
        # Try to load from results file
        result_file = RESULTS_DIR / f"{submission_id}.json"
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        # Fallback: search leaderboard
        data = self._load_leaderboard()
        for bench_data in data.values():
            for item in bench_data:
                if item["submission_id"] == submission_id:
                    return item

        return None

    def get_model_submissions(self, model_name: str) -> List[Dict]:
        """Get all submissions for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            List of submissions
        """
        data = self._load_leaderboard()
        submissions = []

        for bench_data in data.values():
            for item in bench_data:
                if item["model_name"] == model_name:
                    submissions.append(item)

        # Sort by submission date
        submissions.sort(key=lambda x: x["submitted_at"], reverse=True)
        return submissions

    def get_statistics(self, benchmark: Optional[BenchmarkType] = None) -> Dict:
        """Get leaderboard statistics.
        
        Args:
            benchmark: Filter by benchmark (None = all)
            
        Returns:
            Statistics dictionary
        """
        data = self._load_leaderboard()
        all_accuracies = []
        all_submissions = 0
        unique_models = set()

        for bench_key, bench_data in data.items():
            if benchmark and bench_key != benchmark.value:
                continue

            for item in bench_data:
                all_accuracies.append(item["overall_accuracy"])
                all_submissions += 1
                unique_models.add(item["model_name"])

        if not all_accuracies:
            return {
                "total_submissions": 0,
                "unique_models": 0,
                "average_accuracy": 0.0,
                "best_accuracy": 0.0,
                "worst_accuracy": 0.0,
            }

        import statistics
        return {
            "total_submissions": all_submissions,
            "unique_models": len(unique_models),
            "average_accuracy": statistics.mean(all_accuracies),
            "median_accuracy": statistics.median(all_accuracies),
            "best_accuracy": max(all_accuracies),
            "worst_accuracy": min(all_accuracies),
            "std_dev": statistics.stdev(all_accuracies) if len(all_accuracies) > 1 else 0.0,
        }

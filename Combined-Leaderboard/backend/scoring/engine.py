"""
Core scoring engine for evaluating submissions against ground truth.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import statistics

from models.submission import (
    BenchmarkType,
    PredictionScore,
    TaskResults,
    SubmissionScore,
)
from data_handlers.ground_truth import GroundTruthManager, GroundTruthItem
from data_handlers.submission import SubmissionParser, PredictionItem
from utils.answer_extractor import AnswerComparator


class ScoringEngine:
    """Main engine for scoring submissions."""

    def __init__(self, use_ollama: bool = False):
        """Initialize scoring engine.
        
        Args:
            use_ollama: Whether to use Ollama for option extraction
        """
        self.gt_manager = GroundTruthManager()
        self.use_ollama = use_ollama

    def score_submission(
        self,
        submission_file: Path,
        benchmark: BenchmarkType,
        model_name: str,
        task_name: Optional[str] = None,
    ) -> SubmissionScore:
        """Score a complete submission.
        
        Args:
            submission_file: Path to submission file (CSV or JSON)
            benchmark: Benchmark type (Do-You-See-Me or Mind's-Eye)
            model_name: Name of the model/submitter
            task_name: Optional specific task to evaluate (if None, evaluates all)
            
        Returns:
            SubmissionScore with detailed results
        """
        # Parse submission
        predictions = SubmissionParser.parse_submission(submission_file)

        # Score each task
        task_results: Dict[str, TaskResults] = {}
        all_scores: List[float] = []

        if benchmark == BenchmarkType.MINDS_EYE:
            task_results, all_scores = self._score_minds_eye(
                predictions, task_name
            )
        elif benchmark == BenchmarkType.DO_YOU_SEE_ME:
            task_results, all_scores = self._score_do_you_see_me(
                predictions, task_name
            )

        # Calculate overall statistics (WEIGHTED average, not simple mean)
        # This ensures tasks with different sample sizes are properly weighted
        total_samples = sum(r.total_samples for r in task_results.values())
        correct_samples = sum(r.correct_samples for r in task_results.values())
        
        # Calculate weighted accuracy: (total correct) / (total samples)
        # This is the correct statistical approach
        overall_accuracy = (
            correct_samples / total_samples if total_samples > 0 else 0.0
        )

        submission_score = SubmissionScore(
            submission_id=str(uuid.uuid4()),
            model_name=model_name,
            benchmark=benchmark,
            submitted_at=datetime.now(),
            task_results=task_results,
            overall_accuracy=overall_accuracy,
            total_samples=total_samples,
            correct_samples=correct_samples,
            metadata={
                "submission_file": submission_file.name,
                "uses_ollama": self.use_ollama,
            },
        )

        return submission_score

    def _score_minds_eye(
        self,
        predictions: Dict[str, List[PredictionItem]],
        task_name: Optional[str] = None,
    ) -> Tuple[Dict[str, TaskResults], List[float]]:
        """Score Mind's-Eye submission.
        
        Returns:
            Tuple of (task_results, all_scores)
        """
        task_results: Dict[str, TaskResults] = {}
        all_scores: List[float] = []

        for pred_task_name, pred_list in predictions.items():
            # Skip if specific task requested and doesn't match
            if task_name and pred_task_name != task_name:
                continue

            try:
                gt_dict = self.gt_manager.get_minds_eye_ground_truth(pred_task_name)
            except ValueError:
                continue

            # Score each prediction
            pred_scores: List[PredictionScore] = []
            task_scores: List[float] = []

            for pred in pred_list:
                if pred.image_name not in gt_dict:
                    continue

                gt_item = gt_dict[pred.image_name]
                is_correct, reasoning = AnswerComparator.compare_answers(
                    gt_item.answer,
                    pred.prediction,
                    use_ollama=self.use_ollama,
                )

                score = 1.0 if is_correct else 0.0
                task_scores.append(score)
                all_scores.append(score)

                pred_scores.append(
                    PredictionScore(
                        image_name=pred.image_name,
                        question=gt_item.question,
                        ground_truth=gt_item.answer,
                        prediction=pred.prediction,
                        score=score,
                        reasoning=reasoning,
                        task_name=pred_task_name,
                        difficulty=gt_item.difficulty,
                    )
                )

            # Calculate task statistics
            if task_scores:
                accuracy = statistics.mean(task_scores)
                std_dev = (
                    statistics.stdev(task_scores)
                    if len(task_scores) > 1
                    else 0.0
                )

                task_results[pred_task_name] = TaskResults(
                    task_name=pred_task_name,
                    total_samples=len(task_scores),
                    correct_samples=sum(1 for s in task_scores if s == 1.0),
                    accuracy=accuracy,
                    std_dev=std_dev,
                    predictions=pred_scores,
                )

        return task_results, all_scores

    def _score_do_you_see_me(
        self,
        predictions: Dict[str, List[PredictionItem]],
        task_name: Optional[str] = None,
    ) -> Tuple[Dict[str, TaskResults], List[float]]:
        """Score Do-You-See-Me submission.
        
        Returns:
            Tuple of (task_results, all_scores)
        """
        task_results: Dict[str, TaskResults] = {}
        all_scores: List[float] = []

        for pred_task_name, pred_list in predictions.items():
            # Skip if specific task requested and doesn't match
            if task_name and pred_task_name != task_name:
                continue

            # Try both 2D and 3D variants
            gt_dict = None
            is_3d = False

            try:
                gt_dict = self.gt_manager.get_do_you_see_me_ground_truth(
                    pred_task_name, is_3d=False
                )
            except ValueError:
                pass

            if gt_dict is None:
                try:
                    gt_dict = self.gt_manager.get_do_you_see_me_ground_truth(
                        pred_task_name, is_3d=True
                    )
                    is_3d = True
                except ValueError:
                    continue

            # Score each prediction
            pred_scores: List[PredictionScore] = []
            task_scores: List[float] = []

            for pred in pred_list:
                if pred.image_name not in gt_dict:
                    continue

                gt_item = gt_dict[pred.image_name]
                is_correct, reasoning = AnswerComparator.compare_answers(
                    gt_item.answer,
                    pred.prediction,
                    use_ollama=self.use_ollama,
                )

                score = 1.0 if is_correct else 0.0
                task_scores.append(score)
                all_scores.append(score)

                pred_scores.append(
                    PredictionScore(
                        image_name=pred.image_name,
                        question=gt_item.question,
                        ground_truth=gt_item.answer,
                        prediction=pred.prediction,
                        score=score,
                        reasoning=reasoning,
                        task_name=f"{pred_task_name}_{'3d' if is_3d else '2d'}",
                        difficulty=gt_item.difficulty,
                    )
                )

            # Calculate task statistics
            if task_scores:
                accuracy = statistics.mean(task_scores)
                std_dev = (
                    statistics.stdev(task_scores)
                    if len(task_scores) > 1
                    else 0.0
                )

                result_task_name = f"{pred_task_name}_{'3d' if is_3d else '2d'}"
                task_results[result_task_name] = TaskResults(
                    task_name=result_task_name,
                    total_samples=len(task_scores),
                    correct_samples=sum(1 for s in task_scores if s == 1.0),
                    accuracy=accuracy,
                    std_dev=std_dev,
                    predictions=pred_scores,
                )

        return task_results, all_scores

    def save_results(
        self,
        submission_score: SubmissionScore,
        output_dir: Path = None,
    ) -> Path:
        """Save scoring results to disk.
        
        Args:
            submission_score: Scores to save
            output_dir: Output directory (defaults to results/)
            
        Returns:
            Path to saved results
        """
        from config import RESULTS_DIR

        if output_dir is None:
            output_dir = RESULTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed JSON results
        result_file = output_dir / f"{submission_score.submission_id}.json"
        with open(result_file, "w") as f:
            json.dump(submission_score.to_dict(), f, indent=2, default=str)

        # Save per-task CSV files
        for task_name, task_result in submission_score.task_results.items():
            csv_file = output_dir / f"{submission_score.submission_id}_{task_name}.csv"
            
            import pandas as pd
            df = pd.DataFrame([
                {
                    "image_name": pred.image_name,
                    "question": pred.question,
                    "ground_truth": pred.ground_truth,
                    "prediction": pred.prediction,
                    "score": pred.score,
                    "reasoning": pred.reasoning,
                    "difficulty": pred.difficulty,
                }
                for pred in task_result.predictions
            ])
            df.to_csv(csv_file, index=False)

        return result_file

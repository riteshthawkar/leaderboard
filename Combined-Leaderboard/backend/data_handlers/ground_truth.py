"""
Data handlers for loading ground truth from CSV and JSON formats.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from config import (
    DO_YOU_SEE_ME_ROOT,
    DO_YOU_SEE_ME_2D_ROOT,
    DO_YOU_SEE_ME_3D_ROOT,
    MINDS_EYE_DATA_ROOT,
    DO_YOU_SEE_ME_TASKS,
    MINDS_EYE_TASKS,
    MINDS_EYE_ANSWER_FIELDS,
)

# Get logger
logger = logging.getLogger(__name__)


@dataclass
class GroundTruthItem:
    """Single ground truth item."""
    image_name: str
    question: str
    answer: str
    task_name: str
    difficulty: Optional[str] = None


class DoYouSeeMeHandler:
    """Handles Do-You-See-Me CSV ground truth format."""

    @staticmethod
    def load_ground_truth(task_name: str, is_3d: bool = False) -> Dict[str, GroundTruthItem]:
        """Load ground truth for a Do-You-See-Me task.
        
        Args:
            task_name: Task name (e.g., 'visual_spatial')
            is_3d: Whether to load 3D version
            
        Returns:
            Dictionary mapping image_name to GroundTruthItem
            
        Raises:
            FileNotFoundError: If ground truth file not found
            ValueError: If CSV is malformed or missing required columns
        """
        try:
            if is_3d:
                variant = "3D"
                # The downloaded 3D dataset may use either a shared
                # 3D_DoYouSeeMe/<task>/ folder or a per-task 3D_<task>/ folder;
                # accept whichever layout is present.
                candidates = [
                    DO_YOU_SEE_ME_3D_ROOT / task_name / "dataset_info.csv",
                    DO_YOU_SEE_ME_ROOT / f"3D_{task_name}" / "dataset_info.csv",
                ]
                csv_path = next((p for p in candidates if p.exists()), candidates[0])
            else:
                variant = "2D"
                csv_path = DO_YOU_SEE_ME_2D_ROOT / task_name / "dataset_info.csv"

            if not csv_path.exists():
                logger.debug(f"Ground truth file not present: {csv_path} ({variant})")
                raise FileNotFoundError(f"Ground truth file not found: {csv_path}")

            logger.info(f"Loading {variant} ground truth for task '{task_name}' from {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.ParserError as e:
                logger.error(f"CSV parsing error for {csv_path}: {e}")
                raise ValueError(f"Corrupted or malformed CSV file: {e}")
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_path}: {e}")
                raise

            # Validate required columns (question is optional; only used for context)
            required_cols = {"filename", "answer"}
            actual_cols = set(df.columns)
            missing = required_cols - actual_cols
            if missing:
                logger.error(f"Missing required columns in {csv_path}: {missing}")
                raise ValueError(f"CSV missing required columns: {missing}")

            ground_truth = {}
            skipped_rows = 0
            
            for idx, row in df.iterrows():
                try:
                    image_name = str(row["filename"]).strip()
                    if not image_name:
                        logger.warning(f"Row {idx}: Empty filename")
                        skipped_rows += 1
                        continue
                    
                    if pd.isna(row.get("answer")):
                        logger.warning(f"Row {idx} ({image_name}): Missing answer")
                        skipped_rows += 1
                        continue
                    
                    question = str(row.get("question", "")).strip()
                    answer = str(row["answer"]).strip()
                    
                    difficulty = None
                    if "sweep" in df.columns:
                        difficulty = str(row.get("sweep", "")).strip() if not pd.isna(row.get("sweep")) else None

                    ground_truth[image_name] = GroundTruthItem(
                        image_name=image_name,
                        question=question,
                        answer=answer,
                        task_name=task_name,
                        difficulty=difficulty,
                    )

                except Exception as e:
                    logger.error(f"Error processing row {idx} in {csv_path}: {e}")
                    skipped_rows += 1
                    continue

            if not ground_truth:
                logger.error(f"No valid ground truth loaded from {csv_path}")
                raise ValueError(f"No valid ground truth found in {csv_path}")
            
            logger.info(f"Loaded {len(ground_truth)} ground truth items from {csv_path} ({skipped_rows} rows skipped)")
            return ground_truth

        except FileNotFoundError:
            # Expected when an optional dataset (e.g. 3D variant) is not
            # installed; the caller decides whether to skip it.
            raise
        except Exception:
            logger.error(f"Failed to load ground truth for {task_name} ({variant})", exc_info=True)
            raise

    @staticmethod
    def load_all_ground_truth() -> Dict[str, Dict[str, GroundTruthItem]]:
        """Load all Do-You-See-Me ground truth.
        
        Returns:
            Dictionary mapping task_name to ground_truth dictionaries
        """
        all_gt = {}
        for task in DO_YOU_SEE_ME_TASKS:
            # Load each task independently so one malformed dataset does not
            # prevent the rest of the benchmark from loading.
            try:
                # Try 2D first
                all_gt[f"{task}_2d"] = DoYouSeeMeHandler.load_ground_truth(task, is_3d=False)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Skipping 2D ground truth for '{task}': {e}")

            try:
                # Try 3D
                all_gt[f"{task}_3d"] = DoYouSeeMeHandler.load_ground_truth(task, is_3d=True)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Skipping 3D ground truth for '{task}': {e}")

        return all_gt


class MindsEyeHandler:
    """Handles Mind's-Eye JSON ground truth format."""

    @staticmethod
    def load_ground_truth(task_name: str) -> Dict[str, GroundTruthItem]:
        """Load ground truth for a Mind's-Eye task.
        
        Args:
            task_name: Task name (e.g., 'dynamic_isomorph')
            
        Returns:
            Dictionary mapping image_name to GroundTruthItem
        """
        json_path = MINDS_EYE_DATA_ROOT / task_name / "annotations.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        ground_truth = {}
        answer_field = MINDS_EYE_ANSWER_FIELDS.get(task_name, "answer")

        for image_name, item in data.items():
            question = item.get("question", "")
            answer = str(item.get(answer_field, "")).strip()

            # Normalize answer format (remove parentheses for options)
            if answer.startswith("(") and answer.endswith(")"):
                answer = answer[1:-1].strip()

            ground_truth[image_name] = GroundTruthItem(
                image_name=image_name,
                question=question,
                answer=answer,
                task_name=task_name,
                difficulty=None,
            )

        return ground_truth

    @staticmethod
    def load_all_ground_truth() -> Dict[str, Dict[str, GroundTruthItem]]:
        """Load all Mind's-Eye ground truth.
        
        Returns:
            Dictionary mapping task_name to ground_truth dictionaries
        """
        all_gt = {}
        for task in MINDS_EYE_TASKS:
            try:
                all_gt[task] = MindsEyeHandler.load_ground_truth(task)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Skipping Mind's-Eye ground truth for '{task}': {e}")

        return all_gt


class GroundTruthManager:
    """Manages ground truth loading from both benchmarks."""

    def __init__(self):
        """Initialize ground truth caches."""
        self._do_you_see_me_cache: Optional[Dict] = None
        self._minds_eye_cache: Optional[Dict] = None

    def get_do_you_see_me_ground_truth(
        self, task_name: str, is_3d: bool = False
    ) -> Dict[str, GroundTruthItem]:
        """Get Do-You-See-Me ground truth with caching."""
        if self._do_you_see_me_cache is None:
            self._do_you_see_me_cache = DoYouSeeMeHandler.load_all_ground_truth()

        key = f"{task_name}_{'3d' if is_3d else '2d'}"
        if key not in self._do_you_see_me_cache:
            raise ValueError(f"Do-You-See-Me task not found: {key}")

        return self._do_you_see_me_cache[key]

    def get_minds_eye_ground_truth(self, task_name: str) -> Dict[str, GroundTruthItem]:
        """Get Mind's-Eye ground truth with caching."""
        if self._minds_eye_cache is None:
            self._minds_eye_cache = MindsEyeHandler.load_all_ground_truth()

        if task_name not in self._minds_eye_cache:
            raise ValueError(f"Mind's-Eye task not found: {task_name}")

        return self._minds_eye_cache[task_name]

    def list_available_tasks(self) -> Dict[str, List[str]]:
        """List all available tasks."""
        return {
            "do_you_see_me": list(self.get_all_do_you_see_me().keys()),
            "minds_eye": list(self.get_all_minds_eye().keys()),
        }

    def get_all_do_you_see_me(self) -> Dict[str, Dict[str, GroundTruthItem]]:
        """Get all Do-You-See-Me tasks."""
        if self._do_you_see_me_cache is None:
            self._do_you_see_me_cache = DoYouSeeMeHandler.load_all_ground_truth()
        return self._do_you_see_me_cache

    def get_all_minds_eye(self) -> Dict[str, Dict[str, GroundTruthItem]]:
        """Get all Mind's-Eye tasks."""
        if self._minds_eye_cache is None:
            self._minds_eye_cache = MindsEyeHandler.load_all_ground_truth()
        return self._minds_eye_cache

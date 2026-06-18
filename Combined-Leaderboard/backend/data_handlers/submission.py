"""
Handlers for parsing user submissions in CSV and JSON formats.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PredictionItem:
    """Single prediction from submission."""
    image_name: str
    prediction: str
    task_name: Optional[str] = None


class SubmissionParser:
    """Parses user submission files."""

    @staticmethod
    def parse_csv(file_path: Path) -> Dict[str, List[PredictionItem]]:
        """Parse CSV submission file.
        
        Expected CSV columns:
        - image_name (or filename)
        - prediction (or response, answer)
        - task_name (optional)
        
        Returns:
            Dictionary mapping task_name to predictions
        """
        df = pd.read_csv(file_path)

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Find the right column names
        image_col = None
        pred_col = None
        task_col = None

        for col in df.columns:
            if "image" in col or "filename" in col or "file" in col:
                image_col = col
            if "pred" in col or "response" in col or "answer" in col or "output" in col:
                pred_col = col
            if "task" in col:
                task_col = col

        if image_col is None or pred_col is None:
            raise ValueError(
                f"CSV must contain 'image_name'/'filename' and 'prediction'/'response'/'answer' columns. "
                f"Found columns: {list(df.columns)}"
            )

        predictions = {}

        for _, row in df.iterrows():
            image_name = str(row[image_col]).strip()
            prediction = str(row[pred_col]).strip()
            task_name = str(row[task_col]).strip() if task_col else "unknown"

            if task_name not in predictions:
                predictions[task_name] = []

            predictions[task_name].append(
                PredictionItem(
                    image_name=image_name,
                    prediction=prediction,
                    task_name=task_name,
                )
            )

        return predictions

    @staticmethod
    def parse_json(file_path: Path) -> Dict[str, List[PredictionItem]]:
        """Parse JSON submission file.
        
        Supported formats:
        1. {task_name: {image_name: prediction, ...}, ...}
        2. {task_name: {image_name: {prediction: ..., ...}, ...}, ...}
        3. [{"image_name": ..., "prediction": ..., "task_name": ...}, ...]
        
        Returns:
            Dictionary mapping task_name to predictions
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        predictions = {}

        # Handle list format
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("List items must be dictionaries")

                image_name = str(item.get("image_name", "")).strip()
                prediction = str(item.get("prediction", item.get("answer", item.get("response", "")))).strip()
                task_name = str(item.get("task_name", "unknown")).strip()

                if not image_name or not prediction:
                    continue

                if task_name not in predictions:
                    predictions[task_name] = []

                predictions[task_name].append(
                    PredictionItem(
                        image_name=image_name,
                        prediction=prediction,
                        task_name=task_name,
                    )
                )

        # Handle nested dict format
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    # Could be {task_name: {image_name: prediction}}
                    task_name = str(key).strip()
                    if task_name not in predictions:
                        predictions[task_name] = []

                    for img_name, pred in value.items():
                        image_name = str(img_name).strip()
                        
                        # Handle both direct values and nested objects
                        if isinstance(pred, dict):
                            prediction = str(pred.get("prediction", pred.get("answer", pred.get("response", "")))).strip()
                        else:
                            prediction = str(pred).strip()

                        if image_name and prediction:
                            predictions[task_name].append(
                                PredictionItem(
                                    image_name=image_name,
                                    prediction=prediction,
                                    task_name=task_name,
                                )
                            )

        if not predictions:
            raise ValueError("No valid predictions found in JSON file")

        return predictions

    @staticmethod
    def parse_submission(file_path: Path) -> Dict[str, List[PredictionItem]]:
        """Parse submission file (auto-detect format).
        
        Args:
            file_path: Path to submission file
            
        Returns:
            Dictionary mapping task_name to predictions
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Submission file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return SubmissionParser.parse_csv(file_path)
        elif suffix == ".json":
            return SubmissionParser.parse_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .json")

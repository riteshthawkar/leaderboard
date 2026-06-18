"""
Persistence and ranking for the unified golden-set leaderboard.

Stores submissions as a single JSON file (results/golden_leaderboard.json) to
keep the unified pipeline self-contained and independent of the legacy database.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from config import GOLDEN_LEADERBOARD_FILE
from models.golden import GoldenSubmissionScore


class GoldenLeaderboardManager:
    def __init__(self, store_file: Path = GOLDEN_LEADERBOARD_FILE):
        self.store_file = Path(store_file)
        self._lock = threading.Lock()
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_file.exists():
            self._write([])

    def _read(self) -> List[dict]:
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, entries: List[dict]):
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def add_submission(self, score: GoldenSubmissionScore) -> dict:
        record = score.to_dict()
        with self._lock:
            entries = self._read()
            entries.append(record)
            self._write(entries)
        return record

    def get_leaderboard(self, limit: int = 100, sort_by: str = "vci") -> List[dict]:
        entries = self._read()
        key = sort_by if sort_by in {"vci", "overall_accuracy"} else "vci"
        entries.sort(key=lambda e: e.get(key, 0.0), reverse=True)
        ranked = []
        for i, e in enumerate(entries[:limit], start=1):
            layer = e.get("layer_results", {})
            ranked.append({
                "rank": i,
                "submission_id": e.get("submission_id"),
                "model_name": e.get("model_name"),
                "model_meta": e.get("model_meta", {}),
                "vci": e.get("vci", 0.0),
                "overall_accuracy": e.get("overall_accuracy", 0.0),
                "total_samples": e.get("total_samples", 0),
                "correct_samples": e.get("correct_samples", 0),
                "submitted_at": e.get("submitted_at"),
                "layer_accuracy": {l: r.get("accuracy", 0.0) for l, r in layer.items()},
                "diagnostics": e.get("diagnostics"),
            })
        return ranked

    def get_submission(self, submission_id: str) -> Optional[dict]:
        for e in self._read():
            if e.get("submission_id") == submission_id:
                return e
        return None

    def get_diagnostics(self) -> List[dict]:
        """Return models that submitted at least one optional condition."""
        out = []
        for e in self._read():
            diag = e.get("diagnostics")
            if diag:
                out.append({
                    "model_name": e.get("model_name"),
                    "submission_id": e.get("submission_id"),
                    "vci": e.get("vci", 0.0),
                    "overall_accuracy": e.get("overall_accuracy", 0.0),
                    "model_meta": e.get("model_meta", {}),
                    "diagnostics": diag,
                })
        out.sort(key=lambda x: x.get("vci", 0.0), reverse=True)
        return out

    def get_statistics(self) -> dict:
        entries = self._read()
        if not entries:
            return {"total_submissions": 0, "best_vci": 0.0, "avg_vci": 0.0,
                    "with_diagnostics": 0}
        vcis = [e.get("vci", 0.0) for e in entries]
        return {
            "total_submissions": len(entries),
            "best_vci": max(vcis),
            "avg_vci": sum(vcis) / len(vcis),
            "with_diagnostics": sum(1 for e in entries if e.get("diagnostics")),
        }

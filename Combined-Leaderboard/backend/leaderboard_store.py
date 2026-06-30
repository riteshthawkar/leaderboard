"""
Per-model leaderboard store for the three-task system.

Submissions are keyed by model name so a model can submit do_you_see_me,
minds_eye and spatial at different times. The store combines the two Visual
Cognition tasks into the Visual Cognition Index (VCI) and exposes a separate
Spatial Reasoning ranking with diagnostics.

Backed by a single JSON file (results/leaderboard_store.json) and guarded by a
lock so concurrent submissions don't corrupt it.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    LEADERBOARD_STORE_FILE,
    VCI_LAYER_WEIGHTS,
    LAYER_LABELS,
    TASKS,
    SECTIONS,
)
from models.tasks import TaskScore

logger = logging.getLogger(__name__)


def _norm_name(name: str) -> str:
    return " ".join(str(name).strip().split())


class LeaderboardStore:
    def __init__(self, store_file: Path = LEADERBOARD_STORE_FILE):
        self.store_file = Path(store_file)
        self._lock = threading.Lock()
        self.store_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_file.exists():
            self._write({"models": {}})

    # ----------------------------------------------------------------- io
    def _read(self) -> dict:
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "models" in data:
                    return data
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return {"models": {}}

    def _write(self, data: dict):
        # Write to a temp file then atomically replace to prevent corruption
        # on crash mid-write (os.replace / Path.replace is atomic on POSIX).
        tmp = self.store_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            import os
            os.fsync(f.fileno())
        tmp.replace(self.store_file)

    # ------------------------------------------------------------- mutate
    def add_result(self, score: TaskScore, submitted_by: str = None) -> dict:
        """Upsert a task result under its model name.

        Only the original submitter can update an existing model entry.
        Raises PermissionError if a different user tries to overwrite.
        """
        record = score.to_dict()
        key = _norm_name(score.model_name)
        with self._lock:
            data = self._read()
            existing = data["models"].get(key)
            if existing:
                owner = existing.get("submitted_by")
                if owner and submitted_by and owner != submitted_by:
                    raise PermissionError(
                        f"Model '{score.model_name}' was registered by a different account."
                    )
            entry = data["models"].setdefault(
                key, {"model_name": score.model_name, "model_meta": {}, "tasks": {}}
            )
            entry["model_name"] = score.model_name
            if submitted_by and not entry.get("submitted_by"):
                entry["submitted_by"] = submitted_by
            if score.model_meta:
                entry["model_meta"] = {**entry.get("model_meta", {}), **score.model_meta}
            entry["tasks"][score.task_id] = record
            self._write(data)
            self._append_history(record, submitted_by)
        return record

    def _append_history(self, record: dict, submitted_by: Optional[str]) -> None:
        """Append an immutable audit-log line for every accepted submission.

        The main store is an upsert keyed by model name (no history retained),
        so this JSONL file is the permanent record of who submitted what and
        when. Best-effort: a logging failure must never fail a submission.
        """
        try:
            history_file = self.store_file.parent / "submission_history.jsonl"
            entry = {
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "model_name": record.get("model_name"),
                "task_id": record.get("task_id"),
                "submission_id": record.get("submission_id"),
                "submitted_by": submitted_by,
                "submitted_at": record.get("submitted_at"),
                "accuracy": record.get("accuracy"),
                "total_samples": record.get("total_samples"),
                "correct_samples": record.get("correct_samples"),
            }
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.warning("Failed to append submission history", exc_info=True)

    # ------------------------------------------------------------- compute
    @staticmethod
    def _vci(perception: Optional[float], imagery: Optional[float]) -> Optional[float]:
        """Layer-weighted mean over the Visual Cognition layers present."""
        present = {}
        if perception is not None:
            present["perception"] = perception
        if imagery is not None:
            present["imagery"] = imagery
        if not present:
            return None
        weights = {l: VCI_LAYER_WEIGHTS.get(l, 0.0) for l in present}
        wsum = sum(weights.values())
        if wsum <= 0:
            weights = {l: 1.0 for l in present}
            wsum = float(len(present))
        return sum(present[l] * w for l, w in weights.items()) / wsum

    def _vc_row(self, key: str, entry: dict) -> Optional[dict]:
        tasks = entry.get("tasks", {})
        dysm = tasks.get("do_you_see_me")
        me = tasks.get("minds_eye")
        if not dysm and not me:
            return None
        perception = dysm["accuracy"] if dysm else None
        imagery = me["accuracy"] if me else None
        vci = self._vci(perception, imagery)
        return {
            "model_name": entry.get("model_name", key),
            "model_meta": entry.get("model_meta", {}),
            "vci": round(vci, 4) if vci is not None else None,
            "perception_accuracy": round(perception, 4) if perception is not None else None,
            "imagery_accuracy": round(imagery, 4) if imagery is not None else None,
            "has_perception": dysm is not None,
            "has_imagery": me is not None,
            "complete": dysm is not None and me is not None,
            "perception_submission": dysm["submission_id"] if dysm else None,
            "imagery_submission": me["submission_id"] if me else None,
            "perception_groups": dysm.get("groups", {}) if dysm else {},
            "imagery_groups": me.get("groups", {}) if me else {},
            "perception_grading": dysm.get("grading") if dysm else None,
            "imagery_grading": me.get("grading") if me else None,
        }

    def visual_cognition_leaderboard(self, limit: int = 100) -> List[dict]:
        data = self._read()
        rows = []
        for key, entry in data["models"].items():
            row = self._vc_row(key, entry)
            if row:
                rows.append(row)
        rows.sort(key=lambda r: (r["complete"], r["vci"] if r["vci"] is not None else -1),
                  reverse=True)
        for i, r in enumerate(rows[:limit], start=1):
            r["rank"] = i
        return rows[:limit]

    def spatial_leaderboard(self, limit: int = 100) -> List[dict]:
        data = self._read()
        rows = []
        for key, entry in data["models"].items():
            sp = entry.get("tasks", {}).get("spatial")
            if not sp:
                continue
            rows.append({
                "model_name": entry.get("model_name", key),
                "model_meta": entry.get("model_meta", {}),
                "accuracy": sp.get("accuracy", 0.0),
                "macro_accuracy": sp.get("macro_accuracy"),
                "accuracy_std": sp.get("accuracy_std"),
                "total_samples": sp.get("total_samples", 0),
                "correct_samples": sp.get("correct_samples", 0),
                "submission_id": sp.get("submission_id"),
                "submitted_at": sp.get("submitted_at"),
                "groups": sp.get("groups", {}),
                "diagnostics": sp.get("diagnostics"),
                "grading": sp.get("grading"),
            })
        rows.sort(key=lambda r: r.get("accuracy", 0.0), reverse=True)
        for i, r in enumerate(rows[:limit], start=1):
            r["rank"] = i
        return rows[:limit]

    def get_model(self, model_name: str) -> Optional[dict]:
        data = self._read()
        entry = data["models"].get(_norm_name(model_name))
        if not entry:
            return None
        tasks = entry.get("tasks", {})
        dysm = tasks.get("do_you_see_me")
        me = tasks.get("minds_eye")
        return {
            "model_name": entry.get("model_name"),
            "model_meta": entry.get("model_meta", {}),
            "tasks": tasks,
            "visual_cognition": self._vc_row(model_name, entry),
            "layer_labels": LAYER_LABELS,
        }

    def statistics(self) -> dict:
        data = self._read()
        models = data["models"]
        vc = self.visual_cognition_leaderboard()
        sp = self.spatial_leaderboard()
        best_vci = max((r["vci"] for r in vc if r["vci"] is not None), default=0.0)
        best_spatial = max((r["accuracy"] for r in sp), default=0.0)
        return {
            "total_models": len(models),
            "visual_cognition_models": len(vc),
            "spatial_models": len(sp),
            "best_vci": round(best_vci, 4),
            "best_spatial_accuracy": round(best_spatial, 4),
            "with_diagnostics": sum(1 for r in sp if r.get("diagnostics")),
        }

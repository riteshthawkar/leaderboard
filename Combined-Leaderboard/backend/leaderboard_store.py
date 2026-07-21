"""
Per-model leaderboard store for the three-task system.

Submissions are keyed by stable registered model IDs so a model can submit
do_you_see_me, minds_eye and spatial at different times. The store combines the two visual
cognition tasks into the Visual Cognition Index (VCI) and exposes a separate
Spatial Reasoning ranking with diagnostics.

Backed by a single JSON file (results/leaderboard_store.json) and guarded by a
lock so concurrent submissions don't corrupt it.
"""

import hashlib
import json
import logging
import math
import shutil
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    LEADERBOARD_STORE_FILE,
    VCI_LAYER_WEIGHTS,
    LAYER_LABELS,
    TASKS,
    SECTIONS,
    MINDS_EYE_ART_BY_CAPABILITY,
)
from models.tasks import TaskScore

try:
    from sqlite_runtime import harden_private_directory, harden_private_file
except ImportError:  # pragma: no cover - package import fallback
    from .sqlite_runtime import harden_private_directory, harden_private_file

logger = logging.getLogger(__name__)

_CANONICAL_MODEL_META_FIELDS = frozenset({
    "organization",
    "org",
    "access",
    "type",
    "parameter_count",
    "params",
    "params_b",
    "base_model",
    "family",
    "training_data",
    "paper_url",
})

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


def _norm_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name or ""))
    return " ".join(value.split()).casefold()


def _canonical_model_meta(meta: Optional[dict]) -> dict:
    """Keep identity metadata separate from benchmark-specific run metadata."""
    if not isinstance(meta, dict):
        return {}
    return {
        key: value
        for key, value in meta.items()
        if key in _CANONICAL_MODEL_META_FIELDS
    }


class LeaderboardStore:
    def __init__(self, store_file: Path = LEADERBOARD_STORE_FILE):
        self.store_file = Path(store_file)
        self.lock_file = self.store_file.with_suffix(self.store_file.suffix + ".lock")
        self._lock = threading.Lock()
        harden_private_directory(self.store_file.parent)
        if not self.store_file.exists():
            self._write({"models": {}})
        else:
            harden_private_file(self.store_file)

    @contextmanager
    def _exclusive_lock(self):
        """Process-safe mutation lock for JSON store updates."""
        harden_private_directory(self.lock_file.parent)
        with self._lock:
            with open(self.lock_file, "a+", encoding="utf-8") as lock_handle:
                harden_private_file(self.lock_file)
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    # ----------------------------------------------------------------- io
    def _read(self) -> dict:
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "models" in data:
                    return data
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            self._backup_corrupt_store()
        return {"models": {}}

    def _backup_corrupt_store(self) -> None:
        try:
            if not self.store_file.exists():
                return
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_file = self.store_file.with_name(
                f"{self.store_file.name}.corrupt-{timestamp}"
            )
            shutil.copy2(self.store_file, backup_file)
            harden_private_file(backup_file)
            logger.error(
                "Leaderboard store contains invalid JSON; copied corrupt file to %s",
                backup_file,
                exc_info=True,
            )
        except Exception:
            logger.error("Failed to back up corrupt leaderboard store", exc_info=True)

    def _write(self, data: dict):
        # Write to a temp file then atomically replace to prevent corruption
        # on crash mid-write (os.replace / Path.replace is atomic on POSIX).
        tmp = self.store_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            import os
            os.fsync(f.fileno())
        harden_private_file(tmp)
        tmp.replace(self.store_file)
        harden_private_file(self.store_file)

    # ------------------------------------------------------------- mutate
    def add_result(self, score: TaskScore, submitted_by: str = None) -> dict:
        """Upsert a task result under its stable registered model ID.

        Only the original submitter can update an existing model entry.
        Raises PermissionError if a different user tries to overwrite.
        """
        record = score.to_dict()
        key = str(score.model_id or "").strip() or _norm_name(score.model_name)
        with self._exclusive_lock():
            data = self._read()
            existing = data["models"].get(key)
            if existing:
                owner = existing.get("submitted_by")
                if owner and submitted_by and owner != submitted_by:
                    raise PermissionError(
                        f"Model '{score.model_name}' was registered by a different account."
                    )
            entry = data["models"].setdefault(
                key, {
                    "model_id": score.model_id,
                    "model_name": score.model_name,
                    "model_meta": {},
                    "tasks": {},
                }
            )
            if score.model_id:
                entry["model_id"] = score.model_id
            entry["model_name"] = score.model_name
            if submitted_by and not entry.get("submitted_by"):
                entry["submitted_by"] = submitted_by
            entry["model_meta"] = _canonical_model_meta({
                **entry.get("model_meta", {}),
                **(score.model_meta or {}),
            })
            entry["tasks"][score.task_id] = record
            self._write(data)
            self._append_history(record, submitted_by)
        return record

    def migrate_model_keys(self, registered_models: List[dict]) -> int:
        """Move legacy name-keyed cache entries under their registered IDs."""
        migrated = 0
        by_owner_and_name = {
            (
                str(model.get("owner_email") or "").strip().lower(),
                _norm_name(model.get("model_name")),
            ): model
            for model in registered_models
            if model.get("model_id") and model.get("model_name")
        }
        with self._exclusive_lock():
            data = self._read()
            models = data.get("models", {})
            for legacy_key, entry in list(models.items()):
                if entry.get("model_id"):
                    continue
                lookup = (
                    str(entry.get("submitted_by") or "").strip().lower(),
                    _norm_name(entry.get("model_name") or legacy_key),
                )
                registered = by_owner_and_name.get(lookup)
                if not registered:
                    continue
                model_id = registered["model_id"]
                destination = models.get(model_id)
                if destination is None:
                    destination = entry
                    models[model_id] = destination
                else:
                    destination.setdefault("tasks", {}).update(entry.get("tasks", {}))
                    destination["model_meta"] = _canonical_model_meta({
                        **entry.get("model_meta", {}),
                        **destination.get("model_meta", {}),
                    })
                destination["model_meta"] = _canonical_model_meta(
                    destination.get("model_meta", {})
                )
                destination["model_id"] = model_id
                destination["model_name"] = registered.get("model_name") or entry.get("model_name")
                destination["submitted_by"] = registered.get("owner_email") or entry.get("submitted_by")
                if legacy_key != model_id:
                    models.pop(legacy_key, None)
                migrated += 1
            if migrated:
                self._write(data)
        return migrated

    def remove_submission(self, submission_id: str) -> bool:
        """Remove a task result from the public leaderboard by submission id."""
        submission_id = str(submission_id or "").strip()
        if not submission_id:
            return False
        removed = False
        with self._exclusive_lock():
            data = self._read()
            empty_models = []
            for key, entry in data.get("models", {}).items():
                tasks = entry.get("tasks", {})
                for task_id, record in list(tasks.items()):
                    if record.get("submission_id") == submission_id:
                        del tasks[task_id]
                        removed = True
                if not tasks:
                    empty_models.append(key)
            for key in empty_models:
                data["models"].pop(key, None)
            if removed:
                self._write(data)
        return removed

    def remove_model_task(self, model_id: str, task_id: str) -> bool:
        """Remove one model/benchmark slot when no visible run remains."""
        model_id = str(model_id or "").strip()
        task_id = str(task_id or "").strip()
        if not model_id or not task_id:
            return False
        removed = False
        with self._exclusive_lock():
            data = self._read()
            entry = data.get("models", {}).get(model_id)
            if entry and task_id in entry.get("tasks", {}):
                del entry["tasks"][task_id]
                removed = True
                if not entry["tasks"]:
                    data["models"].pop(model_id, None)
                self._write(data)
        return removed

    def public_submission_ids(self) -> List[str]:
        """Return the submission IDs currently represented in public rows."""
        data = self._read()
        return sorted({
            str(record.get("submission_id"))
            for entry in data.get("models", {}).values()
            for record in entry.get("tasks", {}).values()
            if record.get("submission_id")
        })

    def public_submission_fingerprints(self) -> Dict[str, str]:
        """Return canonical fingerprints for every currently published score."""
        data = self._read()
        fingerprints: Dict[str, str] = {}
        for entry in data.get("models", {}).values():
            for record in entry.get("tasks", {}).values():
                submission_id = str(record.get("submission_id") or "").strip()
                if not submission_id:
                    continue
                canonical = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                fingerprints[submission_id] = hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest()
        return fingerprints

    def replace_all_results(self, scored_records: List[tuple[TaskScore, Optional[str]]]) -> int:
        """Replace public leaderboard contents from scored submissions.

        Used by admin re-score/rebuild jobs. This intentionally does not append
        to the immutable submission history log; the source of truth is the
        database-backed submission table.
        """
        with self._exclusive_lock():
            data = {"models": {}}
            for score, submitted_by in scored_records:
                record = score.to_dict()
                key = str(score.model_id or "").strip() or _norm_name(score.model_name)
                entry = data["models"].setdefault(
                    key, {
                        "model_id": score.model_id,
                        "model_name": score.model_name,
                        "model_meta": {},
                        "tasks": {},
                    }
                )
                if score.model_id:
                    entry["model_id"] = score.model_id
                entry["model_name"] = score.model_name
                if submitted_by:
                    entry["submitted_by"] = submitted_by
                entry["model_meta"] = _canonical_model_meta({
                    **entry.get("model_meta", {}),
                    **(score.model_meta or {}),
                })
                entry["tasks"][score.task_id] = record
            self._write(data)
        return len(scored_records)

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
                "macro_accuracy": record.get("macro_accuracy"),
                "score_method": record.get("score_method"),
                "total_samples": record.get("total_samples"),
                "correct_samples": record.get("correct_samples"),
            }
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            harden_private_file(history_file)
        except Exception:
            logger.warning("Failed to append submission history", exc_info=True)

    # ------------------------------------------------------------- compute
    @staticmethod
    def _vci(perception: Optional[float], cognition: Optional[float]) -> Optional[float]:
        """Return VPCI only when both visual benchmark layers are present."""
        if perception is None or cognition is None:
            return None
        present = {"perception": perception, "cognition": cognition}
        weights = {layer: VCI_LAYER_WEIGHTS.get(layer, 0.0) for layer in present}
        wsum = sum(weights.values())
        if wsum <= 0:
            weights = {layer: 1.0 for layer in present}
            wsum = float(len(present))
        return sum(present[layer] * weight for layer, weight in weights.items()) / wsum

    @staticmethod
    def _mean_group_accuracy(groups: Optional[dict]) -> Optional[float]:
        values = [
            float(group["accuracy"])
            for group in (groups or {}).values()
            if isinstance(group, dict) and group.get("accuracy") is not None
        ]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _task_spread(task: Optional[dict]) -> Optional[float]:
        if not task:
            return None
        value = task.get("task_spread")
        if value is None:
            value = task.get("accuracy_std")
        try:
            spread = float(value)
        except (TypeError, ValueError):
            return None
        return spread if math.isfinite(spread) and spread >= 0 else None

    @classmethod
    def _dysm_dimensions(cls, task: Optional[dict]) -> dict:
        """Return dimension scores as unweighted means of task variants."""
        if not task:
            return {}
        analysis = task.get("analysis", {})
        variants = analysis.get("task_variant", {})
        by_dimension: Dict[str, List[dict]] = {}
        for label, result in variants.items():
            if not isinstance(result, dict) or result.get("accuracy") is None:
                continue
            dimension = str(label).split(":", 1)[0]
            by_dimension.setdefault(dimension, []).append(result)
        if by_dimension:
            dimensions = {}
            for dimension in ("2D", "3D"):
                results = by_dimension.get(dimension, [])
                if not results:
                    continue
                dimensions[dimension] = {
                    "name": dimension,
                    "total_samples": sum(
                        int(result.get("total_samples", 0) or 0)
                        for result in results
                    ),
                    "correct_samples": sum(
                        int(result.get("correct_samples", 0) or 0)
                        for result in results
                    ),
                    "accuracy": sum(float(result["accuracy"]) for result in results)
                    / len(results),
                    "meta": {
                        "aggregation": "unweighted_task_mean",
                        "task_count": len(results),
                    },
                }
            return dimensions

        # Legacy score records predate per-task-variant analysis. The released
        # suite is balanced within each dimension, so its stored dimension
        # accuracy is the closest reproducible fallback.
        stored = analysis.get("dimension", {})
        return {
            dimension: {
                **result,
                "meta": {
                    **(result.get("meta", {}) if isinstance(result, dict) else {}),
                    "aggregation": "legacy_dimension_accuracy",
                },
            }
            for dimension, result in stored.items()
            if isinstance(result, dict)
        }

    @classmethod
    def _task_headline_accuracy(
        cls,
        task: Optional[dict],
        task_id: str,
    ) -> Optional[float]:
        if not task:
            return None
        if task_id == "do_you_see_me":
            dimensions = cls._dysm_dimensions(task)
            values = [
                float(result["accuracy"])
                for result in dimensions.values()
                if result.get("accuracy") is not None
            ]
            if values:
                return sum(values) / len(values)
        group_macro = cls._mean_group_accuracy(task.get("groups"))
        if group_macro is not None:
            return group_macro
        stored_macro = task.get("macro_accuracy")
        if stored_macro is not None:
            return float(stored_macro)
        accuracy = task.get("accuracy")
        return float(accuracy) if accuracy is not None else None

    @classmethod
    def _public_task_record(cls, task_id: str, task: dict) -> dict:
        record = dict(task)
        record["micro_accuracy"] = task.get("accuracy")
        record["macro_accuracy"] = cls._task_headline_accuracy(task, task_id)
        record["task_spread"] = task.get("task_spread", task.get("accuracy_std"))
        record["score_method"] = (
            task.get("score_method") or TASKS.get(task_id, {}).get("score_method")
        )
        return record

    @staticmethod
    def _art_groups(task: Optional[dict]) -> dict:
        """Return ART scores as unweighted means of their task accuracies."""
        if not task:
            return {}
        totals: Dict[str, dict] = {}
        for capability, group in task.get("groups", {}).items():
            art = MINDS_EYE_ART_BY_CAPABILITY.get(capability)
            if not art or not isinstance(group, dict) or group.get("accuracy") is None:
                continue
            bucket = totals.setdefault(
                art,
                {"accuracies": [], "correct": 0, "total": 0},
            )
            bucket["accuracies"].append(float(group["accuracy"]))
            bucket["correct"] += int(group.get("correct_samples", 0) or 0)
            bucket["total"] += int(group.get("total_samples", 0) or 0)
        if not totals:
            return task.get("analysis", {}).get("art", {})
        return {
            art: {
                "name": art,
                "total_samples": values["total"],
                "correct_samples": values["correct"],
                "accuracy": sum(values["accuracies"]) / len(values["accuracies"]),
                "meta": {
                    "aggregation": "unweighted_task_mean",
                    "task_count": len(values["accuracies"]),
                },
            }
            for art, values in totals.items()
        }

    def _vc_row(self, key: str, entry: dict) -> Optional[dict]:
        tasks = entry.get("tasks", {})
        dysm = tasks.get("do_you_see_me")
        me = tasks.get("minds_eye")
        if not dysm and not me:
            return None
        perception = self._task_headline_accuracy(dysm, "do_you_see_me")
        cognition = self._task_headline_accuracy(me, "minds_eye")
        vci = self._vci(perception, cognition)
        perception_task_spread = self._task_spread(dysm)
        cognition_task_spread = self._task_spread(me)
        task_spread = (
            (perception_task_spread + cognition_task_spread) / 2
            if perception_task_spread is not None
            and cognition_task_spread is not None
            else None
        )
        cognition_fields = {
            "cognition_accuracy": round(cognition, 4) if cognition is not None else None,
            "cognition_macro_accuracy": round(cognition, 4) if cognition is not None else None,
            "cognition_micro_accuracy": me.get("accuracy") if me else None,
            "cognition_task_spread": round(cognition_task_spread, 4)
            if cognition_task_spread is not None
            else None,
            "cognition_score_method": (
                me.get("score_method") if me else None
            ) or TASKS["minds_eye"].get("score_method"),
            "has_cognition": me is not None,
            "cognition_submission": me["submission_id"] if me else None,
            "cognition_groups": me.get("groups", {}) if me else {},
            "cognition_grading": me.get("grading") if me else None,
            "cognition_art": self._art_groups(me),
        }
        perception_analysis = dysm.get("analysis", {}) if dysm else {}
        perception_dimensions = self._dysm_dimensions(dysm)
        return {
            "model_id": entry.get("model_id"),
            "model_name": entry.get("model_name", key),
            "model_meta": _canonical_model_meta(entry.get("model_meta", {})),
            "vci": round(vci, 4) if vci is not None else None,
            "task_spread": round(task_spread, 5) if task_spread is not None else None,
            "perception_accuracy": round(perception, 4) if perception is not None else None,
            "perception_macro_accuracy": round(perception, 4) if perception is not None else None,
            "perception_micro_accuracy": dysm.get("accuracy") if dysm else None,
            "perception_task_spread": round(perception_task_spread, 4)
            if perception_task_spread is not None
            else None,
            "perception_score_method": (
                dysm.get("score_method") if dysm else None
            ) or TASKS["do_you_see_me"].get("score_method"),
            "has_perception": dysm is not None,
            "complete": dysm is not None and me is not None,
            "perception_submission": dysm["submission_id"] if dysm else None,
            "perception_groups": dysm.get("groups", {}) if dysm else {},
            "perception_grading": dysm.get("grading") if dysm else None,
            "perception_dimensions": perception_dimensions,
            "perception_sample_dimensions": perception_analysis.get("dimension", {}),
            "perception_task_variants": perception_analysis.get("task_variant", {}),
            "perception_difficulty": perception_analysis.get("difficulty", {}),
            **cognition_fields,
            # Backward-compatible aliases for existing frontend snapshots/API clients.
            "imagery_accuracy": cognition_fields["cognition_accuracy"],
            "has_imagery": cognition_fields["has_cognition"],
            "imagery_submission": cognition_fields["cognition_submission"],
            "imagery_groups": cognition_fields["cognition_groups"],
            "imagery_grading": cognition_fields["cognition_grading"],
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
            macro_accuracy = self._task_headline_accuracy(sp, "spatial")
            rows.append({
                "model_id": entry.get("model_id"),
                "model_name": entry.get("model_name", key),
                "model_meta": _canonical_model_meta(entry.get("model_meta", {})),
                "accuracy": sp.get("accuracy", 0.0),
                "micro_accuracy": sp.get("micro_accuracy", sp.get("accuracy", 0.0)),
                "macro_accuracy": macro_accuracy,
                "task_spread": sp.get("task_spread", sp.get("accuracy_std")),
                "accuracy_std": sp.get("accuracy_std"),
                "score_method": sp.get("score_method") or TASKS["spatial"].get("score_method"),
                "total_samples": sp.get("total_samples", 0),
                "correct_samples": sp.get("correct_samples", 0),
                "submission_id": sp.get("submission_id"),
                "evidence_url": (
                    f"/api/public/submissions/{sp.get('submission_id')}/evidence"
                    if sp.get("submission_id")
                    else None
                ),
                "submitted_at": sp.get("submitted_at"),
                "groups": sp.get("groups", {}),
                "diagnostics": sp.get("diagnostics"),
                "grading": sp.get("grading"),
            })
        rows.sort(
            key=lambda r: (
                r.get("macro_accuracy")
                if r.get("macro_accuracy") is not None
                else r.get("accuracy", 0.0)
            ),
            reverse=True,
        )
        for i, r in enumerate(rows[:limit], start=1):
            r["rank"] = i
        return rows[:limit]

    def get_model(self, model_name: str) -> Optional[dict]:
        data = self._read()
        entry = data["models"].get(str(model_name))
        if not entry:
            normalized = _norm_name(model_name)
            entry = next(
                (
                    candidate
                    for candidate in data["models"].values()
                    if _norm_name(candidate.get("model_name")) == normalized
                ),
                None,
            )
        if not entry:
            return None
        tasks = {
            task_id: self._public_task_record(task_id, task)
            for task_id, task in entry.get("tasks", {}).items()
        }
        return {
            "model_id": entry.get("model_id"),
            "model_name": entry.get("model_name"),
            "model_meta": _canonical_model_meta(entry.get("model_meta", {})),
            "tasks": tasks,
            "visual_cognition": self._vc_row(model_name, entry),
            "layer_labels": LAYER_LABELS,
        }

    def statistics(self) -> dict:
        data = self._read()
        models = data["models"]
        model_limit = max(len(models), 1)
        vc = self.visual_cognition_leaderboard(limit=model_limit)
        sp = self.spatial_leaderboard(limit=model_limit)
        task_maps = [
            entry.get("tasks", {})
            for entry in models.values()
            if isinstance(entry, dict) and isinstance(entry.get("tasks", {}), dict)
        ]
        visual_cognition_models = sum(
            1
            for tasks in task_maps
            if tasks.get("do_you_see_me") or tasks.get("minds_eye")
        )
        spatial_models = sum(1 for tasks in task_maps if tasks.get("spatial"))
        ranked_models = sum(
            1
            for tasks in task_maps
            if tasks.get("do_you_see_me")
            or tasks.get("minds_eye")
            or tasks.get("spatial")
        )
        best_vci = max((r["vci"] for r in vc if r["vci"] is not None), default=None)
        best_spatial = max(
            (
                r.get("macro_accuracy")
                if r.get("macro_accuracy") is not None
                else r.get("accuracy", 0.0)
                for r in sp
            ),
            default=0.0,
        )
        return {
            "total_models": len(models),
            "ranked_models": ranked_models,
            "visual_cognition_models": visual_cognition_models,
            "spatial_models": spatial_models,
            "best_vci": round(best_vci, 4) if best_vci is not None else None,
            "best_spatial_accuracy": round(best_spatial, 4),
            "with_diagnostics": sum(1 for r in sp if r.get("diagnostics")),
        }

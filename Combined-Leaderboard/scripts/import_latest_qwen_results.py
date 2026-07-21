#!/usr/bin/env python3
"""Validate and publish the latest finalized Qwen visual evaluations.

The command is dry-run only unless ``--apply`` is supplied. Applying creates a
database backup, publishes the exact Qwen2.5 and Qwen3.5 variant runs, then
soft-deletes and archives the superseded generic Qwen comparison records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
for import_root in (PROJECT_DIR, BACKEND_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from backup import write_backup_archive
from config import AUTO_BACKUP_DIR, AUTO_BACKUP_RETENTION_COUNT, LEADERBOARD_STORE_FILE
from evaluation.finalize_visual_results import (
    expected_question_ids,
    read_json,
    sha256,
    validate_diagnostics,
    validate_submission,
)
from leaderboard_store import LeaderboardStore
from scoring.task_scorer import TaskScorer
from submission_store import (
    archive_registered_model,
    create_registered_model,
    finalize_submission,
    latest_visible_scored_submission_ids,
    list_registered_models,
    list_submissions,
    normalize_model_name,
    set_moderation_status,
    store_submission_answers,
    submission_integrity_status,
    try_consume_quota,
)


TRACKS = ("do_you_see_me", "minds_eye")
OWNER_EMAIL = "admin-import@ms-vista.local"
DATASET_REPOSITORY = "amolharsh/visual-intelligence-leaderboard"
DATASET_REVISION = "cc41be90e74679a9d3c9dd295834b2cee9100b9d"
QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"
QWEN35_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SUPERSEDED_MODEL_NAMES = {"Qwen2.5-VL", "Qwen3.5"}


@dataclass
class TrackPlan:
    task_id: str
    submission_path: Path
    diagnostics_path: Path
    file_sha256: str
    answer_records: list[dict[str, Any]]
    score: Any
    provenance: dict[str, Any]


@dataclass
class ModelPlan:
    display_name: str
    organization: str
    parameter_count: str
    repository: str
    revision: str
    compute_dtype: str
    weight_loading: str
    thinking: bool
    tracks: dict[str, TrackPlan]

    @property
    def vpci(self) -> float:
        return sum(self.tracks[task].score.macro_accuracy for task in TRACKS) / 2


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def _model_meta(model: ModelPlan, task_id: str, generation: dict[str, Any]) -> dict[str, Any]:
    prompt_mode = str(generation.get("prompt_mode") or "").strip().lower()
    mode_label = "thinking" if model.thinking else "non-thinking"
    return {
        "organization": model.organization,
        "org": model.organization,
        "access": "open_weights",
        "type": "open_weights",
        "parameter_count": model.parameter_count,
        "method_description": (
            "Official MS-VISTA visual evaluation using the pinned checkpoint, "
            f"full unquantized weights, {model.compute_dtype.upper()} compute, and "
            f"the recorded {mode_label} generation configuration. Invalid answer "
            "formats receive zero credit."
        ),
        "cot_used": "Yes" if prompt_mode == "cot" else "No",
        "prompt_template": (
            "Official benchmark prompt and chat-template configuration identified "
            "by the retained run configuration and hashes."
        ),
        "changes_from_previous": (
            "Replaces the superseded generic Qwen comparison entry with the latest "
            "fully identified finalized evaluation."
        ),
        "model_repository": model.repository,
        "model_revision": model.revision,
        "weight_loading": model.weight_loading,
        "compute_dtype": model.compute_dtype,
        "thinking_mode": model.thinking,
        "submission_track": task_id,
    }


def _score_track(
    model: ModelPlan,
    task_id: str,
    submission_path: Path,
    diagnostics_path: Path,
    generation: dict[str, Any],
    provenance: dict[str, Any],
) -> TrackPlan:
    expected_ids = expected_question_ids(PROJECT_DIR, task_id)
    submission_rows = validate_submission(submission_path, expected_ids)
    diagnostics_rows, _ = validate_diagnostics(diagnostics_path, expected_ids)
    if len(diagnostics_rows) != len(submission_rows):
        raise ValueError(f"{model.display_name}/{task_id} diagnostics count mismatch")

    payload = submission_path.read_bytes()
    try:
        submission_text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{submission_path} is not valid UTF-8") from exc
    scorer = TaskScorer(task_id)
    predictions, parsed_meta, answer_records = scorer.parse_submission_text_with_records(
        submission_text
    )
    score = scorer.score_predictions(
        predictions,
        model_name=model.display_name,
        parsed_meta=parsed_meta,
        model_meta=_model_meta(model, task_id, generation),
    )
    score.metadata.update({
        "evaluation_source": "latest_finalized_qwen_suite",
        "source_submission_sha256": hashlib.sha256(payload).hexdigest(),
        **provenance,
    })
    return TrackPlan(
        task_id=task_id,
        submission_path=submission_path,
        diagnostics_path=diagnostics_path,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        answer_records=answer_records,
        score=score,
        provenance=provenance,
    )


def _qwen25_plan(result_root: Path) -> ModelPlan:
    source_dir = result_root / "qwen25-vl-7b"
    manifest_path = source_dir / "final_manifest.json"
    manifest = read_json(manifest_path)
    _assert_equal(manifest.get("model_id"), "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5 model")
    _assert_equal(
        manifest.get("model_revision"),
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "Qwen2.5 revision",
    )
    _assert_equal(manifest.get("weight_loading"), "unquantized", "Qwen2.5 weights")
    _assert_equal(manifest.get("compute_dtype"), "bfloat16", "Qwen2.5 dtype")
    model = ModelPlan(
        display_name="Qwen2.5-VL-7B-Instruct",
        organization="Qwen",
        parameter_count="7B",
        repository=manifest["model_id"],
        revision=manifest["model_revision"],
        compute_dtype=manifest["compute_dtype"],
        weight_loading=manifest["weight_loading"],
        thinking=False,
        tracks={},
    )
    for task_id in TRACKS:
        track_manifest = manifest["tracks"][task_id]
        for artifact_name, expected in track_manifest["artifacts"].items():
            artifact = source_dir / artifact_name
            _assert_equal(artifact.stat().st_size, expected["bytes"], f"{artifact} size")
            _assert_equal(sha256(artifact), expected["sha256"], f"{artifact} hash")
        generation = dict(track_manifest["generation"])
        track = _score_track(
            model,
            task_id,
            source_dir / f"{task_id}_submission.jsonl",
            source_dir / f"{task_id}.diagnostics.jsonl",
            generation,
            {
                "source_manifest_sha256": sha256(manifest_path),
                "source_run": track_manifest.get("source_run"),
                "pipeline_revision": manifest["selection_policy"]["pipeline_revision"],
                "invalid_format_count": track_manifest["invalid_format_count"],
            },
        )
        actual_invalid = int(
            track.score.grading.get("method_counts", {}).get("invalid_format", 0)
        )
        _assert_equal(
            actual_invalid,
            int(track_manifest["invalid_format_count"]),
            f"Qwen2.5 {task_id} invalid-format count",
        )
        model.tracks[task_id] = track
    return model


def _qwen35_plan(result_root: Path, *, thinking: bool) -> ModelPlan:
    if thinking:
        parent = result_root / "qwen35-thinking-enabled"
        run_names = {
            "do_you_see_me": "qwen35-9b-fp16-dysm-thinking-enabled",
            "minds_eye": "qwen35-9b-fp16-mindseye-thinking-enabled",
        }
        display_name = "Qwen3.5-9B (Thinking)"
    else:
        parent = result_root / "qwen-35-thinking-disabled"
        run_names = {
            "do_you_see_me": "qwen35-9b-fp16-dysm-thinking-disabled",
            "minds_eye": "qwen35-9b-fp16-mindseye-thinking-disabled",
        }
        display_name = "Qwen3.5-9B (Non-thinking)"
    model = ModelPlan(
        display_name=display_name,
        organization="Qwen",
        parameter_count="9B",
        repository=QWEN35_MODEL_ID,
        revision=QWEN35_MODEL_REVISION,
        compute_dtype="float16",
        weight_loading="full",
        thinking=thinking,
        tracks={},
    )
    for task_id, run_name in run_names.items():
        source_dir = parent / run_name
        config_path = source_dir / ".run_config.json"
        manifest_path = source_dir / "run_manifest.json"
        config = read_json(config_path)
        manifest = read_json(manifest_path)
        for source, label in ((config, "config"), (manifest, "manifest")):
            _assert_equal(source.get("model_id"), model.repository, f"{display_name} {label} model")
            _assert_equal(source.get("model_revision"), model.revision, f"{display_name} {label} revision")
            _assert_equal(source.get("weight_loading"), model.weight_loading, f"{display_name} {label} weights")
            _assert_equal(source.get("compute_dtype"), model.compute_dtype, f"{display_name} {label} dtype")
        _assert_equal(config.get("allowed_tracks"), [task_id], f"{display_name} track")
        _assert_equal(
            config.get("chat_template_kwargs", {}).get("enable_thinking"),
            thinking,
            f"{display_name} thinking mode",
        )
        _assert_equal(config.get("prompt_mode"), "cot" if thinking else "noncot", f"{display_name} prompt mode")
        dataset = manifest.get("dataset", {})
        _assert_equal(dataset.get("repo_id"), DATASET_REPOSITORY, f"{display_name} dataset")
        _assert_equal(dataset.get("revision"), DATASET_REVISION, f"{display_name} dataset revision")
        run = manifest["tracks"][task_id]["runs"]
        _assert_equal(len(run), 1, f"{display_name} {task_id} run count")
        run = run[0]
        submission_path = source_dir / run["submission_file"]
        diagnostics_path = source_dir / run["diagnostics_file"]
        _assert_equal(sha256(submission_path), run["sha256"], f"{display_name} {task_id} submission hash")
        _assert_equal(run["row_count"], len(expected_question_ids(PROJECT_DIR, task_id)), f"{display_name} {task_id} row count")
        _assert_equal(
            manifest["tracks"][task_id]["question_bundle_sha256"],
            sha256(PROJECT_DIR / "tasks" / task_id / "questions.jsonl"),
            f"{display_name} {task_id} question hash",
        )
        model.tracks[task_id] = _score_track(
            model,
            task_id,
            submission_path,
            diagnostics_path,
            dict(manifest["generation"]),
            {
                "source_manifest_sha256": sha256(manifest_path),
                "source_run_config_sha256": sha256(config_path),
                "source_diagnostics_sha256": sha256(diagnostics_path),
                "pipeline_schema_version": manifest.get("schema_version"),
                "dataset_revision": DATASET_REVISION,
                "thinking_mode": thinking,
            },
        )
    return model


def build_plan(result_root: Path) -> list[ModelPlan]:
    result_root = result_root.expanduser().resolve()
    return [
        _qwen25_plan(result_root),
        _qwen35_plan(result_root, thinking=True),
        _qwen35_plan(result_root, thinking=False),
    ]


def _existing_models_by_name() -> dict[str, dict[str, Any]]:
    return {
        normalize_model_name(model["model_name"]): model
        for model in list_registered_models()
    }


def _delete_submission(
    leaderboard: LeaderboardStore,
    submission: dict[str, Any],
    *,
    reason: str,
) -> bool:
    submission_id = str(submission.get("submission_id") or "")
    if not submission_id or submission.get("moderation_status") == "deleted":
        return False
    changed = set_moderation_status(
        submission_id,
        "deleted",
        reason=reason,
        moderated_by=OWNER_EMAIL,
    )
    if changed is None:
        raise RuntimeError(f"Could not delete superseded submission {submission_id}")
    leaderboard.remove_submission(submission_id)
    return True


def apply_plan(plan: list[ModelPlan], quota_limit: int) -> dict[str, Any]:
    before = submission_integrity_status()
    if not before["healthy"]:
        raise RuntimeError("Submission database is unhealthy before Qwen replacement")
    backup_path, backup_manifest = write_backup_archive(
        AUTO_BACKUP_DIR,
        retention_count=AUTO_BACKUP_RETENTION_COUNT,
    )
    leaderboard = LeaderboardStore(LEADERBOARD_STORE_FILE)
    existing_by_name = _existing_models_by_name()
    imported_models = []
    imported_submissions = []
    skipped_submissions = []
    superseded_submission_ids = []
    archived_model_ids = []

    for model in plan:
        normalized_name = normalize_model_name(model.display_name)
        registered = existing_by_name.get(normalized_name)
        if registered is not None and registered["owner_email"] != OWNER_EMAIL:
            raise RuntimeError(f"Model '{model.display_name}' belongs to another account")
        if registered is None:
            registered = create_registered_model(
                OWNER_EMAIL,
                model.display_name,
                {
                    "organization": model.organization,
                    "access": "open_weights",
                    "parameter_count": model.parameter_count,
                },
            )
            existing_by_name[normalized_name] = registered
            imported_models.append(registered["model_id"])

        model_id = registered["model_id"]
        existing_submissions = [
            row
            for row in list_submissions(
                user_email=OWNER_EMAIL,
                limit=500,
                include_failed=False,
                include_deleted=True,
            )
            if row.get("model_id") == model_id
        ]
        for task_id in TRACKS:
            track = model.tracks[task_id]
            identical = next(
                (
                    row
                    for row in existing_submissions
                    if row.get("task_id") == task_id
                    and row.get("file_sha256") == track.file_sha256
                    and row.get("moderation_status") != "deleted"
                ),
                None,
            )
            if identical is not None:
                skipped_submissions.append(identical["submission_id"])
                continue

            reservation = try_consume_quota(
                OWNER_EMAIL,
                task_id,
                model_name=model.display_name,
                model_id=model_id,
                request_id=f"latest-qwen-import-{uuid.uuid4().hex}",
                ip="local-admin-import",
                limit=quota_limit,
            )
            if not reservation.allowed or not reservation.submission_id:
                raise RuntimeError(f"Could not reserve import for {model.display_name}/{task_id}")
            score = track.score
            score.model_id = model_id
            try:
                store_submission_answers(
                    reservation.submission_id,
                    score_submission_id=score.submission_id,
                    file_sha256=track.file_sha256,
                    records=track.answer_records,
                    model_meta=score.model_meta,
                    score_json=score.to_dict(),
                )
                finalize_submission(reservation.submission_id, True)
                leaderboard.add_result(score, submitted_by=OWNER_EMAIL)
            except Exception:
                finalize_submission(reservation.submission_id, False)
                raise
            imported_submissions.append(score.submission_id)
            for previous in existing_submissions:
                if previous.get("task_id") == task_id and previous.get("moderation_status") != "deleted":
                    if _delete_submission(
                        leaderboard,
                        previous,
                        reason="Superseded by the latest finalized evaluation",
                    ):
                        superseded_submission_ids.append(previous["submission_id"])

    current_models = _existing_models_by_name()
    all_submissions = list_submissions(
        user_email=OWNER_EMAIL,
        limit=500,
        include_failed=True,
        include_hidden=True,
        include_deleted=True,
    )
    for old_name in SUPERSEDED_MODEL_NAMES:
        old_model = current_models.get(normalize_model_name(old_name))
        if old_model is None:
            continue
        for submission in all_submissions:
            if submission.get("model_id") != old_model["model_id"]:
                continue
            if _delete_submission(
                leaderboard,
                submission,
                reason="Superseded by the latest fully identified Qwen evaluation",
            ):
                superseded_submission_ids.append(submission["submission_id"])
        archived = archive_registered_model(old_model["model_id"], OWNER_EMAIL)
        if archived is not None:
            archived_model_ids.append(old_model["model_id"])

    after = submission_integrity_status()
    if not after["healthy"]:
        raise RuntimeError("Submission database is unhealthy after Qwen replacement")
    expected_public = set(latest_visible_scored_submission_ids())
    actual_public = set(leaderboard.public_submission_ids())
    if actual_public != expected_public:
        raise RuntimeError(
            "Public leaderboard cache differs from the latest visible database submissions"
        )
    return {
        "backup": str(backup_path),
        "backup_sqlite_snapshots": backup_manifest["validation"]["sqlite_snapshots"],
        "imported_model_count": len(imported_models),
        "imported_submission_count": len(imported_submissions),
        "skipped_submission_count": len(skipped_submissions),
        "superseded_submission_count": len(set(superseded_submission_ids)),
        "archived_model_count": len(archived_model_ids),
        "database_integrity": after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "results" / "final",
    )
    parser.add_argument("--quota-limit", type=int, default=10_000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.quota_limit <= 0:
        parser.error("--quota-limit must be positive")
    plan = build_plan(args.result_root)
    summary: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "superseded_models": sorted(SUPERSEDED_MODEL_NAMES),
        "models": [
            {
                "display_name": model.display_name,
                "repository": model.repository,
                "revision": model.revision,
                "thinking": model.thinking,
                "vpci": round(model.vpci, 6),
                "tracks": {
                    task_id: {
                        "macro_accuracy": round(model.tracks[task_id].score.macro_accuracy, 6),
                        "micro_accuracy": round(model.tracks[task_id].score.accuracy, 6),
                        "invalid_format_count": int(
                            model.tracks[task_id]
                            .score.grading.get("method_counts", {})
                            .get("invalid_format", 0)
                        ),
                        "submission_sha256": model.tracks[task_id].file_sha256,
                    }
                    for task_id in TRACKS
                },
            }
            for model in plan
        ],
    }
    if args.apply:
        summary["import"] = apply_plan(plan, args.quota_limit)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

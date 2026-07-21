#!/usr/bin/env python3
"""Score and import verified canonical visual-evaluation results.

The command is dry-run only unless ``--apply`` is supplied. It validates the
canonical result tree, scores every selected file with the production scorer,
creates a backup, then uses the same model/submission stores as the API.
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
from config import (
    AUTO_BACKUP_DIR,
    AUTO_BACKUP_RETENTION_COUNT,
    LEADERBOARD_STORE_FILE,
)
from evaluation.finalize_visual_results import verify_canonical_results
from leaderboard_store import LeaderboardStore
from scoring.task_scorer import TaskScorer
from submission_store import (
    create_registered_model,
    finalize_submission,
    list_registered_models,
    normalize_model_name,
    store_submission_answers,
    submission_integrity_status,
    try_consume_quota,
)


TRACKS = ("do_you_see_me", "minds_eye")
DEFAULT_OWNER = "admin-import@ms-vista.local"


MODEL_CATALOG: dict[str, dict[str, str]] = {
    "qwen35-9b": {
        "repository": "Qwen/Qwen3.5-9B",
        "display_name": "Qwen3.5-9B",
        "organization": "Qwen",
        "parameter_count": "9B",
    },
    "internvl35-8b": {
        "repository": "OpenGVLab/InternVL3_5-8B",
        "display_name": "InternVL3.5-8B (Thinking)",
        "organization": "OpenGVLab",
        "parameter_count": "8B",
    },
    "qwen25-vl-7b": {
        "repository": "Qwen/Qwen2.5-VL-7B-Instruct",
        "display_name": "Qwen2.5-VL-7B-Instruct",
        "organization": "Qwen",
        "parameter_count": "7B",
    },
    "qwen3-vl-8b": {
        "repository": "Qwen/Qwen3-VL-8B-Instruct",
        "display_name": "Qwen3-VL-8B-Instruct",
        "organization": "Qwen",
        "parameter_count": "8B",
    },
    "qwen36-27b": {
        "repository": "Qwen/Qwen3.6-27B",
        "display_name": "Qwen3.6-27B",
        "organization": "Qwen",
        "parameter_count": "27B",
    },
    "deepseek-vl2": {
        "repository": "deepseek-ai/deepseek-vl2",
        "display_name": "DeepSeek-VL2",
        "organization": "DeepSeek",
        "parameter_count": "",
    },
    "gemma3-12b-it": {
        "repository": "google/gemma-3-12b-it",
        "display_name": "Gemma 3 12B IT",
        "organization": "Google",
        "parameter_count": "12B",
    },
    "gemma3-27b-it": {
        "repository": "google/gemma-3-27b-it",
        "display_name": "Gemma 3 27B IT",
        "organization": "Google",
        "parameter_count": "27B",
    },
    "llama32-11b-vision-instruct": {
        "repository": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "display_name": "Llama 3.2 11B Vision Instruct",
        "organization": "Meta",
        "parameter_count": "11B",
    },
    "phi4-multimodal": {
        "repository": "microsoft/Phi-4-multimodal-instruct",
        "display_name": "Phi-4 Multimodal Instruct",
        "organization": "Microsoft",
        "parameter_count": "",
    },
    "kimi-vl-a3b-instruct": {
        "repository": "moonshotai/Kimi-VL-A3B-Instruct",
        "display_name": "Kimi-VL-A3B-Instruct",
        "organization": "Moonshot AI",
        "parameter_count": "",
    },
    "minicpm-v46": {
        "repository": "openbmb/MiniCPM-V-4.6",
        "display_name": "MiniCPM-V 4.6",
        "organization": "OpenBMB",
        "parameter_count": "",
    },
    "glm46v-flash": {
        "repository": "zai-org/GLM-4.6V-Flash",
        "display_name": "GLM-4.6V-Flash",
        "organization": "Z.ai",
        "parameter_count": "",
    },
}


@dataclass
class TrackImport:
    task_id: str
    path: Path
    payload: bytes
    file_sha256: str
    answer_records: list[dict[str, Any]]
    score: Any
    invalid_format_count: int


@dataclass
class ModelImport:
    slug: str
    manifest: dict[str, Any]
    catalog: dict[str, str]
    manifest_sha256: str
    tracks: dict[str, TrackImport]

    @property
    def vpci(self) -> float:
        return sum(
            float(self.tracks[task_id].score.macro_accuracy)
            for task_id in TRACKS
        ) / len(TRACKS)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _submission_model_meta(
    model: ModelImport,
    task_id: str,
) -> dict[str, Any]:
    track_manifest = model.manifest["tracks"][task_id]
    prompt_mode = str(track_manifest["generation"]["prompt_mode"])
    organization = model.catalog["organization"]
    access = "open_weights"
    return {
        "organization": organization,
        "org": organization,
        "access": access,
        "type": access,
        "parameter_count": model.catalog.get("parameter_count", ""),
        "method_description": (
            "Official MS-VISTA canonical visual evaluation using the pinned model "
            "revision, original unquantized weights, BF16 compute, benchmark prompt, "
            "and deterministic final-answer contract recorded in the retained manifest."
        ),
        "cot_used": "Yes" if prompt_mode == "cot" else "No",
        "prompt_template": (
            "Official shared MS-VISTA prompt identified by the prompt hash in the "
            "canonical final manifest."
        ),
        "changes_from_previous": "Initial import from the verified canonical evaluation set.",
        "model_repository": model.manifest["model_id"],
        "model_revision": model.manifest["model_revision"],
        "weight_loading": model.manifest["weight_loading"],
        "compute_dtype": model.manifest["compute_dtype"],
        "submission_track": task_id,
    }


def _submission_metadata(
    model: ModelImport,
    track: TrackImport,
) -> dict[str, Any]:
    track_manifest = model.manifest["tracks"][track.task_id]
    return {
        "evaluation_source": "canonical_visual_suite",
        "canonical_slug": model.slug,
        "canonical_model_id": model.manifest["model_id"],
        "canonical_model_revision": model.manifest["model_revision"],
        "final_manifest_sha256": model.manifest_sha256,
        "source_submission_sha256": track.file_sha256,
        "source_run": track_manifest.get("source_run"),
        "strict_answer_count": track_manifest["strict_answer_count"],
        "invalid_format_count": track_manifest["invalid_format_count"],
    }


def build_import_plan(
    result_root: Path,
    excluded_slugs: set[str],
) -> list[ModelImport]:
    result_root = result_root.expanduser().resolve()
    verify_canonical_results(result_root, PROJECT_DIR)
    index = json.loads((result_root / "index.json").read_text(encoding="utf-8"))
    scorers = {task_id: TaskScorer(task_id) for task_id in TRACKS}
    plan = []

    for index_model in index["models"]:
        slug = str(index_model["slug"])
        if slug in excluded_slugs:
            continue
        catalog = MODEL_CATALOG.get(slug)
        if catalog is None:
            raise ValueError(f"No trusted model metadata is configured for '{slug}'.")

        model_dir = result_root / slug
        manifest_path = model_dir / "final_manifest.json"
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        if manifest["model_id"] != catalog["repository"]:
            raise ValueError(
                f"Catalog repository mismatch for '{slug}': "
                f"{catalog['repository']} != {manifest['model_id']}"
            )

        tracks = {}
        for task_id in TRACKS:
            path = model_dir / f"{task_id}_submission.jsonl"
            payload = path.read_bytes()
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path} is not valid UTF-8.") from exc
            scorer = scorers[task_id]
            predictions, parsed_meta, answer_records = (
                scorer.parse_submission_text_with_records(text)
            )
            provisional_model = ModelImport(
                slug=slug,
                manifest=manifest,
                catalog=catalog,
                manifest_sha256=_sha256(manifest_payload),
                tracks={},
            )
            score = scorer.score_predictions(
                predictions,
                model_name=catalog["display_name"],
                parsed_meta=parsed_meta,
                model_meta=_submission_model_meta(provisional_model, task_id),
                submission_metadata={
                    "evaluation_source": "canonical_visual_suite",
                    "canonical_slug": slug,
                },
            )
            invalid_format_count = int(
                score.grading.get("method_counts", {}).get("invalid_format", 0)
            )
            expected_invalid = int(manifest["tracks"][task_id]["invalid_format_count"])
            if invalid_format_count != expected_invalid:
                raise ValueError(
                    f"{slug}/{task_id} scorer found {invalid_format_count} invalid "
                    f"answers; manifest declares {expected_invalid}."
                )
            tracks[task_id] = TrackImport(
                task_id=task_id,
                path=path,
                payload=payload,
                file_sha256=_sha256(payload),
                answer_records=answer_records,
                score=score,
                invalid_format_count=invalid_format_count,
            )
        model = ModelImport(
            slug=slug,
            manifest=manifest,
            catalog=catalog,
            manifest_sha256=_sha256(manifest_payload),
            tracks=tracks,
        )
        for track in tracks.values():
            track.score.metadata.update(_submission_metadata(model, track))
        plan.append(model)
    return plan


def _existing_models_by_name() -> dict[str, dict[str, Any]]:
    return {
        normalize_model_name(model["model_name"]): model
        for model in list_registered_models()
    }


def apply_import_plan(
    plan: list[ModelImport],
    *,
    owner_email: str,
    quota_limit: int,
) -> dict[str, Any]:
    before = submission_integrity_status()
    if not before["healthy"]:
        raise RuntimeError(
            f"Submission database is unhealthy before import: {before['issue_count']} issue(s)."
        )
    backup_path, backup_manifest = write_backup_archive(
        AUTO_BACKUP_DIR,
        retention_count=AUTO_BACKUP_RETENTION_COUNT,
    )
    leaderboard = LeaderboardStore(LEADERBOARD_STORE_FILE)
    existing_by_name = _existing_models_by_name()
    imported_models = []
    imported_submissions = []
    skipped_submissions = []

    for model in plan:
        normalized_name = normalize_model_name(model.catalog["display_name"])
        registered = existing_by_name.get(normalized_name)
        if registered is not None and registered["owner_email"] != owner_email:
            raise RuntimeError(
                f"Model '{model.catalog['display_name']}' belongs to another account."
            )
        if registered is None:
            registered = create_registered_model(
                owner_email,
                model.catalog["display_name"],
                {
                    "organization": model.catalog["organization"],
                    "access": "open_weights",
                    "parameter_count": model.catalog.get("parameter_count", ""),
                },
            )
            existing_by_name[normalized_name] = registered
            imported_models.append(registered["model_id"])

        model_id = registered["model_id"]
        existing_benchmarks = registered.get("benchmarks") or {}
        for task_id in TRACKS:
            track = model.tracks[task_id]
            existing = existing_benchmarks.get(task_id)
            if existing is not None:
                if existing.get("file_sha256") == track.file_sha256:
                    skipped_submissions.append(existing["submission_id"])
                    continue
                raise RuntimeError(
                    f"Model '{model.catalog['display_name']}' already has a different "
                    f"{task_id} submission. Refusing to overwrite it."
                )

            reservation = try_consume_quota(
                owner_email,
                task_id,
                model_name=model.catalog["display_name"],
                model_id=model_id,
                request_id=f"canonical-import-{uuid.uuid4().hex}",
                ip="local-admin-import",
                limit=quota_limit,
            )
            if not reservation.allowed or not reservation.submission_id:
                raise RuntimeError(
                    f"Could not reserve the trusted import slot for "
                    f"{model.catalog['display_name']}/{task_id}."
                )

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
                leaderboard.add_result(score, submitted_by=owner_email)
            except Exception:
                finalize_submission(reservation.submission_id, False)
                raise
            imported_submissions.append(score.submission_id)

    after = submission_integrity_status()
    if not after["healthy"]:
        raise RuntimeError(
            f"Submission database is unhealthy after import: {after['issue_count']} issue(s)."
        )
    public_ids = set(leaderboard.public_submission_ids())
    missing_public_ids = sorted(set(imported_submissions) - public_ids)
    if missing_public_ids:
        raise RuntimeError(
            "Imported submissions are missing from the public leaderboard: "
            + ", ".join(missing_public_ids)
        )
    return {
        "backup": str(backup_path),
        "backup_sqlite_snapshots": backup_manifest["validation"]["sqlite_snapshots"],
        "imported_model_count": len(imported_models),
        "imported_submission_count": len(imported_submissions),
        "skipped_submission_count": len(skipped_submissions),
        "database_integrity": after,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "results" / "final",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SLUG",
        help="Canonical model slug to score but not import (repeatable).",
    )
    parser.add_argument("--owner-email", default=DEFAULT_OWNER)
    parser.add_argument("--quota-limit", type=int, default=10_000)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.quota_limit <= 0:
        raise SystemExit("--quota-limit must be positive")
    plan = build_import_plan(args.result_root, set(args.exclude))
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "excluded_slugs": sorted(set(args.exclude)),
        "models": [
            {
                "slug": model.slug,
                "model_id": model.manifest["model_id"],
                "display_name": model.catalog["display_name"],
                "organization": model.catalog["organization"],
                "vpci": round(model.vpci, 6),
                "do_you_see_me_macro": round(
                    model.tracks["do_you_see_me"].score.macro_accuracy, 6
                ),
                "minds_eye_macro": round(
                    model.tracks["minds_eye"].score.macro_accuracy, 6
                ),
            }
            for model in plan
        ],
    }
    if args.apply:
        summary["import"] = apply_import_plan(
            plan,
            owner_email=args.owner_email.strip().lower(),
            quota_limit=args.quota_limit,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

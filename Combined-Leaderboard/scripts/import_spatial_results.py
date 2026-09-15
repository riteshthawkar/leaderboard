#!/usr/bin/env python3
"""Validate and import completed Spatial Track 3 evidence packages.

The command is dry-run only unless ``--apply`` is supplied. Every package is
validated against the installed public contract before a backup or database
write occurs. Artifact-backed packages remain explicitly self-reported; the
import checks package integrity, public coverage, and claimed score arithmetic,
not semantic correctness. Imports may map a harness repository name to an
existing leaderboard display name so one model is not split across rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
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
from leaderboard_store import LeaderboardStore
from spatial_submission import (
    SPATIAL_ARTIFACT_ANSWERS_MEMBER,
    SPATIAL_ARTIFACT_ARCHIVE_NAME,
    SPATIAL_ARTIFACT_CHECKSUMS_MEMBER,
    SPATIAL_ARTIFACT_MANIFEST_MEMBER,
    SPATIAL_ARTIFACT_SCORES_MEMBER,
    SPATIAL_MANIFEST_MEMBER,
    SPATIAL_REPORT_MEMBER,
    SPATIAL_SUBMISSION_ARCHIVE_NAME,
    SPATIAL_SUBMISSION_MEMBER,
    build_spatial_task_score,
    parse_spatial_artifact_evidence,
    parse_spatial_evidence,
    read_spatial_artifact_archive,
    read_spatial_submission_archive,
    spatial_bundle_health,
    _load_public_spatial_contract,
    validate_run_manifest,
    validate_spatial_report,
)
from submission_store import (
    create_registered_model,
    finalize_submission,
    latest_visible_scored_submission_fingerprints,
    latest_visible_scored_submission_ids,
    list_registered_models,
    normalize_model_name,
    set_moderation_status,
    store_submission_answers,
    submission_integrity_status,
    try_consume_quota,
)


DEFAULT_OWNER = "admin-import@ms-vista.local"
DEFAULT_MODEL_NAMES = {
    "OpenGVLab/InternVL3_5-8B": "InternVL3.5-8B (Thinking)",
    "Qwen/Qwen3.6-27B": "Qwen3.6-27B",
}
ORGANIZATION_NAMES = {
    "OpenGVLab": "OpenGVLab",
    "Qwen": "Qwen",
    "google": "Google",
    "meta-llama": "Meta",
    "microsoft": "Microsoft",
    "openbmb": "OpenBMB",
    "zai-org": "Z.ai",
}


@dataclass
class SpatialImport:
    package_path: Path
    package_bytes: bytes
    package_sha256: str
    source_model_name: str
    display_name: str
    organization: str
    parameter_count: str
    model_revision: str
    records: list[dict[str, Any]]
    artifacts: dict[str, bytes]
    contract: dict[str, bytes | str]
    run_metadata: dict[str, Any]
    report: dict[str, Any]
    score: Any
    declared_model: dict[str, Any] = field(default_factory=dict)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_model_names(values: list[str]) -> dict[str, str]:
    result = dict(DEFAULT_MODEL_NAMES)
    for value in values:
        source, separator, display = value.partition("=")
        source = source.strip()
        display = display.strip()
        if not separator or not source or not display:
            raise ValueError(
                f"Invalid --model-name value {value!r}; expected HARNESS_NAME=DISPLAY_NAME."
            )
        result[source] = display
    return result


def _organization(model_name: str) -> str:
    namespace = model_name.split("/", 1)[0] if "/" in model_name else ""
    return ORGANIZATION_NAMES.get(namespace, namespace or "Unknown")


def _parameter_count(model_name: str) -> str:
    match = re.search(r"(?:^|[-_])(\d+(?:\.\d+)?B)(?:$|[-_])", model_name, re.I)
    return match.group(1).upper() if match else ""


def _default_display_name(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _model_metadata(item: SpatialImport) -> dict[str, Any]:
    artifact_backed = (
        item.run_metadata.get("verification_level")
        == "self_reported_artifact_backed"
    )
    declared_access = item.declared_model.get("access")
    if declared_access is None:
        declared_access = "open_weights" if item.source_model_name.split("/", 1)[0] in ORGANIZATION_NAMES else "unknown"
    if declared_access not in {"open_weights", "closed", "unknown"}:
        raise ValueError("Invalid declared model access type")
    metadata = {
        "organization": item.organization,
        "org": item.organization,
        "access": declared_access,
        "type": declared_access,
        "parameter_count": item.parameter_count,
        "method_description": (
            "Submitter-reported MS-VISTA Track 3 evaluation with complete public "
            "sample coverage, retained final answers and original model outputs, "
            "and aggregate counts reproduced from per-sample claimed credit."
            if artifact_backed
            else
            "Official MS-VISTA paper-aligned Track 3 evaluation using the pinned "
            "model revision, unquantized BF16 serving, deterministic pass@1 "
            "decoding, and the pinned paper judge."
        ),
        "cot_used": "Both",
        "prompt_template": (
            "Direct and chain-of-thought conditions; see the retained manifest for submitter-declared configuration. Prompt/model revision attestation is not implied."
            if artifact_backed else
            "Official Track 3 direct and chain-of-thought prompts identified by the retained run manifest."
        ),
        "changes_from_previous": (
            "Imported from the self-reported, artifact-backed Track 3 package."
            if artifact_backed
            else "Initial import from a canonical Track 3 package."
        ),
        "model_repository": item.source_model_name,
        "model_revision": item.model_revision,
        "submission_track": "spatial",
    }
    if not artifact_backed:
        metadata.update(weight_loading="unquantized", compute_dtype="bfloat16")
    if item.run_metadata.get("missing_output_rows"):
        metadata["method_description"] += f" {item.run_metadata['missing_output_rows']} source outputs are missing and retain zero claimed credit; their failure cause is unknown."
    return metadata


def _contract_sources(contract_dir: Path, *, allow_submitted_cohort: bool = False) -> dict[str, bytes | str]:
    contract_dir = contract_dir.expanduser().resolve()
    paths = {
        "manifest": contract_dir / "manifest.json",
        "questions": contract_dir / "questions.jsonl",
        "template": contract_dir / "submission_template.jsonl",
    }
    if allow_submitted_cohort:
        _load_public_spatial_contract(paths["manifest"], paths["template"], paths["questions"], allow_submitted_cohort=True)
    else:
        status, details = spatial_bundle_health(
            paths["manifest"], paths["template"], paths["questions"]
        )
        if status != "healthy" or details.get("production_ready") is not True:
            raise ValueError(f"The installed Spatial public contract is not production ready: {details.get('error') or details}")
    result: dict[str, bytes | str] = {
        key: path.read_bytes() for key, path in paths.items()
    }
    result["manifest_sha256"] = _sha256(result["manifest"])
    return result


def build_import_plan(
    package_paths: list[Path],
    *,
    contract_dir: Path,
    model_names: dict[str, str] | None = None,
    allow_submitted_cohort: bool = False,
) -> list[SpatialImport]:
    if not package_paths:
        raise ValueError("At least one --package or package under --package-root is required.")
    contract = _contract_sources(contract_dir, allow_submitted_cohort=allow_submitted_cohort)
    model_names = {**DEFAULT_MODEL_NAMES, **(model_names or {})}
    plan: list[SpatialImport] = []
    seen_paths: set[Path] = set()
    seen_display_names: set[str] = set()

    for supplied_path in package_paths:
        package_path = supplied_path.expanduser().resolve()
        if package_path in seen_paths:
            continue
        seen_paths.add(package_path)
        if package_path.name not in {
            SPATIAL_SUBMISSION_ARCHIVE_NAME,
            SPATIAL_ARTIFACT_ARCHIVE_NAME,
        }:
            raise ValueError(
                f"{package_path} is not a supported Track 3 package filename."
            )
        package_bytes = package_path.read_bytes()
        artifact_backed = package_path.name == SPATIAL_ARTIFACT_ARCHIVE_NAME
        if artifact_backed:
            artifact_package = read_spatial_artifact_archive(package_bytes)
            manifest = artifact_package.manifest
        else:
            submission_bytes, manifest_bytes, report_bytes = (
                read_spatial_submission_archive(package_bytes)
            )
            try:
                manifest = json.loads(manifest_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{package_path} has an unreadable run manifest.") from exc
        source_model_name = str((manifest.get("model") or {}).get("name") or "").strip()
        if not source_model_name:
            raise ValueError(f"{package_path} does not declare a model name.")
        display_name = model_names.get(
            source_model_name, (manifest.get("model") or {}).get("display_name") or _default_display_name(source_model_name)
        )
        normalized_display = normalize_model_name(display_name)
        if normalized_display in seen_display_names:
            raise ValueError(
                f"More than one supplied package maps to the model {display_name!r}."
            )
        seen_display_names.add(normalized_display)

        if artifact_backed:
            records, report, _benchmark, run_metadata = (
                parse_spatial_artifact_evidence(
                    artifact_package,
                    source_model_name,
                    contract["manifest"],
                    contract["template"],
                    contract["questions"],
                    allow_submitted_cohort=allow_submitted_cohort,
                )
            )
            artifact_contents = {
                SPATIAL_ARTIFACT_ARCHIVE_NAME: package_bytes,
                SPATIAL_ARTIFACT_MANIFEST_MEMBER: artifact_package.members[
                    SPATIAL_ARTIFACT_MANIFEST_MEMBER
                ],
                SPATIAL_ARTIFACT_SCORES_MEMBER: artifact_package.members[
                    SPATIAL_ARTIFACT_SCORES_MEMBER
                ],
                SPATIAL_ARTIFACT_ANSWERS_MEMBER: artifact_package.members[
                    SPATIAL_ARTIFACT_ANSWERS_MEMBER
                ],
                SPATIAL_ARTIFACT_CHECKSUMS_MEMBER: artifact_package.members[
                    SPATIAL_ARTIFACT_CHECKSUMS_MEMBER
                ],
            }
        else:
            records, computed_report, _benchmark = parse_spatial_evidence(
                submission_bytes,
                contract["manifest"],
                contract["template"],
                contract["questions"],
            )
            run_metadata = validate_run_manifest(
                manifest_bytes,
                submission_bytes,
                report_bytes,
                source_model_name,
                records,
                contract["manifest"],
            )
            report = validate_spatial_report(
                report_bytes,
                source_model_name,
                computed_report,
            )
            artifact_contents = {
                SPATIAL_SUBMISSION_ARCHIVE_NAME: package_bytes,
                SPATIAL_SUBMISSION_MEMBER: submission_bytes,
                SPATIAL_MANIFEST_MEMBER: manifest_bytes,
                SPATIAL_REPORT_MEMBER: report_bytes,
            }
        model_revision = str((manifest.get("model") or {}).get("revision") or "").strip()
        provisional = SpatialImport(
            package_path=package_path,
            package_bytes=package_bytes,
            package_sha256=_sha256(package_bytes),
            source_model_name=source_model_name,
            display_name=display_name,
            organization=(manifest.get("model") or {}).get("organization") or _organization(source_model_name),
            parameter_count=_parameter_count(source_model_name),
            model_revision=model_revision,
            records=records,
            artifacts=artifact_contents,
            contract=dict(contract),
            run_metadata=run_metadata,
            report=report,
            score=None,
            declared_model=dict(manifest.get("model") or {}),
        )
        provisional.score = build_spatial_task_score(
            report,
            display_name,
            _model_metadata(provisional),
            run_metadata,
        )
        provisional.score.metadata.update(
            {
                "evaluation_source": (
                    "self_reported_artifact_backed_track3"
                    if run_metadata.get("verification_level")
                    == "self_reported_artifact_backed"
                    else "canonical_spatial_track3"
                ),
                "verification_level": run_metadata.get("verification_level"),
                "source_model_name": source_model_name,
                "source_model_revision": model_revision,
                "source_package_sha256": provisional.package_sha256,
                "benchmark_manifest_sha256": contract["manifest_sha256"],
            }
        )
        plan.append(provisional)
    return plan


def _existing_models_by_name() -> dict[str, dict[str, Any]]:
    return {
        normalize_model_name(model["model_name"]): model
        for model in list_registered_models()
    }


def apply_import_plan(
    plan: list[SpatialImport],
    *,
    owner_email: str,
    quota_limit: int,
    replace_existing: bool,
) -> dict[str, Any]:
    before = submission_integrity_status()
    if not before["healthy"]:
        raise RuntimeError(
            f"Submission database is unhealthy before import: {before['issue_count']} issue(s)."
        )
    existing_by_name = _existing_models_by_name()
    for item in plan:
        registered = existing_by_name.get(normalize_model_name(item.display_name))
        if registered is None:
            continue
        if registered["owner_email"] != owner_email:
            raise RuntimeError(f"Model {item.display_name!r} belongs to another account.")
        existing = (registered.get("benchmarks") or {}).get("spatial")
        if (
            existing is not None
            and existing.get("file_sha256") != item.package_sha256
            and not replace_existing
        ):
            raise RuntimeError(
                f"Model {item.display_name!r} already has a different Spatial submission. "
                "Pass --replace-existing to supersede it."
            )

    backup_path, backup_manifest = write_backup_archive(
        AUTO_BACKUP_DIR,
        retention_count=AUTO_BACKUP_RETENTION_COUNT,
    )
    leaderboard = LeaderboardStore(LEADERBOARD_STORE_FILE)
    imported_models: list[str] = []
    imported_submissions: list[str] = []
    skipped_submissions: list[str] = []
    replaced_submissions: list[str] = []

    for item in plan:
        normalized_name = normalize_model_name(item.display_name)
        registered = existing_by_name.get(normalized_name)
        if registered is None:
            registered = create_registered_model(
                owner_email,
                item.display_name,
                {
                    "organization": item.organization,
                    "access": item.score.model_meta["access"],
                    "parameter_count": item.parameter_count,
                },
            )
            existing_by_name[normalized_name] = registered
            imported_models.append(registered["model_id"])
        elif registered["owner_email"] != owner_email:
            raise RuntimeError(f"Model {item.display_name!r} belongs to another account.")

        model_id = registered["model_id"]
        existing = (registered.get("benchmarks") or {}).get("spatial")
        if existing is not None:
            if existing.get("file_sha256") == item.package_sha256 and not replace_existing:
                skipped_submissions.append(existing["submission_id"])
                continue
            if not replace_existing:
                raise RuntimeError(
                    f"Model {item.display_name!r} already has a different Spatial submission."
                )

        reservation = try_consume_quota(
            owner_email,
            "spatial",
            model_name=item.display_name,
            model_id=model_id,
            request_id=f"spatial-import-{uuid.uuid4().hex}",
            ip="local-admin-import",
            limit=quota_limit,
        )
        if not reservation.allowed or not reservation.submission_id:
            raise RuntimeError(
                f"Could not reserve the trusted import slot for {item.display_name!r}."
            )

        score = item.score
        score.model_id = model_id
        artifacts = [
            {
                "artifact_name": name,
                "media_type": (
                    "application/zip"
                    if name in {
                        SPATIAL_SUBMISSION_ARCHIVE_NAME,
                        SPATIAL_ARTIFACT_ARCHIVE_NAME,
                    }
                    else "application/x-ndjson"
                    if name == SPATIAL_SUBMISSION_MEMBER
                    else "application/gzip"
                    if name == SPATIAL_ARTIFACT_ANSWERS_MEMBER
                    else "application/json"
                ),
                "content": content,
            }
            for name, content in item.artifacts.items()
        ]
        try:
            store_submission_answers(
                reservation.submission_id,
                score_submission_id=score.submission_id,
                file_sha256=item.package_sha256,
                records=item.records,
                model_meta=score.model_meta,
                score_json=score.to_dict(),
                artifacts=artifacts,
                spatial_contract=item.contract,
            )
            finalize_submission(reservation.submission_id, True)
            leaderboard.add_result(score, submitted_by=owner_email)
        except Exception:
            finalize_submission(reservation.submission_id, False)
            raise
        imported_submissions.append(score.submission_id)

        if existing is not None and existing.get("submission_id"):
            replaced = set_moderation_status(
                existing["submission_id"],
                "deleted",
                reason="Superseded by the current artifact-backed Track 3 package",
                moderated_by=owner_email,
            )
            if replaced is None:
                raise RuntimeError(
                    f"Could not supersede the previous Spatial submission for {item.display_name!r}."
                )
            leaderboard.remove_submission(existing["submission_id"])
            replaced_submissions.append(existing["submission_id"])

    after = submission_integrity_status()
    if not after["healthy"]:
        raise RuntimeError(
            f"Submission database is unhealthy after import: {after['issue_count']} issue(s)."
        )
    expected_ids = set(latest_visible_scored_submission_ids())
    if set(leaderboard.public_submission_ids()) != expected_ids:
        raise RuntimeError(
            "Public leaderboard cache differs from the latest visible database submissions."
        )
    expected_fingerprints = latest_visible_scored_submission_fingerprints()
    if leaderboard.public_submission_fingerprints() != expected_fingerprints:
        raise RuntimeError(
            "Public leaderboard scores differ from the latest visible database scores."
        )
    return {
        "backup": str(backup_path),
        "backup_sqlite_snapshots": backup_manifest["validation"]["sqlite_snapshots"],
        "imported_model_count": len(imported_models),
        "imported_submission_count": len(imported_submissions),
        "skipped_submission_count": len(skipped_submissions),
        "replaced_submission_count": len(replaced_submissions),
        "database_integrity": after,
    }


def _package_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.package)
    for root in args.package_root:
        paths.extend(root.expanduser().resolve().rglob(SPATIAL_SUBMISSION_ARCHIVE_NAME))
        paths.extend(root.expanduser().resolve().rglob(SPATIAL_ARTIFACT_ARCHIVE_NAME))
    return sorted(set(paths), key=lambda path: str(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", type=Path, default=[])
    parser.add_argument("--package-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=PROJECT_DIR / "tasks" / "spatial",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        metavar="HARNESS_NAME=DISPLAY_NAME",
    )
    parser.add_argument("--owner-email", default=DEFAULT_OWNER)
    parser.add_argument("--quota-limit", type=int, default=10_000)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--submitted-cohort", action="store_true", help="Explicitly allow a reviewed, separately ranked submitted-cohort contract; does not change public upload admission.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.quota_limit <= 0:
        raise SystemExit("--quota-limit must be positive")
    try:
        model_names = _parse_model_names(args.model_name)
        plan = build_import_plan(
            _package_paths(args),
            contract_dir=args.contract_dir,
            model_names=model_names,
            allow_submitted_cohort=args.submitted_cohort,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    summary: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "contract_dir": str(args.contract_dir.expanduser().resolve()),
        "models": [
            {
                "source_model_name": item.source_model_name,
                "display_name": item.display_name,
                "model_revision": item.model_revision,
                "package": str(item.package_path),
                "package_sha256": item.package_sha256,
                "evidence_rows": len(item.records),
                "macro_accuracy": round(float(item.score.macro_accuracy), 6),
                "cot_delta": round(float(item.score.diagnostics.cot_delta), 6),
            }
            for item in plan
        ],
    }
    if args.apply:
        summary["import"] = apply_import_plan(
            plan,
            owner_email=args.owner_email.strip().lower(),
            quota_limit=args.quota_limit,
            replace_existing=args.replace_existing,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

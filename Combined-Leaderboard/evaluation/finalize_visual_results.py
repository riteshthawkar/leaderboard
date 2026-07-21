from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.common.vllm_runner import ANSWER_EXTRACTION_METHOD
from evaluation.common.visual_pipeline import (
    MISSING_ANSWER_TOKEN,
    record_answer,
)
from visual_answer_contract import (
    INVALID_FORMAT_TOKEN,
    PRODUCTION_EXTRACTION_METHOD,
    is_canonical_visual_answer,
    task_from_question_id,
)


TRACKS = ("do_you_see_me", "minds_eye")
CURRENT_PIPELINE_REVISION = "unquantized-bf16-evidence-extraction-v12"
SUPPORTED_PIPELINE_REVISIONS = {
    "unquantized-bf16-smoke-and-full-text-extraction-v10",
    "unquantized-bf16-mandatory-extraction-v11",
    "unquantized-bf16-split-inference-extraction-v12",
    "unquantized-bf16-single-evidence-extraction-v13",
    CURRENT_PIPELINE_REVISION,
}
SUBMISSION_FIELDS = {"question_id", "condition", "answer"}


class FinalizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    source_dir: Path
    source_run: str
    track: str
    submission: Path
    diagnostics: Path
    run_config: Path
    source_manifest: Path | None
    model_id: str
    model_revision: str
    slug: str
    modified_at: float
    config: dict[str, Any]


def live_active_roots(results_root: Path) -> list[Path]:
    active_roots: list[Path] = []
    for marker in results_root.resolve().rglob(".active-run.json"):
        try:
            pid = int(read_json(marker).get("pid") or 0)
            os.kill(pid, 0)
        except (FinalizationError, OSError, ValueError):
            continue
        active_roots.append(marker.parent.resolve())
    return active_roots


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"Expected a JSON object in {path}.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FinalizationError(f"Cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalizationError(
                f"{path} line {line_number} is invalid JSON: {exc.msg}."
            ) from exc
        if not isinstance(row, dict):
            raise FinalizationError(f"{path} line {line_number} must be an object.")
        rows.append(row)
    return rows


def expected_question_ids(project_root: Path, track: str) -> list[str]:
    questions = project_root / "tasks" / track / "questions.jsonl"
    rows = read_jsonl(questions)
    question_ids = [str(row.get("question_id") or "") for row in rows]
    if not question_ids or any(not question_id for question_id in question_ids):
        raise FinalizationError(f"Question bundle {questions} has missing identifiers.")
    if len(set(question_ids)) != len(question_ids):
        raise FinalizationError(f"Question bundle {questions} has duplicate identifiers.")
    return question_ids


def validate_submission(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != len(expected_ids):
        raise FinalizationError(
            f"{path} has {len(rows)} rows; expected {len(expected_ids)}."
        )
    actual_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        if set(row) != SUBMISSION_FIELDS:
            raise FinalizationError(
                f"{path} row {index} has fields {sorted(row)}; expected "
                f"{sorted(SUBMISSION_FIELDS)}."
            )
        question_id = str(row["question_id"])
        answer = row["answer"]
        if row["condition"] != "standard":
            raise FinalizationError(f"{path} row {index} has a nonstandard condition.")
        if not isinstance(answer, str) or not answer.strip():
            raise FinalizationError(f"{path} row {index} has an empty answer.")
        actual_ids.append(question_id)
    if len(set(actual_ids)) != len(actual_ids):
        raise FinalizationError(f"{path} contains duplicate question identifiers.")
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise FinalizationError(
            f"{path} does not match question order and coverage; "
            f"missing={missing[:3]}, extra={extra[:3]}."
        )
    return rows


def validate_diagnostics(
    path: Path, expected_ids: list[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in by_id:
            raise FinalizationError(f"{path} has a missing or duplicate question identifier.")
        by_id[question_id] = row
    if set(by_id) != set(expected_ids):
        raise FinalizationError(f"{path} does not cover the complete question bundle.")
    return rows, by_id


def answer_provenance_counts(
    submission_rows: list[dict[str, Any]], diagnostics_by_id: dict[str, dict[str, Any]]
) -> tuple[int, int, int, int]:
    strict_count = 0
    unresolved_count = 0
    invalid_commitment_count = 0
    exact_raw_count = 0
    for submission in submission_rows:
        question_id = str(submission["question_id"])
        diagnostic = diagnostics_by_id[question_id]
        answer_type = str(diagnostic.get("answer_type") or "text")
        mandatory = (
            diagnostic.get("answer_extraction_method")
            in {ANSWER_EXTRACTION_METHOD, PRODUCTION_EXTRACTION_METHOD}
        )
        parsed = ""
        if mandatory and "extracted_answer" in diagnostic:
            parsed = str(diagnostic.get("extracted_answer") or "").strip()
        elif not mandatory:
            parsed = record_answer(diagnostic, answer_type)
        if parsed:
            if submission["answer"] != parsed:
                raise FinalizationError(
                    f"Submission answer for {question_id} differs from its parsed diagnostic."
                )
            if parsed == MISSING_ANSWER_TOKEN:
                unresolved_count += 1
            elif parsed == INVALID_FORMAT_TOKEN:
                invalid_commitment_count += 1
            elif is_canonical_visual_answer(
                parsed,
                answer_type=answer_type,
                task=task_from_question_id(question_id),
            ):
                strict_count += 1
            else:
                invalid_commitment_count += 1
        elif (
            not diagnostic.get("error")
            and diagnostic.get("output")
            and submission["answer"] == diagnostic["output"]
        ):
            exact_raw_count += 1
        else:
            raise FinalizationError(
                f"Submission answer for {question_id} has no verified diagnostic source."
            )
    return strict_count, unresolved_count, invalid_commitment_count, exact_raw_count


def discover_candidates(
    results_root: Path, output_root: Path, project_root: Path
) -> list[Candidate]:
    expected = {
        track: expected_question_ids(project_root, track) for track in TRACKS
    }
    candidates: list[Candidate] = []
    resolved_output = output_root.resolve()
    active_roots = live_active_roots(results_root)
    for track in TRACKS:
        for submission in results_root.rglob(f"{track}_submission.jsonl"):
            source_dir = submission.parent
            if resolved_output == source_dir.resolve() or resolved_output in source_dir.resolve().parents:
                continue
            if any(
                active_root == source_dir.resolve()
                or active_root in source_dir.resolve().parents
                for active_root in active_roots
            ):
                continue
            diagnostics = source_dir / f"{track}.diagnostics.jsonl"
            run_config = source_dir / ".run_config.json"
            if not diagnostics.is_file() or not run_config.is_file():
                continue
            config = read_json(run_config)
            if (
                config.get("weight_loading") != "unquantized"
                or config.get("compute_dtype") != "bfloat16"
                or config.get("pipeline_revision") not in SUPPORTED_PIPELINE_REVISIONS
            ):
                continue
            model_id = str(config.get("model_id") or "")
            model_revision = str(config.get("model_revision") or "")
            if not model_id or not model_revision:
                continue
            validate_submission(submission, expected[track])
            validate_diagnostics(diagnostics, expected[track])
            manifest = source_dir / "run_manifest.json"
            candidates.append(
                Candidate(
                    source_dir=source_dir,
                    source_run=str(source_dir.relative_to(results_root)),
                    track=track,
                    submission=submission,
                    diagnostics=diagnostics,
                    run_config=run_config,
                    source_manifest=manifest if manifest.is_file() else None,
                    model_id=model_id,
                    model_revision=model_revision,
                    slug=source_dir.name,
                    modified_at=submission.stat().st_mtime,
                    config=config,
                )
            )
    return candidates


def discover_canonical_candidates(
    output_root: Path, project_root: Path
) -> list[Candidate]:
    output_root = output_root.resolve()
    if not (output_root / "index.json").is_file():
        return []
    verify_canonical_results(output_root, project_root)
    index = read_json(output_root / "index.json")
    candidates: list[Candidate] = []
    for model_record in index["models"]:
        manifest_path = output_root / str(model_record["manifest"])
        manifest = read_json(manifest_path)
        model_dir = manifest_path.parent
        slug = str(model_record["slug"])
        for track in TRACKS:
            track_record = manifest["tracks"][track]
            run_config = model_dir / f"{track}.run_config.json"
            config = read_json(run_config)
            source_manifest_name = track_record.get("source_manifest")
            source_manifest = (
                model_dir / source_manifest_name
                if isinstance(source_manifest_name, str) and source_manifest_name
                else None
            )
            modified_at = datetime.fromisoformat(
                str(track_record["source_submission_modified_at"])
            ).timestamp()
            candidates.append(
                Candidate(
                    source_dir=model_dir,
                    source_run=str(
                        track_record.get("source_run") or f"final/{slug}"
                    ),
                    track=track,
                    submission=model_dir / f"{track}_submission.jsonl",
                    diagnostics=model_dir / f"{track}.diagnostics.jsonl",
                    run_config=run_config,
                    source_manifest=source_manifest,
                    model_id=str(manifest["model_id"]),
                    model_revision=str(manifest["model_revision"]),
                    slug=slug,
                    modified_at=modified_at,
                    config=config,
                )
            )
    return candidates


def select_complete_models(candidates: list[Candidate]) -> list[dict[str, Candidate]]:
    latest_by_track: dict[tuple[str, str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.model_id, candidate.model_revision, candidate.track)
        current = latest_by_track.get(key)
        if current is None or candidate.modified_at > current.modified_at:
            latest_by_track[key] = candidate

    revisions: dict[tuple[str, str], dict[str, Candidate]] = {}
    for (model_id, revision, track), candidate in latest_by_track.items():
        revisions.setdefault((model_id, revision), {})[track] = candidate

    complete_by_model: dict[str, tuple[float, dict[str, Candidate]]] = {}
    for (model_id, _revision), tracks in revisions.items():
        if set(tracks) != set(TRACKS):
            continue
        contract = {
            (
                candidate.config.get("weight_loading"),
                candidate.config.get("compute_dtype"),
                candidate.config.get("pipeline_revision"),
            )
            for candidate in tracks.values()
        }
        if len(contract) != 1:
            continue
        newest = max(candidate.modified_at for candidate in tracks.values())
        current = complete_by_model.get(model_id)
        if current is None or newest > current[0]:
            complete_by_model[model_id] = (newest, tracks)
    return [
        tracks
        for _model_id, (_modified_at, tracks) in sorted(complete_by_model.items())
    ]


def artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def copy_track(candidate: Candidate, destination: Path, results_root: Path) -> dict[str, Any]:
    submission_rows = read_jsonl(candidate.submission)
    _diagnostic_rows, diagnostics_by_id = validate_diagnostics(
        candidate.diagnostics,
        [str(row["question_id"]) for row in submission_rows],
    )
    (
        strict_count,
        unresolved_count,
        invalid_commitment_count,
        exact_raw_count,
    ) = answer_provenance_counts(
        submission_rows, diagnostics_by_id
    )

    source_files = [candidate.submission, candidate.diagnostics]
    source_files.extend(sorted(candidate.source_dir.glob(f"{candidate.track}.attempt-*.diagnostics.jsonl")))
    source_files.extend(sorted(candidate.source_dir.glob(f"{candidate.track}.smoke*.diagnostics.jsonl")))
    source_files.extend(sorted(candidate.source_dir.glob(f"{candidate.track}.inference.diagnostics.jsonl")))
    source_files.extend(sorted(candidate.source_dir.glob(f"{candidate.track}.inference.smoke*.diagnostics.jsonl")))
    source_files.extend(sorted(candidate.source_dir.glob(f"{candidate.track}.evidence_extraction.jsonl")))
    copied: dict[str, dict[str, Any]] = {}
    for source in source_files:
        target = destination / source.name
        shutil.copy2(source, target)
        copied[target.name] = artifact_record(target)

    run_config_target = destination / f"{candidate.track}.run_config.json"
    shutil.copy2(candidate.run_config, run_config_target)
    copied[run_config_target.name] = artifact_record(run_config_target)

    source_manifest_name = None
    if candidate.source_manifest is not None:
        source_manifest_target = destination / f"{candidate.track}.source_manifest.json"
        shutil.copy2(candidate.source_manifest, source_manifest_target)
        copied[source_manifest_target.name] = artifact_record(source_manifest_target)
        source_manifest_name = source_manifest_target.name

    return {
        "row_count": len(submission_rows),
        "strict_answer_count": strict_count,
        "unresolved_answer_count": unresolved_count,
        "invalid_commitment_count": invalid_commitment_count,
        "invalid_format_count": unresolved_count + invalid_commitment_count,
        "exact_raw_output_fallback_count": exact_raw_count,
        "source_run": candidate.source_run,
        "source_submission_modified_at": datetime.fromtimestamp(
            candidate.modified_at, timezone.utc
        ).isoformat(),
        "source_run_config": run_config_target.name,
        "source_manifest": source_manifest_name,
        "generation": candidate.config.get("generation", {}).get(candidate.track),
        "serving_engine": candidate.config.get("serving_engine"),
        "tensor_parallel_size": candidate.config.get("tensor_parallel_size"),
        "data_parallel_size": candidate.config.get("data_parallel_size"),
        "request_concurrency": candidate.config.get("request_concurrency"),
        "max_model_len": candidate.config.get("max_model_len"),
        "artifacts": copied,
    }


def verify_canonical_results(output_root: Path, project_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    index = read_json(output_root / "index.json")
    models = index.get("models")
    if not isinstance(models, list) or index.get("model_count") != len(models):
        raise FinalizationError("Canonical index model_count does not match its model list.")
    expected = {
        track: expected_question_ids(project_root.resolve(), track) for track in TRACKS
    }
    verified_models: list[str] = []
    for model_record in models:
        if not isinstance(model_record, dict):
            raise FinalizationError("Canonical index contains a non-object model record.")
        manifest_relative = Path(str(model_record.get("manifest") or ""))
        manifest_path = (output_root / manifest_relative).resolve()
        if output_root not in manifest_path.parents or not manifest_path.is_file():
            raise FinalizationError(f"Invalid canonical manifest path: {manifest_relative}.")
        if sha256(manifest_path) != model_record.get("manifest_sha256"):
            raise FinalizationError(f"Canonical manifest hash mismatch: {manifest_relative}.")
        manifest = read_json(manifest_path)
        if (
            manifest.get("model_id") != model_record.get("model_id")
            or manifest.get("model_revision") != model_record.get("model_revision")
        ):
            raise FinalizationError(f"Canonical index identity mismatch: {manifest_relative}.")
        tracks = manifest.get("tracks")
        if not isinstance(tracks, dict) or set(tracks) != set(TRACKS):
            raise FinalizationError(f"Canonical manifest is missing a track: {manifest_relative}.")
        model_dir = manifest_path.parent
        for track in TRACKS:
            track_record = tracks[track]
            artifacts = track_record.get("artifacts")
            if not isinstance(artifacts, dict):
                raise FinalizationError(f"Canonical {track} artifact map is missing.")
            for filename, expected_artifact in artifacts.items():
                artifact = (model_dir / filename).resolve()
                if model_dir not in artifact.parents or not artifact.is_file():
                    raise FinalizationError(f"Invalid canonical artifact path: {filename}.")
                if (
                    artifact.stat().st_size != expected_artifact.get("bytes")
                    or sha256(artifact) != expected_artifact.get("sha256")
                ):
                    raise FinalizationError(f"Canonical artifact hash mismatch: {artifact}.")
            submission = model_dir / f"{track}_submission.jsonl"
            diagnostics = model_dir / f"{track}.diagnostics.jsonl"
            submission_rows = validate_submission(submission, expected[track])
            _diagnostic_rows, diagnostics_by_id = validate_diagnostics(
                diagnostics, expected[track]
            )
            (
                strict_count,
                unresolved_count,
                invalid_commitment_count,
                exact_raw_count,
            ) = answer_provenance_counts(
                submission_rows, diagnostics_by_id
            )
            if (
                track_record.get("row_count") != len(submission_rows)
                or track_record.get("strict_answer_count") != strict_count
                or track_record.get("unresolved_answer_count", 0)
                != unresolved_count
                or track_record.get("invalid_commitment_count", 0)
                != invalid_commitment_count
                or track_record.get("invalid_format_count")
                != unresolved_count + invalid_commitment_count
                or track_record.get("exact_raw_output_fallback_count")
                != exact_raw_count
            ):
                raise FinalizationError(
                    f"Canonical answer provenance count mismatch for "
                    f"{manifest.get('model_id')}/{track}."
                )
        verified_models.append(str(manifest["model_id"]))
    return {"model_count": len(verified_models), "verified_models": verified_models}


def build_canonical_results(
    results_root: Path, output_root: Path, project_root: Path, dry_run: bool = False
) -> dict[str, Any]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    project_root = project_root.resolve()
    candidates = [
        *discover_canonical_candidates(output_root, project_root),
        *discover_candidates(results_root, output_root, project_root),
    ]
    selected = select_complete_models(candidates)
    if not selected:
        raise FinalizationError("No model has valid final submissions for both tracks.")

    plan = {
        tracks[TRACKS[0]].model_id: {
            track: candidate.source_run
            for track, candidate in tracks.items()
        }
        for tracks in selected
    }
    if dry_run:
        return {"selection": plan, "model_count": len(selected)}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        index_models: list[dict[str, Any]] = []
        for tracks in selected:
            representative = max(tracks.values(), key=lambda candidate: candidate.modified_at)
            destination = staging / representative.slug
            destination.mkdir(parents=True)
            track_records = {
                track: copy_track(candidate, destination, results_root)
                for track, candidate in sorted(tracks.items())
            }
            manifest = {
                "schema_version": 1,
                "finalized_at": datetime.now(timezone.utc).isoformat(),
                "selection_policy": {
                    "precision": "original-unquantized-bf16",
                    "pipeline_revision": CURRENT_PIPELINE_REVISION,
                    "track_selection": "newest-valid-submission-per-model-revision",
                    "required_tracks": list(TRACKS),
                },
                "model_id": representative.model_id,
                "model_revision": representative.model_revision,
                "weight_loading": "unquantized",
                "compute_dtype": "bfloat16",
                "tracks": track_records,
            }
            manifest_path = destination / "final_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            index_models.append(
                {
                    "slug": representative.slug,
                    "model_id": representative.model_id,
                    "model_revision": representative.model_revision,
                    "manifest": f"{representative.slug}/final_manifest.json",
                    "manifest_sha256": sha256(manifest_path),
                    "tracks": {
                        track: {
                            "row_count": record["row_count"],
                            "strict_answer_count": record["strict_answer_count"],
                            "unresolved_answer_count": record[
                                "unresolved_answer_count"
                            ],
                            "invalid_commitment_count": record[
                                "invalid_commitment_count"
                            ],
                            "invalid_format_count": record[
                                "invalid_format_count"
                            ],
                            "exact_raw_output_fallback_count": record[
                                "exact_raw_output_fallback_count"
                            ],
                        }
                        for track, record in track_records.items()
                    },
                }
            )

        index = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_count": len(index_models),
            "models": sorted(index_models, key=lambda item: item["model_id"]),
        }
        (staging / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_canonical_results(staging, project_root)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging, output_root)
        return index
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prune_source_runs(results_root: Path, output_root: Path) -> list[str]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    active_roots = live_active_roots(results_root)
    if active_roots:
        relative = ", ".join(str(path.relative_to(results_root)) for path in active_roots)
        raise FinalizationError(f"Refusing to prune active evaluation runs: {relative}.")
    removed: list[str] = []
    for path in sorted(results_root.glob("visual_suite*")):
        if not path.is_dir() or path.resolve() == output_root:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def prune_cache(results_root: Path) -> bool:
    results_root = results_root.resolve()
    active_roots = live_active_roots(results_root)
    if active_roots:
        relative = ", ".join(str(path.relative_to(results_root)) for path in active_roots)
        raise FinalizationError(f"Refusing to prune cache during active runs: {relative}.")
    cache = results_root / ".cache"
    if not cache.exists():
        return False
    shutil.rmtree(cache)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select, validate, and consolidate final DYS and Mind's Eye results."
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--results-root", type=Path, default=project_root / "evaluation" / "results"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "evaluation" / "results" / "final",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prune-source-runs", action="store_true")
    parser.add_argument("--prune-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.verify_only:
            if args.dry_run or args.prune_source_runs or args.prune_cache:
                raise FinalizationError(
                    "--verify-only cannot be combined with --dry-run or pruning."
                )
            result = verify_canonical_results(args.output_root, args.project_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        result = build_canonical_results(
            args.results_root, args.output_root, args.project_root, args.dry_run
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.prune_source_runs or args.prune_cache:
            if args.dry_run:
                raise FinalizationError("Pruning cannot be used with --dry-run.")
            cleanup: dict[str, Any] = {}
            if args.prune_source_runs:
                cleanup["pruned_source_runs"] = prune_source_runs(
                    args.results_root, args.output_root
                )
            if args.prune_cache:
                cleanup["pruned_cache"] = prune_cache(args.results_root)
            print(json.dumps(cleanup, indent=2))
    except FinalizationError as exc:
        raise SystemExit(f"Finalization failed: {exc}") from exc


if __name__ == "__main__":
    main()
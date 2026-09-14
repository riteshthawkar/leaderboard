"""Build a frozen canonical-extraction source tree from retained API outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.closed_source_catalog import CLOSED_SOURCE_MODEL_CATALOG


TRACKS = ("do_you_see_me", "minds_eye")
SUBSET_TO_TRACK = {
    "dysm_2d_v1": "do_you_see_me",
    "dysm_3d_v1": "do_you_see_me",
    "minds_eye_fresh_v1": "minds_eye",
}
SOURCE_PIPELINE_REVISION = "provider-api-retained-output-v1"
FINAL_PIPELINE_REVISION = "provider-api-gold-blind-evidence-extraction-v1"
PENDING_ANSWER = "__PENDING_EXTRACTION__"


class ClosedSourcePreparationError(RuntimeError):
    """The retained source bundle does not satisfy the frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClosedSourcePreparationError(
                f"{path} line {line_number} is invalid JSON: {exc.msg}."
            ) from exc
        if not isinstance(row, dict):
            raise ClosedSourcePreparationError(
                f"{path} line {line_number} must be a JSON object."
            )
        rows.append(row)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def artifact(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def question_contract(project_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    questions: dict[str, dict[str, Any]] = {}
    ordered: dict[str, list[str]] = {}
    for track in TRACKS:
        path = project_root / "tasks" / track / "questions.jsonl"
        rows = read_jsonl(path)
        ids = []
        for row in rows:
            question_id = str(row.get("question_id") or "").strip()
            if not question_id or question_id in questions:
                raise ClosedSourcePreparationError(
                    f"Question bundle has a missing or duplicate ID: {question_id!r}."
                )
            questions[question_id] = {**row, "track": track}
            ids.append(question_id)
        ordered[track] = ids
    return questions, ordered


def _source_files(
    bundle_root: Path,
    expected_models: set[str],
) -> dict[str, Path]:
    manifest_path = bundle_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosedSourcePreparationError(
            f"Cannot read source manifest {manifest_path}: {exc}."
        ) from exc
    if not isinstance(manifest, list):
        raise ClosedSourcePreparationError("Source manifest must contain a list.")
    files: dict[str, Path] = {}
    seen_models: set[str] = set()
    for record in manifest:
        if not isinstance(record, dict):
            raise ClosedSourcePreparationError("Source manifest contains a non-object row.")
        model = str(record.get("model") or "").strip()
        if model in seen_models or model not in CLOSED_SOURCE_MODEL_CATALOG:
            raise ClosedSourcePreparationError(f"Unexpected or duplicate model {model!r}.")
        seen_models.add(model)
        if model not in expected_models:
            continue
        path = bundle_root / str(record.get("file") or "")
        if not path.is_file():
            raise ClosedSourcePreparationError(f"Missing retained output file: {path}.")
        files[model] = path
    if set(files) != expected_models:
        missing = sorted(expected_models - set(files))
        raise ClosedSourcePreparationError(
            f"Source manifest does not cover the trusted catalog: missing={missing}."
        )
    return files


def _validated_rows(
    source: Path,
    questions: dict[str, dict[str, Any]],
    ordered: dict[str, list[str]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], Counter[str]]:
    by_track = {track: {} for track in TRACKS}
    blank_counts: Counter[str] = Counter()
    required_fields = {"question_id", "subset", "task", "answer_type", "output"}
    for line_number, row in enumerate(read_jsonl(source), start=1):
        if set(row) != required_fields:
            raise ClosedSourcePreparationError(
                f"{source} line {line_number} has fields {sorted(row)}; expected "
                f"{sorted(required_fields)}."
            )
        question_id = str(row["question_id"]).strip()
        question = questions.get(question_id)
        if question is None:
            raise ClosedSourcePreparationError(
                f"{source} contains unknown question_id {question_id!r}."
            )
        track = SUBSET_TO_TRACK.get(str(row["subset"]).strip())
        if track != question["track"]:
            raise ClosedSourcePreparationError(
                f"{source} has a subset/track mismatch for {question_id}."
            )
        if str(row["task"]).strip() != str(question.get("task") or "").strip():
            raise ClosedSourcePreparationError(
                f"{source} has a task mismatch for {question_id}."
            )
        if str(row["answer_type"]).strip() != str(
            question.get("answer_type") or ""
        ).strip():
            raise ClosedSourcePreparationError(
                f"{source} has an answer_type mismatch for {question_id}."
            )
        if question_id in by_track[track]:
            raise ClosedSourcePreparationError(
                f"{source} repeats question_id {question_id!r}."
            )
        output = row["output"]
        if isinstance(output, (dict, list)):
            raise ClosedSourcePreparationError(
                f"{source} has a structured output for {question_id}."
            )
        normalized_output = "" if output is None else str(output)
        if not normalized_output.strip():
            blank_counts[track] += 1
        by_track[track][question_id] = {
            **row,
            "output": normalized_output,
            "source_line_number": line_number,
        }
    for track in TRACKS:
        actual = set(by_track[track])
        expected = set(ordered[track])
        if actual != expected:
            raise ClosedSourcePreparationError(
                f"{source}/{track} coverage mismatch: "
                f"missing={len(expected - actual)}, unknown={len(actual - expected)}."
            )
    return by_track, blank_counts


def prepare_closed_source_results(
    project_root: Path,
    bundle_root: Path,
    output_root: Path,
    *,
    force: bool = False,
    source_models: set[str] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    bundle_root = bundle_root.resolve()
    output_root = output_root.resolve()
    questions, ordered = question_contract(project_root)
    expected_models = (
        set(CLOSED_SOURCE_MODEL_CATALOG)
        if source_models is None
        else {str(model).strip() for model in source_models if str(model).strip()}
    )
    if not expected_models:
        raise ClosedSourcePreparationError("At least one trusted model is required.")
    unknown_models = expected_models - set(CLOSED_SOURCE_MODEL_CATALOG)
    if unknown_models:
        raise ClosedSourcePreparationError(
            f"Unknown trusted models requested: {sorted(unknown_models)}."
        )
    source_files = _source_files(bundle_root, expected_models)
    if output_root.exists() and not force:
        raise ClosedSourcePreparationError(
            f"Output root already exists: {output_root}. Use --force after review."
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        variants = []
        total_rows = 0
        total_blank = 0
        for source_model, source in sorted(source_files.items()):
            catalog = CLOSED_SOURCE_MODEL_CATALOG[source_model]
            slug = str(catalog["slug"])
            model_dir = staging / slug
            model_dir.mkdir()
            source_hash = sha256(source)
            retained_source = model_dir / "retained_provider_outputs.jsonl"
            shutil.copy2(source, retained_source)
            rows_by_track, blank_counts = _validated_rows(
                source, questions, ordered
            )
            track_records = {}
            for track in TRACKS:
                diagnostics_path = model_dir / f"{track}.diagnostics.jsonl"
                submission_path = model_dir / f"{track}_submission.jsonl"
                run_config_path = model_dir / f"{track}.run_config.json"
                source_manifest_path = model_dir / f"{track}.source_manifest.json"
                diagnostics = []
                submissions = []
                for question_id in ordered[track]:
                    source_row = rows_by_track[track][question_id]
                    diagnostics.append(
                        {
                            **source_row,
                            "source_file": source.name,
                            "source_file_sha256": source_hash,
                            "provider_output_retained_verbatim": True,
                        }
                    )
                    submissions.append(
                        {
                            "question_id": question_id,
                            "condition": "standard",
                            "answer": PENDING_ANSWER,
                        }
                    )
                write_jsonl(diagnostics_path, diagnostics)
                write_jsonl(submission_path, submissions)
                generation = {
                    "prompt_mode": "noncot",
                    "temperature": catalog["temperature"],
                    "source": "retained_provider_api_output",
                }
                for field in (
                    "source_declared_temperature",
                    "temperature_behavior",
                    "thinking_level",
                ):
                    if field in catalog:
                        generation[field] = catalog[field]
                run_config = {
                    "schema_version": 1,
                    "pipeline_revision": SOURCE_PIPELINE_REVISION,
                    "final_pipeline_revision": FINAL_PIPELINE_REVISION,
                    "model_id": catalog["model_id"],
                    "model_revision": catalog["model_revision"],
                    "organization": catalog["organization"],
                    "access": "closed",
                    "weight_loading": "provider_managed",
                    "compute_dtype": "provider_managed",
                    "reasoning_profile": catalog["reasoning_profile"],
                    "serving_engine": "provider_api",
                    "generation": {track: generation},
                }
                for field in (
                    "evaluation_date",
                    "provider_version_pinning",
                ):
                    if field in catalog:
                        run_config[field] = catalog[field]
                write_json(run_config_path, run_config)
                source_manifest = {
                    "schema_version": 1,
                    "model_id": catalog["model_id"],
                    "model_revision": catalog["model_revision"],
                    "track": track,
                    "row_count": len(diagnostics),
                    "blank_output_count": blank_counts[track],
                    "retained_source": artifact(retained_source),
                    "diagnostics": artifact(diagnostics_path),
                    "submission": artifact(submission_path),
                    "run_config": artifact(run_config_path),
                }
                write_json(source_manifest_path, source_manifest)
                track_records[track] = {
                    "relative_dir": slug,
                    "diagnostics": diagnostics_path.name,
                    "diagnostics_sha256": sha256(diagnostics_path),
                    "submission": submission_path.name,
                    "submission_sha256": sha256(submission_path),
                    "source_run_config_sha256": sha256(run_config_path),
                    "row_count": len(diagnostics),
                    "blank_output_count": blank_counts[track],
                }
                total_rows += len(diagnostics)
                total_blank += blank_counts[track]
            variants.append(
                {
                    "variant_id": slug,
                    "source_model": source_model,
                    "model_id": catalog["model_id"],
                    "model_revision": catalog["model_revision"],
                    "display_name": catalog["display_name"],
                    "organization": catalog["organization"],
                    "access": "closed",
                    "weight_loading": "provider_managed",
                    "compute_dtype": "provider_managed",
                    "reasoning_profile": catalog["reasoning_profile"],
                    "selection_precision": "provider-api-retained-output",
                    "final_pipeline_revision": FINAL_PIPELINE_REVISION,
                    "source_file": source.name,
                    "source_file_sha256": source_hash,
                    "tracks": track_records,
                }
            )
        index = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_pipeline_revision": SOURCE_PIPELINE_REVISION,
            "final_pipeline_revision": FINAL_PIPELINE_REVISION,
            "model_count": len(variants),
            "response_count": total_rows,
            "blank_output_count": total_blank,
            "variants": variants,
        }
        write_json(staging / "index.json", index)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(staging, output_root)
        return index
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        dest="source_models",
        action="append",
        help="Trusted source model to prepare. Repeat to prepare multiple models.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare_closed_source_results(
            args.project_root,
            args.bundle_root,
            args.output_root,
            force=args.force,
            source_models=(set(args.source_models) if args.source_models else None),
        )
    except ClosedSourcePreparationError as exc:
        raise SystemExit(f"Closed-source preparation failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

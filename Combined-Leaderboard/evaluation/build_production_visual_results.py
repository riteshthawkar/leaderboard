"""Build a production canonical visual result tree from a completed evidence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.extract_canonical_answers import (
    DEFAULT_EXCLUDED_VARIANTS,
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
    TERMINAL_FALLBACK_METHOD,
    candidate_key,
    classify_extractor_output,
    finalize_persistent_extractor_failure,
    load_candidates,
)
from evaluation.finalize_visual_results import (
    CURRENT_PIPELINE_REVISION,
    TRACKS,
    answer_provenance_counts,
    artifact_record,
    expected_question_ids,
    read_json,
    read_jsonl,
    sha256,
    validate_diagnostics,
    validate_submission,
    verify_canonical_results,
)
from visual_answer_contract import INVALID_FORMAT_TOKEN, UNRESOLVED_TOKEN


FINAL_STATUSES = {
    "committed",
    "invalid_format_committed",
    "unresolved",
    "unsupported_by_evidence",
    "unresolved_truncated_response",
}
UNRESOLVED_STATUSES = {
    "unresolved",
    "unsupported_by_evidence",
    "unresolved_truncated_response",
}
EXTRACTION_IDENTITY_FIELDS = {
    "answer_extraction_method",
    "extracted_answer",
    "ground_truth_available_to_validator",
    "ground_truth_loaded",
    "ground_truth_loaded_by_extractor_process",
    "ground_truth_supplied",
    "ground_truth_supplied_to_extractor",
    "independent_extraction_status",
}


class ProductionBuildError(RuntimeError):
    pass


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def selected_variants(
    source_root: Path,
    excluded_variants: set[str],
) -> list[dict[str, Any]]:
    index = read_json(source_root / "index.json")
    variants = index.get("variants")
    if not isinstance(variants, list):
        raise ProductionBuildError("Source index has no v11 variants list.")
    selected = [
        variant
        for variant in variants
        if str(variant.get("variant_id") or "") not in excluded_variants
    ]
    slugs = [str(variant.get("variant_id") or "") for variant in selected]
    if not slugs or any(not slug for slug in slugs) or len(set(slugs)) != len(slugs):
        raise ProductionBuildError("Selected source variants have invalid identities.")
    for variant in selected:
        if set(variant.get("tracks", {})) != set(TRACKS):
            raise ProductionBuildError(
                f"Variant {variant.get('variant_id')} lacks a complete track pair."
            )
    return selected


def validate_source_artifact(path: Path, expected_sha256: Any) -> None:
    if not path.is_file():
        raise ProductionBuildError(f"Missing source artifact: {path}.")
    if sha256(path) != str(expected_sha256 or ""):
        raise ProductionBuildError(f"Source artifact hash mismatch: {path}.")


def load_completed_audit(
    audit_path: Path,
    candidates: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], str]:
    candidates_by_key = {candidate_key(item): item for item in candidates}
    if len(candidates_by_key) != len(candidates):
        raise ProductionBuildError("Expected evidence candidates are not unique.")
    audit: dict[tuple[str, str, str], dict[str, Any]] = {}
    contracts = set()
    failures = []
    for row in read_jsonl(audit_path):
        key = candidate_key(row)
        if key in audit:
            raise ProductionBuildError(f"Evidence audit repeats {key}.")
        candidate = candidates_by_key.get(key)
        if candidate is None:
            raise ProductionBuildError(f"Evidence audit contains unexpected row {key}.")
        if row.get("method") != METHOD:
            raise ProductionBuildError(f"Evidence audit method mismatch for {key}.")
        if row.get("extractor_model") != DEFAULT_EXTRACTOR_MODEL:
            raise ProductionBuildError(f"Evidence extractor model mismatch for {key}.")
        if row.get("extractor_revision") != DEFAULT_EXTRACTOR_REVISION:
            raise ProductionBuildError(f"Evidence extractor revision mismatch for {key}.")
        if row.get("ground_truth_loaded") is not False:
            raise ProductionBuildError(f"Extractor loaded ground truth for {key}.")
        if row.get("ground_truth_supplied_to_extractor") is not False:
            raise ProductionBuildError(f"Extractor received ground truth for {key}.")
        if row.get("response_sha256") != candidate["response_sha256"]:
            raise ProductionBuildError(f"Candidate response hash changed for {key}.")
        contract = str(row.get("extractor_contract_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", contract):
            raise ProductionBuildError(f"Evidence contract hash is missing for {key}.")
        contracts.add(contract)
        status = str(row.get("status") or "")
        if status not in FINAL_STATUSES:
            failures.append((key, status))
        recomputed = classify_extractor_output(
            candidate,
            str(row.get("extractor_output") or ""),
            str(row.get("error") or "") or None,
        )
        fallback_method = row.get("terminal_fallback_method")
        if fallback_method:
            if fallback_method != TERMINAL_FALLBACK_METHOD:
                raise ProductionBuildError(
                    f"Terminal fallback method mismatch for {key}."
                )
            if row.get("terminal_fallback_from_status") != recomputed.get("status"):
                raise ProductionBuildError(
                    f"Terminal fallback source status mismatch for {key}."
                )
            recomputed = finalize_persistent_extractor_failure(
                candidate,
                {
                    **row,
                    **recomputed,
                },
            )
            if recomputed is None:
                raise ProductionBuildError(
                    f"Terminal fallback is not reproducible for {key}."
                )
        elif row.get("terminal_fallback_from_status"):
            raise ProductionBuildError(
                f"Terminal fallback method is missing for {key}."
            )
        for field in ("status", "answer", "evidence", "extractor_verdict"):
            if recomputed.get(field) != row.get(field):
                raise ProductionBuildError(
                    f"Evidence classification changed for {key}: {field}."
                )
        audit[key] = row
    missing = set(candidates_by_key) - set(audit)
    if missing:
        raise ProductionBuildError(
            f"Evidence audit is incomplete: {len(missing)} rows are missing."
        )
    if failures:
        preview = ", ".join(f"{key}:{status}" for key, status in failures[:5])
        raise ProductionBuildError(
            f"Evidence audit has {len(failures)} blocking extractor failures: {preview}."
        )
    if len(contracts) != 1:
        raise ProductionBuildError("Evidence audit uses more than one extractor contract.")
    return audit, contracts.pop()


def replace_extraction(
    diagnostic: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    status = str(audit["status"])
    if status == "committed":
        answer = str(audit["answer"])
    elif status == "invalid_format_committed":
        answer = INVALID_FORMAT_TOKEN
    elif status in UNRESOLVED_STATUSES:
        answer = UNRESOLVED_TOKEN
    else:
        raise ProductionBuildError(
            f"Cannot apply extraction status {status!r} for {diagnostic.get('question_id')}."
        )

    prior = {
        key: value
        for key, value in diagnostic.items()
        if key.startswith("extractor_") or key in EXTRACTION_IDENTITY_FIELDS
    }
    prior.pop("extractor_attempts", None)
    attempts = list(diagnostic.get("extractor_attempts") or [])
    if prior:
        attempts.append(prior)
    for key in list(diagnostic):
        if key.startswith("extractor_") or key in EXTRACTION_IDENTITY_FIELDS:
            diagnostic.pop(key, None)

    diagnostic.update(
        {
            "answer_extraction_method": METHOD,
            "extractor_model": DEFAULT_EXTRACTOR_MODEL,
            "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
            "extractor_contract_sha256": str(audit["extractor_contract_sha256"]),
            "extractor_output": str(audit.get("extractor_output") or ""),
            "extractor_evidence": str(audit.get("evidence") or ""),
            "extractor_verdict": str(audit.get("extractor_verdict") or ""),
            "extractor_status": status,
            "extractor_finish_reason": audit.get("finish_reason"),
            "extractor_completion_tokens": audit.get("completion_tokens"),
            "extractor_source_output_sha256": str(audit["response_sha256"]),
            "extractor_ground_truth_loaded": False,
            "extractor_ground_truth_supplied": False,
            "extracted_answer": answer,
        }
    )
    if audit.get("proposed_answer"):
        diagnostic["extractor_proposed_answer"] = str(audit["proposed_answer"])
    if attempts:
        diagnostic["extractor_attempts"] = attempts
    return answer


def copy_optional_source_files(
    source_dir: Path,
    destination: Path,
    track: str,
) -> list[Path]:
    copied = []
    for pattern in (
        f"{track}.attempt-*.diagnostics.jsonl",
        f"{track}.smoke*.diagnostics.jsonl",
        f"{track}.inference.diagnostics.jsonl",
        f"{track}.inference.smoke*.diagnostics.jsonl",
    ):
        for source in sorted(source_dir.glob(pattern)):
            target = destination / source.name
            shutil.copy2(source, target)
            copied.append(target)
    source_manifest = source_dir / f"{track}.source_manifest.json"
    if source_manifest.is_file():
        target = destination / source_manifest.name
        shutil.copy2(source_manifest, target)
        copied.append(target)
    return copied


def build_track(
    *,
    source_root: Path,
    destination: Path,
    variant: dict[str, Any],
    track: str,
    audit: dict[tuple[str, str, str], dict[str, Any]],
    expected_ids: list[str],
    audit_sha256: str,
    contract_sha256: str,
) -> dict[str, Any]:
    slug = str(variant["variant_id"])
    source_record = variant["tracks"][track]
    source_dir = source_root / str(source_record["relative_dir"])
    source_diagnostics = source_dir / str(source_record["diagnostics"])
    source_submission = source_dir / str(source_record["submission"])
    source_run_config = source_dir / f"{track}.run_config.json"
    validate_source_artifact(source_diagnostics, source_record["diagnostics_sha256"])
    validate_source_artifact(source_submission, source_record["submission_sha256"])
    validate_source_artifact(source_run_config, source_record["source_run_config_sha256"])

    diagnostics = read_jsonl(source_diagnostics)
    submissions = read_jsonl(source_submission)
    diagnostics_by_id = {str(row["question_id"]): row for row in diagnostics}
    submissions_by_id = {str(row["question_id"]): row for row in submissions}
    if set(diagnostics_by_id) != set(expected_ids) or set(submissions_by_id) != set(
        expected_ids
    ):
        raise ProductionBuildError(f"Source coverage mismatch for {slug}/{track}.")

    track_audit = []
    for question_id in expected_ids:
        result = audit[(slug, track, question_id)]
        answer = replace_extraction(diagnostics_by_id[question_id], result)
        submissions_by_id[question_id]["answer"] = answer
        track_audit.append(result)

    diagnostics_path = destination / f"{track}.diagnostics.jsonl"
    submission_path = destination / f"{track}_submission.jsonl"
    audit_path = destination / f"{track}.evidence_extraction.jsonl"
    write_jsonl(diagnostics_path, diagnostics)
    write_jsonl(submission_path, submissions)
    write_jsonl(audit_path, track_audit)

    source_config_copy = destination / f"{track}.v11_run_config.json"
    shutil.copy2(source_run_config, source_config_copy)
    run_config = read_json(source_run_config)
    run_config["source_pipeline_revision"] = run_config.get("pipeline_revision")
    run_config["pipeline_revision"] = CURRENT_PIPELINE_REVISION
    run_config["schema_version"] = 12
    run_config["answer_extraction"] = {
        "method": METHOD,
        "model": DEFAULT_EXTRACTOR_MODEL,
        "revision": DEFAULT_EXTRACTOR_REVISION,
        "extractor_contract_sha256": contract_sha256,
        "audit_sha256": audit_sha256,
        "input_fields": [
            "question",
            "answer_type",
            "task",
            "response_metadata",
            "candidate_response",
        ],
        "image_supplied": False,
        "ground_truth_loaded": False,
        "ground_truth_supplied": False,
        "evidence_requirement": "literal-source-quote-and-commitment-validation",
    }
    run_config_path = destination / f"{track}.run_config.json"
    write_json(run_config_path, run_config)
    optional_artifacts = copy_optional_source_files(
        source_dir, destination, track
    )

    submission_rows = validate_submission(submission_path, expected_ids)
    _diagnostics, validated_by_id = validate_diagnostics(
        diagnostics_path, expected_ids
    )
    (
        strict_count,
        unresolved_count,
        invalid_commitment_count,
        exact_raw_count,
    ) = answer_provenance_counts(submission_rows, validated_by_id)
    if strict_count + unresolved_count + invalid_commitment_count != len(expected_ids):
        raise ProductionBuildError(f"Provenance count mismatch for {slug}/{track}.")
    if exact_raw_count:
        raise ProductionBuildError(f"Raw-output fallback survived for {slug}/{track}.")

    artifacts = [
        diagnostics_path,
        submission_path,
        audit_path,
        source_config_copy,
        run_config_path,
        *optional_artifacts,
    ]
    status_counts = Counter(str(item["status"]) for item in track_audit)
    return {
        "row_count": len(expected_ids),
        "strict_answer_count": strict_count,
        "unresolved_answer_count": unresolved_count,
        "invalid_commitment_count": invalid_commitment_count,
        "invalid_format_count": unresolved_count + invalid_commitment_count,
        "exact_raw_output_fallback_count": exact_raw_count,
        "source_run": str(source_record["relative_dir"]),
        "source_submission_modified_at": datetime.fromtimestamp(
            source_submission.stat().st_mtime, timezone.utc
        ).isoformat(),
        "source_run_config": source_config_copy.name,
        "source_manifest": (
            f"{track}.source_manifest.json"
            if (destination / f"{track}.source_manifest.json").is_file()
            else None
        ),
        "generation": run_config.get("generation", {}).get(track),
        "serving_engine": run_config.get("serving_engine"),
        "tensor_parallel_size": run_config.get("tensor_parallel_size"),
        "data_parallel_size": run_config.get("data_parallel_size"),
        "request_concurrency": run_config.get("request_concurrency"),
        "max_model_len": run_config.get("max_model_len"),
        "evidence_extraction": {
            "method": METHOD,
            "extractor_model": DEFAULT_EXTRACTOR_MODEL,
            "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
            "extractor_contract_sha256": contract_sha256,
            "audit_sha256": audit_sha256,
            "status_counts": dict(status_counts),
            "image_supplied": False,
            "ground_truth_loaded": False,
            "ground_truth_supplied": False,
        },
        "artifacts": {path.name: artifact_record(path) for path in artifacts},
    }


def build_production_results(
    project_root: Path,
    source_root: Path,
    audit_path: Path,
    output_root: Path,
    excluded_variants: set[str],
) -> dict[str, Any]:
    project_root = project_root.resolve()
    source_root = source_root.resolve()
    audit_path = audit_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise ProductionBuildError(f"Output root already exists: {output_root}.")
    variants = selected_variants(source_root, excluded_variants)
    candidates = load_candidates(
        project_root,
        source_root,
        "all",
        excluded_variants=excluded_variants,
    )
    audit, contract_sha256 = load_completed_audit(audit_path, candidates)
    audit_digest = sha256(audit_path)
    expected = {
        track: expected_question_ids(project_root, track) for track in TRACKS
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        index_models = []
        for variant in variants:
            slug = str(variant["variant_id"])
            destination = staging / slug
            destination.mkdir()
            track_records = {
                track: build_track(
                    source_root=source_root,
                    destination=destination,
                    variant=variant,
                    track=track,
                    audit=audit,
                    expected_ids=expected[track],
                    audit_sha256=audit_digest,
                    contract_sha256=contract_sha256,
                )
                for track in TRACKS
            }
            manifest = {
                "schema_version": 2,
                "finalized_at": datetime.now(timezone.utc).isoformat(),
                "selection_policy": {
                    "precision": "original-unquantized-bf16",
                    "pipeline_revision": CURRENT_PIPELINE_REVISION,
                    "required_tracks": list(TRACKS),
                    "excluded_variants": sorted(excluded_variants),
                },
                "model_id": str(variant["model_id"]),
                "model_revision": str(variant["model_revision"]),
                "weight_loading": "unquantized",
                "compute_dtype": "bfloat16",
                "evidence_extraction": {
                    "method": METHOD,
                    "extractor_model": DEFAULT_EXTRACTOR_MODEL,
                    "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
                    "extractor_contract_sha256": contract_sha256,
                    "source_audit_sha256": audit_digest,
                    "image_supplied": False,
                    "ground_truth_loaded": False,
                    "ground_truth_supplied": False,
                },
                "tracks": track_records,
            }
            manifest_path = destination / "final_manifest.json"
            write_json(manifest_path, manifest)
            index_models.append(
                {
                    "slug": slug,
                    "model_id": manifest["model_id"],
                    "model_revision": manifest["model_revision"],
                    "manifest": f"{slug}/final_manifest.json",
                    "manifest_sha256": sha256(manifest_path),
                    "tracks": {
                        track: {
                            key: track_records[track][key]
                            for key in (
                                "row_count",
                                "strict_answer_count",
                                "unresolved_answer_count",
                                "invalid_commitment_count",
                                "invalid_format_count",
                                "exact_raw_output_fallback_count",
                            )
                        }
                        for track in TRACKS
                    },
                }
            )

        index = {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_revision": CURRENT_PIPELINE_REVISION,
            "model_count": len(index_models),
            "excluded_variants": sorted(excluded_variants),
            "evidence_extraction": {
                "method": METHOD,
                "extractor_model": DEFAULT_EXTRACTOR_MODEL,
                "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
                "extractor_contract_sha256": contract_sha256,
                "source_audit_sha256": audit_digest,
                "response_count": len(candidates),
                "image_supplied": False,
                "ground_truth_loaded": False,
                "ground_truth_supplied": False,
            },
            "models": sorted(index_models, key=lambda item: item["model_id"]),
        }
        write_json(staging / "index.json", index)
        verification = verify_canonical_results(staging, project_root)
        if verification["model_count"] != len(variants):
            raise ProductionBuildError("Final verification model count changed.")
        os.replace(staging, output_root)
        return index
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "evaluation/results/final-extracted-v11",
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "evaluation/results/final-extracted-v12",
    )
    parser.add_argument(
        "--exclude-variant",
        action="append",
        dest="exclude_variants",
        default=list(DEFAULT_EXCLUDED_VARIANTS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = build_production_results(
            args.project_root,
            args.source_root,
            args.audit,
            args.output_root,
            set(args.exclude_variants),
        )
    except ProductionBuildError as exc:
        raise SystemExit(f"Production build failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
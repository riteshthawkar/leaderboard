from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from evaluation.common.visual_pipeline import (
    MISSING_ANSWER_TOKEN,
    extracted_record_answer,
)
from evaluation.common.vllm_runner import (
    ANSWER_EXTRACTION_METHOD,
    EXTRACTOR_SYSTEM_PROMPT,
)


TRACK_MODULES = {
    "do_you_see_me": "evaluation.do_you_see_me.run_vllm",
    "minds_eye": "evaluation.minds_eye.run_vllm",
}
DEFAULT_EXTRACTOR_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EXTRACTOR_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


class BatchExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractionJob:
    job_id: str
    variant_id: str
    track: str
    relative_dir: Path
    diagnostics: Path
    submission: Path
    run_config: Path
    questions: Path
    model_id: str
    model_revision: str
    mode: str
    row_count: int
    source_diagnostics_sha256: str
    source_submission_sha256: str
    source_run_config_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_id": self.variant_id,
            "track": self.track,
            "relative_dir": self.relative_dir.as_posix(),
            "diagnostics": self.diagnostics.name,
            "submission": self.submission.name,
            "run_config": self.run_config.name,
            "questions": self.questions.as_posix(),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "mode": self.mode,
            "row_count": self.row_count,
            "source_diagnostics_sha256": self.source_diagnostics_sha256,
            "source_submission_sha256": self.source_submission_sha256,
            "source_run_config_sha256": self.source_run_config_sha256,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise BatchExtractionError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchExtractionError(f"Expected a JSON object in {path}.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise BatchExtractionError(f"Cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BatchExtractionError(
                f"{path} line {line_number} is invalid JSON: {exc.msg}."
            ) from exc
        if not isinstance(row, dict):
            raise BatchExtractionError(f"{path} line {line_number} must be an object.")
        rows.append(row)
    return rows


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def question_ids(path: Path) -> list[str]:
    rows = read_jsonl(path)
    identifiers = [str(row.get("question_id") or "") for row in rows]
    if not identifiers or any(not value for value in identifiers):
        raise BatchExtractionError(f"Question bundle {path} has missing IDs.")
    if len(set(identifiers)) != len(identifiers):
        raise BatchExtractionError(f"Question bundle {path} has duplicate IDs.")
    return identifiers


def validate_ids(path: Path, expected: list[str]) -> int:
    rows = read_jsonl(path)
    actual = [str(row.get("question_id") or "") for row in rows]
    if actual != expected:
        raise BatchExtractionError(
            f"{path} coverage/order mismatch: expected {len(expected)}, got {len(actual)}."
        )
    return len(rows)


def mode_from_config(config: dict[str, Any], relative_dir: Path, track: str) -> str:
    first = relative_dir.parts[0]
    if first == "qwen-35-thinking-disabled":
        return "thinking-disabled"
    if first == "qwen35-thinking-enabled":
        return "thinking-enabled"
    reasoning_profile = str(config.get("reasoning_profile") or "").strip()
    if reasoning_profile:
        return reasoning_profile
    chat_kwargs = config.get("chat_template_kwargs")
    if isinstance(chat_kwargs, dict) and "enable_thinking" in chat_kwargs:
        return "thinking-enabled" if chat_kwargs["enable_thinking"] else "thinking-disabled"
    generation = config.get("generation", {}).get(track, {})
    if isinstance(generation, dict):
        kwargs = generation.get("chat_template_kwargs")
        if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
            return "thinking-enabled" if kwargs["enable_thinking"] else "thinking-disabled"
    return "unspecified"


def discover_jobs(source_root: Path, project_root: Path) -> list[ExtractionJob]:
    expected_by_track = {
        track: question_ids(project_root / "tasks" / track / "questions.jsonl")
        for track in TRACK_MODULES
    }
    jobs: list[ExtractionJob] = []
    seen: set[tuple[str, str]] = set()
    for diagnostics in sorted(source_root.rglob("*.diagnostics.jsonl")):
        track = diagnostics.name.removesuffix(".diagnostics.jsonl")
        if track not in TRACK_MODULES:
            continue
        relative_dir = diagnostics.parent.relative_to(source_root)
        variant_id = relative_dir.parts[0]
        key = (variant_id, track)
        if key in seen:
            raise BatchExtractionError(
                f"Variant {variant_id} has more than one {track} diagnostics file."
            )
        seen.add(key)
        submission = diagnostics.with_name(f"{track}_submission.jsonl")
        if not submission.is_file():
            raise BatchExtractionError(f"Missing submission beside {diagnostics}.")
        run_config = diagnostics.parent / ".run_config.json"
        if not run_config.is_file():
            run_config = diagnostics.parent / f"{track}.run_config.json"
        if not run_config.is_file():
            raise BatchExtractionError(f"Missing run configuration beside {diagnostics}.")
        config = read_json(run_config)
        model_id = str(config.get("model_id") or "").strip()
        model_revision = str(config.get("model_revision") or "").strip()
        if not model_id or not model_revision:
            raise BatchExtractionError(f"Missing model identity in {run_config}.")
        expected = expected_by_track[track]
        row_count = validate_ids(diagnostics, expected)
        validate_ids(submission, expected)
        jobs.append(
            ExtractionJob(
                job_id=f"{variant_id}:{track}",
                variant_id=variant_id,
                track=track,
                relative_dir=relative_dir,
                diagnostics=diagnostics,
                submission=submission,
                run_config=run_config,
                questions=project_root / "tasks" / track / "questions.jsonl",
                model_id=model_id,
                model_revision=model_revision,
                mode=mode_from_config(config, relative_dir, track),
                row_count=row_count,
                source_diagnostics_sha256=sha256(diagnostics),
                source_submission_sha256=sha256(submission),
                source_run_config_sha256=sha256(run_config),
            )
        )
    variants: dict[str, set[str]] = {}
    for job in jobs:
        variants.setdefault(job.variant_id, set()).add(job.track)
    incomplete = {
        variant: sorted(set(TRACK_MODULES) - tracks)
        for variant, tracks in variants.items()
        if tracks != set(TRACK_MODULES)
    }
    if incomplete:
        raise BatchExtractionError(f"Incomplete model variants: {incomplete}.")
    if not jobs:
        raise BatchExtractionError(f"No complete diagnostics found under {source_root}.")
    return jobs


def contract_record(args: argparse.Namespace) -> dict[str, Any]:
    contract = {
        "method": ANSWER_EXTRACTION_METHOD,
        "model": args.model,
        "revision": args.revision,
        "system_prompt": EXTRACTOR_SYSTEM_PROMPT,
        "input_fields": [
            "question",
            "answer_type",
            "answer_domain",
            "required_output_format",
            "response_metadata",
            "candidate_response",
        ],
        "image_supplied": False,
        "ground_truth_supplied": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "stop_sequences": ["</answer>"],
        "chat_template_kwargs": {"enable_thinking": False},
        "unresolved_token": MISSING_ANSWER_TOKEN,
        "support_validation": "answer-must-be-stated-in-candidate-response",
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def inventory_fingerprint(jobs: list[ExtractionJob], contract: dict[str, Any]) -> str:
    payload = {
        "jobs": [job.record() for job in jobs],
        "extractor_contract_sha256": contract["contract_sha256"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preserve_source_manifests(output_root: Path) -> None:
    index = output_root / "index.json"
    if index.is_file() and not (output_root / "source_index.json").exists():
        index.rename(output_root / "source_index.json")
    for name, preserved in (
        ("final_manifest.json", "source_final_manifest.json"),
        ("run_manifest.json", "source_run_manifest.json"),
    ):
        for path in sorted(output_root.rglob(name)):
            target = path.with_name(preserved)
            if target.exists():
                raise BatchExtractionError(f"Manifest preservation target exists: {target}.")
            path.rename(target)


def initial_manifest(
    source_root: Path,
    output_root: Path,
    jobs: list[ExtractionJob],
    contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "prepared",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_inventory_sha256": inventory_fingerprint(jobs, contract),
        "extractor": contract,
        "variant_count": len({job.variant_id for job in jobs}),
        "track_job_count": len(jobs),
        "total_response_count": sum(job.row_count for job in jobs),
        "jobs": [{**job.record(), "status": "pending"} for job in jobs],
    }


def prepare_output(
    source_root: Path,
    output_root: Path,
    jobs: list[ExtractionJob],
    contract: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = output_root / "batch_manifest.json"
    expected_fingerprint = inventory_fingerprint(jobs, contract)
    if output_root.exists():
        if not manifest_path.is_file():
            raise BatchExtractionError(
                f"Output root exists without batch_manifest.json: {output_root}."
            )
        manifest = read_json(manifest_path)
        if manifest.get("source_inventory_sha256") != expected_fingerprint:
            raise BatchExtractionError(
                "Source artifacts or extractor contract changed; refusing an unsafe resume."
            )
        return manifest

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.with_name(f".{output_root.name}.copying-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source_root, temporary, copy_function=shutil.copy2)
        preserve_source_manifests(temporary)
        manifest = initial_manifest(source_root, output_root, jobs, contract)
        atomic_write_json(temporary / "batch_manifest.json", manifest)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_json(manifest_path)


def wait_for_endpoint(endpoint: str, model: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    models_url = endpoint.rstrip("/") + "/models"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(models_url, timeout=10) as response:
                payload = json.load(response)
            model_ids = {
                str(item.get("id") or "")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            }
            if model in model_ids:
                return
            last_error = f"served models: {sorted(model_ids)}"
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(5)
    raise BatchExtractionError(
        f"Extractor endpoint {endpoint} did not serve {model} within {timeout}s ({last_error})."
    )


def update_manifest_job(
    manifest_path: Path,
    job_id: str,
    **updates: Any,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    for record in manifest["jobs"]:
        if record["job_id"] == job_id:
            record.update(updates)
            break
    else:
        raise BatchExtractionError(f"Unknown manifest job: {job_id}.")
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_extracted_job(
    output_root: Path,
    job: ExtractionJob,
    model: str,
    revision: str,
) -> dict[str, Any]:
    directory = output_root / job.relative_dir
    diagnostics = directory / job.diagnostics.name
    submission = directory / job.submission.name
    diagnostic_rows = read_jsonl(diagnostics)
    submission_rows = read_jsonl(submission)
    if len(diagnostic_rows) != job.row_count or len(submission_rows) != job.row_count:
        raise BatchExtractionError(f"Extracted row count mismatch for {job.job_id}.")
    resolved = 0
    unresolved = 0
    extractor_failed = 0
    for diagnostic, submitted in zip(diagnostic_rows, submission_rows, strict=True):
        question_id = str(diagnostic.get("question_id") or "")
        if question_id != str(submitted.get("question_id") or ""):
            raise BatchExtractionError(f"ID/order mismatch for {job.job_id}/{question_id}.")
        if diagnostic.get("answer_extraction_method") != ANSWER_EXTRACTION_METHOD:
            raise BatchExtractionError(f"Missing extractor method for {job.job_id}/{question_id}.")
        if diagnostic.get("extractor_model") != model:
            raise BatchExtractionError(f"Extractor model mismatch for {job.job_id}/{question_id}.")
        if diagnostic.get("extractor_revision") != revision:
            raise BatchExtractionError(f"Extractor revision mismatch for {job.job_id}/{question_id}.")
        expected_hash = hashlib.sha256(
            str(diagnostic.get("output") or "").encode("utf-8")
        ).hexdigest()
        if diagnostic.get("extractor_source_output_sha256") != expected_hash:
            raise BatchExtractionError(f"Response hash mismatch for {job.job_id}/{question_id}.")
        answer = extracted_record_answer(
            diagnostic, str(diagnostic.get("answer_type") or "text")
        )
        if not answer or submitted.get("answer") != answer:
            raise BatchExtractionError(f"Submission mismatch for {job.job_id}/{question_id}.")
        if answer == MISSING_ANSWER_TOKEN:
            unresolved += 1
            if diagnostic.get("extractor_status") == "failed":
                extractor_failed += 1
        else:
            resolved += 1
    return {
        "status": "completed",
        "completed_at": utc_now(),
        "resolved_answer_count": resolved,
        "unresolved_answer_count": unresolved,
        "extractor_failed_count": extractor_failed,
        "diagnostics_sha256": sha256(diagnostics),
        "submission_sha256": sha256(submission),
    }


def run_job(args: argparse.Namespace, output_root: Path, job: ExtractionJob) -> None:
    directory = output_root / job.relative_dir
    command = [
        sys.executable,
        "-m",
        TRACK_MODULES[job.track],
        "--model",
        args.model,
        "--endpoints",
        args.endpoint,
        "--api-key",
        args.api_key,
        "--extractor-model",
        args.model,
        "--extractor-revision",
        args.revision,
        "--extractor-endpoints",
        args.endpoint,
        "--extractor-api-key",
        args.api_key,
        "--questions",
        str(job.questions),
        "--resume",
        "--extract-existing-diagnostics",
        "--extractor-max-tokens",
        str(args.max_tokens),
        "--extractor-seed",
        str(args.seed),
        "--extractor-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--max-final-answer-tokens",
        str(args.final_answer_max_tokens),
        "--concurrency",
        str(args.concurrency),
        "--request-timeout",
        str(args.timeout),
        "--max-retries",
        str(args.retries),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--out",
        str(directory / job.submission.name),
        "--diagnostics",
        str(directory / job.diagnostics.name),
    ]
    print(f"[{utc_now()}] Starting {job.job_id} ({job.row_count} responses)", flush=True)
    result = subprocess.run(command, cwd=args.project_root, check=False)
    if result.returncode != 0:
        raise BatchExtractionError(
            f"Extractor command failed for {job.job_id} with exit code {result.returncode}."
        )


def write_variant_manifests(
    output_root: Path,
    jobs: list[ExtractionJob],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    by_job = {record["job_id"]: record for record in manifest["jobs"]}
    variants: dict[str, dict[str, Any]] = {}
    for job in jobs:
        record = by_job[job.job_id]
        variant = variants.setdefault(
            job.variant_id,
            {
                "variant_id": job.variant_id,
                "model_id": job.model_id,
                "model_revision": job.model_revision,
                "mode": job.mode,
                "tracks": {},
            },
        )
        if variant["model_id"] != job.model_id or variant["model_revision"] != job.model_revision:
            raise BatchExtractionError(f"Model identity differs within {job.variant_id}.")
        variant["tracks"][job.track] = {
            key: record[key]
            for key in (
                "relative_dir",
                "diagnostics",
                "submission",
                "row_count",
                "resolved_answer_count",
                "unresolved_answer_count",
                "extractor_failed_count",
                "diagnostics_sha256",
                "submission_sha256",
                "source_diagnostics_sha256",
                "source_submission_sha256",
                "source_run_config_sha256",
            )
        }
    for variant in variants.values():
        variant_root = output_root / variant["variant_id"]
        atomic_write_json(
            variant_root / "mandatory_extraction_manifest.json",
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "extractor": manifest["extractor"],
                **variant,
            },
        )
    index = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_root": manifest["source_root"],
        "extractor": manifest["extractor"],
        "variant_count": len(variants),
        "unique_model_count": len({value["model_id"] for value in variants.values()}),
        "track_job_count": len(jobs),
        "total_response_count": sum(job.row_count for job in jobs),
        "resolved_answer_count": sum(
            record["resolved_answer_count"] for record in by_job.values()
        ),
        "unresolved_answer_count": sum(
            record["unresolved_answer_count"] for record in by_job.values()
        ),
        "variants": [variants[key] for key in sorted(variants)],
    }
    atomic_write_json(output_root / "index.json", index)
    return index


def parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description="Apply mandatory gold-blind extraction to every completed visual result."
    )
    result.add_argument("--project-root", type=Path, default=project_root)
    result.add_argument("--source-root", type=Path, default=project_root / "evaluation/results/final")
    result.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "evaluation/results/final-extracted-v11",
    )
    result.add_argument("--endpoint", default="http://127.0.0.1:8035/v1")
    result.add_argument("--api-key", default="EMPTY")
    result.add_argument("--model", default=DEFAULT_EXTRACTOR_MODEL)
    result.add_argument("--revision", default=DEFAULT_EXTRACTOR_REVISION)
    result.add_argument("--concurrency", type=int, default=4)
    result.add_argument("--max-tokens", type=int, default=200)
    result.add_argument("--final-answer-max-tokens", type=int, default=200)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--timeout", type=float, default=600)
    result.add_argument("--retries", type=int, default=2)
    result.add_argument("--checkpoint-every", type=int, default=25)
    result.add_argument("--endpoint-start-timeout", type=float, default=1800)
    result.add_argument("--prepare-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.source_root == args.output_root or args.source_root in args.output_root.parents:
        raise BatchExtractionError("Output root must be outside the source root.")
    if args.concurrency < 1 or args.max_tokens < 1 or args.final_answer_max_tokens < 1:
        raise BatchExtractionError("Concurrency and token limits must be positive.")
    if args.timeout <= 0 or args.endpoint_start_timeout <= 0 or args.retries < 0:
        raise BatchExtractionError("Timeouts must be positive and retries non-negative.")
    if len(args.revision) != 40 or any(value not in "0123456789abcdef" for value in args.revision):
        raise BatchExtractionError("Extractor revision must be a 40-character lowercase commit hash.")

    jobs = discover_jobs(args.source_root, args.project_root)
    contract = contract_record(args)
    manifest = prepare_output(args.source_root, args.output_root, jobs, contract)
    print(
        json.dumps(
            {
                "variant_count": manifest["variant_count"],
                "track_job_count": manifest["track_job_count"],
                "total_response_count": manifest["total_response_count"],
                "output_root": str(args.output_root),
                "extractor_contract_sha256": contract["contract_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.prepare_only:
        return 0

    active_marker = args.output_root / ".active-run.json"
    atomic_write_json(
        active_marker,
        {"pid": os.getpid(), "started_at": utc_now(), "operation": "mandatory-extraction"},
    )
    manifest_path = args.output_root / "batch_manifest.json"
    try:
        manifest["status"] = "waiting-for-endpoint"
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        print("Input preparation complete; waiting for extractor endpoint.", flush=True)
        wait_for_endpoint(args.endpoint, args.model, args.endpoint_start_timeout)
        manifest = read_json(manifest_path)
        manifest["status"] = "running"
        manifest["started_at"] = manifest.get("started_at") or utc_now()
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)

        for job in jobs:
            current = next(
                record for record in read_json(manifest_path)["jobs"] if record["job_id"] == job.job_id
            )
            if current.get("status") == "completed":
                validate_extracted_job(args.output_root, job, args.model, args.revision)
                print(f"[{utc_now()}] Verified completed {job.job_id}; skipping", flush=True)
                continue
            update_manifest_job(
                manifest_path, job.job_id, status="in-progress", started_at=utc_now()
            )
            run_job(args, args.output_root, job)
            validation = validate_extracted_job(
                args.output_root, job, args.model, args.revision
            )
            update_manifest_job(manifest_path, job.job_id, **validation)
            print(
                f"[{utc_now()}] Completed {job.job_id}: "
                f"resolved={validation['resolved_answer_count']} "
                f"unresolved={validation['unresolved_answer_count']}",
                flush=True,
            )

        manifest = read_json(manifest_path)
        index = write_variant_manifests(args.output_root, jobs, manifest)
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        manifest["resolved_answer_count"] = index["resolved_answer_count"]
        manifest["unresolved_answer_count"] = index["unresolved_answer_count"]
        atomic_write_json(manifest_path, manifest)
        print(json.dumps(index, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception:
        manifest = read_json(manifest_path)
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        active_marker.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

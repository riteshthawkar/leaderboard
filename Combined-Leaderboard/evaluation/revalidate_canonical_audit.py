"""Revalidate a complete visual-answer audit without making new LLM requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.extract_canonical_answers import (
    METHOD,
    candidate_key,
    classify_extractor_output,
    load_candidates,
    load_gold_answers,
)


class RevalidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RevalidationError(
                f"Invalid JSON on {path} line {line_number}: {exc.msg}."
            ) from exc
        if not isinstance(row, dict):
            raise RevalidationError(f"Audit line {line_number} is not an object.")
        rows.append(row)
    return rows


def revalidate_row(
    candidate: dict[str, Any],
    source: dict[str, Any],
    ground_truth_sha256: str,
    source_audit_sha256: str,
) -> dict[str, Any]:
    if candidate_key(source) != candidate_key(candidate):
        raise RevalidationError(f"Candidate key mismatch: {candidate_key(source)}.")
    if source.get("response_sha256") != candidate["response_sha256"]:
        raise RevalidationError(
            f"Response hash changed for {candidate_key(candidate)}."
        )
    source_ground_truth_sha256 = source.get("ground_truth_sha256")
    if (
        source_ground_truth_sha256 is not None
        and source_ground_truth_sha256 != ground_truth_sha256
    ):
        raise RevalidationError(
            f"Ground truth changed for {candidate_key(candidate)}."
        )
    if source.get("ground_truth_loaded") is not False:
        raise RevalidationError(
            f"Source audit process loaded ground truth for {candidate_key(candidate)}."
        )
    if source.get("ground_truth_supplied_to_extractor") is not False:
        raise RevalidationError(
            f"Source audit is not gold-blind for {candidate_key(candidate)}."
        )
    extractor_contract_sha256 = str(
        source.get("extractor_contract_sha256") or ""
    )
    if len(extractor_contract_sha256) != 64:
        raise RevalidationError(
            f"Source audit lacks an extractor contract hash for "
            f"{candidate_key(candidate)}."
        )

    extractor_output = str(source.get("extractor_output") or "")
    error = str(source.get("error") or "") or None
    classification = classify_extractor_output(candidate, extractor_output, error)
    result = {
        key: candidate[key]
        for key in (
            "model_slug",
            "track",
            "question_id",
            "answer_type",
            "category",
            "response_sha256",
        )
    } | {
        "method": METHOD,
        "extractor_contract_sha256": extractor_contract_sha256,
        "ground_truth_sha256": ground_truth_sha256,
        "ground_truth_available_to_validator": True,
        "ground_truth_loaded_by_extractor_process": False,
        "ground_truth_supplied_to_extractor": False,
        "extractor_model": str(source.get("extractor_model") or ""),
        **classification,
        "extractor_output": extractor_output,
        "finish_reason": source.get("finish_reason"),
        "completion_tokens": source.get("completion_tokens"),
        "revalidated_without_llm_request": True,
        "revalidated_from_method": str(source.get("method") or ""),
        "revalidated_from_audit_sha256": source_audit_sha256,
    }
    if error:
        result["error"] = error
    return result


def revalidate_audit(
    project_root: Path,
    canonical_root: Path,
    ground_truth_paths: list[Path],
    source_audit: Path,
    output: Path,
    policy: str,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    canonical_root = canonical_root.resolve()
    source_audit = source_audit.resolve()
    output = output.resolve()
    if output.exists():
        raise RevalidationError(f"Output already exists: {output}")

    gold_answers, ground_truth_sha256 = load_gold_answers(
        project_root, ground_truth_paths
    )
    candidates = load_candidates(
        project_root, canonical_root, policy, gold_answers
    )
    candidates_by_key = {candidate_key(row): row for row in candidates}
    source_rows = read_jsonl(source_audit)
    source_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in source_rows:
        key = candidate_key(row)
        if key in source_by_key:
            raise RevalidationError(f"Source audit repeats candidate {key}.")
        source_by_key[key] = row

    missing = set(candidates_by_key) - set(source_by_key)
    extra = set(source_by_key) - set(candidates_by_key)
    if missing or extra:
        raise RevalidationError(
            f"Source audit is incomplete or mismatched: {len(missing)} missing, "
            f"{len(extra)} unexpected records."
        )

    source_audit_sha256 = sha256(source_audit)
    revalidated = [
        revalidate_row(
            candidates_by_key[candidate_key(row)],
            row,
            ground_truth_sha256,
            source_audit_sha256,
        )
        for row in source_rows
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in revalidated:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return {
        "candidate_count": len(revalidated),
        "ground_truth_sha256": ground_truth_sha256,
        "method": METHOD,
        "output": str(output),
        "output_sha256": sha256(output),
        "source_audit": str(source_audit),
        "source_audit_sha256": source_audit_sha256,
        "status_counts": dict(Counter(row["status"] for row in revalidated)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--canonical-root", type=Path, default=project_root / "evaluation/results/final"
    )
    parser.add_argument(
        "--ground-truth",
        action="append",
        dest="ground_truth_paths",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("unresolved", "high_risk", "all_nonexact"),
        default="high_risk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = revalidate_audit(
            args.project_root,
            args.canonical_root,
            args.ground_truth_paths,
            args.source_audit,
            args.output,
            args.policy,
        )
    except (RevalidationError, ValueError) as exc:
        raise SystemExit(f"Audit revalidation failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

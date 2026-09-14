"""Revalidate persisted gold-blind LLM evidence without calling or replacing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from evaluation.extract_canonical_answers import (
    DEFAULT_EXCLUDED_VARIANTS,
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    FAIL_CLOSED_FALLBACK_METHOD,
    METHOD,
    candidate_key,
    load_candidates,
    revalidate_extractor_row,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def revalidate_audit(
    project_root: Path,
    source_root: Path,
    input_audit: Path,
    output_audit: Path,
    excluded_variants: set[str],
) -> dict:
    project_root = project_root.resolve()
    source_root = source_root.resolve()
    input_audit = input_audit.resolve()
    output_audit = output_audit.resolve()
    if input_audit == output_audit:
        raise ValueError("Input and output audit paths must differ.")
    if output_audit.exists():
        raise ValueError(f"Output audit already exists: {output_audit}.")

    candidates = load_candidates(
        project_root,
        source_root,
        "all",
        excluded_variants=excluded_variants,
    )
    candidates_by_key = {candidate_key(item): item for item in candidates}
    if len(candidates_by_key) != len(candidates):
        raise ValueError("Evidence candidates are not unique.")

    source_digest = _sha256(input_audit)
    seen = set()
    transitions = Counter()
    row_count = 0

    def revalidated_rows() -> Iterable[dict]:
        nonlocal row_count
        with input_audit.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = candidate_key(row)
                if key in seen:
                    raise ValueError(
                        f"Input audit repeats {key} at line {line_number}."
                    )
                seen.add(key)
                candidate = candidates_by_key.get(key)
                if candidate is None:
                    raise ValueError(f"Input audit contains unexpected row {key}.")
                if row.get("method") != METHOD:
                    raise ValueError(f"Extractor method mismatch for {key}.")
                if row.get("extractor_model") != DEFAULT_EXTRACTOR_MODEL:
                    raise ValueError(f"Extractor model mismatch for {key}.")
                if row.get("extractor_revision") != DEFAULT_EXTRACTOR_REVISION:
                    raise ValueError(f"Extractor revision mismatch for {key}.")
                if row.get("ground_truth_loaded") is not False:
                    raise ValueError(f"Ground truth was loaded for {key}.")
                if row.get("ground_truth_supplied_to_extractor") is not False:
                    raise ValueError(f"Ground truth was supplied for {key}.")
                if row.get("response_sha256") != candidate["response_sha256"]:
                    raise ValueError(f"Candidate response changed for {key}.")
                if row.get("terminal_fallback_method") not in {
                    None,
                    FAIL_CLOSED_FALLBACK_METHOD,
                }:
                    raise ValueError(f"Unknown terminal fallback for {key}.")

                revalidated = revalidate_extractor_row(
                    candidate,
                    row,
                    source_audit_sha256=source_digest,
                )
                transitions[
                    (str(row.get("status")), str(revalidated.get("status")))
                ] += 1
                row_count += 1
                yield revalidated

        missing = set(candidates_by_key) - seen
        if missing:
            raise ValueError(f"Input audit is missing {len(missing)} candidates.")

    _atomic_write(output_audit, revalidated_rows())
    return {
        "input_audit": str(input_audit),
        "output_audit": str(output_audit),
        "source_audit_sha256": source_digest,
        "output_audit_sha256": _sha256(output_audit),
        "rows": row_count,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": False,
        "transitions": {
            f"{source}->{target}": count
            for (source, target), count in sorted(transitions.items())
        },
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
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
        result = revalidate_audit(
            args.project_root,
            args.source_root,
            args.input_audit,
            args.output_audit,
            set(args.exclude_variants),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Evidence revalidation failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

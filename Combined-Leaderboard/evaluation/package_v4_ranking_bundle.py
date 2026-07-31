#!/usr/bin/env python3
"""Score and package a canonical v4 visual-ranking result tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
    extractor_contract_sha256,
)
from evaluation.finalize_visual_results import (
    CURRENT_PIPELINE_REVISION,
    FinalizationError,
    TRACKS,
    read_json,
    read_jsonl,
    sha256,
    verify_canonical_results,
)
from scripts.import_canonical_visual_results import build_import_plan
from visual_answer_contract import INVALID_FORMAT_TOKEN, UNRESOLVED_TOKEN


EXPECTED_ROWS = {"do_you_see_me": 4500, "minds_eye": 799}
EXTRACTOR_MAX_TOKENS = 256
CHECKSUM_FILE = "SHA256SUMS"
UNRESOLVED_STATUSES = {
    "unresolved",
    "unsupported_by_evidence",
    "unresolved_truncated_response",
}


class PackagingError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def extraction_identity(config: dict[str, Any]) -> tuple[Any, ...]:
    extraction = config.get("answer_extraction")
    if not isinstance(extraction, dict):
        raise PackagingError("Run configuration has no answer-extraction record.")
    return (
        extraction.get("method"),
        extraction.get("model"),
        extraction.get("revision"),
        extraction.get("extractor_contract_sha256"),
        tuple(extraction.get("input_fields") or ()),
        extraction.get("image_supplied"),
        extraction.get("ground_truth_loaded"),
        extraction.get("ground_truth_supplied"),
    )


def validate_extraction_identity(identity: tuple[Any, ...]) -> None:
    expected_contract = extractor_contract_sha256(
        DEFAULT_EXTRACTOR_MODEL,
        EXTRACTOR_MAX_TOKENS,
        DEFAULT_EXTRACTOR_REVISION,
    )
    if identity[:4] != (
        METHOD,
        DEFAULT_EXTRACTOR_MODEL,
        DEFAULT_EXTRACTOR_REVISION,
        expected_contract,
    ):
        raise PackagingError(f"Unexpected v4 extraction identity: {identity[:4]}.")
    expected_fields = (
        "question",
        "answer_type",
        "task",
        "response_metadata",
        "candidate_response",
    )
    if identity[4] != expected_fields:
        raise PackagingError(f"Unexpected extractor input fields: {identity[4]}.")
    if identity[5:] != (False, False, False):
        raise PackagingError("The evidence extractor was not recorded as gold-blind.")


def validate_track_evidence(
    root: Path,
    slug: str,
    track: str,
    track_manifest: dict[str, Any],
) -> dict[str, Any]:
    model_dir = root / slug
    diagnostics = read_jsonl(model_dir / f"{track}.diagnostics.jsonl")
    audit = read_jsonl(model_dir / f"{track}.evidence_extraction.jsonl")
    if len(diagnostics) != EXPECTED_ROWS[track] or len(audit) != EXPECTED_ROWS[track]:
        raise PackagingError(f"Unexpected evidence coverage for {slug}/{track}.")

    diagnostics_by_id = {
        str(row.get("question_id") or ""): row for row in diagnostics
    }
    audit_by_id = {str(row.get("question_id") or ""): row for row in audit}
    if (
        "" in diagnostics_by_id
        or "" in audit_by_id
        or len(diagnostics_by_id) != len(diagnostics)
        or len(audit_by_id) != len(audit)
        or set(diagnostics_by_id) != set(audit_by_id)
    ):
        raise PackagingError(f"Evidence identifiers are invalid for {slug}/{track}.")

    status_counts: Counter[str] = Counter()
    source_hash_mismatches = 0
    terminal_fallback_count = 0
    for question_id, evidence in audit_by_id.items():
        diagnostic = diagnostics_by_id[question_id]
        status = str(evidence.get("status") or "")
        status_counts[status] += 1
        if evidence.get("terminal_fallback_method"):
            terminal_fallback_count += 1

        raw_output = str(diagnostic.get("output") or "")
        raw_hash = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        recorded_hashes = {
            str(evidence.get("response_sha256") or ""),
            str(diagnostic.get("extractor_source_output_sha256") or ""),
        }
        if recorded_hashes != {raw_hash}:
            source_hash_mismatches += 1

        extracted = str(diagnostic.get("extracted_answer") or "")
        if status == "committed":
            expected_answer = str(evidence.get("answer") or "")
        elif status == "invalid_format_committed":
            expected_answer = INVALID_FORMAT_TOKEN
        elif status in UNRESOLVED_STATUSES:
            expected_answer = UNRESOLVED_TOKEN
        else:
            raise PackagingError(
                f"Unexpected evidence status {status!r} for {slug}/{track}/{question_id}."
            )
        if extracted != expected_answer:
            raise PackagingError(
                f"Evidence answer mismatch for {slug}/{track}/{question_id}."
            )
        if diagnostic.get("answer_extraction_method") != METHOD:
            raise PackagingError(
                f"Extraction method mismatch for {slug}/{track}/{question_id}."
            )

    if source_hash_mismatches:
        raise PackagingError(
            f"Found {source_hash_mismatches} source-response hash mismatches "
            f"for {slug}/{track}."
        )
    declared = Counter(
        {
            str(key): int(value)
            for key, value in track_manifest["evidence_extraction"][
                "status_counts"
            ].items()
        }
    )
    if status_counts != declared:
        raise PackagingError(f"Evidence status counts changed for {slug}/{track}.")
    if status_counts["committed"] != int(track_manifest["strict_answer_count"]):
        raise PackagingError(f"Strict-answer count changed for {slug}/{track}.")
    if status_counts["invalid_format_committed"] != int(
        track_manifest["invalid_commitment_count"]
    ):
        raise PackagingError(f"Invalid-commitment count changed for {slug}/{track}.")
    unresolved = sum(status_counts[status] for status in UNRESOLVED_STATUSES)
    if unresolved != int(track_manifest["unresolved_answer_count"]):
        raise PackagingError(f"Unresolved-answer count changed for {slug}/{track}.")
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "terminal_fallback_count": terminal_fallback_count,
        "source_response_sha256_mismatches": source_hash_mismatches,
    }


def package_bundle(root: Path, project_root: Path) -> dict[str, Any]:
    root = root.resolve()
    project_root = project_root.resolve()
    verified = verify_canonical_results(root, project_root)
    index = read_json(root / "index.json")
    if index.get("pipeline_revision") != CURRENT_PIPELINE_REVISION:
        raise PackagingError("Canonical index has an unexpected pipeline revision.")

    import_plan = build_import_plan(root, set())
    imports_by_slug = {model.slug: model for model in import_plan}
    if set(imports_by_slug) != {
        str(model.get("slug") or "") for model in index["models"]
    }:
        raise PackagingError("Importer coverage differs from the canonical index.")

    identities: set[tuple[Any, ...]] = set()
    audit_hashes: set[str] = set()
    total_rows = 0
    total_strict = 0
    total_invalid_commitments = 0
    total_unresolved = 0
    total_terminal_fallbacks = 0
    models = []

    for index_model in index["models"]:
        slug = str(index_model["slug"])
        manifest = read_json(root / slug / "final_manifest.json")
        scored = imports_by_slug[slug]
        if index_model.get("reasoning_profile") != manifest.get(
            "reasoning_profile"
        ):
            raise PackagingError(f"Reasoning profile differs in the index for {slug}.")
        manifest_extraction = manifest.get("evidence_extraction") or {}
        audit_hash = str(manifest_extraction.get("source_audit_sha256") or "")
        if len(audit_hash) != 64:
            raise PackagingError(f"Source audit hash is invalid for {slug}.")
        audit_hashes.add(audit_hash)

        tracks = {}
        for track in TRACKS:
            track_manifest = manifest["tracks"][track]
            config = read_json(root / slug / f"{track}.run_config.json")
            identity = extraction_identity(config)
            identities.add(identity)
            evidence = validate_track_evidence(
                root,
                slug,
                track,
                track_manifest,
            )
            expected_rows = EXPECTED_ROWS[track]
            row_count = int(track_manifest["row_count"])
            if row_count != expected_rows:
                raise PackagingError(f"Unexpected row count for {slug}/{track}.")

            strict = int(track_manifest["strict_answer_count"])
            invalid_commitments = int(track_manifest["invalid_commitment_count"])
            unresolved = int(track_manifest["unresolved_answer_count"])
            if strict + invalid_commitments + unresolved != row_count:
                raise PackagingError(f"Provenance counts do not sum for {slug}/{track}.")

            total_rows += row_count
            total_strict += strict
            total_invalid_commitments += invalid_commitments
            total_unresolved += unresolved
            total_terminal_fallbacks += evidence["terminal_fallback_count"]
            score = scored.tracks[track].score
            tracks[track] = {
                "macro_accuracy": round(float(score.macro_accuracy), 6),
                "row_count": row_count,
                "strict_answer_count": strict,
                "invalid_commitment_count": invalid_commitments,
                "invalid_format_count": int(track_manifest["invalid_format_count"]),
                "unresolved_answer_count": unresolved,
                "evidence_status_counts": evidence["status_counts"],
                "terminal_fallback_count": evidence["terminal_fallback_count"],
                "submission": f"{slug}/{track}_submission.jsonl",
                "submission_sha256": sha256(
                    root / slug / f"{track}_submission.jsonl"
                ),
            }

        models.append(
            {
                "display_name": scored.catalog["display_name"],
                "model_id": manifest["model_id"],
                "model_revision": manifest["model_revision"],
                "reasoning_profile": manifest["reasoning_profile"],
                "slug": slug,
                "tracks": tracks,
                "vpci": round(scored.vpci, 6),
            }
        )

    if len(identities) != 1:
        raise PackagingError(
            f"Expected one extraction contract, found {len(identities)}."
        )
    identity = identities.pop()
    validate_extraction_identity(identity)
    expected_total = len(models) * sum(EXPECTED_ROWS.values())
    if total_rows != expected_total:
        raise PackagingError("Bundle coverage does not match its canonical index.")

    generated_at = datetime.now(timezone.utc).isoformat()
    ranked_models = sorted(
        models,
        key=lambda item: (-item["vpci"], item["display_name"]),
    )
    ranking = [
        {
            "rank": rank,
            "slug": model["slug"],
            "display_name": model["display_name"],
            "model_id": model["model_id"],
            "reasoning_profile": model["reasoning_profile"],
            "do_you_see_me_macro": model["tracks"]["do_you_see_me"][
                "macro_accuracy"
            ],
            "minds_eye_macro": model["tracks"]["minds_eye"]["macro_accuracy"],
            "vpci": model["vpci"],
        }
        for rank, model in enumerate(ranked_models, start=1)
    ]
    write_json(
        root / "RANKING_SCORES.json",
        {
            "schema_version": 2,
            "generated_at": generated_at,
            "metric": (
                "VPCI: arithmetic mean of Do You See Me and Mind's Eye "
                "macro accuracy"
            ),
            "import_mode": "dry-run",
            "model_count": len(ranking),
            "ranking": ranking,
        },
    )

    bundle_manifest = {
        "schema_version": 2,
        "bundle_name": root.name,
        "generated_at": generated_at,
        "purpose": "Verified MS-VISTA visual leaderboard ranking bundle",
        "model_count": len(models),
        "models": sorted(models, key=lambda item: item["slug"]),
        "coverage": {
            "do_you_see_me_rows_per_model": EXPECTED_ROWS["do_you_see_me"],
            "minds_eye_rows_per_model": EXPECTED_ROWS["minds_eye"],
            "total_rows": total_rows,
            "strict_answer_count": total_strict,
            "invalid_commitment_count": total_invalid_commitments,
            "invalid_format_count": total_invalid_commitments + total_unresolved,
            "unresolved_answer_count": total_unresolved,
            "terminal_fallback_count": total_terminal_fallbacks,
        },
        "extraction": {
            "method": identity[0],
            "model": identity[1],
            "model_revision": identity[2],
            "contract_sha256": identity[3],
            "max_tokens": EXTRACTOR_MAX_TOKENS,
            "input_fields": list(identity[4]),
            "image_supplied": identity[5],
            "ground_truth_loaded": identity[6],
            "ground_truth_supplied": identity[7],
            "source_audit_sha256s": sorted(audit_hashes),
            "invalid_commitment_submission_token": INVALID_FORMAT_TOKEN,
            "unresolved_submission_token": UNRESOLVED_TOKEN,
        },
        "pipeline_revision": CURRENT_PIPELINE_REVISION,
        "validation": {
            "canonical_verifier": "passed",
            "importer_dry_run": "passed",
            "verified_model_count": verified["model_count"],
            "source_response_sha256_mismatches": 0,
        },
        "checksum_inventory": CHECKSUM_FILE,
        "ranking_report": "RANKING_SCORES.json",
    }
    write_json(root / "BUNDLE_MANIFEST.json", bundle_manifest)

    ranking_rows = "\n".join(
        f"| {item['rank']} | {item['display_name']} | "
        f"{item['do_you_see_me_macro']:.6f} | "
        f"{item['minds_eye_macro']:.6f} | {item['vpci']:.6f} |"
        for item in ranking
    )
    readme = f"""# MS-VISTA Evidence-Audited Ranking Bundle

This directory contains {len(models)} canonical model variants evaluated on Do
You See Me and Mind's Eye. Every one of the {total_rows:,} responses was graded
from the same production submission format and the same gold-blind v4 evidence
audit contract.

The extractor received the original question, answer type, task, response
metadata, and candidate response. It received no image and no reference answer.
Committed out-of-domain answers are retained as `{INVALID_FORMAT_TOKEN}` and
responses without defensible final commitment are retained as
`{UNRESOLVED_TOKEN}`. Both are scored as incorrect.

## Ranking

| Rank | Model variant | Perception macro | Cognition macro | VPCI |
| ---: | --- | ---: | ---: | ---: |
{ranking_rows}

## Verification

```bash
BUNDLE=/path/to/{root.name}

(cd "$BUNDLE" && shasum -a 256 -c SHA256SUMS)
python -m evaluation.finalize_visual_results --verify-only --output-root "$BUNDLE"
PYTHONPATH="$PWD:$PWD/backend" \\
  GROUND_TRUTHS_DIR="$PWD/Ground_truths" \\
  python scripts/import_canonical_visual_results.py --result-root "$BUNDLE"
```
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    checksum_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    )
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in checksum_paths
    ]
    (root / CHECKSUM_FILE).write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "bundle": str(root),
        "checksum_entries": len(checksum_lines),
        "model_count": len(models),
        "total_rows": total_rows,
        "strict_answer_count": total_strict,
        "invalid_commitment_count": total_invalid_commitments,
        "unresolved_answer_count": total_unresolved,
        "terminal_fallback_count": total_terminal_fallbacks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("root", type=Path)
    parser.add_argument("--project-root", type=Path, default=project_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = package_bundle(args.root, args.project_root)
    except (
        FinalizationError,
        KeyError,
        OSError,
        PackagingError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"Ranking bundle packaging failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

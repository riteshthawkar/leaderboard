"""Assemble one canonical ranking tree from verified v4 evidence-audited runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.build_production_visual_results import (
    ProductionBuildError,
    build_production_results,
)
from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
)
from evaluation.finalize_visual_results import (
    CURRENT_PIPELINE_REVISION,
    FinalizationError,
    TRACKS,
    read_json,
    sha256,
    verify_canonical_results,
)


class AssemblyError(RuntimeError):
    pass


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_identity(source_dir: Path) -> tuple[str, str, str]:
    manifest_path = source_dir / "final_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    configs = {
        track: read_json(source_dir / f"{track}.run_config.json")
        for track in TRACKS
    }
    model_ids = {
        str(value)
        for value in (
            manifest.get("model_id"),
            *(config.get("model_id") for config in configs.values()),
        )
        if value
    }
    revisions = {
        str(value)
        for value in (
            manifest.get("model_revision"),
            *(config.get("model_revision") for config in configs.values()),
        )
        if value
    }
    profiles = {
        str(value).strip().lower()
        for value in (
            manifest.get("reasoning_profile"),
            *(config.get("reasoning_profile") for config in configs.values()),
        )
        if value
    }
    if len(model_ids) != 1 or len(revisions) != 1 or len(profiles) != 1:
        raise AssemblyError(
            f"Source identity or reasoning profile is inconsistent in {source_dir}."
        )
    profile = profiles.pop()
    if profile not in {"nonthinking", "thinking"}:
        raise AssemblyError(f"Unsupported reasoning profile {profile!r}.")
    return model_ids.pop(), revisions.pop(), profile


def make_single_source_tree(
    parent: Path,
    source_dir: Path,
    slug: str,
) -> Path:
    source_root = parent / "additional-source"
    source_root.mkdir()
    linked_dir = source_root / slug
    linked_dir.symlink_to(source_dir, target_is_directory=True)
    model_id, model_revision, reasoning_profile = source_identity(source_dir)
    tracks = {}
    for track in TRACKS:
        diagnostics = source_dir / f"{track}.diagnostics.jsonl"
        submission = source_dir / f"{track}_submission.jsonl"
        run_config = source_dir / f"{track}.run_config.json"
        for path in (diagnostics, submission, run_config):
            if not path.is_file():
                raise AssemblyError(f"Additional source artifact is missing: {path}.")
        tracks[track] = {
            "relative_dir": slug,
            "diagnostics": diagnostics.name,
            "submission": submission.name,
            "diagnostics_sha256": sha256(diagnostics),
            "submission_sha256": sha256(submission),
            "source_run_config_sha256": sha256(run_config),
        }
    write_json(
        source_root / "index.json",
        {
            "schema_version": 1,
            "variant_count": 1,
            "variants": [
                {
                    "variant_id": slug,
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "reasoning_profile": reasoning_profile,
                    "tracks": tracks,
                }
            ],
        },
    )
    return source_root


def evidence_identity(manifest: dict[str, Any]) -> tuple[str, str, str, str]:
    extraction = manifest.get("evidence_extraction")
    if not isinstance(extraction, dict):
        raise AssemblyError("Canonical manifest has no evidence-extraction record.")
    identity = (
        str(extraction.get("method") or ""),
        str(extraction.get("extractor_model") or ""),
        str(extraction.get("extractor_revision") or ""),
        str(extraction.get("extractor_contract_sha256") or ""),
    )
    if identity[:3] != (METHOD, DEFAULT_EXTRACTOR_MODEL, DEFAULT_EXTRACTOR_REVISION):
        raise AssemblyError(f"Unexpected evidence-extraction identity: {identity}.")
    if len(identity[3]) != 64:
        raise AssemblyError("Evidence-extraction contract hash is invalid.")
    for field in ("image_supplied", "ground_truth_loaded", "ground_truth_supplied"):
        if extraction.get(field) is not False:
            raise AssemblyError(f"Evidence audit is not gold-blind: {field}.")
    return identity


def model_reasoning_profile(model_dir: Path) -> str:
    profiles = {
        str(
            read_json(model_dir / f"{track}.run_config.json").get(
                "reasoning_profile"
            )
            or ""
        )
        .strip()
        .lower()
        for track in TRACKS
    }
    profiles.discard("")
    if len(profiles) != 1:
        raise AssemblyError(f"Reasoning profile is inconsistent in {model_dir}.")
    profile = profiles.pop()
    if profile not in {"nonthinking", "thinking"}:
        raise AssemblyError(f"Unsupported reasoning profile {profile!r}.")
    return profile


def normalize_source_config_names(
    model_dir: Path,
    manifest: dict[str, Any],
) -> None:
    for track in TRACKS:
        track_record = manifest["tracks"][track]
        source_name = str(track_record.get("source_run_config") or "")
        if not source_name:
            raise AssemblyError(f"Source run configuration is missing in {model_dir}.")
        source = model_dir / source_name
        if not source.is_file():
            raise AssemblyError(f"Source run configuration is missing: {source}.")
        target_name = f"{track}.source_run_config.json"
        target = model_dir / target_name
        if source != target:
            if target.exists():
                raise AssemblyError(f"Source run configuration collides: {target}.")
            source.rename(target)
        artifacts = track_record.get("artifacts")
        if not isinstance(artifacts, dict):
            raise AssemblyError(f"Artifact map is missing for {model_dir}/{track}.")
        artifact = artifacts.pop(source_name, None)
        if not isinstance(artifact, dict):
            raise AssemblyError(
                f"Source run configuration is absent from the artifact map: {source}."
            )
        artifacts[target_name] = artifact
        track_record["source_run_config"] = target_name


def normalized_model_record(
    root: Path,
    record: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, str, str, str], str]:
    slug = str(record["slug"])
    model_dir = root / slug
    manifest_path = model_dir / "final_manifest.json"
    manifest = read_json(manifest_path)
    profile = model_reasoning_profile(model_dir)
    manifest["reasoning_profile"] = profile
    normalize_source_config_names(model_dir, manifest)
    write_json(manifest_path, manifest)
    normalized = {
        **record,
        "reasoning_profile": profile,
        "manifest": f"{slug}/final_manifest.json",
        "manifest_sha256": sha256(manifest_path),
    }
    extraction = evidence_identity(manifest)
    audit_sha = str(manifest["evidence_extraction"].get("source_audit_sha256") or "")
    if len(audit_sha) != 64:
        raise AssemblyError(f"Source audit hash is invalid for {slug}.")
    return normalized, extraction, audit_sha


def assemble_bundle(
    project_root: Path,
    base_root: Path,
    additional_source_dir: Path,
    additional_audit: Path,
    additional_slug: str,
    output_root: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    base_root = base_root.resolve()
    additional_source_dir = additional_source_dir.resolve()
    additional_audit = additional_audit.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise AssemblyError(f"Output root already exists: {output_root}.")
    verify_canonical_results(base_root, project_root)
    base_index = read_json(base_root / "index.json")
    if any(model.get("slug") == additional_slug for model in base_index["models"]):
        raise AssemblyError(f"Additional slug already exists: {additional_slug}.")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.assemble-", dir=output_root.parent)
    )
    assembled = workspace / "assembled"
    try:
        source_root = make_single_source_tree(
            workspace,
            additional_source_dir,
            additional_slug,
        )
        additional_root = workspace / "additional-v4"
        build_production_results(
            project_root,
            source_root,
            additional_audit,
            additional_root,
            excluded_variants=set(),
        )
        additional_index = read_json(additional_root / "index.json")
        if additional_index.get("model_count") != 1:
            raise AssemblyError("Additional v4 build did not produce one model.")

        shutil.copytree(base_root, assembled)
        for filename in ("BUNDLE_MANIFEST.json", "RANKING_SCORES.json", "README.md", "SHA256SUMS"):
            path = assembled / filename
            if path.exists():
                path.unlink()
        shutil.copytree(
            additional_root / additional_slug,
            assembled / additional_slug,
        )

        records = [*base_index["models"], *additional_index["models"]]
        normalized_records = []
        identities = set()
        audit_hashes = set()
        response_count = 0
        for record in records:
            normalized, identity, audit_hash = normalized_model_record(
                assembled,
                record,
            )
            normalized_records.append(normalized)
            identities.add(identity)
            audit_hashes.add(audit_hash)
            response_count += sum(
                int(normalized["tracks"][track]["row_count"]) for track in TRACKS
            )
        if len(identities) != 1:
            raise AssemblyError(
                f"Canonical models use {len(identities)} extraction contracts."
            )
        method, model, revision, contract = identities.pop()
        index = {
            "schema_version": 3,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_revision": CURRENT_PIPELINE_REVISION,
            "model_count": len(normalized_records),
            "excluded_variants": sorted(
                set(base_index.get("excluded_variants", []))
                | set(additional_index.get("excluded_variants", []))
            ),
            "evidence_extraction": {
                "method": method,
                "extractor_model": model,
                "extractor_revision": revision,
                "extractor_contract_sha256": contract,
                "source_audit_sha256s": sorted(audit_hashes),
                "response_count": response_count,
                "image_supplied": False,
                "ground_truth_loaded": False,
                "ground_truth_supplied": False,
            },
            "models": sorted(
                normalized_records,
                key=lambda item: (
                    str(item["model_id"]),
                    str(item["reasoning_profile"]),
                ),
            ),
        }
        write_json(assembled / "index.json", index)
        verification = verify_canonical_results(assembled, project_root)
        if verification["model_count"] != len(normalized_records):
            raise AssemblyError("Canonical verification changed the model count.")
        os.replace(assembled, output_root)
        return {
            "output_root": str(output_root),
            "model_count": len(normalized_records),
            "response_count": response_count,
            "evidence_method": method,
            "extractor_contract_sha256": contract,
            "source_audit_sha256s": sorted(audit_hashes),
        }
    except (ProductionBuildError, OSError, ValueError) as exc:
        raise AssemblyError(str(exc)) from exc
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--additional-source-dir", type=Path, required=True)
    parser.add_argument("--additional-audit", type=Path, required=True)
    parser.add_argument(
        "--additional-slug",
        default="qwen35-9b-thinking-enabled",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = assemble_bundle(
            args.project_root,
            args.base_root,
            args.additional_source_dir,
            args.additional_audit,
            args.additional_slug,
            args.output_root,
        )
    except (AssemblyError, FinalizationError, OSError, ValueError) as exc:
        raise SystemExit(f"Canonical v4 assembly failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

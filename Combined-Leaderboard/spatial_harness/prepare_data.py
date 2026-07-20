"""Build the explicit 13-dataset Track-3 v2 LMUData bundle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import constants as hf_constants
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError

from spatial_harness.loaders.common import (
    OPTS,
    normalize_options,
    read_tsv,
    resolve_mcq_answer,
    rows_to_frame,
)
from spatial_harness.loaders.prepare_custom import (
    mindcube_rows,
    mmvp_rows,
    omnispatial_rows,
    realworldqa_rows,
    sat_real_rows,
    spatialbench_rows,
)
from spatial_harness.run_track3_vllm import DATASETS


VLMEVALKIT_COMMIT = "7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f"
HOSTED_DATASETS = {
    "BLINK": "BLINK",
    "CV-Bench-2D": "CV-Bench-2D",
    "CV-Bench-3D": "CV-Bench-3D",
    "VSR_MCQ": "VSR-zeroshot",
}
PINNED_TSV_DATASETS = {
    "VStarBench": (
        "xjtupanda/VStar_Bench",
        "VStarBench.tsv",
        "b1312e6fbf89ace5a50cc5b0214010aeb0f69bdc",
    ),
    "MMSIBench_wo_circular": (
        "lmms-lab-si/EASI-Leaderboard-Data",
        "MMSIBench_wo_circular.tsv",
        "52bb1dd2bc5cd58a291b3a780dab08d19c0ccf0d",
    ),
}
CUSTOM_REVISIONS = {
    "MMVP": "37eafecab8a3940c50c2ade5b36de69dbc99a8cf",
    "RealWorldQA": "17e7f75e092e47169732462ea3cdfebe911105dd",
    "3DSRBench": "5c90e422b68d0484d50a3ff06b3096ea7d288026",
    "MindCubeTSV": "52bb1dd2bc5cd58a291b3a780dab08d19c0ccf0d",
    "MindCubeAssets": "9c941b46a6bd65b6914669ef7a579948fc9c8467",
    "SpatialBench": "dcfdaec5ed04aaee1ff57eef44fe25d8b530fae5",
    "OmniSpatial": "6691f3288bb1ff207d6ead4d841b505de08a6fd8",
    "SAT-Real": "bda5dde942d7f7b41bff7935f086ed9a9e348ae3",
}
MINDCUBE_TSV = "MindCubeBench_tiny_raw_qa.tsv"
EXPECTED_COUNTS = {
    "MMVP": 300,
    "RealWorldQA": 765,
    "MindCube": 1040,
    "SpatialBench": 174,
    "OmniSpatial": 1533,
    "SAT-Real": 150,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_frame(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, destination)


def _load_manifest(path: Path, lmudata: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "vlmevalkit_commit": VLMEVALKIT_COMMIT,
        "datasets": {},
    }
    if not path.is_file():
        return manifest
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return manifest
    if (
        previous.get("schema_version") != 2
        or previous.get("vlmevalkit_commit") != VLMEVALKIT_COMMIT
        or not isinstance(previous.get("datasets"), dict)
    ):
        return manifest
    manifest["datasets"] = {
        dataset: record
        for dataset, record in previous["datasets"].items()
        if dataset in DATASETS
        and isinstance(record, dict)
        and (lmudata / f"{dataset}.tsv").is_file()
    }
    return manifest


def _atomic_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["dataset_count"] = len(manifest["datasets"])
    manifest["total_rows"] = sum(
        int(item["rows"]) for item in manifest["datasets"].values()
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def verify_bundle(lmudata: Path, datasets: tuple[str, ...]) -> dict[str, Any]:
    manifest_path = lmudata / "track3_data_manifest.json"
    manifest = _load_manifest(manifest_path, lmudata)
    if not manifest_path.is_file():
        raise ValueError(f"Track-3 data manifest not found: {manifest_path}")
    missing_records = sorted(set(datasets) - set(manifest["datasets"]))
    if missing_records:
        raise ValueError(
            f"Track-3 manifest has no records for: {', '.join(missing_records)}"
        )
    for dataset in datasets:
        destination = lmudata / f"{dataset}.tsv"
        if not destination.is_file():
            raise ValueError(f"Track-3 TSV not found: {destination}")
        record = manifest["datasets"][dataset]
        actual_hash = sha256(destination)
        if actual_hash != record.get("tsv_sha256"):
            raise ValueError(
                f"{dataset} SHA-256 mismatch: {actual_hash} != {record.get('tsv_sha256')}"
            )
        frame = read_tsv(destination)
        validate_frame(frame, dataset)
        if dataset in EXPECTED_COUNTS and len(frame) != EXPECTED_COUNTS[dataset]:
            raise ValueError(
                f"{dataset} has {len(frame)} rows; v2 requires {EXPECTED_COUNTS[dataset]}"
            )
        counts = {
            str(key): int(value)
            for key, value in frame["answer_type"].value_counts().to_dict().items()
        }
        if len(frame) != record.get("rows") or counts != record.get("answer_types"):
            raise ValueError(f"{dataset} manifest counts do not match its TSV")
        print(f"Verified {dataset}: {len(frame)} rows ({counts})")
    return manifest


def _image_cell(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps([str(item) for item in value])
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return json.dumps([str(item) for item in parsed])
        except (ValueError, SyntaxError):
            pass
    return str(value)


def normalize_hosted_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    frame = frame.copy()
    if "question" not in frame and "input_prompt" in frame:
        frame["question"] = frame["input_prompt"]
    if "image" not in frame and "image_path" in frame:
        frame["image"] = frame["image_path"].map(_image_cell)
    if "image" not in frame:
        raise ValueError(f"{target} has no image/image_path column")
    if target == "VSR_MCQ":
        frame["A"] = "Yes"
        frame["B"] = "No"
        frame["answer"] = frame["answer"].map(
            lambda value: "A" if str(value).strip().casefold() in {"yes", "true", "1"} else "B"
        )
        frame["answer_type"] = "mcq"
    else:
        answer_types = []
        answers = []
        for _, row in frame.iterrows():
            option_map, _letters = normalize_options(
                {letter: row[letter] for letter in OPTS if letter in frame.columns and pd.notna(row[letter])}
            )
            if len(option_map) >= 2:
                answer, error = resolve_mcq_answer(row.get("answer"), option_map, 0)
                if error:
                    answer = str(row.get("answer") or "").strip().upper()
                answer_types.append("mcq")
                answers.append(answer)
            else:
                answer_types.append("vqa")
                answers.append(str(row.get("answer") or "").strip())
        frame["answer"] = answers
        frame["answer_type"] = answer_types
    required = {"index", "image", "question", "answer", "answer_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{target} is missing columns: {sorted(missing)}")
    return frame


def validate_frame(frame: pd.DataFrame, dataset: str) -> None:
    required = {"index", "image", "question", "answer", "answer_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{dataset} is empty")
    if frame["index"].isna().any() or frame["index"].astype(str).duplicated().any():
        raise ValueError(f"{dataset} has missing or duplicate indices")
    for column in ("image", "question", "answer"):
        empty = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if empty.any():
            raise ValueError(f"{dataset} has {int(empty.sum())} empty {column} values")
    answer_types = set(frame["answer_type"].astype(str).str.strip().str.lower())
    if not answer_types or not answer_types <= {"mcq", "vqa"}:
        raise ValueError(f"{dataset} has invalid answer types: {sorted(answer_types)}")
    mcq_mask = frame["answer_type"].astype(str).str.strip().str.lower().eq("mcq")
    for row_index, row in frame[mcq_mask].iterrows():
        options = {
            letter
            for letter in OPTS
            if letter in frame.columns and pd.notna(row[letter]) and str(row[letter]).strip()
        }
        answer = str(row["answer"]).strip().upper()
        if len(options) < 2 or answer not in options:
            raise ValueError(
                f"{dataset} MCQ row {row_index} has answer {answer!r} and options {sorted(options)}"
            )


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def build_hosted(target: str) -> pd.DataFrame:
    from vlmeval.dataset import build_dataset

    source = HOSTED_DATASETS[target]
    dataset = build_dataset(source)
    if dataset is None:
        raise RuntimeError(f"VLMEvalKit could not build {source}")
    return normalize_hosted_frame(dataset.data, target)


def build_pinned_tsv(target: str, cache_root: Path, token: str | None):
    repo_id, filename, revision = PINNED_TSV_DATASETS[target]
    source = Path(
        hf_hub_download(
            repo_id,
            filename,
            repo_type="dataset",
            revision=revision,
            token=token,
            cache_dir=cache_root,
        )
    )
    frame = normalize_hosted_frame(read_tsv(source), target)
    return frame, {
        "repo_id": repo_id,
        "revision": revision,
        "source_file": filename,
        "source_sha256": sha256(source),
    }


def build_spatialbench(cache_root: Path, token: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_disable_xet = hf_constants.HF_HUB_DISABLE_XET
    hf_constants.HF_HUB_DISABLE_XET = True
    try:
        try:
            snapshot = Path(
                snapshot_download(
                    "RussRobin/SpatialBench",
                    repo_type="dataset",
                    revision=CUSTOM_REVISIONS["SpatialBench"],
                    token=token,
                    cache_dir=cache_root,
                    allow_patterns=[
                        "*.json",
                        "counting/*",
                        "existence/*",
                        "positional/*",
                        "reach/*",
                        "size/*",
                    ],
                )
            )
        except GatedRepoError as exc:
            raise SystemExit(
                "SpatialBench access is gated. Accept the terms at "
                "https://huggingface.co/datasets/RussRobin/SpatialBench and rerun "
                "with HF_TOKEN or --hf-token."
            ) from exc
    finally:
        hf_constants.HF_HUB_DISABLE_XET = previous_disable_xet
    rows, skipped = spatialbench_rows(snapshot)
    return rows, {
        "repo_id": "RussRobin/SpatialBench",
        "revision": CUSTOM_REVISIONS["SpatialBench"],
        "transport": "https",
        "skipped": skipped,
    }


def build_mmvp(cache_root: Path, token: str | None):
    snapshot = Path(
        snapshot_download(
            "MMVP/MMVP",
            repo_type="dataset",
            revision=CUSTOM_REVISIONS["MMVP"],
            token=token,
            cache_dir=cache_root,
            allow_patterns=["Questions.csv", "MMVP Images/*"],
        )
    )
    return mmvp_rows(snapshot), {
        "repo_id": "MMVP/MMVP",
        "revision": CUSTOM_REVISIONS["MMVP"],
    }


def build_realworldqa(cache_root: Path, token: str | None):
    files = [
        Path(
            hf_hub_download(
                "xai-org/RealworldQA",
                f"data/test-{shard:05d}-of-00002.parquet",
                repo_type="dataset",
                revision=CUSTOM_REVISIONS["RealWorldQA"],
                token=token,
                cache_dir=cache_root,
            )
        )
        for shard in range(2)
    ]
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    return realworldqa_rows(frame), {
        "repo_id": "xai-org/RealworldQA",
        "revision": CUSTOM_REVISIONS["RealWorldQA"],
        "parquet_sha256": [sha256(path) for path in files],
    }


def build_3dsrbench(cache_root: Path, token: str | None):
    source = Path(
        hf_hub_download(
            "ccvl/3DSRBench",
            "3dsrbench_v1_vlmevalkit.tsv",
            repo_type="dataset",
            revision=CUSTOM_REVISIONS["3DSRBench"],
            token=token,
            cache_dir=cache_root,
        )
    )
    frame = normalize_hosted_frame(read_tsv(source), "3DSRBench")
    return frame, {
        "repo_id": "ccvl/3DSRBench",
        "revision": CUSTOM_REVISIONS["3DSRBench"],
        "source_file": "3dsrbench_v1_vlmevalkit.tsv",
        "source_sha256": sha256(source),
    }


def build_mindcube(cache_root: Path, extract_root: Path, token: str | None):
    source = Path(
        hf_hub_download(
            "lmms-lab-si/EASI-Leaderboard-Data",
            MINDCUBE_TSV,
            repo_type="dataset",
            revision=CUSTOM_REVISIONS["MindCubeTSV"],
            token=token,
            cache_dir=cache_root,
        )
    )
    archive = Path(
        hf_hub_download(
            "MLL-Lab/MindCube",
            "data.zip",
            repo_type="dataset",
            revision=CUSTOM_REVISIONS["MindCubeAssets"],
            token=token,
            cache_dir=cache_root,
        )
    )
    destination = extract_root / f"mindcube-{CUSTOM_REVISIONS['MindCubeAssets'][:12]}"
    sentinel = destination / ".complete"
    if not sentinel.is_file():
        shutil.rmtree(destination, ignore_errors=True)
        _safe_extract(archive, destination)
        sentinel.write_text("complete\n", encoding="utf-8")
    rows = mindcube_rows(read_tsv(source), destination)
    return rows, {
        "tsv_repo_id": "lmms-lab-si/EASI-Leaderboard-Data",
        "tsv_revision": CUSTOM_REVISIONS["MindCubeTSV"],
        "tsv_source_file": MINDCUBE_TSV,
        "tsv_source_sha256": sha256(source),
        "asset_repo_id": "MLL-Lab/MindCube",
        "asset_revision": CUSTOM_REVISIONS["MindCubeAssets"],
        "asset_archive_sha256": sha256(archive),
    }


def build_omnispatial(cache_root: Path, extract_root: Path, token: str | None):
    archive = Path(
        hf_hub_download(
            "qizekun/OmniSpatial",
            "OmniSpatial-test.zip",
            repo_type="dataset",
            revision=CUSTOM_REVISIONS["OmniSpatial"],
            token=token,
            cache_dir=cache_root,
        )
    )
    destination = extract_root / f"omnispatial-{CUSTOM_REVISIONS['OmniSpatial'][:12]}"
    sentinel = destination / ".complete"
    if not sentinel.is_file():
        shutil.rmtree(destination, ignore_errors=True)
        _safe_extract(archive, destination)
        sentinel.write_text("complete\n", encoding="utf-8")
    return omnispatial_rows(destination), {
        "repo_id": "qizekun/OmniSpatial",
        "revision": CUSTOM_REVISIONS["OmniSpatial"],
        "archive_sha256": sha256(archive),
    }


def build_sat_real(cache_root: Path, token: str | None):
    parquet = Path(
        hf_hub_download(
            "array/SAT",
            "SAT_test.parquet",
            repo_type="dataset",
            revision=CUSTOM_REVISIONS["SAT-Real"],
            token=token,
            cache_dir=cache_root,
        )
    )
    return sat_real_rows(pd.read_parquet(parquet)), {
        "repo_id": "array/SAT",
        "revision": CUSTOM_REVISIONS["SAT-Real"],
        "parquet_sha256": sha256(parquet),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Track-3 v2 13-dataset bundle")
    parser.add_argument("--lmudata", type=Path, default=Path("LMUData"))
    parser.add_argument("--cache", type=Path, default=Path("track3_cache"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    requested = tuple(value for value in args.datasets.split(",") if value)
    unknown = sorted(set(requested) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown Track-3 datasets: {', '.join(unknown)}")
    args.lmudata.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    source_lmudata = args.cache / "LMUData-source"
    source_lmudata.mkdir(parents=True, exist_ok=True)
    os.environ["LMUData"] = str(source_lmudata.resolve())
    if args.verify_only:
        verify_bundle(args.lmudata, requested)
        print(f"Verified Track-3 v2 data manifest at {args.lmudata / 'track3_data_manifest.json'}")
        return
    manifest_path = args.lmudata / "track3_data_manifest.json"
    manifest = _load_manifest(manifest_path, args.lmudata)
    for dataset in requested:
        destination = args.lmudata / f"{dataset}.tsv"
        publish = not (destination.is_file() and not args.force)
        if destination.is_file() and not args.force:
            frame = read_tsv(destination)
            previous = manifest["datasets"].get(dataset, {})
            metadata: dict[str, Any] = {
                key: value
                for key, value in previous.items()
                if key not in {"rows", "answer_types", "tsv_sha256"}
            }
            if not metadata and dataset in HOSTED_DATASETS:
                metadata = {
                    "source": "VLMEvalKit",
                    "registry_name": HOSTED_DATASETS[dataset],
                    "vlmevalkit_commit": VLMEVALKIT_COMMIT,
                }
            elif not metadata:
                metadata = {"source": "existing-validated"}
        elif dataset in HOSTED_DATASETS:
            frame = build_hosted(dataset)
            metadata = {
                "source": "VLMEvalKit",
                "registry_name": HOSTED_DATASETS[dataset],
                "vlmevalkit_commit": VLMEVALKIT_COMMIT,
            }
        elif dataset in PINNED_TSV_DATASETS:
            frame, metadata = build_pinned_tsv(dataset, args.cache, args.hf_token)
        elif dataset == "MMVP":
            rows, metadata = build_mmvp(args.cache, args.hf_token)
            frame = rows_to_frame(rows)
        elif dataset == "RealWorldQA":
            rows, metadata = build_realworldqa(args.cache, args.hf_token)
            frame = rows_to_frame(rows)
        elif dataset == "3DSRBench":
            frame, metadata = build_3dsrbench(args.cache, args.hf_token)
        elif dataset == "MindCube":
            rows, metadata = build_mindcube(
                args.cache, args.cache / "extracted", args.hf_token
            )
            frame = rows_to_frame(rows)
        elif dataset == "SpatialBench":
            rows, metadata = build_spatialbench(args.cache, args.hf_token)
            frame = rows_to_frame(rows)
        elif dataset == "OmniSpatial":
            rows, metadata = build_omnispatial(args.cache, args.cache / "extracted", args.hf_token)
            frame = rows_to_frame(rows)
        elif dataset == "SAT-Real":
            rows, metadata = build_sat_real(args.cache, args.hf_token)
            frame = rows_to_frame(rows)
        else:
            raise AssertionError(dataset)
        if dataset in EXPECTED_COUNTS and len(frame) != EXPECTED_COUNTS[dataset]:
            raise SystemExit(
                f"{dataset} has {len(frame)} rows; v2 requires {EXPECTED_COUNTS[dataset]}."
            )
        validate_frame(frame, dataset)
        if publish:
            _atomic_frame(frame, destination)
        counts = frame["answer_type"].value_counts().to_dict()
        manifest["datasets"][dataset] = {
            **metadata,
            "rows": len(frame),
            "answer_types": {str(key): int(value) for key, value in counts.items()},
            "tsv_sha256": sha256(destination),
        }
        _atomic_manifest(manifest_path, manifest)
        print(f"{dataset}: {len(frame)} rows ({counts})")
    print(f"Wrote Track-3 v2 data manifest to {manifest_path}")


if __name__ == "__main__":
    main()
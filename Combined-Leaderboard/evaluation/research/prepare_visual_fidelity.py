#!/usr/bin/env python3
"""Prepare fixed benchmark subsets at controlled visual fidelity levels."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.research.common import (  # noqa: E402
    build_manifest,
    read_jsonl,
    sha256_file,
    stable_seed,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/research/results/visual_fidelity"
DEFAULT_QUESTIONS = {
    "do_you_see_me": PROJECT_ROOT / "tasks/do_you_see_me/questions.jsonl",
    "minds_eye": PROJECT_ROOT / "tasks/minds_eye/questions.jsonl",
}
VARIANTS = {
    "native": None,
    "downsample_50": 0.5,
    "downsample_25": 0.25,
}
MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_IMAGE_PIXELS = 150_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def stratum(track: str, row: dict[str, Any]) -> str:
    if track == "do_you_see_me":
        return ":".join(
            (
                str(row.get("source_subset") or row.get("track") or ""),
                str(row.get("task") or ""),
                str(row.get("difficulty") or ""),
            )
        )
    return str(row.get("task") or "")


def select_stratified(
    rows: list[dict[str, Any]], track: str, per_stratum: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stratum(track, row)].append(row)
    selected: list[dict[str, Any]] = []
    for label, candidates in sorted(grouped.items()):
        ordered = sorted(
            candidates,
            key=lambda row: (
                stable_seed(seed, track, label, row["question_id"]),
                str(row["question_id"]),
            ),
        )
        selected.extend(ordered[: min(per_stratum, len(ordered))])
    return sorted(selected, key=lambda row: str(row["question_id"]))


def local_source(row: dict[str, Any], dataset_root: Path | None) -> Path | None:
    if dataset_root is None:
        return None
    relative = Path(str(row.get("image") or ""))
    candidates = (
        dataset_root / str(row.get("source_subset") or "") / relative,
        dataset_root / relative,
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(dataset_root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def download_source(url: str, destination: Path, retries: int = 4) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported image URL: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    for attempt in range(retries):
        try:
            with requests.get(
                url,
                headers={"User-Agent": "MS-VISTA-research-fidelity/1.0"},
                timeout=(15, 60),
                stream=True,
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image exceeds {MAX_IMAGE_BYTES} bytes: {url}")
                total = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            raise ValueError(
                                f"Image exceeds {MAX_IMAGE_BYTES} bytes: {url}"
                            )
                        handle.write(chunk)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(destination)
            return destination
        except (OSError, requests.RequestException) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise RuntimeError(f"Could not download {url}: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def source_cache_path(output: Path, row: dict[str, Any]) -> Path:
    url = str(row.get("image_url") or "")
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        suffix = ".img"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return output / "_source_cache" / f"{digest}{suffix}"


def materialize_sources(
    selected: list[tuple[str, dict[str, Any]]],
    output: Path,
    dataset_root: Path | None,
    workers: int,
) -> dict[tuple[str, str], Path]:
    resolved: dict[tuple[str, str], Path] = {}
    downloads: dict[Any, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for track, row in selected:
            key = (track, str(row["question_id"]))
            if local := local_source(row, dataset_root):
                resolved[key] = local
                continue
            url = str(row.get("image_url") or "").strip()
            if not url:
                raise ValueError(f"{row['question_id']} has no usable image source")
            destination = source_cache_path(output, row)
            future = executor.submit(download_source, url, destination)
            downloads[future] = key
        for completed, future in enumerate(as_completed(downloads), start=1):
            key = downloads[future]
            resolved[key] = future.result()
            if completed % 50 == 0 or completed == len(downloads):
                print(
                    f"Downloaded or validated {completed}/{len(downloads)} source images",
                    flush=True,
                )
    return resolved


def normalized_copy(source: Path, destination: Path, scale: float | None) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as raw:
        original_size = raw.size
        if original_size[0] * original_size[1] > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit: {source}"
            )
        image = ImageOps.exif_transpose(raw).convert("RGB")
        original_size = image.size
        if scale is not None:
            reduced_size = (
                max(1, round(original_size[0] * scale)),
                max(1, round(original_size[1] * scale)),
            )
            image = image.resize(reduced_size, Image.Resampling.LANCZOS)
            image = image.resize(original_size, Image.Resampling.BICUBIC)
        image.save(destination, format="PNG", optimize=True)
    return original_size


def prepare(
    *,
    output: Path,
    dysm_questions: Path,
    minds_eye_questions: Path,
    dysm_per_stratum: int,
    minds_eye_per_task: int,
    seed: int,
    dataset_root: Path | None,
    workers: int,
) -> dict[str, Any]:
    input_paths = {
        "do_you_see_me_questions": dysm_questions,
        "minds_eye_questions": minds_eye_questions,
    }
    selections = {
        "do_you_see_me": select_stratified(
            read_jsonl(dysm_questions),
            "do_you_see_me",
            dysm_per_stratum,
            seed,
        ),
        "minds_eye": select_stratified(
            read_jsonl(minds_eye_questions),
            "minds_eye",
            minds_eye_per_task,
            seed,
        ),
    }
    flattened = [
        (track, row) for track, rows in selections.items() for row in rows
    ]
    sources = materialize_sources(flattened, output, dataset_root, workers)
    output_files: dict[str, Path] = {}
    item_records: list[dict[str, Any]] = []
    for track, rows in selections.items():
        per_variant_questions: dict[str, list[dict[str, Any]]] = {
            variant: [] for variant in VARIANTS
        }
        for item_index, row in enumerate(rows, start=1):
            question_id = str(row["question_id"])
            source = sources[(track, question_id)]
            source_hash = sha256_file(source)
            variant_hashes: dict[str, str] = {}
            original_size = (0, 0)
            for variant, scale in VARIANTS.items():
                relative = Path("images") / f"{question_id}.png"
                destination = output / track / variant / relative
                original_size = normalized_copy(source, destination, scale)
                variant_hashes[variant] = sha256_file(destination)
                per_variant_questions[variant].append(
                    {
                        "question_id": question_id,
                        "question": row["question"],
                        "answer_type": row.get("answer_type", "text"),
                        "image": str(relative),
                    }
                )
            item_records.append(
                {
                    "track": track,
                    "question_id": question_id,
                    "stratum": stratum(track, row),
                    "source_subset": row.get("source_subset"),
                    "source_image": row.get("image"),
                    "source_sha256": source_hash,
                    "original_width": original_size[0],
                    "original_height": original_size[1],
                    "variant_sha256": variant_hashes,
                }
            )
            if item_index % 100 == 0 or item_index == len(rows):
                print(
                    f"Prepared {item_index}/{len(rows)} {track} items",
                    flush=True,
                )
        for variant, question_rows in per_variant_questions.items():
            path = output / track / variant / "questions.jsonl"
            write_jsonl(path, question_rows)
            output_files[f"{track}_{variant}_questions"] = path

    item_map = output / "items.jsonl"
    write_jsonl(item_map, item_records)
    output_files["item_map"] = item_map
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        experiment="visual_fidelity",
        parameters={
            "seed": seed,
            "do_you_see_me_per_stratum": dysm_per_stratum,
            "minds_eye_per_task": minds_eye_per_task,
            "variants": VARIANTS,
            "resampling": {
                "downsample": "Pillow LANCZOS",
                "restore_original_dimensions": "Pillow BICUBIC",
            },
            "max_source_bytes": MAX_IMAGE_BYTES,
            "max_source_pixels": MAX_IMAGE_PIXELS,
            "native_note": (
                "Native preserves original spatial dimensions and pixels after "
                "EXIF orientation and lossless PNG normalization."
            ),
        },
        inputs=input_paths,
        outputs=output_files,
        item_count=len(item_records),
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--do-you-see-me-questions",
        type=Path,
        default=DEFAULT_QUESTIONS["do_you_see_me"],
    )
    parser.add_argument(
        "--minds-eye-questions",
        type=Path,
        default=DEFAULT_QUESTIONS["minds_eye"],
    )
    parser.add_argument("--do-you-see-me-per-stratum", type=int, default=10)
    parser.add_argument("--minds-eye-per-task", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.do_you_see_me_per_stratum < 1 or args.minds_eye_per_task < 1:
        raise ValueError("Per-stratum sample counts must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    manifest = prepare(
        output=args.output.resolve(),
        dysm_questions=args.do_you_see_me_questions.resolve(),
        minds_eye_questions=args.minds_eye_questions.resolve(),
        dysm_per_stratum=args.do_you_see_me_per_stratum,
        minds_eye_per_task=args.minds_eye_per_task,
        seed=args.seed,
        dataset_root=args.dataset_root.resolve() if args.dataset_root else None,
        workers=args.workers,
    )
    print(
        f"Prepared {manifest['item_count']} source items across "
        f"{len(VARIANTS)} fidelity conditions at {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the six-condition spatial evaluation against OpenAI-compatible endpoints.

The script performs inference only. It writes one raw prediction JSONL file per
condition plus ``inference_manifest.json``. ``judge_track3.py`` converts those
raw generations into the compact final-answer file accepted by the leaderboard.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import io
import json
import math
import os
import string
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from spatial_contract import (
    ABLATION_MODES,
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    DATASETS,
    HARNESS_VERSION,
    MODES,
    PROMPT_MODES,
    REQUIRED_CONDITIONS,
    condition_for,
    evaluation_group_id,
    load_ablation_manifest,
    parse_csv_values,
    question_id,
    sha256_file,
)


OPTS = list(string.ascii_uppercase)
CANNOT = "Cannot determine from the image"
GRAY = (128, 128, 128)
CIRCULAR_BASE_ONLY = {"3DSRBench"}
CIRCULAR = {"SpatialBench", "SAT-Real"}
HERE = Path(__file__).resolve().parent

# Keep Pillow's decompression-bomb protection enabled with a generous ceiling.
Image.MAX_IMAGE_PIXELS = 200_000_000

DATASET_PROMPT = {
    "OmniSpatial": (
        "Task\n-----\nYou will receive 1. Image - a single RGB frame depicting a scene.\n"
        "2. Question - a natural-language query about spatial relationships between objects in the image.\n"
        "3. Options - at least two answer candidates, each tagged by a capital letter.\n"
        "Based on the image and question, provide your answer. Always ground your answer in the visual "
        "evidence; do not hallucinate unseen objects. If uncertain, pick the most plausible option and "
        "do not refuse or reply insufficient information."
    ),
    "MindCube": (
        "Your task is to analyze the spatial arrangement of objects in the scene by examining the provided "
        "images, which show the scene from different viewpoints."
    ),
}


def _clear_stale_run_artifacts(output_dir: Path) -> None:
    """Remove artifacts that could make a failed rerun look successful."""
    names = {
        "inference_manifest.json",
        "inference_errors.json",
        "submission.jsonl",
        "submission.jsonl.tmp",
        "run_manifest.json",
        "leaderboard.json",
        "judge_errors.json",
        "spatial_reasoning_submission.zip",
        "spatial_reasoning_submission.zip.tmp",
    }
    for mode in MODES:
        for prompt_mode in PROMPT_MODES:
            names.update(
                {
                    f"pred_{mode}_{prompt_mode}.jsonl",
                    f"pred_{mode}_{prompt_mode}.jsonl.tmp",
                    f"judged_{mode}_{prompt_mode}.jsonl",
                    f"judged_{mode}_{prompt_mode}.jsonl.tmp",
                }
            )
    for name in names:
        (output_dir / name).unlink(missing_ok=True)


def data_url(encoded: str, gray: bool = False) -> str:
    """Return a clean image data URL, optionally replacing the image with gray."""
    raw = base64.b64decode("".join(str(encoded).split()), validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    if gray:
        with Image.open(io.BytesIO(raw)) as image:
            size = image.size
        output = io.BytesIO()
        Image.new("RGB", size, GRAY).save(output, format="PNG")
        raw = output.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def image_cells(cell) -> list[str]:
    text = str(cell).strip()
    if text.startswith("["):
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(text)
            except Exception:
                continue
            if isinstance(value, list) and value:
                return [str(item) for item in value]
    return [text]


def present_opts(row) -> list[str]:
    return [
        letter
        for letter in OPTS
        if letter in row.index
        and pd.notna(row[letter])
        and str(row[letter]).strip()
    ]


def build_question(stem: str, options: dict[str, str], dataset: str) -> str:
    lines = [f"Question:{stem}", "Options:"]
    lines.extend(f"{letter}.{options[letter]}" for letter in sorted(options))
    lines.append(
        "Please select the correct answer (letter and option text) from the options above."
    )
    question = "\n".join(lines)
    prefix = DATASET_PROMPT.get(dataset)
    return f"{prefix}\n{question}" if prefix else question


def circular_rotations(options: dict[str, str], ground_truth: str):
    letters = sorted(options)
    texts = [options[letter] for letter in letters]
    answer_index = letters.index(ground_truth)
    for rotation in range(len(letters)):
        rotated = {
            letters[index]: texts[(index - rotation) % len(letters)]
            for index in range(len(letters))
        }
        yield rotation, rotated, letters[(answer_index + rotation) % len(letters)]


def _base_circular_id(value) -> str:
    text = str(value)
    for suffix in ("-flip-1", "-flip", "-1"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _read_dataset(
    lmudata: Path,
    dataset: str,
    include_image_lookup: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    path = lmudata / f"{dataset}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset TSV is missing: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {"index", "image", "question", "answer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset} is missing required TSV columns: {', '.join(missing)}")
    image_by_index = {}
    if include_image_lookup:
        image_by_index = {
            str(row["index"]).strip(): str(row["image"])
            for _, row in frame.iterrows()
            if len(str(row["image"])) > 64
        }
    if dataset in CIRCULAR_BASE_ONLY and "qid" in frame.columns:
        frame = (
            frame.assign(_base_id=frame["qid"].map(_base_circular_id))
            .drop_duplicates("_base_id")
            .drop(columns="_base_id")
            .reset_index(drop=True)
        )
    return frame, image_by_index


def _apply_selected_indices(
    frame: pd.DataFrame,
    dataset: str,
    selected_indices: set[str] | None,
) -> pd.DataFrame:
    if selected_indices is None:
        return frame
    normalized = frame["index"].map(lambda value: str(value).strip())
    available = set(normalized)
    missing = sorted(selected_indices - available)
    if missing:
        preview = ", ".join(missing[:8])
        remainder = len(missing) - min(len(missing), 8)
        suffix = f" and {remainder} more" if remainder else ""
        raise ValueError(
            f"{dataset} is missing {len(missing)} frozen ablation indices: {preview}{suffix}"
        )
    selected = frame.loc[normalized.isin(selected_indices)].copy()
    if len(selected) != len(selected_indices):
        raise ValueError(
            f"{dataset} ablation selection produced {len(selected)} rows; "
            f"expected {len(selected_indices)}"
        )
    return selected


def build_records(
    lmudata: str | Path,
    dataset: str,
    mode: str,
    limit: int = 0,
    selected_indices: set[str] | None = None,
    include_payload: bool = True,
) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    frame, image_by_index = _read_dataset(
        Path(lmudata),
        dataset,
        include_image_lookup=include_payload,
    )
    frame = _apply_selected_indices(frame, dataset, selected_indices)
    if limit:
        frame = frame.head(limit)

    records = []
    for _, row in frame.iterrows():
        source_index = str(row["index"]).strip()
        stem = str(row["question"]).strip()
        option_letters = present_opts(row)
        options = {letter: str(row[letter]).strip() for letter in option_letters}
        ground_truth = str(row["answer"]).strip().upper()
        if len(options) < 2:
            raise ValueError(f"{dataset}:{source_index} has fewer than two answer options")
        if ground_truth not in options:
            raise ValueError(
                f"{dataset}:{source_index} has ground-truth option {ground_truth!r} "
                "that is not present in the options"
            )

        cannot_label = None
        if mode == "noimgpp":
            if len(options) >= len(OPTS):
                raise ValueError(f"{dataset}:{source_index} has too many answer options")
            cannot_label = OPTS[len(options)]
            options = {**options, cannot_label: CANNOT}
            layouts = [(None, options, cannot_label)]
        elif mode == "main" and dataset in CIRCULAR:
            layouts = list(circular_rotations(options, ground_truth))
        else:
            layouts = [(None, options, ground_truth)]

        group_id = evaluation_group_id(dataset, source_index)
        for rotation, layout_options, answer in layouts:
            record = {
                "dataset": dataset,
                "question_id": question_id(dataset, source_index, rotation),
                "source_index": source_index,
                "evaluation_group": group_id,
                "rotation": rotation,
                "gt": answer,
                "cannot_label": cannot_label,
            }
            if include_payload:
                encoded_images = str(row["image"])
                image_reference = encoded_images.strip()
                if len(image_reference) < 32 and image_reference in image_by_index:
                    encoded_images = image_by_index[image_reference]
                record.update(
                    {
                        "question": build_question(stem, layout_options, dataset),
                        "options": layout_options,
                        "images": image_cells(encoded_images),
                        "gray": mode in ABLATION_MODES,
                    }
                )
            records.append(record)
    return records


async def infer_one(
    client,
    semaphore: asyncio.Semaphore,
    item: dict,
    system_prompt: str,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> dict:
    content = [{"type": "text", "text": item["question"]}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": data_url(encoded, gray=item["gray"])},
        }
        for encoded in item["images"]
    )
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    top_p=1.0,
                    max_tokens=max_tokens,
                    extra_body={"repetition_penalty": 1.0, "top_k": -1},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                )
            output = response.choices[0].message.content
            if output is None or not str(output).strip():
                raise RuntimeError("model endpoint returned an empty response")
            item["output"] = str(output)
            item["error"] = None
            item["attempts"] = attempt + 1
            return item
        except Exception as exc:
            last_error = str(exc)[:500]
            if attempt < max_retries:
                await asyncio.sleep(min(2**attempt, 8))
    item["output"] = None
    item["error"] = last_error or "unknown inference error"
    item["attempts"] = max_retries + 1
    return item


async def run_combo(
    clients,
    items: list[dict],
    system_prompt: str,
    model: str,
    max_tokens: int,
    concurrency: int,
    max_retries: int,
    tag: str,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        infer_one(
            clients[index % len(clients)],
            semaphore,
            item,
            system_prompt,
            model,
            max_tokens,
            max_retries,
        )
        for index, item in enumerate(items)
    ]
    completed = 0
    started = time.time()
    for future in asyncio.as_completed(tasks):
        await future
        completed += 1
        if completed % 200 == 0 or completed == len(tasks):
            rate = completed / max(time.time() - started, 1e-6)
            print(f"  [{tag}] {completed}/{len(tasks)} ({rate:.1f}/s)", flush=True)
    return items


def _prediction_record(item: dict, mode: str, prompt_mode: str) -> dict:
    return {
        "dataset": item["dataset"],
        "question_id": item["question_id"],
        "source_index": item["source_index"],
        "evaluation_group": item["evaluation_group"],
        "rotation": item["rotation"],
        "options": item["options"],
        "gt": item["gt"],
        "cannot_label": item["cannot_label"],
        "mode": mode,
        "pmode": prompt_mode,
        "condition": condition_for(mode, prompt_mode),
        "output": item.get("output"),
        "error": item.get("error"),
        "attempts": item.get("attempts"),
    }


def _load_benchmark_manifest(path: str) -> dict | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Benchmark manifest must be a JSON object")
    return manifest


def _verify_benchmark_hashes(
    benchmark_manifest: dict | None,
    dataset_files: dict,
    prompt_files: dict,
    ablation_sha256: str,
) -> None:
    if not benchmark_manifest:
        return
    if benchmark_manifest.get("schema_version") != BENCHMARK_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Official benchmark manifest has an unsupported schema_version")
    if benchmark_manifest.get("demo") is not False:
        raise ValueError("Demo benchmark manifests cannot be used for leaderboard runs")
    if benchmark_manifest.get("datasets") != list(DATASETS):
        raise ValueError("Official benchmark manifest does not declare the canonical 13 datasets")
    if benchmark_manifest.get("required_conditions") != list(REQUIRED_CONDITIONS):
        raise ValueError("Official benchmark manifest does not declare the six required conditions")
    expected_datasets = benchmark_manifest.get("dataset_files") or {}
    for dataset, actual in dataset_files.items():
        expected = expected_datasets.get(dataset, {}).get("sha256")
        if not expected:
            raise ValueError(f"Official benchmark manifest is missing the {dataset} TSV hash")
        if expected != actual["sha256"]:
            raise ValueError(
                f"{dataset} TSV hash does not match the official benchmark manifest"
            )
    expected_prompts = benchmark_manifest.get("prompts") or {}
    for prompt_mode, actual in prompt_files.items():
        expected = expected_prompts.get(prompt_mode, {}).get("sha256")
        if not expected:
            raise ValueError(f"Official benchmark manifest is missing the {prompt_mode} prompt hash")
        if expected != actual["sha256"]:
            raise ValueError(
                f"{prompt_mode} prompt hash does not match the official benchmark manifest"
            )
    expected_ablation = (benchmark_manifest.get("ablation_manifest") or {}).get("sha256")
    if not expected_ablation:
        raise ValueError("Official benchmark manifest is missing the ablation manifest hash")
    if expected_ablation != ablation_sha256:
        raise ValueError("Ablation manifest hash does not match the official benchmark manifest")


def resolve_model_names(
    endpoint_model: str,
    leaderboard_model_name: str,
    legacy_model: str,
) -> tuple[str, str]:
    """Resolve the served API identifier and stable public model name."""
    endpoint_model = str(endpoint_model or "").strip()
    leaderboard_model_name = str(leaderboard_model_name or "").strip()
    legacy_model = str(legacy_model or "").strip()
    if legacy_model and (endpoint_model or leaderboard_model_name):
        raise ValueError(
            "--model cannot be combined with --endpoint-model or --leaderboard-model-name"
        )
    if legacy_model:
        endpoint_model = legacy_model
        leaderboard_model_name = legacy_model
    if not endpoint_model:
        raise ValueError("--endpoint-model is required")
    if not leaderboard_model_name:
        leaderboard_model_name = endpoint_model
    if len(leaderboard_model_name) > 255:
        raise ValueError("--leaderboard-model-name cannot exceed 255 characters")
    return endpoint_model, leaderboard_model_name


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmudata", required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--ablation-manifest",
        default=str(HERE / "ablation_manifest.json"),
        help="Frozen per-dataset indices used for No-Image and No-Image++.",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="",
        help="Optional official manifest whose dataset and prompt hashes must match.",
    )
    parser.add_argument(
        "--endpoint-model",
        default="",
        help="Exact model identifier sent to the OpenAI-compatible inference endpoint.",
    )
    parser.add_argument(
        "--leaderboard-model-name",
        default="",
        help="Public registered-model name embedded in the submission package.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Compatibility alias that uses one value for both model names.",
    )
    parser.add_argument("--endpoints", default="http://localhost:8000/v1")
    parser.add_argument(
        "--api-key",
        default=os.getenv("SPATIAL_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY",
        help="Endpoint API key. Prefer SPATIAL_MODEL_API_KEY instead of command history.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=os.getenv("SPATIAL_MODEL_TIMEOUT_SECONDS", "120"),
        help="Per-request timeout for model inference.",
    )
    parser.add_argument("--prompts-dir", default=str(HERE / "prompts"))
    parser.add_argument("--base-prompt-file", default="")
    parser.add_argument("--cot-prompt-file", default="")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--pmodes", default=",".join(PROMPT_MODES))
    parser.add_argument("--max-tokens-noncot", type=int, default=16384)
    parser.add_argument("--max-tokens-cot", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=64,
        help="Maximum generated responses retained in memory before writing JSONL.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.concurrency < 1 or args.request_batch_size < 1:
        parser.error("--concurrency and --request-batch-size must be at least 1")
    if args.max_tokens_noncot < 1 or args.max_tokens_cot < 1:
        parser.error("--max-tokens-noncot and --max-tokens-cot must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    if (
        not math.isfinite(args.request_timeout_seconds)
        or args.request_timeout_seconds <= 0
    ):
        parser.error("--request-timeout-seconds must be a positive finite number")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    try:
        endpoint_model, leaderboard_model_name = resolve_model_names(
            args.endpoint_model,
            args.leaderboard_model_name,
            args.model,
        )
    except ValueError as exc:
        parser.error(str(exc))
    datasets = parse_csv_values(args.datasets, DATASETS, "dataset")
    modes = parse_csv_values(args.modes, MODES, "mode")
    prompt_modes = parse_csv_values(args.pmodes, PROMPT_MODES, "prompt mode")
    endpoints = [value.strip() for value in args.endpoints.split(",") if value.strip()]
    if not endpoints:
        parser.error("At least one model endpoint is required")

    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lmudata = Path(args.lmudata).resolve()
    ablation_manifest = load_ablation_manifest(args.ablation_manifest)
    benchmark_manifest = _load_benchmark_manifest(args.benchmark_manifest)
    if benchmark_manifest:
        if datasets != list(DATASETS):
            raise ValueError("Official leaderboard runs must include all 13 datasets in canonical order")
        if modes != list(MODES) or prompt_modes != list(PROMPT_MODES):
            raise ValueError("Official leaderboard runs must include all six conditions in canonical order")

    prompt_paths = {
        "noncot": Path(args.base_prompt_file).resolve()
        if args.base_prompt_file
        else Path(args.prompts_dir).resolve() / "base_noncot.txt",
        "cot": Path(args.cot_prompt_file).resolve()
        if args.cot_prompt_file
        else Path(args.prompts_dir).resolve() / "cot_default.txt",
    }
    prompts = {}
    prompt_files = {}
    for prompt_mode, path in prompt_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Prompt file is missing: {path}")
        prompts[prompt_mode] = path.read_text(encoding="utf-8").strip()
        if not prompts[prompt_mode]:
            raise ValueError(f"Prompt file is empty: {path}")
        prompt_files[prompt_mode] = {
            "filename": path.name,
            "sha256": sha256_file(path),
        }

    dataset_files = {}
    for dataset in datasets:
        path = lmudata / f"{dataset}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset TSV is missing: {path}")
        dataset_files[dataset] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _verify_benchmark_hashes(
        benchmark_manifest,
        dataset_files,
        prompt_files,
        sha256_file(args.ablation_manifest),
    )
    if benchmark_manifest:
        # Content hashes above prove these local files are the official files.
        # Persist canonical server metadata so harmless local filenames cannot
        # produce a package that the backend rejects later.
        dataset_files = benchmark_manifest["dataset_files"]
        prompt_files = benchmark_manifest["prompts"]
        ablation_file = benchmark_manifest["ablation_manifest"]
    else:
        ablation_file = {
            "filename": Path(args.ablation_manifest).name,
            "sha256": sha256_file(args.ablation_manifest),
        }

    from openai import AsyncOpenAI

    clients = [
        AsyncOpenAI(
            base_url=endpoint,
            api_key=args.api_key,
            timeout=args.request_timeout_seconds,
            max_retries=0,
        )
        for endpoint in endpoints
    ]
    _clear_stale_run_artifacts(output_dir)
    condition_counts = {}
    error_counts = {}
    artifacts = {}

    print(
        f"datasets={datasets}\nconditions={[condition_for(m, p) for m in modes for p in prompt_modes]}\n"
        f"{len(clients)} endpoint(s), concurrency={args.concurrency}",
        flush=True,
    )

    for mode in modes:
        for prompt_mode in prompt_modes:
            condition = condition_for(mode, prompt_mode)
            max_tokens = (
                args.max_tokens_cot if prompt_mode == "cot" else args.max_tokens_noncot
            )
            target = output_dir / f"pred_{mode}_{prompt_mode}.jsonl"
            temporary = target.with_suffix(target.suffix + ".tmp")
            count = 0
            failures = 0
            fatal_errors = []
            with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
                for dataset in datasets:
                    selected = (
                        ablation_manifest[dataset] if mode in ABLATION_MODES else None
                    )
                    base_items = build_records(
                        lmudata,
                        dataset,
                        mode,
                        limit=args.limit,
                        selected_indices=selected,
                    )
                    print(
                        f">>> {condition} / {dataset}: {len(base_items)} items, max_tokens={max_tokens}",
                        flush=True,
                    )
                    for start in range(0, len(base_items), args.request_batch_size):
                        items = [
                            dict(item)
                            for item in base_items[start : start + args.request_batch_size]
                        ]
                        await run_combo(
                            clients,
                            items,
                            prompts[prompt_mode],
                            endpoint_model,
                            max_tokens,
                            args.concurrency,
                            args.max_retries,
                            f"{condition}:{dataset}:{start + 1}-{start + len(items)}",
                        )
                        for item in items:
                            record = _prediction_record(item, mode, prompt_mode)
                            failures += int(record["error"] is not None)
                            count += 1
                            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                            if record["error"] is not None:
                                fatal_errors.append(
                                    {
                                        "condition": condition,
                                        "dataset": dataset,
                                        "question_id": record["question_id"],
                                        "error": record["error"],
                                    }
                                )
                        if fatal_errors:
                            break
                    if fatal_errors:
                        break
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
            condition_counts[condition] = count
            error_counts[condition] = failures
            artifacts[target.name] = {
                "sha256": sha256_file(target),
                "rows": count,
            }
            if fatal_errors:
                error_path = output_dir / "inference_errors.json"
                error_path.write_text(
                    json.dumps(
                        {"count": len(fatal_errors), "errors": fatal_errors[:100]},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(
                    f"Inference failed for {len(fatal_errors)} item(s) in {condition}. "
                    f"Stopped early to avoid an incomplete run. Inspect {error_path} and rerun."
                )

    if benchmark_manifest:
        expected_counts = benchmark_manifest.get("condition_counts") or {}
        if condition_counts != expected_counts:
            raise ValueError(
                "Generated condition counts do not match the official benchmark manifest: "
                f"generated={condition_counts}, expected={expected_counts}"
            )

    inference_manifest = {
        "schema_version": "ms-vista-spatial-inference/v1",
        "harness_version": HARNESS_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": leaderboard_model_name,
            "endpoint_model": endpoint_model,
        },
        "datasets": datasets,
        "conditions": [condition_for(mode, prompt) for mode in modes for prompt in prompt_modes],
        "dataset_files": dataset_files,
        "prompts": prompt_files,
        "decoding": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens_noncot": args.max_tokens_noncot,
            "max_tokens_cot": args.max_tokens_cot,
        },
        "ablation_manifest": ablation_file,
        "benchmark_manifest_sha256": (
            sha256_file(args.benchmark_manifest) if args.benchmark_manifest else None
        ),
        "condition_counts": condition_counts,
        "error_counts": error_counts,
        "artifacts": artifacts,
        "debug": bool(args.limit),
    }
    inference_manifest_path = output_dir / "inference_manifest.json"
    inference_manifest_path.write_text(
        json.dumps(inference_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total_failures = sum(error_counts.values())
    if total_failures:
        raise SystemExit(
            f"Inference failed for {total_failures} item(s). No leaderboard submission can be produced. "
            f"Inspect {inference_manifest_path} and the prediction error fields, then rerun."
        )
    print(f"\nInference complete. Manifest: {inference_manifest_path}")
    print("Next: run judge_track3.py to create spatial_reasoning_submission.zip")


if __name__ == "__main__":
    asyncio.run(main())

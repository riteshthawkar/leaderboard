"""Run Track-3 v2 inference against OpenAI-compatible VLM endpoints."""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import hashlib
import io
import json
import os
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from PIL import Image

from spatial_harness.loaders.common import read_tsv


DATASETS = (
    "BLINK",
    "CV-Bench-2D",
    "CV-Bench-3D",
    "MMVP",
    "RealWorldQA",
    "VStarBench",
    "MMSIBench_wo_circular",
    "3DSRBench",
    "VSR_MCQ",
    "SpatialBench",
    "MindCube",
    "OmniSpatial",
    "SAT-Real",
)
MODES = ("main", "noimgpp")
PROMPT_MODES = ("noncot", "cot")
CIRCULAR_BASE_ONLY = {"3DSRBench"}
CIRCULAR = {"SpatialBench", "SAT-Real"}
OPTS = list(string.ascii_uppercase)
CANNOT = "Cannot determine from the image"
DATASET_PROMPT = {
    "MindCube": (
        "Your task is to analyze the spatial arrangement of objects in the scene "
        "by examining the provided images, which show the scene from different viewpoints."
    )
}
NONCOT_SUFFIX = "End with a clear final answer."
COT_SUFFIX = "Let's think step by step, then end with a clear final answer."


def circular_rotations(optmap: dict[str, str], gt: str):
    """Yield every cyclic option-text rotation and its new correct letter."""
    letters = sorted(optmap)
    texts = [optmap[letter] for letter in letters]
    answer_index = letters.index(gt)
    for rotation in range(len(letters)):
        rotated = {
            letters[index]: texts[(index - rotation) % len(letters)]
            for index in range(len(letters))
        }
        yield rotated, letters[(answer_index + rotation) % len(letters)]


def build_question(stem: str, optmap: dict[str, str], dataset: str) -> str:
    lines = [f"Question:{stem}"]
    if optmap:
        lines.append("Options:")
        for letter in sorted(optmap):
            lines.append(f"{letter}.{optmap[letter]}")
        lines.append(
            "Please select the correct answer (letter and option text) from the options above."
        )
    else:
        lines.append("Answer the question directly with a short final answer.")
    question = "\n".join(lines)
    prefix = DATASET_PROMPT.get(dataset)
    return f"{prefix}\n{question}" if prefix else question


def present_opts(row: pd.Series) -> list[str]:
    return [
        letter
        for letter in OPTS
        if letter in row.index and pd.notna(row[letter]) and str(row[letter]).strip()
    ]


def _split_image_cell(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [value]


def image_cells(value: str) -> list[str]:
    """Normalize a TSV image cell into one or more image strings."""
    return _split_image_cell(str(value))


def _image_bytes(value: str) -> tuple[bytes, str]:
    if value.startswith("data:image/") and "," in value:
        header, payload = value.split(",", 1)
        mime = header.split(";", 1)[0].split(":", 1)[1]
        return base64.b64decode(payload), mime
    path = Path(value).expanduser()
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        data = path.read_bytes()
    else:
        data = base64.b64decode(value)
    try:
        with Image.open(io.BytesIO(data)) as image:
            mime = Image.MIME.get(image.format or "", "image/png")
    except Exception:
        mime = "image/png"
    return data, mime


def data_url(value: str) -> str:
    if value.startswith("data:image/"):
        return value
    data, mime = _image_bytes(value)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def gray_data_url(value: str) -> str:
    data, _mime = _image_bytes(value)
    with Image.open(io.BytesIO(data)) as image:
        gray = Image.new("RGB", image.size, (128, 128, 128))
    buffer = io.BytesIO()
    gray.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def build_records(lmudir: str | os.PathLike[str], dataset: str, mode: str, limit: int = 0):
    frame = read_tsv(Path(lmudir) / f"{dataset}.tsv")
    image_by_index = {
        str(row["index"]): str(row["image"])
        for _, row in frame.iterrows()
        if len(str(row["image"])) > 64
    }
    if dataset in CIRCULAR_BASE_ONLY:
        def base_question_id(value: Any) -> str:
            value = str(value)
            for suffix in ("-flip-1", "-flip", "-1"):
                if value.endswith(suffix):
                    return value[: -len(suffix)]
            return value

        identifier_column = "qid" if "qid" in frame.columns else "index"
        frame = (
            frame.assign(_base=frame[identifier_column].map(base_question_id))
            .drop_duplicates("_base")
            .drop(columns="_base")
            .reset_index(drop=True)
        )
    if limit:
        frame = frame.head(limit)
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        image_value = str(row["image"])
        if len(image_value) < 32 and image_value in image_by_index:
            image_value = image_by_index[image_value]
        images = image_cells(image_value)
        stem = str(row["question"]).strip()
        base = str(row["index"])
        option_letters = present_opts(row)
        explicit_type = row.get("answer_type")
        answer_type = (
            str(explicit_type).strip().lower()
            if explicit_type is not None and pd.notna(explicit_type)
            else ("mcq" if len(option_letters) >= 2 else "vqa")
        )

        if answer_type == "vqa":
            if mode == "noimgpp":
                continue
            records.append(
                {
                    "dataset": dataset,
                    "index": base,
                    "group": base,
                    "answer_type": "vqa",
                    "question": build_question(stem, {}, dataset),
                    "options": {},
                    "imgs": images,
                    "gray": False,
                    "gt": str(row["answer"]).strip(),
                    "cannot_label": None,
                }
            )
            continue

        option_map = {letter: str(row[letter]).strip() for letter in option_letters}
        ground_truth = str(row["answer"]).strip().upper()
        gray = mode == "noimgpp"
        if mode == "noimgpp":
            cannot_label = OPTS[len(option_letters)]
            augmented = dict(option_map)
            augmented[cannot_label] = CANNOT
            layouts = [(base, augmented, cannot_label, cannot_label)]
        elif dataset in CIRCULAR and ground_truth in option_map:
            layouts = [
                (f"{base}_r{rotation}", rotated, correct, None)
                for rotation, (rotated, correct) in enumerate(
                    circular_rotations(option_map, ground_truth)
                )
            ]
        else:
            layouts = [(base, dict(option_map), ground_truth, None)]
        for index, options, correct, cannot_label in layouts:
            records.append(
                {
                    "dataset": dataset,
                    "index": index,
                    "group": base,
                    "answer_type": "mcq",
                    "question": build_question(stem, options, dataset),
                    "options": options,
                    "imgs": images,
                    "gray": gray,
                    "gt": correct,
                    "cannot_label": cannot_label,
                }
            )
    return records


def prediction_payload(item: dict[str, Any], mode: str, prompt_mode: str) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "dataset",
            "index",
            "group",
            "answer_type",
            "options",
            "gt",
            "cannot_label",
        )
    } | {
        "mode": mode,
        "pmode": prompt_mode,
        "output": item.get("output"),
        **(
            {"finish_reason": item["finish_reason"]}
            if item.get("finish_reason") is not None
            else {}
        ),
        **(
            {"completion_tokens": item["completion_tokens"]}
            if item.get("completion_tokens") is not None
            else {}
        ),
        **({"error": item["error"]} if item.get("error") else {}),
    }


def _record_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("dataset")),
        str(item.get("index")),
        str(item.get("mode")),
        str(item.get("pmode")),
    )


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_existing(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    existing = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            existing[_record_key(row)] = row
    return existing


def _message_content(item: dict[str, Any], prompt_mode: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in item["imgs"]:
        url = gray_data_url(image) if item["gray"] else data_url(image)
        content.append({"type": "image_url", "image_url": {"url": url}})
    suffix = COT_SUFFIX if prompt_mode == "cot" else NONCOT_SUFFIX
    content.append({"type": "text", "text": f"{item['question']}\n{suffix}"})
    return content


def normalize_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    for suffix in ("/chat/completions", "/v1"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return f"{endpoint}/v1"


async def _validate_endpoints(clients: list[AsyncOpenAI], model: str) -> None:
    for index, client in enumerate(clients):
        models = await client.models.list()
        names = {entry.id for entry in models.data}
        if model not in names:
            raise RuntimeError(
                f"Endpoint {index} serves {sorted(names)}, not requested model {model}."
            )


async def _infer_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
    model: str,
    prompt_mode: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    retries: int,
    chat_template_kwargs: dict[str, Any],
) -> None:
    async with semaphore:
        last_error = ""
        for attempt in range(retries + 1):
            try:
                extra_body = (
                    {"chat_template_kwargs": chat_template_kwargs}
                    if chat_template_kwargs
                    else {}
                )
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": _message_content(item, prompt_mode)}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + attempt,
                    extra_body=extra_body,
                )
                choice = response.choices[0]
                content = choice.message.content
                item["output"] = str(content or "")
                item["finish_reason"] = getattr(choice, "finish_reason", None)
                usage = getattr(response, "usage", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if completion_tokens is not None:
                    item["completion_tokens"] = int(completion_tokens)
                item.pop("error", None)
                return
            except Exception as exc:  # noqa: BLE001 - preserve request failure
                last_error = f"{type(exc).__name__}: {exc}"[:500]
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
        item["output"] = ""
        item["error"] = last_error


async def run_inference(args: argparse.Namespace) -> None:
    datasets = tuple(item for item in args.datasets.split(",") if item)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown Track-3 datasets: {', '.join(unknown)}")
    endpoints = [normalize_endpoint(value) for value in args.endpoints.split(",") if value.strip()]
    clients = [AsyncOpenAI(base_url=endpoint, api_key=args.api_key, timeout=args.timeout) for endpoint in endpoints]
    await _validate_endpoints(clients, args.model)
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "schema_version": 2,
        "datasets": list(datasets),
        "model": args.model,
        "model_revision": args.model_revision,
        "endpoints": endpoints,
        "server_metadata": args.server_metadata,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "chat_template_kwargs": {
            "noncot": args.chat_template_kwargs_noncot,
            "cot": args.chat_template_kwargs_cot,
        },
        "max_tokens_noncot": args.max_tokens_noncot,
        "max_tokens_cot": args.max_tokens_cot,
        "modes": list(args.modes),
        "prompt_modes": list(args.prompt_modes),
        "limit": args.limit,
        "lmudata": str(Path(args.lmudata).resolve()),
        "dataset_sha256": {
            dataset: hashlib.sha256((Path(args.lmudata) / f"{dataset}.tsv").read_bytes()).hexdigest()
            for dataset in datasets
        },
    }
    run_config_path = output_root / "run_config.json"
    if run_config_path.is_file():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        existing_contract = {
            key: value
            for key, value in existing.items()
            if key not in {"created_at", "execution_migrations"}
        }
        if existing_contract != run_contract:
            execution_keys = {"endpoints", "server_metadata"}
            existing_semantics = {
                key: value
                for key, value in existing_contract.items()
                if key not in execution_keys
            }
            new_semantics = {
                key: value for key, value in run_contract.items() if key not in execution_keys
            }
            if existing_semantics != new_semantics:
                raise SystemExit(
                    f"Run contract changed for {output_root}; refusing to mix checkpoints."
                )
            migration = {
                "changed_at": datetime.now(timezone.utc).isoformat(),
                "previous": {
                    key: existing_contract.get(key) for key in sorted(execution_keys)
                },
                "current": {key: run_contract[key] for key in sorted(execution_keys)},
                "reason": "throughput-only-endpoint-scaling",
            }
            updated = {
                **existing,
                "endpoints": run_contract["endpoints"],
                "server_metadata": run_contract["server_metadata"],
                "execution_migrations": [
                    *existing.get("execution_migrations", []),
                    migration,
                ],
            }
            temporary = run_config_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, run_config_path)
            print("Recorded throughput-only endpoint scaling; checkpoints preserved.")
    else:
        run_config = {
            **run_contract,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        run_config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    failures = 0
    for mode in args.modes:
        for prompt_mode in args.prompt_modes:
            tag = f"{mode}_{prompt_mode}"
            path = output_root / f"pred_{tag}.jsonl"
            checkpoint_path = output_root / f"pred_{tag}.checkpoint.jsonl"
            existing = _load_existing(path)
            existing.update(_load_existing(checkpoint_path))
            items = [
                item
                for dataset in datasets
                for item in build_records(args.lmudata, dataset, mode, args.limit)
            ]
            for item in items:
                key = (item["dataset"], item["index"], mode, prompt_mode)
                old = existing.get(key)
                if old and old.get("output") and not old.get("error"):
                    item["output"] = old["output"]
            pending = [item for item in items if not item.get("output")]
            max_tokens = (
                args.max_tokens_cot if prompt_mode == "cot" else args.max_tokens_noncot
            )
            print(f"{tag}: {len(items) - len(pending)}/{len(items)} resumed; {len(pending)} pending")
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            for item in pending:
                queue.put_nowait(item)
            checkpoint_buffer: list[dict[str, Any]] = []
            completed_new = 0

            async def worker(worker_index: int) -> None:
                nonlocal completed_new
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        await _infer_one(
                            clients[worker_index % len(clients)],
                            semaphore,
                            item,
                            args.model,
                            prompt_mode,
                            max_tokens,
                            args.temperature,
                            args.top_p,
                            args.seed,
                            args.request_retries,
                            (
                                args.chat_template_kwargs_cot
                                if prompt_mode == "cot"
                                else args.chat_template_kwargs_noncot
                            ),
                        )
                        checkpoint_buffer.append(
                            prediction_payload(item, mode, prompt_mode)
                        )
                        completed_new += 1
                        if len(checkpoint_buffer) >= args.checkpoint_every:
                            _append_jsonl(checkpoint_path, checkpoint_buffer)
                            checkpoint_buffer.clear()
                            print(
                                f"{tag}: {completed_new}/{len(pending)} new",
                                flush=True,
                            )
                    finally:
                        queue.task_done()

            await asyncio.gather(
                *(
                    worker(worker_index)
                    for worker_index in range(min(args.concurrency, len(pending)))
                )
            )
            if checkpoint_buffer:
                _append_jsonl(checkpoint_path, checkpoint_buffer)
                checkpoint_buffer.clear()
                print(f"{tag}: {completed_new}/{len(pending)} new", flush=True)
            _atomic_jsonl(
                path,
                [prediction_payload(item, mode, prompt_mode) for item in items],
            )
            checkpoint_path.unlink(missing_ok=True)
            failures += sum(not item.get("output") for item in items)
    if failures:
        raise SystemExit(f"Track-3 inference completed with {failures} empty responses.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track-3 v2 spatial inference runner")
    parser.add_argument("--model", required=True, help="Exact served model name")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--endpoints", required=True, help="Comma-separated OpenAI-compatible base URLs")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--lmudata", type=Path, default=Path("LMUData"))
    parser.add_argument("--out", type=Path, default=Path("track3_results"))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--prompt-modes", nargs="+", choices=PROMPT_MODES, default=list(PROMPT_MODES)
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--chat-template-kwargs-noncot",
        type=json.loads,
        default={},
        metavar="JSON",
        help="JSON chat-template arguments sent with non-CoT requests",
    )
    parser.add_argument(
        "--chat-template-kwargs-cot",
        type=json.loads,
        default={},
        metavar="JSON",
        help="JSON chat-template arguments sent with CoT requests",
    )
    parser.add_argument(
        "--server-metadata",
        type=json.loads,
        default={},
        metavar="JSON",
        help="JSON serving provenance recorded in the immutable run contract",
    )
    parser.add_argument("--max-tokens-noncot", type=int, default=16384)
    parser.add_argument("--max-tokens-cot", type=int, default=16384)
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")
    if args.max_tokens_noncot != args.max_tokens_cot:
        parser.error("Track-3 requires equal CoT and non-CoT completion budgets")
    if not isinstance(args.chat_template_kwargs_noncot, dict):
        parser.error("--chat-template-kwargs-noncot must be a JSON object")
    if not isinstance(args.chat_template_kwargs_cot, dict):
        parser.error("--chat-template-kwargs-cot must be a JSON object")
    if not isinstance(args.server_metadata, dict):
        parser.error("--server-metadata must be a JSON object")
    return args


def main() -> None:
    asyncio.run(run_inference(parse_args()))


if __name__ == "__main__":
    main()
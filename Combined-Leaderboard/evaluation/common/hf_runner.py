"""Direct Hugging Face inference runner for one visual benchmark track."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .visual_pipeline import (
    EvaluationPipelineError,
    VisualTrackConfig,
    export_submission,
    image_for_hf,
    load_prompt,
    load_questions,
    write_diagnostics,
)


def _parser(track: VisualTrackConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run {track.label} with transformers and create canonical JSONL."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", type=Path, default=track.questions_path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--prompt-mode", choices=("noncot", "cot"), default="noncot")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Diagnostic partial run; partial runs never create an upload file",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    return parser


def _validate_args(args) -> None:
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise EvaluationPipelineError("Batch size and max-new-tokens must be positive.")
    if args.nshards < 1 or args.shard < 0 or args.shard >= args.nshards:
        raise EvaluationPipelineError("Shard must be in the range 0 <= shard < nshards.")
    if args.limit < 0:
        raise EvaluationPipelineError("--limit cannot be negative.")


def _chat_text(processor, prompt: str, question: str, prompt_mode: str) -> str:
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=prompt_mode == "cot",
        )
    except TypeError:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _run(args, track: VisualTrackConfig) -> tuple[list[dict], list[dict]]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise EvaluationPipelineError(
            "torch and transformers are required. Install evaluation/requirements-hf.txt."
        ) from exc

    questions = load_questions(args.questions, track)
    selected = questions[args.shard :: args.nshards]
    if args.limit:
        selected = selected[: args.limit]
    prompt = load_prompt(track, args.prompt_mode)

    print(f"[{track.task_id}] loading {args.model}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"

    records: list[dict] = []
    started = time.monotonic()
    for start in range(0, len(selected), args.batch_size):
        source_batch = selected[start : start + args.batch_size]
        prepared: list[tuple[dict, str, object]] = []
        for item in source_batch:
            result = dict(item)
            try:
                image = image_for_hf(item, args.image_root)
                text = _chat_text(processor, prompt, item["question"], args.prompt_mode)
                prepared.append((result, text, image))
            except Exception as exc:
                result["output"] = None
                result["error"] = f"{type(exc).__name__}: {exc}"[:500]
                records.append(result)

        if prepared:
            batch_records = [item[0] for item in prepared]
            texts = [item[1] for item in prepared]
            images = [item[2] for item in prepared]
            try:
                inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                prompt_length = inputs["input_ids"].shape[1]
                outputs = processor.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
                for result, output in zip(batch_records, outputs):
                    result["output"] = str(output or "").strip()
                    if not result["output"]:
                        result["error"] = "The model returned an empty response."
                    records.append(result)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"[:500]
                for result in batch_records:
                    result["output"] = None
                    result["error"] = message
                    records.append(result)
            finally:
                for image in images:
                    close = getattr(image, "close", None)
                    if close:
                        close()

        done = min(start + len(source_batch), len(selected))
        rate = done / max(time.monotonic() - started, 0.001)
        print(f"[{track.task_id}] {done}/{len(selected)} ({rate:.2f} samples/s)", flush=True)

    order = {item["question_id"]: index for index, item in enumerate(selected)}
    records.sort(key=lambda item: order[item["question_id"]])
    return questions, records


def main(track: VisualTrackConfig, argv: list[str] | None = None) -> int:
    args = _parser(track).parse_args(argv)
    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.prompt_mode == "cot" else 384
    try:
        _validate_args(args)
        all_questions, records = _run(args, track)
        output_path = args.out or track.default_output_path()
        if args.diagnostics:
            diagnostics_path = args.diagnostics
        elif args.nshards == 1:
            diagnostics_path = output_path.with_name(
                f"{output_path.stem}.{args.prompt_mode}.diagnostics.jsonl"
            )
        else:
            diagnostics_path = track.results_dir / (
                f"{track.task_id}_{args.prompt_mode}_shard_{args.shard}_of_{args.nshards}.diagnostics.jsonl"
            )
        write_diagnostics(diagnostics_path, records)

        if args.limit or args.nshards > 1:
            reason = "partial diagnostic run" if args.limit else "sharded run"
            print(
                f"Completed {reason}. Wrote {len(records)} rows to {diagnostics_path}. "
                "Merge every shard before creating a submission file.",
                flush=True,
            )
            return 0
        report = export_submission(records, all_questions, output_path)
    except (EvaluationPipelineError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

"""Judge Track-3 v2 predictions with the paper's Appendix A.3 prompts."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from spatial_harness.run_track3_vllm import DATASETS, normalize_endpoint


JUDGE_SYS = (
    "You are a helpful assistant.\n\n Your task: given (1) a free-form \"Response\" and (2) a list "
    "of \"Options\", decide which option the response most likely corresponds to and return the option "
    "letter. If no option clearly matches, output \"0\".\n\n Inputs:\n - Response: free-form text "
    "that may include a letter, a phrase, or an explanation.\n - Options: A series of choices, each "
    "starting with a single uppercase letter followed by \".\", one option in each line.\n\n Output "
    "format:\n - STRICTLY OUTPUT EXACTLY ONE CHARACTER: a single uppercase option letter from the "
    "allowed set, or \"0\".\n - Do not output any explanation, spaces, punctuation, or additional "
    "text.\n\n Rules:\n 1) If the response explicitly names exactly one letter (patterns like \"A\", "
    "\"A)\", \"Option A\", \"Answer is C\"), return that letter immediately.\n 2) Only evaluate the "
    "explicitly provided choice. If the response is long and complex without an explicit final choice, "
    "return \"0\".\n 3) If multiple choices appear in the response, the last unambiguous one is the "
    "final choice.\n 4) Never judge factual correctness--only map the response to the best matching "
    "option letter from the given options.\n 5) If no explicit letter can be extracted from the response, "
    "compare the response's meaning to option texts. If exactly one option clearly restates or is a "
    "synonym/number/name/unit match for the response, return its letter.\n 6) If the response uses standard "
    "MCQ phrases such as \"none of the above\" or \"all of the above\" and a matching option exists, "
    "map them. If there is no matching option, output \"0\".\n 7) If the response contains both an "
    "explicit letter and a conflicting phrase, prefer the explicit letter. If conflicts remain or are "
    "unclear, output \"0\".\n 8) If the response says \"I don't know\", \"Cannot determine\", or "
    "similar, output \"0\"."
)

JUDGE_VQA = (
    "You are a helpful assistant.\n\n Task: Given a short free-form \"Response\" and a gold-standard \"Gold\", "
    "decide if the Response expresses the SAME answer as Gold. Output \"1\" for match, \"0\" otherwise.\n\n "
    "Inputs:\n - Gold: the gold-standard answer which is either (i) a short phrase, (ii) an integer, or (iii) "
    "\"Yes\"/\"No\".\n - Response: a few words or a short phrase, possibly will include reasoning steps before "
    "the final answer.\n\n Output format:\n - STRICTLY OUTPUT EXACTLY ONE CHARACTER: \"1\" if matching, \"0\" "
    "if not.\n - Do not output any explanation, spaces, punctuation, or additional text.\n\n Rules:\n 1) Compare "
    "only the final answer in the Response to Gold. Ignore any reasoning steps or intermediate answers present "
    "in the Response.\n 2) If multiple conflicting answers or uncertainty like \"I don't know\" appear in the "
    "Response, output \"0\".\n 3) Do not use external knowledge; judge only based on the text in Gold and "
    "Response.\n 4) Punctuation, grammar, and minor spelling errors should be ignored.\n - uppercase/lowercase "
    "differences should be ignored.\n - hyphen and underscore are ignored. For ex, \"double-bus\" and \"double "
    "bus\" are considered the same.\n - synonyms of \"Yes\"/\"No\" like \"Y\"/\"N\", \"True\"/\"False\" must be "
    "considered the same.\n - word representations of numbers like \"one\"/\"two\"/\"three\" must be considered "
    "the same as \"1\"/\"2\"/\"3\".\n 5) Core concept and critical attributes must match. For example, \"New "
    "York City\" and \"New York State\" do not match. Other examples of non-matches are \"bus\" vs \"double "
    "bus\"; \"red\" vs \"light red\"; \"dog\" vs \"golden retriever\"; \"apple\" vs \"green apple\".\n 6) If the "
    "response says \"I don't know\", \"Cannot determine\", or similar, output \"0\".\n\n Examples:\n - Gold: "
    "Double Bus | Response: This is a bus -> 0\n - Gold: Double Bus | Response: I can see a double-bus -> 1\n - "
    "Gold: Yes | Response: Y -> 1\n - Gold: 10 | Response: ten -> 1\n - Gold: red | Response: light red -> 0\n - "
    "Gold: stop sign | Response: a stop sign on a pole -> 1\n - Gold: person | Response: man -> 0\n\n Now read "
    "the following Gold and Response and output exactly one character: \"1\" or \"0\".\n"
)


def judge_user(item: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {text}" for letter, text in sorted((item.get("options") or {}).items())
    )
    return f"Response:\n{item.get('output') or ''}\n\nOptions:\n{options}"


def judge_user_vqa(item: dict[str, Any]) -> str:
    return f"Gold: {item.get('gt')}\nResponse: {item.get('output') or ''}"


def parse_letter(value: str | None) -> str | None:
    value = (value or "").strip().upper()
    return value[0] if value and value[0] in "0ABCDEFGHIJKLMNOPQRSTUVWXYZ" else None


def parse_bit(value: str | None) -> str | None:
    value = (value or "").strip()
    return value[0] if value and value[0] in "01" else None


async def judge_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
    model: str,
):
    vqa = item.get("answer_type") == "vqa"
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4,
                messages=[
                    {"role": "system", "content": JUDGE_VQA if vqa else JUDGE_SYS},
                    {
                        "role": "user",
                        "content": judge_user_vqa(item) if vqa else judge_user(item),
                    },
                ],
            )
            content = response.choices[0].message.content
            item["judged"] = parse_bit(content) if vqa else parse_letter(content)
            item.pop("jerr", None)
        except Exception as exc:  # noqa: BLE001 - preserve judge failure
            item["judged"] = None
            item["jerr"] = f"{type(exc).__name__}: {exc}"[:500]
    return item


def correct(item: dict[str, Any]) -> bool | None:
    judged = item.get("judged")
    if judged is None:
        return None
    if item.get("answer_type") == "vqa":
        return judged == "1"
    if item["mode"] == "noimgpp":
        return judged == str(item.get("cannot_label") or "").upper()
    return judged == str(item["gt"]).upper()


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[bool | None]] = collections.defaultdict(list)
    for item in items:
        key = (
            item["dataset"],
            item["mode"],
            item["pmode"],
            str(item.get("group", item["index"])),
        )
        groups[key].append(correct(item))

    per_slice: dict[tuple[str, str, str], dict[str, int | float]] = {}
    for (dataset, mode, prompt_mode, _group), values in groups.items():
        key = (dataset, mode, prompt_mode)
        stats = per_slice.setdefault(key, {"correct": 0, "total": 0, "unresolved": 0})
        if any(value is None for value in values):
            stats["unresolved"] += 1
            continue
        stats["total"] += 1
        stats["correct"] += int(all(values))

    datasets: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        result: dict[str, Any] = {}
        for mode, prefix in (("main", "main"), ("noimgpp", "npp")):
            for prompt_mode in ("noncot", "cot"):
                metric = f"{prefix}_{prompt_mode}"
                stats = per_slice.get((dataset, mode, prompt_mode), {})
                total = int(stats.get("total", 0))
                result[metric] = (
                    int(stats.get("correct", 0)) / total if total else None
                )
                result[f"{metric}_n"] = total
                result[f"{metric}_unresolved"] = int(stats.get("unresolved", 0))
        if result["main_noncot"] is not None and result["main_cot"] is not None:
            result["main_delta"] = result["main_cot"] - result["main_noncot"]
        else:
            result["main_delta"] = None
        datasets[dataset] = result

    def macro(metric: str) -> float | None:
        values = [result[metric] for result in datasets.values() if result[metric] is not None]
        return sum(values) / len(values) if values else None

    return {
        "schema_version": 2,
        "datasets": datasets,
        "macro": {
            metric: macro(metric)
            for metric in ("main_noncot", "main_cot", "main_delta", "npp_noncot", "npp_cot")
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _item_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (item["dataset"], item["index"], item["mode"], item["pmode"])


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def run_judge(args: argparse.Namespace) -> None:
    input_root = Path(args.input)
    prediction_files = sorted(input_root.glob("pred_*.jsonl"))
    if not prediction_files:
        raise SystemExit(f"No pred_*.jsonl files found under {input_root}.")
    items = [item for path in prediction_files for item in _read_jsonl(path)]
    existing = {
        _item_key(item): item
        for item in (_read_jsonl(args.judged) if args.judged.is_file() else [])
    }
    for item in items:
        old = existing.get(_item_key(item))
        if old and old.get("judged") is not None:
            item["judged"] = old["judged"]
    endpoint = normalize_endpoint(args.endpoint)
    client = AsyncOpenAI(base_url=endpoint, api_key=args.api_key, timeout=args.timeout)
    served = {entry.id for entry in (await client.models.list()).data}
    if args.model not in served:
        raise SystemExit(f"Judge endpoint serves {sorted(served)}, not {args.model}.")
    semaphore = asyncio.Semaphore(args.concurrency)
    pending = [item for item in items if item.get("judged") is None]
    print(f"Judge: {len(items) - len(pending)}/{len(items)} resumed; {len(pending)} pending")
    for offset in range(0, len(pending), args.checkpoint_every):
        batch = pending[offset : offset + args.checkpoint_every]
        await asyncio.gather(
            *(judge_one(client, semaphore, item, args.model) for item in batch)
        )
        _atomic_jsonl(args.judged, items)
        print(f"Judge: {min(offset + len(batch), len(pending))}/{len(pending)} new")
    leaderboard = aggregate(items)
    leaderboard["judge"] = {"model": args.model, "endpoint": endpoint, "temperature": 0}
    _atomic_json(args.leaderboard, leaderboard)
    unresolved = sum(item.get("judged") is None for item in items)
    if unresolved:
        raise SystemExit(f"Judge completed with {unresolved} unresolved records.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track-3 v2 paper-faithful judge")
    parser.add_argument("--input", type=Path, default=Path("track3_results"))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judged", type=Path)
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    args.judged = args.judged or args.input / "judged.jsonl"
    args.leaderboard = args.leaderboard or args.input / "leaderboard.json"
    if args.concurrency < 1 or args.checkpoint_every < 1:
        parser.error("concurrency and checkpoint-every must be positive")
    return args


def main() -> None:
    asyncio.run(run_judge(parse_args()))


if __name__ == "__main__":
    main()
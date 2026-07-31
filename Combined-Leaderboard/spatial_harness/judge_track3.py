"""Judge standardized Track-3 predictions with the paper's prompts."""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from spatial_harness.run_track3_vllm import (
    DATASETS,
    HARNESS_CONTRACT,
    MODES,
    PROMPT_MODES,
    normalize_endpoint,
)
from spatial_harness.submission_contract import (
    INFERENCE_FAILURE_DISPOSITION,
    INFERENCE_FAILURE_JUDGE_METHOD,
    INFERENCE_FAILURE_POLICY,
    INFERENCE_FAILURE_SCHEMA_VERSION,
)


PAPER_JUDGE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PAPER_JUDGE_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
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

ABSTENTION_PHRASES = {
    "cannot determine",
    "cannot determine from the image",
    "can not determine",
    "can not determine from the image",
    "cannot tell",
    "cannot tell from the image",
    "i cannot determine",
    "i cannot determine from the image",
    "i cannot tell",
    "i cannot tell from the image",
    "it cannot be determined",
    "it cannot be determined from the image",
    "insufficient information",
    "insufficient visual information",
    "there is insufficient information",
    "not enough information",
    "not enough visual information",
    "there is not enough information",
    "unknown",
    "unanswerable",
}

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
    return value if re.fullmatch(r"[0A-Z]", value) else None


def parse_bit(value: str | None) -> str | None:
    value = (value or "").strip()
    return value if re.fullmatch(r"[01]", value) else None


def _normalized_answer_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(
        r"^\s*(final\s+answer|answer|option|choice)\s*(?:is|=)?\s*[:\-]?\s*",
        "",
        text,
    )
    text = text.strip().strip("<>()[]{}.,:;!?\"'")
    return re.sub(r"\s+", " ", text)


def explicit_abstention_letter(item: dict[str, Any]) -> str | None:
    """Map a clear No-Image++ abstention to its injected option letter."""
    if item.get("mode") != "noimgpp" or not item.get("cannot_label"):
        return None
    output = str(item.get("output") or "").strip()
    answer_tags = re.findall(
        r"<answer>(.*?)</answer>",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = answer_tags[-1] if answer_tags else output if len(output) <= 240 else ""
    return (
        str(item["cannot_label"]).upper()
        if _normalized_answer_text(candidate) in ABSTENTION_PHRASES
        else None
    )


def is_terminal_inference_failure(item: dict[str, Any]) -> bool:
    marker = item.get("terminal_failure")
    if not isinstance(marker, dict):
        return False
    try:
        input_length = int(marker.get("input_length") or 0)
        max_model_len = int(marker.get("max_model_len") or 0)
    except (TypeError, ValueError):
        return False
    return (
        marker.get("schema_version") == INFERENCE_FAILURE_SCHEMA_VERSION
        and marker.get("policy") == INFERENCE_FAILURE_POLICY
        and marker.get("category") == "input_context_exceeded"
        and marker.get("disposition") == INFERENCE_FAILURE_DISPOSITION
        and input_length > max_model_len > 0
        and not str(item.get("output") or "").strip()
        and bool(str(item.get("error") or "").strip())
        and item.get("finish_reason") == "inference_error"
    )


async def judge_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
    model: str,
    retries: int,
):
    if is_terminal_inference_failure(item):
        item["judged"] = "0"
        item["judge_method"] = INFERENCE_FAILURE_JUDGE_METHOD
        item["judge_attempts"] = 0
        item.pop("jerr", None)
        return item
    vqa = item.get("answer_type") == "vqa"
    abstention = explicit_abstention_letter(item)
    if abstention:
        item["judged"] = abstention
        item["judge_method"] = "explicit_abstention"
        item["judge_attempts"] = 0
        item.pop("jerr", None)
        return item
    last_error = ""
    for attempt in range(retries + 1):
        try:
            allowed = (
                {"0", "1"}
                if vqa
                else set((item.get("options") or {}).keys()) | {"0"}
            )
            async with semaphore:
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
                    extra_body={
                        "structured_outputs": {"choice": sorted(allowed)}
                    },
                )
            content = response.choices[0].message.content
            judged = parse_bit(content) if vqa else parse_letter(content)
            if judged is None or (not vqa and judged not in allowed):
                raise RuntimeError(f"judge returned invalid value: {content!r}")
            item["judged"] = judged
            item["judge_method"] = "paper_llm_judge"
            item["judge_attempts"] = attempt + 1
            item.pop("jerr", None)
            return item
        except Exception as exc:  # noqa: BLE001 - preserve judge failure
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt < retries:
                await asyncio.sleep(min(2**attempt, 8))
    item["judged"] = None
    item["judge_method"] = "paper_llm_judge"
    item["judge_attempts"] = retries + 1
    item["jerr"] = last_error
    return item


def correct(item: dict[str, Any]) -> bool | None:
    if item.get("judge_method") == INFERENCE_FAILURE_JUDGE_METHOD:
        return False
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
        for mode in ("main", "noimage", "noimgpp"):
            for prompt_mode in ("noncot", "cot"):
                metric = f"{mode}_{prompt_mode}"
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
        "schema_version": 5,
        "datasets": datasets,
        "macro": {
            metric: macro(metric)
            for metric in (
                "main_noncot",
                "main_cot",
                "main_delta",
                "noimage_noncot",
                "noimage_cot",
                "noimgpp_noncot",
                "noimgpp_cot",
            )
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
    expected_names = {
        f"pred_{mode}_{prompt_mode}.jsonl"
        for mode in MODES
        for prompt_mode in PROMPT_MODES
    }
    actual_names = {path.name for path in input_root.glob("pred_*.jsonl")}
    if actual_names != expected_names:
        missing = ", ".join(sorted(expected_names - actual_names)) or "none"
        extra = ", ".join(sorted(actual_names - expected_names)) or "none"
        raise SystemExit(
            f"Track-3 prediction set is incomplete or mixed; missing: {missing}; extra: {extra}."
        )
    config_path = input_root / "run_config.json"
    if not config_path.is_file():
        raise SystemExit("Track-3 run_config.json is missing.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 5
        or config.get("harness_contract") != HARNESS_CONTRACT
        or config.get("modes") != list(MODES)
        or config.get("prompt_modes") != list(PROMPT_MODES)
    ):
        raise SystemExit("Track-3 run_config.json does not match the v5 six-condition contract.")
    prediction_files = [input_root / name for name in sorted(expected_names)]
    items = [item for path in prediction_files for item in _read_jsonl(path)]
    unexpected = [
        item
        for item in items
        if item.get("answer_type") not in {"mcq", "vqa"}
        or (item.get("answer_type") == "vqa" and item.get("mode") == "noimgpp")
        or item.get("mode") not in {"main", "noimage", "noimgpp"}
        or item.get("pmode") not in {"noncot", "cot"}
    ]
    if unexpected:
        raise SystemExit(
            "Prediction set violates the standardized mixed MCQ/VQA six-condition contract."
        )
    invalid_failures = []
    for item in items:
        terminal = item.get("terminal_failure") is not None
        has_inference_failure = bool(item.get("error")) or not str(
            item.get("output") or ""
        ).strip()
        if terminal and not is_terminal_inference_failure(item):
            invalid_failures.append((*_item_key(item), "invalid terminal marker"))
        elif has_inference_failure and not is_terminal_inference_failure(item):
            invalid_failures.append(
                (*_item_key(item), "unfinalized inference failure")
            )
    if invalid_failures:
        raise SystemExit(
            "Prediction set contains unapproved inference failures; "
            f"examples={invalid_failures[:5]}"
        )
    item_keys = [_item_key(item) for item in items]
    if len(item_keys) != len(set(item_keys)):
        raise SystemExit("Track-3 prediction set contains duplicate sample-condition keys.")
    existing = {
        _item_key(item): item
        for item in (_read_jsonl(args.judged) if args.judged.is_file() else [])
    }
    for item in items:
        old = existing.get(_item_key(item))
        if old and old.get("judged") is not None:
            for field in ("judged", "judge_method", "judge_attempts"):
                if old.get(field) is not None:
                    item[field] = old[field]
            item.setdefault("judge_method", "paper_llm_judge")
            item.setdefault("judge_attempts", 1)
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
            *(
                judge_one(
                    client,
                    semaphore,
                    item,
                    args.model,
                    args.request_retries,
                )
                for item in batch
            )
        )
        _atomic_jsonl(args.judged, items)
        print(f"Judge: {min(offset + len(batch), len(pending))}/{len(pending)} new")
    leaderboard = aggregate(items)
    leaderboard["judge"] = {
        "model": args.model,
        "model_revision": args.model_revision,
        "endpoint": endpoint,
        "temperature": 0,
        "max_tokens": 4,
        "decoding_constraint": "structured_choice",
        "method_counts": dict(
            collections.Counter(
                str(item.get("judge_method") or "")
                for item in items
                if item.get("judged") is not None
            )
        ),
        "attempt_count": sum(
            int(item.get("judge_attempts") or 0)
            for item in items
            if item.get("judged") is not None
        ),
        "terminal_inference_failures": sum(
            is_terminal_inference_failure(item) for item in items
        ),
    }
    if args.server_max_model_len is not None:
        leaderboard["judge"]["server_max_model_len"] = args.server_max_model_len
    _atomic_json(args.leaderboard, leaderboard)
    unresolved = sum(item.get("judged") is None for item in items)
    if unresolved:
        raise SystemExit(f"Judge completed with {unresolved} unresolved records.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track-3 v2 paper-faithful judge")
    parser.add_argument("--input", type=Path, default=Path("track3_results"))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default=PAPER_JUDGE_MODEL)
    parser.add_argument("--model-revision", default=PAPER_JUDGE_REVISION)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judged", type=Path)
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--server-max-model-len", type=int)
    args = parser.parse_args(argv)
    args.judged = args.judged or args.input / "judged.jsonl"
    args.leaderboard = args.leaderboard or args.input / "leaderboard.json"
    if args.concurrency < 1 or args.checkpoint_every < 1 or args.request_retries < 0:
        parser.error(
            "concurrency and checkpoint-every must be positive; request-retries cannot be negative"
        )
    if args.model != PAPER_JUDGE_MODEL:
        parser.error(
            f"Track-3 requires the fixed paper judge {PAPER_JUDGE_MODEL!r}"
        )
    if args.model_revision != PAPER_JUDGE_REVISION:
        parser.error(
            f"Track-3 requires judge revision {PAPER_JUDGE_REVISION}"
        )
    return args


def main() -> None:
    asyncio.run(run_judge(parse_args()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only review of external v2 Track 3 bundles; never publishes or executes ZIP code.

Reproducing submitter-claimed arithmetic is not independent answer validation.
Only exact public IDs, conditions, types and groups can establish compatibility;
numeric indices are never guessed or substituted for release identifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import stat
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

COMBOS = ("main_noncot", "main_cot", "noimg_noncot", "noimg_cot", "noimgpp_noncot", "noimgpp_cot")
MAIN_COUNTS = dict(zip(
    ("3DSRBench", "BLINK", "CV-Bench-2D", "CV-Bench-3D", "MMSIBench_wo_circular", "MMVP", "MindCube", "OmniSpatial", "RealWorldQA", "SAT-Real", "SpatialBench", "VSR_MCQ", "VStarBench"),
    (3997, 1901, 1438, 1200, 1000, 300, 1050, 1533, 765, 300, 473, 1222, 191),
))
PLUS_COUNTS = {**MAIN_COUNTS, "OmniSpatial": 1304, "SAT-Real": 150, "SpatialBench": 154}
META_KEYS = ("dataset", "index", "group", "answer_type", "mode", "pmode")


def check_archive(archive: zipfile.ZipFile) -> None:
    entries = archive.infolist()
    names = [entry.filename for entry in entries]
    if len(entries) > 200 or len(names) != len(set(names)):
        raise ValueError("Too many ZIP entries or duplicate member names.")
    if sum(entry.file_size for entry in entries) > 256 * 1024 ** 2:
        raise ValueError("ZIP exceeds the 256 MiB expanded review limit.")
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError("ZIP contains an unsafe path or symlink.")
        if entry.flag_bits & 1 or entry.file_size > 64 * 1024 ** 2:
            raise ValueError("ZIP contains an encrypted or oversized member.")


def load_rows(archive: zipfile.ZipFile, name: str, combo: str, *, verdict: bool) -> dict:
    mode, pmode = combo.split("_", 1)
    required = set(META_KEYS) | ({"gt", "cannot_label", "judged"} if verdict else {"output", "options"})
    indexed = {}
    with archive.open(name) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError("A row has missing required fields.")
            if any(not isinstance(row[key], str) or not row[key] for key in META_KEYS):
                raise ValueError("A row has invalid identifying metadata.")
            if (row["mode"], row["pmode"]) != (mode, pmode) or row["answer_type"] not in {"mcq", "vqa"}:
                raise ValueError("Condition or answer type does not match its file.")
            if verdict:
                if not isinstance(row["judged"], str) or not row["judged"].strip():
                    raise ValueError("A verdict is missing or empty.")
            elif {"gt", "cannot_label"} & row.keys() or (row["output"] is not None and not isinstance(row["output"], str)) or not isinstance(row["options"], dict):
                raise ValueError("Prediction contains answer keys or invalid output/options types.")
            key = (row["dataset"], row["index"])
            if key in indexed:
                raise ValueError("Duplicate (dataset, index) row.")
            indexed[key] = row
    return indexed


def claimed_credit(row: dict) -> bool:
    if row["answer_type"] == "vqa":
        return row["judged"] == "1"
    expected = row["cannot_label"] if row["mode"] == "noimgpp" else row["gt"]
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("MCQ verdict has no target label.")
    return row["judged"].upper() == expected.upper()


def review_model(archive: zipfile.ZipFile, base: str, questions: dict) -> dict:
    metadata = json.loads(archive.read(base + "metadata.json"))
    # Explicit whitelist: metadata includes a private contact email.
    report = {"model": {key: metadata.get(key) for key in ("model_display_name", "model_id", "model_type", "kit_version", "judge_model", "date")}, "conditions": {}, "errors": []}
    scores = list(csv.DictReader(io.StringIO(archive.read(base + "scores.csv").decode("utf-8-sig"))))
    if len(scores) != 14 or {row["dataset"] for row in scores} != set(MAIN_COUNTS) | {"AVERAGE"}:
        raise ValueError("scores.csv must contain each of 13 datasets and AVERAGE exactly once.")
    by_dataset = {row["dataset"]: row for row in scores}
    computed = {}
    for combo in COMBOS:
        pred = load_rows(archive, base + f"predictions/pred_{combo}.jsonl", combo, verdict=False)
        verdicts = load_rows(archive, base + f"verdicts/judged_{combo}.jsonl", combo, verdict=True)
        if pred.keys() != verdicts.keys():
            raise ValueError("Prediction and verdict key sets differ.")
        counts = dict(Counter(row["dataset"] for row in pred.values()))
        if counts != (PLUS_COUNTS if combo.startswith("noimgpp_") else MAIN_COUNTS):
            raise ValueError("Row counts differ from the supplied v2 format's declared scope.")
        groups = defaultdict(list)
        empty_wins = 0
        for key, answer in pred.items():
            verdict = verdicts[key]
            if any(answer[field] != verdict[field] for field in META_KEYS):
                raise ValueError("Paired prediction/verdict metadata differ.")
            credit = claimed_credit(verdict)
            empty_wins += int(credit and not (answer["output"] or "").strip())
            groups[(answer["dataset"], answer["group"])].append(credit)
        totals = defaultdict(lambda: [0, 0])
        for (dataset, _group), credits in groups.items():
            totals[dataset][0] += int(all(credits))
            totals[dataset][1] += 1
        column = combo.replace("noimgpp", "npp")
        computed[column] = {dataset: round(100 * correct / total, 2) for dataset, (correct, total) in totals.items()}
        condition = combo.replace("noimgpp", "no_image_plus").replace("noimg", "no_image")
        expected = {key for key, row in questions.items() if condition in row["conditions"]}
        actual = {dataset + ":" + index: row for (dataset, index), row in pred.items()}
        overlap = expected & actual.keys()
        report["conditions"][condition] = {
            "rows": len(pred), "dataset_rows": counts,
            "expected_release_rows": len(expected), "matching_public_ids": len(overlap),
            "missing_public_ids": len(expected - actual.keys()), "unknown_public_ids": len(actual.keys() - expected),
            "answer_type_mismatches": sum(questions[key]["answer_type"] != actual[key]["answer_type"] for key in overlap),
            "group_mismatches": sum(questions[key]["evaluation_group"] != actual[key]["dataset"] + ":" + actual[key]["group"] for key in overlap),
            "empty_outputs_with_claimed_credit": empty_wins,
            "null_model_outputs": sum(row["output"] is None for row in pred.values()),
        }
        if empty_wins:
            report["errors"].append(f"{combo}: {empty_wins} empty outputs receive claimed credit.")
        if report["conditions"][condition]["null_model_outputs"]:
            report["errors"].append(f"{combo}: model outputs are missing; an explicit inference-failure record is required.")
    mismatches = []
    for column, values in computed.items():
        for dataset, value in {**values, "AVERAGE": round(sum(values.values()) / len(values), 2)}.items():
            supplied = float(by_dataset[dataset][column])
            if not math.isfinite(supplied) or abs(supplied - value) > 0.0200001:
                mismatches.append(f"{dataset}/{column}")
    for dataset, row in by_dataset.items():
        delta = float(row["main_delta"])
        if not math.isfinite(delta) or abs(delta - (float(row["main_cot"]) - float(row["main_noncot"]))) > 0.0200001:
            mismatches.append(f"{dataset}/main_delta")
    report["arithmetic_mismatches"] = mismatches
    report["submitted_arithmetic_consistent"] = not mismatches
    report["compatible_with_live_contract"] = all(
        not row[key] for row in report["conditions"].values()
        for key in ("missing_public_ids", "unknown_public_ids", "answer_type_mismatches", "group_mismatches")
    )
    report["eligible_for_canonical_repackaging"] = report["compatible_with_live_contract"] and report["submitted_arithmetic_consistent"] and not report["errors"]
    return report


def review(path: Path, contract: Path) -> dict:
    question_bytes = (contract / "questions.jsonl").read_bytes()
    questions = {row["question_id"]: row for row in map(json.loads, question_bytes.splitlines())}
    manifest_bytes = (contract / "manifest.json").read_bytes()
    if hashlib.sha256(question_bytes).hexdigest() != json.loads(manifest_bytes)["artifacts"]["questions"]["sha256"]:
        raise ValueError("Public question file does not match the installed manifest hash.")
    with zipfile.ZipFile(path) as archive:
        check_archive(archive)
        bases = sorted(name.removesuffix("metadata.json") for name in archive.namelist() if name.endswith("__track3/metadata.json"))
        if not bases:
            raise ValueError("No external v2 model directories found.")
        reports = [review_model(archive, base, questions) for base in bases]
    with path.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "source_sha256": source_sha256,
        "benchmark_version": json.loads(manifest_bytes)["benchmark_version"],
        "benchmark_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "publication_performed": False,
        "verification": "Submitted verdict arithmetic and exact public-contract compatibility only; no semantic reevaluation or judge-provenance attestation.",
        "models": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--contract-dir", type=Path, default=Path(__file__).resolve().parents[1] / "tasks/spatial")
    args = parser.parse_args()
    result = review(args.archive, args.contract_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(model["eligible_for_canonical_repackaging"] for model in result["models"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

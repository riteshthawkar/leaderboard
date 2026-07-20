#!/usr/bin/env python3
"""Report progress for a running Track-3 inference directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_harness.run_track3_vllm import build_records


SLICES = ("main_noncot", "main_cot", "noimgpp_noncot", "noimgpp_cot")


def keyed_rows(path: Path) -> dict[tuple[str, str, str, str], dict]:
    rows = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row.get("dataset")),
            str(row.get("index")),
            str(row.get("mode")),
            str(row.get("pmode")),
        )
        rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    config_path = result_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        f"{mode}_{prompt_mode}": sum(
            len(
                build_records(
                    config["lmudata"],
                    dataset,
                    mode,
                    int(config.get("limit") or 0),
                )
            )
            for dataset in config["datasets"]
        )
        for mode in config["modes"]
        for prompt_mode in config["prompt_modes"]
    }
    grand_done = 0
    grand_total = 0
    for name in SLICES:
        final_path = result_dir / f"pred_{name}.jsonl"
        checkpoint_path = result_dir / f"pred_{name}.checkpoint.jsonl"
        final_rows = keyed_rows(final_path)
        checkpoint_rows = keyed_rows(checkpoint_path)
        merged = {**final_rows, **checkpoint_rows}
        done = sum(bool(row.get("output")) and not row.get("error") for row in merged.values())
        errors = sum(bool(row.get("error")) for row in merged.values())
        length_stops = sum(row.get("finish_reason") == "length" for row in merged.values())
        total = expected.get(name, len(final_rows))
        state = "complete" if final_rows and not checkpoint_rows else "running" if checkpoint_rows else "waiting"
        print(
            f"{name:<17} {done:>6}/{total:<6} errors={errors:<4} "
            f"length_stops={length_stops:<5} {state}"
        )
        grand_done += done
        grand_total += total
    print(f"total             {grand_done:>6}/{grand_total:<6}")


if __name__ == "__main__":
    main()
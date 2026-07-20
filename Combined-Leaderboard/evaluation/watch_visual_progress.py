#!/usr/bin/env python3
"""Display read-only progress bars for a visual-suite model run."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path


TRACKS = (("do_you_see_me", 4500), ("minds_eye", 799))


def row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def progress_bar(done: int, total: int, width: int = 36) -> str:
    fraction = min(max(done / total, 0.0), 1.0) if total else 0.0
    filled = int(width * fraction)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done:>4}/{total} {fraction:6.2%}"


def track_state(model_dir: Path, track: str, total: int) -> tuple[str, int, Path | None]:
    submission = model_dir / f"{track}_submission.jsonl"
    diagnostics = model_dir / f"{track}.diagnostics.jsonl"
    smoke = model_dir / f"{track}.smoke.diagnostics.jsonl"
    submission_rows = row_count(submission)
    if submission_rows:
        return ("complete" if submission_rows == total else "submission", submission_rows, submission)
    diagnostics_rows = row_count(diagnostics)
    if diagnostics_rows:
        return "full evaluation", diagnostics_rows, diagnostics
    smoke_rows = row_count(smoke)
    if smoke_rows:
        return "smoke", smoke_rows, smoke
    return "waiting", 0, None


def render(
    model_dir: Path,
    previous: dict[str, tuple[str, int]] | None,
    elapsed: float,
) -> dict[str, tuple[str, int]]:
    now = time.time()
    print(f"Visual-suite progress: {model_dir.name}")
    current = {}
    for track, total in TRACKS:
        state, done, source = track_state(model_dir, track, total)
        current[track] = (state, done)
        display_total = 20 if state == "smoke" else total
        prior_state, prior_done = previous.get(track, ("", 0)) if previous else ("", 0)
        rate = (
            (done - prior_done) / elapsed
            if state == prior_state == "full evaluation" and done > prior_done
            else 0.0
        )
        eta = timedelta(seconds=int((total - done) / rate)) if rate else None
        suffix = f" | {state}"
        if rate:
            suffix += f" | {rate:.2f} samples/s | ETA {eta}"
        if source:
            suffix += f" | {source.name}"
        print(f"  {track:<14} {progress_bar(done, display_total)}{suffix}")
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="internvl35-8b")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "visual_suite_bf16",
    )
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    args = parser.parse_args()
    if args.watch is not None and args.watch <= 0:
        parser.error("--watch must be positive")

    model_dir = args.output_root.expanduser().resolve() / args.model
    previous = None
    previous_at = time.monotonic()
    while True:
        if args.watch is not None:
            print("\033[2J\033[H", end="")
        now = time.monotonic()
        previous = render(model_dir, previous, max(now - previous_at, 0.001))
        previous_at = now
        if args.watch is None:
            return 0
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

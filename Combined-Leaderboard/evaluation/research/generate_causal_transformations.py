#!/usr/bin/env python3
"""Generate paired causal and nuisance visual-transformation controls."""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.research.common import (  # noqa: E402
    build_manifest,
    stable_seed,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/research/results/causal_transformations"
OPERATIONS = ("rotate_cw", "rotate_180", "mirror_vertical", "mirror_horizontal")
GRID_SIZE = 5
CANVAS_SIZE = (1400, 920)
QUESTION = (
    "Infer what each abstract operator does from the two demonstrations. "
    "Apply the operator in the query and select the matching option. "
    "Answer with exactly one option letter: A, B, C, or D."
)

PALETTES = {
    "base": {
        "background": "#ffffff",
        "ink": "#151515",
        "grid": "#d9d9d9",
        "cell": "#5f6bff",
        "anchor": "#f05a47",
        "panel": "#f8f8f8",
        "line_width": 3,
    },
    "nuisance": {
        "background": "#f7fbfc",
        "ink": "#20272b",
        "grid": "#c4d4d8",
        "cell": "#248f8d",
        "anchor": "#b74b8b",
        "panel": "#ffffff",
        "line_width": 5,
    },
}


def transform_cell(x: int, y: int, operation: str) -> tuple[int, int]:
    last = GRID_SIZE - 1
    if operation == "rotate_cw":
        return last - y, x
    if operation == "rotate_180":
        return last - x, last - y
    if operation == "mirror_vertical":
        return last - x, y
    if operation == "mirror_horizontal":
        return x, last - y
    raise ValueError(f"Unsupported operation: {operation}")


def transform_motif(
    motif: tuple[tuple[int, int, bool], ...], operation: str
) -> tuple[tuple[int, int, bool], ...]:
    return tuple(
        sorted(
            (*transform_cell(x, y, operation), is_anchor)
            for x, y, is_anchor in motif
        )
    )


def random_motif(rng: random.Random) -> tuple[tuple[int, int, bool], ...]:
    while True:
        count = rng.randint(4, 7)
        coordinates = rng.sample(list(itertools.product(range(5), repeat=2)), count)
        anchor_index = rng.randrange(count)
        motif = tuple(
            sorted(
                (x, y, index == anchor_index)
                for index, (x, y) in enumerate(coordinates)
            )
        )
        variants = {transform_motif(motif, operation) for operation in OPERATIONS}
        if len(variants) == len(OPERATIONS):
            return motif


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_motif(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    motif: tuple[tuple[int, int, bool], ...],
    palette: dict[str, Any],
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, fill=palette["panel"], outline=palette["ink"], width=2)
    padding = 14
    cell_size = min(
        (right - left - 2 * padding) // GRID_SIZE,
        (bottom - top - 2 * padding) // GRID_SIZE,
    )
    width = cell_size * GRID_SIZE
    height = cell_size * GRID_SIZE
    grid_left = left + (right - left - width) // 2
    grid_top = top + (bottom - top - height) // 2
    for step in range(GRID_SIZE + 1):
        x = grid_left + step * cell_size
        y = grid_top + step * cell_size
        draw.line(
            (x, grid_top, x, grid_top + height),
            fill=palette["grid"],
            width=1,
        )
        draw.line(
            (grid_left, y, grid_left + width, y),
            fill=palette["grid"],
            width=1,
        )
    inset = max(4, cell_size // 7)
    for x, y, is_anchor in motif:
        cell_box = (
            grid_left + x * cell_size + inset,
            grid_top + y * cell_size + inset,
            grid_left + (x + 1) * cell_size - inset,
            grid_top + (y + 1) * cell_size - inset,
        )
        fill = palette["anchor"] if is_anchor else palette["cell"]
        if is_anchor:
            draw.ellipse(
                cell_box,
                fill=fill,
                outline=palette["ink"],
                width=palette["line_width"],
            )
        else:
            draw.rectangle(
                cell_box,
                fill=fill,
                outline=palette["ink"],
                width=palette["line_width"],
            )


def draw_operator(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    operator: int,
    palette: dict[str, Any],
) -> None:
    x, y = center
    radius = 34
    width = palette["line_width"]
    if operator == 0:
        points = ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y))
        draw.polygon(points, outline=palette["ink"], width=width)
        draw.ellipse(
            (x - 7, y - 7, x + 7, y + 7),
            fill=palette["anchor"],
            outline=palette["ink"],
            width=max(1, width - 1),
        )
    else:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=palette["ink"],
            width=width,
        )
        draw.line((x - 22, y, x + 22, y), fill=palette["ink"], width=width)
        draw.line((x, y - 22, x, y + 22), fill=palette["ink"], width=width)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    palette: dict[str, Any],
) -> None:
    draw.line((*start, *end), fill=palette["ink"], width=3)
    x, y = end
    draw.polygon(
        ((x, y), (x - 14, y - 8), (x - 14, y + 8)),
        fill=palette["ink"],
    )


def render_problem(
    *,
    destination: Path,
    demo_a: tuple[tuple[int, int, bool], ...],
    demo_b: tuple[tuple[int, int, bool], ...],
    query: tuple[tuple[int, int, bool], ...],
    operation_a: str,
    operation_b: str,
    query_operator: int,
    option_operations: list[str],
    palette_name: str,
) -> None:
    palette = PALETTES[palette_name]
    image = Image.new("RGB", CANVAS_SIZE, palette["background"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    label_font = load_font(26)
    small_font = load_font(21)
    draw.text((55, 28), "Infer the operators, then complete the query", fill=palette["ink"], font=title_font)

    panel_width = 178
    panel_height = 178
    left = 80
    result_left = 510
    row_tops = (90, 292, 494)
    demonstrations = (
        (demo_a, transform_motif(demo_a, operation_a), 0, "Demonstration 1"),
        (demo_b, transform_motif(demo_b, operation_b), 1, "Demonstration 2"),
    )
    for row_index, (source, target, operator, label) in enumerate(demonstrations):
        top = row_tops[row_index]
        draw.text((820, top + 70), label, fill=palette["ink"], font=small_font)
        draw_motif(draw, (left, top, left + panel_width, top + panel_height), source, palette)
        draw_operator(draw, (347, top + panel_height // 2), operator, palette)
        draw_arrow(draw, (282, top + panel_height // 2), (305, top + panel_height // 2), palette)
        draw_arrow(draw, (389, top + panel_height // 2), (475, top + panel_height // 2), palette)
        draw_motif(
            draw,
            (result_left, top, result_left + panel_width, top + panel_height),
            target,
            palette,
        )

    query_top = row_tops[2]
    draw.text((820, query_top + 70), "Query", fill=palette["ink"], font=small_font)
    draw_motif(
        draw,
        (left, query_top, left + panel_width, query_top + panel_height),
        query,
        palette,
    )
    draw_operator(
        draw,
        (347, query_top + panel_height // 2),
        query_operator,
        palette,
    )
    draw_arrow(draw, (282, query_top + panel_height // 2), (305, query_top + panel_height // 2), palette)
    draw_arrow(draw, (389, query_top + panel_height // 2), (475, query_top + panel_height // 2), palette)
    question_box = (
        result_left,
        query_top,
        result_left + panel_width,
        query_top + panel_height,
    )
    draw.rectangle(question_box, fill=palette["panel"], outline=palette["ink"], width=2)
    draw.text((575, query_top + 63), "?", fill=palette["ink"], font=load_font(48))

    option_top = 724
    option_width = 220
    option_height = 160
    option_gap = 76
    for option_index, operation in enumerate(option_operations):
        option_left = 80 + option_index * (option_width + option_gap)
        draw.text(
            (option_left + 96, option_top - 34),
            chr(65 + option_index),
            fill=palette["ink"],
            font=label_font,
        )
        draw_motif(
            draw,
            (
                option_left,
                option_top,
                option_left + option_width,
                option_top + option_height,
            ),
            transform_motif(query, operation),
            palette,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def build_bundle(output: Path, pairs: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for pair_index in range(pairs):
        pair_id = f"ct_{pair_index:04d}"
        operation_a, operation_b = rng.sample(list(OPERATIONS), 2)
        demo_a = random_motif(rng)
        demo_b = random_motif(rng)
        query = random_motif(rng)
        option_operations = list(OPERATIONS)
        rng.shuffle(option_operations)
        variants = (
            ("base", 0, "base"),
            ("causal", 1, "base"),
            ("nuisance", 0, "nuisance"),
        )
        variant_ids: dict[str, str] = {}
        for variant, query_operator, palette_name in variants:
            question_id = f"{pair_id}_{variant}"
            image_relative = Path("images") / f"{question_id}.png"
            render_problem(
                destination=output / image_relative,
                demo_a=demo_a,
                demo_b=demo_b,
                query=query,
                operation_a=operation_a,
                operation_b=operation_b,
                query_operator=query_operator,
                option_operations=option_operations,
                palette_name=palette_name,
            )
            target_operation = operation_a if query_operator == 0 else operation_b
            answer = chr(65 + option_operations.index(target_operation))
            questions.append(
                {
                    "question_id": question_id,
                    "question": QUESTION,
                    "answer_type": "mcq_letter",
                    "image": str(image_relative),
                }
            )
            ground_truth.append(
                {
                    "question_id": question_id,
                    "pair_id": pair_id,
                    "variant": variant,
                    "answer": answer,
                }
            )
            variant_ids[variant] = question_id
        pair_rows.append(
            {
                "pair_id": pair_id,
                "base_question_id": variant_ids["base"],
                "causal_question_id": variant_ids["causal"],
                "nuisance_question_id": variant_ids["nuisance"],
                "base_operation": operation_a,
                "causal_operation": operation_b,
                "causal_edit": "query_operator_swap",
                "nuisance_edit": "palette_background_and_stroke",
            }
        )

    questions_path = output / "questions.jsonl"
    ground_truth_path = output / "private_ground_truth.jsonl"
    pairs_path = output / "pairs.jsonl"
    write_jsonl(questions_path, questions)
    write_jsonl(ground_truth_path, ground_truth)
    write_jsonl(pairs_path, pair_rows)
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        experiment="causal_visual_transformations",
        parameters={
            "seed": seed,
            "pairs": pairs,
            "variants": ["base", "causal", "nuisance"],
            "operations": list(OPERATIONS),
            "causal_edit": "Only the abstract operator in the query changes.",
            "nuisance_edit": (
                "Palette, background, and stroke width change without changing "
                "geometry, operator identity, option order, or answer."
            ),
        },
        inputs={},
        outputs={
            "questions": questions_path,
            "private_ground_truth": ground_truth_path,
            "pairs": pairs_path,
            "images": output / "images",
        },
        item_count=len(questions),
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pairs < 1:
        raise ValueError("--pairs must be positive")
    manifest = build_bundle(args.output.resolve(), args.pairs, args.seed)
    print(
        f"Generated {manifest['item_count']} questions from {args.pairs} paired "
        f"problems at {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

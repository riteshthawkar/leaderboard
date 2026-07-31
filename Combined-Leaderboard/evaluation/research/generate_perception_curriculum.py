#!/usr/bin/env python3
"""Generate paired perception-focused and recognition-control SFT data."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.research.common import (  # noqa: E402
    build_manifest,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/research/results/perception_curriculum"
CANVAS_SIZE = 512
SHAPES = ("circle", "square", "triangle", "pentagon")
COLORS = {
    "red": "#d94a4a",
    "blue": "#3f69d8",
    "green": "#3c9b63",
    "yellow": "#d9ae32",
    "purple": "#8a59b5",
    "gray": "#92979e",
}
TASKS = (
    "shape_counting",
    "feature_binding",
    "spatial_relation",
    "form_constancy",
    "figure_ground",
)


def polygon_points(
    center: tuple[float, float], radius: float, sides: int, angle: float
) -> list[tuple[float, float]]:
    cx, cy = center
    return [
        (
            cx + radius * math.cos(angle + 2 * math.pi * index / sides),
            cy + radius * math.sin(angle + 2 * math.pi * index / sides),
        )
        for index in range(sides)
    ]


def draw_shape(
    draw: ImageDraw.ImageDraw,
    *,
    shape: str,
    center: tuple[int, int],
    radius: int,
    color: str,
    angle: float = 0.0,
    outline: str = "#171717",
    width: int = 3,
) -> None:
    cx, cy = center
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    if shape == "circle":
        draw.ellipse(box, fill=color, outline=outline, width=width)
    elif shape == "square":
        draw.polygon(
            polygon_points(center, radius, 4, angle + math.pi / 4),
            fill=color,
            outline=outline,
        )
    elif shape == "triangle":
        draw.polygon(
            polygon_points(center, radius, 3, angle - math.pi / 2),
            fill=color,
            outline=outline,
        )
    elif shape == "pentagon":
        draw.polygon(
            polygon_points(center, radius, 5, angle - math.pi / 2),
            fill=color,
            outline=outline,
        )
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def spaced_centers(
    rng: random.Random,
    count: int,
    *,
    margin: int = 55,
    minimum_distance: int = 72,
) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for _ in range(2_000):
        candidate = (
            rng.randint(margin, CANVAS_SIZE - margin),
            rng.randint(margin, CANVAS_SIZE - margin),
        )
        if all(
            (candidate[0] - x) ** 2 + (candidate[1] - y) ** 2
            >= minimum_distance**2
            for x, y in centers
        ):
            centers.append(candidate)
            if len(centers) == count:
                return centers
    raise RuntimeError(f"Could not place {count} non-overlapping objects")


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def render_counting(
    rng: random.Random, answer: int, shape: str, condition: str
) -> tuple[Image.Image, str]:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(image)
    if condition == "perception_curriculum":
        distractor_count = 8
        centers = spaced_centers(rng, answer + distractor_count)
        for index, center in enumerate(centers):
            is_target = index < answer
            draw_shape(
                draw,
                shape=shape if is_target else rng.choice([item for item in SHAPES if item != shape]),
                center=center,
                radius=rng.randint(24, 38),
                color=COLORS[rng.choice(tuple(COLORS))],
                angle=rng.random() * math.pi,
                width=2,
            )
        for _ in range(12):
            x = rng.randint(0, CANVAS_SIZE)
            draw.line((x, 0, CANVAS_SIZE - x, CANVAS_SIZE), fill="#dddddd", width=1)
    else:
        columns = max(answer, 1)
        spacing = CANVAS_SIZE // (columns + 1)
        for index in range(answer):
            draw_shape(
                draw,
                shape=shape,
                center=((index + 1) * spacing, CANVAS_SIZE // 2),
                radius=38,
                color=COLORS["blue"],
            )
    return image, f"How many {shape}s are visible? Respond with only a number."


def render_binding(
    rng: random.Random,
    answer: int,
    shape: str,
    color_name: str,
    condition: str,
) -> tuple[Image.Image, str]:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(image)
    distractors = 10 if condition == "perception_curriculum" else 4
    centers = spaced_centers(rng, answer + distractors)
    for index, center in enumerate(centers):
        is_target = index < answer
        if is_target:
            current_shape = shape
            current_color = color_name
        elif condition == "perception_curriculum" and index % 2:
            current_shape = shape
            current_color = rng.choice(
                [name for name in COLORS if name != color_name]
            )
        elif condition == "perception_curriculum":
            current_shape = rng.choice([item for item in SHAPES if item != shape])
            current_color = color_name
        else:
            current_shape = rng.choice([item for item in SHAPES if item != shape])
            current_color = "gray"
        draw_shape(
            draw,
            shape=current_shape,
            center=center,
            radius=rng.randint(24, 37),
            color=COLORS[current_color],
            angle=rng.random() * math.pi,
        )
    return (
        image,
        f"How many {color_name} {shape}s are visible? Respond with only a number.",
    )


def spatial_centers(relation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    anchor = (CANVAS_SIZE // 2, CANVAS_SIZE // 2)
    offsets = {
        "left": (-145, 0),
        "right": (145, 0),
        "above": (0, -145),
        "below": (0, 145),
    }
    dx, dy = offsets[relation]
    return (anchor[0] + dx, anchor[1] + dy), anchor


def render_spatial(
    rng: random.Random, relation: str, condition: str
) -> tuple[Image.Image, str]:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(image)
    red_center, blue_center = spatial_centers(relation)
    if condition == "perception_curriculum":
        for step in range(0, CANVAS_SIZE + 1, 32):
            draw.line((step, 0, step, CANVAS_SIZE), fill="#ececec", width=1)
            draw.line((0, step, CANVAS_SIZE, step), fill="#ececec", width=1)
        occupied = [red_center, blue_center]
        for center in spaced_centers(rng, 8, minimum_distance=65):
            if all(
                (center[0] - x) ** 2 + (center[1] - y) ** 2 > 70**2
                for x, y in occupied
            ):
                draw_shape(
                    draw,
                    shape=rng.choice(SHAPES),
                    center=center,
                    radius=22,
                    color=COLORS[rng.choice(("green", "yellow", "purple", "gray"))],
                )
    draw_shape(
        draw,
        shape="circle",
        center=red_center,
        radius=36,
        color=COLORS["red"],
    )
    draw_shape(
        draw,
        shape="square",
        center=blue_center,
        radius=38,
        color=COLORS["blue"],
    )
    return (
        image,
        "Where is the red circle relative to the blue square? "
        "Answer with one word: left, right, above, or below.",
    )


def render_form(
    rng: random.Random, shape: str, condition: str
) -> tuple[Image.Image, str]:
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(image)
    if condition == "perception_curriculum":
        draw_shape(
            draw,
            shape=shape,
            center=(CANVAS_SIZE // 2, CANVAS_SIZE // 2),
            radius=105,
            color=COLORS["purple"],
            angle=rng.random() * math.pi,
            width=6,
        )
        centers = ((90, 90), (422, 90), (90, 422), (422, 422))
        for center in centers:
            draw_shape(
                draw,
                shape=rng.choice([item for item in SHAPES if item != shape]),
                center=center,
                radius=42,
                color="#e5e5e5",
                angle=rng.random() * math.pi,
                outline="#aaaaaa",
                width=2,
            )
        for _ in range(5):
            y = rng.randint(170, 342)
            draw.line((130, y, 382, y + rng.randint(-30, 30)), fill="#ffffff", width=8)
    else:
        draw_shape(
            draw,
            shape=shape,
            center=(CANVAS_SIZE // 2, CANVAS_SIZE // 2),
            radius=110,
            color=COLORS["purple"],
            angle=0.0,
            width=6,
        )
    return image, "What is the large purple shape? Respond with only its shape name."


def render_figure_ground(
    rng: random.Random, answer: int, shape: str, condition: str
) -> tuple[Image.Image, str]:
    background = "#f4f4f1" if condition == "perception_curriculum" else "white"
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), background)
    draw = ImageDraw.Draw(image)
    if condition == "perception_curriculum":
        for _ in range(45):
            draw.line(
                (
                    rng.randint(0, CANVAS_SIZE),
                    rng.randint(0, CANVAS_SIZE),
                    rng.randint(0, CANVAS_SIZE),
                    rng.randint(0, CANVAS_SIZE),
                ),
                fill=rng.choice(("#c8c8c3", "#d4d4d0", "#dfdfda")),
                width=rng.randint(1, 2),
            )
    centers = spaced_centers(rng, answer, minimum_distance=95)
    for center in centers:
        draw_shape(
            draw,
            shape=shape,
            center=center,
            radius=rng.randint(35, 48),
            color="#f4f4f1" if condition == "perception_curriculum" else "#ffffff",
            angle=rng.random() * math.pi,
            outline="#777777" if condition == "perception_curriculum" else "#111111",
            width=4,
        )
    if condition == "perception_curriculum":
        for _ in range(12):
            draw.line(
                (
                    rng.randint(0, CANVAS_SIZE),
                    rng.randint(0, CANVAS_SIZE),
                    rng.randint(0, CANVAS_SIZE),
                    rng.randint(0, CANVAS_SIZE),
                ),
                fill=rng.choice(("#b8b8b2", "#c6c6c2", "#d2d2ce")),
                width=rng.randint(1, 2),
            )
    return (
        image,
        f"How many outlined {shape}s are present? Respond with only a number.",
    )


def render_pair(
    rng: random.Random, task: str
) -> tuple[dict[str, Image.Image], str, str, dict[str, Any]]:
    if task == "shape_counting":
        answer = rng.randint(1, 5)
        shape = rng.choice(SHAPES)
        metadata = {"answer": str(answer), "shape": shape}
        renderer = lambda condition: render_counting(rng, answer, shape, condition)
    elif task == "feature_binding":
        answer = rng.randint(1, 4)
        shape = rng.choice(SHAPES)
        color = rng.choice(("red", "blue", "green", "yellow"))
        metadata = {"answer": str(answer), "shape": shape, "color": color}
        renderer = lambda condition: render_binding(
            rng, answer, shape, color, condition
        )
    elif task == "spatial_relation":
        answer = rng.choice(("left", "right", "above", "below"))
        metadata = {"answer": answer, "relation": answer}
        renderer = lambda condition: render_spatial(rng, answer, condition)
    elif task == "form_constancy":
        answer = rng.choice(SHAPES)
        metadata = {"answer": answer, "shape": answer}
        renderer = lambda condition: render_form(rng, answer, condition)
    elif task == "figure_ground":
        answer = rng.randint(1, 4)
        shape = rng.choice(SHAPES)
        metadata = {"answer": str(answer), "shape": shape}
        renderer = lambda condition: render_figure_ground(
            rng, answer, shape, condition
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    images: dict[str, Image.Image] = {}
    prompts: dict[str, str] = {}
    for condition in ("perception_curriculum", "recognition_control"):
        image, prompt = renderer(condition)
        images[condition] = image
        prompts[condition] = prompt
    if prompts["perception_curriculum"] != prompts["recognition_control"]:
        raise RuntimeError("Paired curriculum and control prompts drifted")
    return images, prompts["perception_curriculum"], str(metadata["answer"]), metadata


def annotation(image_path: str, prompt: str, answer: str) -> dict[str, Any]:
    return {
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def generate_split(
    *,
    output: Path,
    split: str,
    count: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    annotations = {
        "perception_curriculum": [],
        "recognition_control": [],
    }
    metadata_rows: list[dict[str, Any]] = []
    for index in range(count):
        pair_id = f"{split}_{index:06d}"
        task = TASKS[index % len(TASKS)]
        rng = random.Random(seed + index * 1_000_003)
        images, prompt, answer, metadata = render_pair(rng, task)
        paths: dict[str, str] = {}
        for condition, image in images.items():
            relative = Path("images") / condition / split / f"{pair_id}.png"
            save_image(image, output / relative)
            paths[condition] = str(relative)
            annotations[condition].append(
                annotation(str(relative), prompt, answer)
            )
        metadata_rows.append(
            {
                "pair_id": pair_id,
                "split": split,
                "task": task,
                "seed": seed + index * 1_000_003,
                "prompt": prompt,
                "answer": answer,
                "perception_curriculum_image": paths["perception_curriculum"],
                "recognition_control_image": paths["recognition_control"],
                "specification": metadata,
            }
        )
        if (index + 1) % 500 == 0 or index + 1 == count:
            print(f"Generated {index + 1}/{count} {split} pairs", flush=True)
    return annotations, metadata_rows


def build(output: Path, train_count: int, validation_count: int, seed: int) -> dict[str, Any]:
    train_annotations, train_metadata = generate_split(
        output=output,
        split="train",
        count=train_count,
        seed=seed,
    )
    validation_annotations, validation_metadata = generate_split(
        output=output,
        split="validation",
        count=validation_count,
        seed=seed + 10_000_019,
    )
    output_files: dict[str, Path] = {}
    for condition in ("perception_curriculum", "recognition_control"):
        for split, values in (
            ("train", train_annotations[condition]),
            ("validation", validation_annotations[condition]),
        ):
            path = output / "annotations" / f"{condition}_{split}.json"
            write_json(path, values)
            output_files[f"{condition}_{split}"] = path
    metadata_path = output / "metadata.jsonl"
    write_jsonl(metadata_path, [*train_metadata, *validation_metadata])
    output_files["metadata"] = metadata_path
    output_files["images"] = output / "images"
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        experiment="perception_curriculum_transfer",
        parameters={
            "seed": seed,
            "train_pairs": train_count,
            "validation_pairs": validation_count,
            "tasks": list(TASKS),
            "conditions": {
                "perception_curriculum": (
                    "Clutter, feature binding, variation, or figure-ground load."
                ),
                "recognition_control": (
                    "Matched prompt, answer, task family, sample count, and image "
                    "dimensions with simplified visual evidence."
                ),
            },
            "benchmark_leakage": (
                "All images and labels are procedurally generated. No benchmark "
                "image, prompt, response, or ground truth is used."
            ),
        },
        inputs={},
        outputs=output_files,
        item_count=2 * (train_count + validation_count),
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-pairs", type=int, default=5_000)
    parser.add_argument("--validation-pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_pairs < len(TASKS) or args.validation_pairs < len(TASKS):
        raise ValueError(f"Each split must have at least {len(TASKS)} pairs")
    manifest = build(
        args.output.resolve(),
        args.train_pairs,
        args.validation_pairs,
        args.seed,
    )
    print(
        f"Generated {manifest['item_count']} paired SFT examples at "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Idempotently register generated MS-VISTA datasets in Qwen's trainer clone."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BEGIN = "# BEGIN MS-VISTA GENERATED DATASETS"
END = "# END MS-VISTA GENERATED DATASETS"


def register(qwen_root: Path, dataset_root: Path) -> Path:
    registry = qwen_root / "qwen-vl-finetune/qwenvl/data/__init__.py"
    if not registry.is_file():
        raise FileNotFoundError(
            f"Qwen dataset registry was not found at {registry}"
        )
    expected = (
        dataset_root / "annotations/perception_curriculum_train.json",
        dataset_root / "annotations/recognition_control_train.json",
    )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Generate the curriculum bundle first; missing "
            + ", ".join(str(path) for path in missing)
        )
    text = registry.read_text(encoding="utf-8")
    text = re.sub(
        rf"\n?{re.escape(BEGIN)}.*?{re.escape(END)}\n?",
        "\n",
        text,
        flags=re.DOTALL,
    )
    marker = "\ndef parse_sampling_rate"
    if marker not in text:
        raise RuntimeError("Qwen dataset registry layout is not recognized")
    block = f"""
{BEGIN}
data_dict["ms_vista_perception_curriculum"] = {{
    "annotation_path": {str(expected[0])!r},
    "data_path": {str(dataset_root)!r},
}}
data_dict["ms_vista_recognition_control"] = {{
    "annotation_path": {str(expected[1])!r},
    "data_path": {str(dataset_root)!r},
}}
{END}
"""
    updated = text.replace(marker, f"\n{block}{marker}", 1)
    temporary = registry.with_suffix(".py.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(registry)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = register(args.qwen_root.resolve(), args.dataset_root.resolve())
    print(f"Registered MS-VISTA datasets in {registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

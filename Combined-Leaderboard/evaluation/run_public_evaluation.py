"""One-command public DYS + Mind's Eye evaluation from a model profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    pass


def _read_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Cannot read model profile {path}: {exc}") from exc
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ProfileError("Model profile must be a schema_version 1 JSON object.")
    return profile


def _gpu_ids(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(not re.fullmatch(r"\d+", item) for item in values):
        raise ProfileError("--gpus must be a comma-separated list of GPU indices.")
    if len(set(values)) != len(values):
        raise ProfileError("--gpus contains a duplicate GPU index.")
    return values


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProfileError(f"{field} must be a positive integer.")
    return value


def _custom_environment(profile: dict[str, Any]) -> tuple[str, dict[str, str]]:
    required = ("slug", "model_id", "revision", "max_model_len")
    missing = [field for field in required if not profile.get(field)]
    if missing:
        raise ProfileError(f"Custom profile is missing: {', '.join(missing)}.")
    slug = str(profile["slug"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ProfileError("slug must contain lowercase letters, digits, and hyphens.")
    revision = str(profile["revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProfileError("revision must be a full 40-character commit hash.")
    max_model_len = _positive_int(profile["max_model_len"], "max_model_len")
    reasoning = str(profile.get("reasoning_profile", "nonthinking"))
    if reasoning not in {"thinking", "nonthinking"}:
        raise ProfileError("reasoning_profile must be thinking or nonthinking.")
    kwargs = profile.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ProfileError("chat_template_kwargs must be a JSON object.")
    generation = profile.get("generation", {})
    if not isinstance(generation, dict):
        raise ProfileError("generation must be a JSON object.")
    dys_tokens = _positive_int(
        generation.get("do_you_see_me_max_tokens", 8192),
        "generation.do_you_see_me_max_tokens",
    )
    me_tokens = _positive_int(
        generation.get("minds_eye_max_tokens", 8192),
        "generation.minds_eye_max_tokens",
    )
    hf_overrides = profile.get("hf_overrides", {})
    if not isinstance(hf_overrides, dict):
        raise ProfileError("hf_overrides must be a JSON object.")
    environment = {
        "CUSTOM_MODEL_SLUG": slug,
        "CUSTOM_MODEL_SPEC": (
            f"{slug}|{profile['model_id']}|{revision}|unquantized|{max_model_len}"
        ),
        "CUSTOM_REASONING_PROFILE": reasoning,
        "CUSTOM_CHAT_TEMPLATE_KWARGS": json.dumps(kwargs, separators=(",", ":")),
        "CUSTOM_DYS_MAX_TOKENS": str(dys_tokens),
        "CUSTOM_MINDS_EYE_MAX_TOKENS": str(me_tokens),
        "CUSTOM_VLLM_ENGINE_MODE": str(profile.get("vllm_engine_mode", "v1")),
        "CUSTOM_SERVER_KV_CACHE_DTYPE": str(
            profile.get("server_kv_cache_dtype", "bfloat16")
        ),
        "CUSTOM_HF_OVERRIDES": (
            json.dumps(hf_overrides, separators=(",", ":")) if hf_overrides else ""
        ),
    }
    return slug, environment


def build_environment(
    profile: dict[str, Any],
    *,
    gpus: list[str],
    output_root: Path,
    cache_root: Path | None,
    force: bool,
    dry_run: bool,
) -> dict[str, str]:
    environment = os.environ.copy()
    builtin_slug = str(profile.get("builtin_slug") or "").strip()
    if builtin_slug:
        slug = builtin_slug
        custom: dict[str, str] = {}
    else:
        slug, custom = _custom_environment(profile)
    serving_mode = str(profile.get("serving_mode", "replica"))
    minimum = _positive_int(profile.get("minimum_gpu_count", 1), "minimum_gpu_count")
    if len(gpus) < minimum:
        raise ProfileError(
            f"Profile requires at least {minimum} GPU(s); received {len(gpus)}."
        )
    if serving_mode == "replica":
        tensor_parallel = 1
        data_parallel = len(gpus)
        replica_mode = "independent"
        concurrency = len(gpus)
    elif serving_mode == "tensor_parallel":
        tensor_parallel = len(gpus)
        data_parallel = 1
        replica_mode = "builtin"
        concurrency = 1
    else:
        raise ProfileError("serving_mode must be replica or tensor_parallel.")
    environment.update(custom)
    environment.update(
        {
            "MODELS": slug,
            "TRACKS": "all",
            "PIPELINE_PHASE": "all",
            "GPU_IDS": ",".join(gpus),
            "TENSOR_PARALLEL_SIZE": str(tensor_parallel),
            "DATA_PARALLEL_SIZE": str(data_parallel),
            "SERVING_REPLICA_MODE": replica_mode,
            "CONCURRENCY": str(concurrency),
            "MAX_NUM_SEQS_PER_REPLICA": "1",
            "OUTPUT_ROOT": str(output_root.resolve()),
            "KEEP_MODEL_CACHE": "1",
            "CONTINUE_ON_MODEL_ERROR": "0",
            "FORCE": "1" if force else "0",
            "DRY_RUN": "1" if dry_run else "0",
        }
    )
    if cache_root is not None:
        environment["CACHE_ROOT"] = str(cache_root.resolve())
    return environment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU indices")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "evaluation/results/public-runs",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an earlier run")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile = _read_profile(args.profile.expanduser().resolve())
        gpus = _gpu_ids(args.gpus)
        environment = build_environment(
            profile,
            gpus=gpus,
            output_root=args.output_root,
            cache_root=args.cache_root,
            force=args.force,
            dry_run=args.dry_run,
        )
    except ProfileError as exc:
        print(f"Profile error: {exc}", file=sys.stderr)
        return 2
    project_root = Path(__file__).resolve().parents[1]
    command = ["bash", str(project_root / "evaluation/run_visual_suite.sh")]
    result = subprocess.run(command, cwd=project_root, env=environment, check=False)
    if result.returncode == 0 and not args.dry_run:
        slug = environment["MODELS"]
        result_dir = args.output_root.resolve() / slug
        print(f"Ready submissions: {result_dir}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

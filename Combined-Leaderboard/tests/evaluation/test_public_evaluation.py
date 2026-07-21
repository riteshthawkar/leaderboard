import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.run_public_evaluation import ProfileError, _read_profile, build_environment


PROJECT = Path(__file__).resolve().parents[2]
PROFILES = PROJECT / "evaluation/model_profiles"
RANKED = {
    "deepseek-vl2",
    "gemma3-12b-it",
    "gemma3-27b-it",
    "glm46v-flash",
    "internvl35-8b",
    "kimi-vl-a3b-instruct",
    "llama32-11b-vision-instruct",
    "minicpm-v46",
    "phi4-multimodal",
    "qwen25-vl-7b",
    "qwen3-vl-8b",
    "qwen35-9b",
    "qwen36-27b",
}


def test_ranked_model_profiles_are_complete_and_resolve_one_command_environment(tmp_path):
    profiles = {
        path.stem: _read_profile(path)
        for path in PROFILES.glob("*.json")
        if not path.name.endswith(".template.json")
    }
    assert set(profiles) == RANKED
    for name, profile in profiles.items():
        minimum = profile["minimum_gpu_count"]
        gpus = [str(index) for index in range(minimum)]
        environment = build_environment(
            profile,
            gpus=gpus,
            output_root=tmp_path / "results",
            cache_root=None,
            force=False,
            dry_run=True,
        )
        assert environment["MODELS"] == name
        assert environment["TRACKS"] == "all"
        assert environment["PIPELINE_PHASE"] == "all"
        assert environment["GPU_IDS"] == ",".join(gpus)
        assert environment["DRY_RUN"] == "1"


def test_custom_profile_maps_model_file_to_shared_pipeline(tmp_path):
    profile = {
        "schema_version": 1,
        "slug": "new-model",
        "model_id": "org/new-model",
        "revision": "a" * 40,
        "max_model_len": 32768,
        "reasoning_profile": "nonthinking",
        "chat_template_kwargs": {"enable_thinking": False},
        "generation": {
            "do_you_see_me_max_tokens": 4096,
            "minds_eye_max_tokens": 8192,
        },
        "serving_mode": "replica",
        "minimum_gpu_count": 1,
    }
    environment = build_environment(
        profile,
        gpus=["2", "3"],
        output_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        force=True,
        dry_run=False,
    )

    assert environment["MODELS"] == "new-model"
    assert environment["CUSTOM_MODEL_SPEC"] == (
        f"new-model|org/new-model|{'a' * 40}|unquantized|32768"
    )
    assert environment["CUSTOM_CHAT_TEMPLATE_KWARGS"] == '{"enable_thinking":false}'
    assert environment["CUSTOM_DYS_MAX_TOKENS"] == "4096"
    assert environment["CUSTOM_MINDS_EYE_MAX_TOKENS"] == "8192"
    assert environment["TENSOR_PARALLEL_SIZE"] == "1"
    assert environment["DATA_PARALLEL_SIZE"] == "2"
    assert environment["CONCURRENCY"] == "2"
    assert environment["FORCE"] == "1"


def test_custom_profile_requires_immutable_revision(tmp_path):
    profile = json.loads((PROFILES / "custom-model.template.json").read_text())
    with pytest.raises(ProfileError, match="40-character"):
        build_environment(
            profile,
            gpus=["0"],
            output_root=tmp_path,
            cache_root=None,
            force=False,
            dry_run=True,
        )


def test_custom_profile_public_cli_dry_run(tmp_path):
    profile = tmp_path / "custom.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slug": "new-model",
                "model_id": "org/new-model",
                "revision": "a" * 40,
                "max_model_len": 32768,
                "reasoning_profile": "nonthinking",
                "chat_template_kwargs": {},
                "generation": {
                    "do_you_see_me_max_tokens": 4096,
                    "minds_eye_max_tokens": 8192,
                },
                "serving_mode": "replica",
                "minimum_gpu_count": 1,
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "evaluation/run_public_evaluation.py"),
            "--profile",
            str(profile),
            "--gpus",
            "0,1",
            "--dry-run",
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "org/new-model" in result.stdout
    assert "GPUs: 0,1 (tensor parallel size 1, data parallel size 2" in result.stdout
    assert "Track do_you_see_me" in result.stdout
    assert "Track minds_eye" in result.stdout
    assert "Model-only answer extraction" in result.stdout

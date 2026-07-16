import hashlib
import json
import os
import shlex
import stat
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "evaluation" / "run_visual_suite.sh"
MULTI_GPU_SCRIPT = PROJECT_ROOT / "evaluation" / "run_visual_suite_multi_gpu.sh"


def _dry_run(**overrides: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({"DRY_RUN": "1", **overrides})
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_sourced(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(SCRIPT))}\n{command}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_visual_suite_shell_scripts_are_valid():
    for script in (SCRIPT, MULTI_GPU_SCRIPT):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_visual_suite_has_no_quantized_or_synthetic_answer_path():
    script = SCRIPT.read_text(encoding="utf-8").lower()

    assert "bitsandbytes" not in script
    assert "--quantization" not in script
    assert "--load-format" not in script
    assert "invalid_model_response" not in script
    assert "finalize_visual_diagnostics" not in script
    assert "original image bytes" in script
    assert '--kv-cache-dtype "$vllm_kv_cache_dtype"' in script
    assert "--tensor-parallel-size" in script
    assert ".attempt-${attempt}.diagnostics.jsonl" in script
    assert 'setup_only="${setup_only:-0}"' in script


def test_visual_suite_dry_run_uses_unquantized_paper_profiles():
    result = _dry_run()

    assert result.returncode == 0, result.stderr
    assert "original checkpoint tensors, unquantized; compute: bfloat16; KV cache: bfloat16" in result.stdout
    assert "no runner resize or recompression" in result.stdout
    assert "top_k=-1, min_p=0.0, presence=0.0, frequency=0.0, repetition=1.0" in result.stdout
    assert "prompt=noncot, temperature=1.0, top_p=0.95, max_tokens=200" in result.stdout
    assert "prompt=cot, temperature=0.1, top_p=1.0, max_tokens=1000" in result.stdout
    assert result.stdout.count("unquantized, context=32768") == 7
    assert "Qwen/Qwen3.5-9B" in result.stdout
    assert "microsoft/Phi-4-multimodal-instruct" in result.stdout


def test_visual_suite_dry_run_honors_track_context_and_gpu_topology():
    result = _dry_run(
        MODELS="qwen35-9b",
        TRACKS="minds_eye",
        MAX_MODEL_LEN="16384",
        GPU_IDS="2,3",
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs: 2,3 (tensor parallel size 2)" in result.stdout
    assert "Track minds_eye" in result.stdout
    assert "Track do_you_see_me" not in result.stdout
    assert "context=16384" in result.stdout
    assert "InternVL" not in result.stdout


def test_visual_suite_rejects_quality_reducing_or_inconsistent_overrides():
    dtype = _dry_run(VLLM_DTYPE="float16")
    assert dtype.returncode == 2
    assert "must remain bfloat16" in dtype.stderr

    kv_dtype = _dry_run(VLLM_KV_CACHE_DTYPE="fp8")
    assert kv_dtype.returncode == 2
    assert "VLLM_KV_CACHE_DTYPE must remain bfloat16" in kv_dtype.stderr

    topology = _dry_run(GPU_IDS="0,1", TENSOR_PARALLEL_SIZE="1")
    assert topology.returncode == 2
    assert "must match" in topology.stderr

    unknown = _dry_run(MODELS="not-a-configured-model")
    assert unknown.returncode == 2
    assert "MODELS did not match" in unknown.stderr


def test_server_identity_check_rejects_another_model_on_the_port():
    result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                'PORT="8011"',
                'curl() { printf \'%s\' \'{"data":[{"id":"OpenGVLab/InternVL3_5-8B"}]}\'; }',
                "server_serves_model OpenGVLab/InternVL3_5-8B || exit 7",
                "if server_serves_model Qwen/Qwen3.5-9B; then exit 8; fi",
            )
        )
    )

    assert result.returncode == 0, result.stderr


def test_force_replaces_stale_artifacts_with_schema_v4_fingerprint(tmp_path):
    output_root = tmp_path / "outputs"
    model_dir = output_root / "test-model"
    model_dir.mkdir(parents=True)
    stale_diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
    stale_submission = model_dir / "minds_eye_submission.jsonl"
    stale_manifest = model_dir / "run_manifest.json"
    stale_diagnostics.write_text("stale\n", encoding="utf-8")
    stale_submission.write_text("stale\n", encoding="utf-8")
    stale_manifest.write_text("{}\n", encoding="utf-8")

    result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert not stale_diagnostics.exists()
    assert not stale_submission.exists()
    assert not stale_manifest.exists()
    run_config = json.loads((model_dir / ".run_config.json").read_text(encoding="utf-8"))
    assert run_config["schema_version"] == 4
    assert run_config["weight_loading"] == "unquantized"
    assert run_config["compute_dtype"] == "bfloat16"
    assert run_config["kv_cache_dtype"] == "bfloat16"
    assert run_config["tensor_parallel_size"] == 1
    assert run_config["generation"]["do_you_see_me"]["max_tokens"] == 200
    assert run_config["generation"]["minds_eye"]["prompt_mode"] == "cot"
    assert run_config["image_preprocessing"].startswith("original-bytes")
    assert set(run_config["source_hashes"]["runner"]) == {
        "visual_pipeline",
        "vllm_runner",
    }

    stale_diagnostics.write_text("checkpoint\n", encoding="utf-8")
    mismatch = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-b unquantized 32768",
            )
        )
    )
    assert mismatch.returncode != 0
    assert "Run configuration changed" in mismatch.stderr


def test_manifest_hashes_failed_formatting_attempts(tmp_path):
    output_root = tmp_path / "outputs"
    initialize = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )
    assert initialize.returncode == 0, initialize.stderr

    model_dir = output_root / "test-model"
    (model_dir / "do_you_see_me_submission.jsonl").write_text(
        '{"question_id":"q1","condition":"standard","answer":"1"}\n',
        encoding="utf-8",
    )
    (model_dir / "do_you_see_me.diagnostics.jsonl").write_text(
        '{"question_id":"q1","output":"<answer>1</answer>"}\n',
        encoding="utf-8",
    )
    attempt = model_dir / "do_you_see_me.attempt-1.diagnostics.jsonl"
    attempt.write_text(
        '{"question_id":"q1","output":"unparseable"}\n', encoding="utf-8"
    )

    manifest_result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "GPU_IDS=0",
                'GPU_NAME="GPU 0: NVIDIA A100"',
                'completed_tracks() { printf "do_you_see_me\\n"; }',
                "write_manifest test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert manifest_result.returncode == 0, manifest_result.stderr
    manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
    archived = manifest["tracks"]["do_you_see_me"]["failed_attempt_diagnostics"]
    assert archived == [
        {
            "file": attempt.name,
            "seed": 0,
            "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest(),
        }
    ]


def test_multi_gpu_dry_run_supports_one_model_per_gpu_and_tensor_parallel(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_venv = tmp_path / "venv"
    fake_bin.mkdir()
    (fake_venv / "bin").mkdir(parents=True)
    _make_executable(
        fake_bin / "nvidia-smi",
        "#!/bin/sh\nprintf 'NVIDIA A100-PCIE-40GB, 40960, 40400\\n'\n",
    )
    _make_executable(fake_venv / "bin" / "python", "#!/bin/sh\ncat >/dev/null\nexit 0\n")
    _make_executable(fake_venv / "bin" / "vllm", "#!/bin/sh\nexit 0\n")
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "VENV_DIR": str(fake_venv),
            "GPU_GROUPS": "0,1;2,3",
            "MODEL_LIST": "internvl35-8b,minicpm-v46",
            "MIN_FREE_DISK_GB_PER_MODEL": "1",
            "DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(MULTI_GPU_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs 0,1" in result.stdout and "TP=2" in result.stdout
    assert "GPUs 2,3" in result.stdout
    assert "unquantized BF16" in result.stdout
    assert "no workers were started" in result.stdout

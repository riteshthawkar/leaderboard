import json
import os
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "evaluation" / "run_visual_suite.sh"


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


def test_visual_suite_shell_is_valid():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_visual_suite_uses_current_vllm_request_logging_default():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "--disable-log-requests" not in script
    assert "verify_vllm_cli" in script
    assert "serve --help=all" in script
    assert "report_startup_progress" in script
    assert "Latest vLLM log" in script


def test_visual_suite_dry_run_selects_every_model_and_track():
    result = _dry_run()

    assert result.returncode == 0, result.stderr
    assert "Tracks: do_you_see_me,minds_eye" in result.stdout
    assert result.stdout.count("max_tokens=") == 7
    assert "Qwen/Qwen3.5-9B" in result.stdout
    assert "microsoft/Phi-4-multimodal-instruct" in result.stdout
    assert "phi4-multimodal" in result.stdout
    assert "full" in result.stdout
    assert "context=4096" in result.stdout
    internvl_line = next(
        line for line in result.stdout.splitlines() if "internvl35-8b" in line
    )
    assert "full" in internvl_line
    assert "context=4096" in internvl_line
    for slug in ("qwen35-9b", "glm41v-9b-thinking", "qwen25-vl-7b", "qwen3-vl-8b"):
        line = next(item for item in result.stdout.splitlines() if slug in item)
        assert "context=32768" in line


def test_visual_suite_dry_run_honors_model_track_and_context_overrides():
    result = _dry_run(
        MODELS="qwen35-9b",
        TRACKS="minds_eye",
        MAX_MODEL_LEN="4096",
    )

    assert result.returncode == 0, result.stderr
    assert "Tracks: minds_eye" in result.stdout
    assert "Qwen/Qwen3.5-9B" in result.stdout
    assert "InternVL" not in result.stdout
    assert "context=4096" in result.stdout


def test_visual_suite_rejects_unknown_model_before_gpu_setup():
    result = _dry_run(MODELS="not-a-configured-model")

    assert result.returncode == 2
    assert "MODELS did not match any configured model slug" in result.stderr


def test_visual_suite_rejects_unsafe_numeric_configuration():
    result = _dry_run(GPU_MEMORY_UTILIZATION="1.2")

    assert result.returncode == 2
    assert "GPU_MEMORY_UTILIZATION" in result.stderr


def test_force_removes_stale_checkpoints_before_replacing_run_fingerprint(tmp_path):
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
                "VLLM_DTYPE=bfloat16",
                "FORCE=1",
                "ensure_run_config test-model test/model revision-a bnb4 noncot 256 8192 '{}'",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert not stale_diagnostics.exists()
    assert not stale_submission.exists()
    assert not stale_manifest.exists()
    run_config = json.loads((model_dir / ".run_config.json").read_text(encoding="utf-8"))
    assert run_config["schema_version"] == 2
    assert run_config["model_revision"] == "revision-a"
    assert set(run_config["source_hashes"]["runner"]) == {
        "visual_pipeline",
        "vllm_runner",
    }

    mismatch = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "VLLM_DTYPE=bfloat16",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-b bnb4 noncot 256 8192 '{}'",
            )
        )
    )
    assert mismatch.returncode != 0
    assert "Run configuration changed" in mismatch.stderr

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
    assert '--kv-cache-dtype "$server_kv_cache_dtype"' in script
    assert "--tensor-parallel-size" in script
    assert "--data-parallel-size" in script
    assert 'scipy_version="1.15.3"' in script
    assert '"scipy==$scipy_version"' in script
    assert "from scipy.optimize import linear_sum_assignment" in script
    assert 'timm_version="1.0.28"' in script
    assert '"timm==$timm_version"' in script
    assert "import timm" in script
    assert "from huggingface_hub import get_token" in script
    assert 'print(get_token() or "")' in script
    assert 'vllm_phi4mm_mask_sum_patch_id="vllm-0.25.1-phi4mm-fp32-mask-sum-v1"' in script
    assert "evaluation.common.patch_vllm_phi4mm" in script
    assert ".attempt-${attempt}.diagnostics.jsonl" in script
    assert 'setup_only="${setup_only:-0}"' in script
    assert 'df -pk "$cache_root"' in script
    assert 'hf_hub_enable_hf_transfer="0"' in script
    assert 'active_run_marker="$output_root/.active-run.json"' in script
    assert "write_active_run_marker" in script
    assert "trap cleanup_runner exit" in script


def test_visual_suite_dry_run_uses_unquantized_paper_profiles():
    result = _dry_run()

    assert result.returncode == 0, result.stderr
    assert "original checkpoint tensors, unquantized; compute: bfloat16; KV cache: bfloat16" in result.stdout
    assert "no runner resize or recompression" in result.stdout
    assert "preserve exact nonempty diagnostics.output" in result.stdout
    assert "top_k=-1, min_p=0.0, presence=0.0, frequency=0.0, repetition=1.0" in result.stdout
    assert "prompt=noncot, temperature=1.0, top_p=0.95" in result.stdout
    assert "prompt=cot, temperature=0.1, top_p=1.0" in result.stdout
    assert "retained_stop=</answer>" in result.stdout
    assert (
        "Reasoning profile: thinking; API max_tokens=uncapped "
        "(remaining-model-context); final_answer_max_tokens=200"
    ) in result.stdout
    assert (
        "Reasoning profile: thinking; API max_tokens=8192; "
        "final_answer_max_tokens=200"
    ) in result.stdout
    assert (
        "Reasoning profile: nonthinking; API max_tokens=200; "
        "final_answer_max_tokens=200"
    ) in result.stdout
    assert result.stdout.count("unquantized, context=32768") == 12
    assert "Qwen/Qwen3.5-9B" in result.stdout
    assert "Qwen/Qwen3.6-27B" in result.stdout
    assert "microsoft/Phi-4-multimodal-instruct" in result.stdout
    assert "Request model: vision; bundled adapter: vision-lora" in result.stdout
    assert "google/gemma-3-12b-it" in result.stdout
    assert "google/gemma-3-27b-it" in result.stdout
    assert "moonshotai/Kimi-VL-A3B-Instruct" in result.stdout
    assert "deepseek-ai/deepseek-vl2" in result.stdout
    assert "meta-llama/Llama-3.2-11B-Vision-Instruct" in result.stdout
    assert result.stdout.count("unquantized, context=4096") == 1


def test_gemma3_27b_uses_pinned_tp2_track_budget_profile():
    script = SCRIPT.read_text(encoding="utf-8")
    assert (
        "gemma3-27b-it|google/gemma-3-27b-it|"
        "005ad3404e59d6023443cb575daa05336842228a|unquantized|32768"
    ) in script

    result = _dry_run(
        MODELS="gemma3-27b-it",
        TRACKS="all",
        GPU_IDS="0,1",
        TENSOR_PARALLEL_SIZE="2",
        DATA_PARALLEL_SIZE="1",
        CONCURRENCY="1",
        DISABLE_CUSTOM_ALL_REDUCE="1",
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs: 0,1 (tensor parallel size 2, data parallel size 1" in result.stdout
    assert "google/gemma-3-27b-it" in result.stdout
    assert (
        "API max_tokens=do_you_see_me=200, minds_eye=8192; "
        "final_answer_max_tokens=200"
    ) in result.stdout


def test_deepseek_vl2_uses_pinned_native_context_tp2_profile():
    script = SCRIPT.read_text(encoding="utf-8")
    assert (
        "deepseek-vl2|deepseek-ai/deepseek-vl2|"
        "f363772d1c47f4239dd844015b4bd53beb87951b|unquantized|4096"
    ) in script
    assert (
        '{"text_config":{"kv_lora_rank":512,"num_hidden_layers":30}}'
        in script
    )
    assert 'command+=(--hf-overrides "$hf_overrides")' in script

    result = _dry_run(
        MODELS="deepseek-vl2",
        TRACKS="all",
        GPU_IDS="0,1",
        TENSOR_PARALLEL_SIZE="2",
        DATA_PARALLEL_SIZE="1",
        CONCURRENCY="1",
        MAX_NUM_SEQS_PER_REPLICA="1",
        DISABLE_CUSTOM_ALL_REDUCE="1",
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs: 0,1 (tensor parallel size 2, data parallel size 1" in result.stdout
    assert "deepseek-ai/deepseek-vl2" in result.stdout
    assert "unquantized, context=4096" in result.stdout
    assert (
        "Reasoning profile: nonthinking; API max_tokens=200; "
        "final_answer_max_tokens=200"
    ) in result.stdout


def test_llama32_11b_vision_uses_pinned_legacy_v0_dp2_profile(tmp_path):
    script = SCRIPT.read_text(encoding="utf-8")
    assert (
        "llama32-11b-vision-instruct|meta-llama/Llama-3.2-11B-Vision-Instruct|"
        "9eb2daaa8597bf192a8b0e73f848f3a102794df5|unquantized|32768"
    ) in script
    assert "server_environment+=(VLLM_USE_V1=0)" in script

    result = _dry_run(
        MODELS="llama32-11b-vision-instruct",
        TRACKS="all",
        GPU_IDS="0,1",
        TENSOR_PARALLEL_SIZE="1",
        DATA_PARALLEL_SIZE="2",
        CONCURRENCY="2",
        MAX_NUM_SEQS_PER_REPLICA="1",
        SERVING_REPLICA_MODE="independent",
        VLLM_VERSION="0.10.2",
        HUGGINGFACE_HUB_VERSION="0.34.4",
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs: 0,1 (tensor parallel size 1, data parallel size 2" in result.stdout
    assert "Serving replica mode: independent" in result.stdout
    assert "Serving engine: vLLM 0.10.2" in result.stdout
    assert "meta-llama/Llama-3.2-11B-Vision-Instruct" in result.stdout
    assert "unquantized, context=32768" in result.stdout
    assert "Engine mode: legacy-v0; server KV cache argument: auto" in result.stdout
    assert (
        "API max_tokens=do_you_see_me=200, minds_eye=8192; "
        "final_answer_max_tokens=200"
    ) in result.stdout

    output_root = tmp_path / "outputs"
    sourced = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                "VLLM_VERSION=0.10.2",
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "GPU_IDS=0,1",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=2",
                "CONCURRENCY=2",
                "MAX_NUM_SEQS_PER_REPLICA=1",
                "SERVING_REPLICA_MODE=independent",
                "FORCE=1",
                "ensure_run_config llama32-11b-vision-instruct meta-llama/Llama-3.2-11B-Vision-Instruct revision-a unquantized 32768",
            )
        )
    )
    assert sourced.returncode == 0, sourced.stderr
    config = json.loads(
        (output_root / "llama32-11b-vision-instruct" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["serving_engine"]["version"] == "0.10.2"
    assert config["serving_engine"]["engine_mode"] == "legacy-v0"
    assert config["serving_engine"]["kv_cache_cli_value"] == "auto"
    assert config["kv_cache_dtype"] == "bfloat16"


def test_kimi_vl_uses_pinned_dp3_track_budget_profile(tmp_path):
    script = SCRIPT.read_text(encoding="utf-8")
    assert (
        "kimi-vl-a3b-instruct|moonshotai/Kimi-VL-A3B-Instruct|"
        "398eede0903cd983a2bfa0cc634e9ac1d843f375|unquantized|32768"
    ) in script

    result = _dry_run(
        MODELS="kimi-vl-a3b-instruct",
        TRACKS="all",
        GPU_IDS="0,1,3",
        TENSOR_PARALLEL_SIZE="1",
        DATA_PARALLEL_SIZE="3",
        CONCURRENCY="3",
        MAX_NUM_SEQS_PER_REPLICA="1",
        SERVING_REPLICA_MODE="independent",
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs: 0,1,3 (tensor parallel size 1, data parallel size 3" in result.stdout
    assert "Serving replica mode: independent" in result.stdout
    assert "moonshotai/Kimi-VL-A3B-Instruct" in result.stdout
    assert (
        "API max_tokens=do_you_see_me=200, minds_eye=8192; "
        "final_answer_max_tokens=200"
    ) in result.stdout

    output_root = tmp_path / "outputs"
    sourced = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "GPU_IDS=0,1,3",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=3",
                "CONCURRENCY=3",
                "SERVING_REPLICA_MODE=independent",
                "FORCE=1",
                "ensure_run_config kimi-vl-a3b-instruct moonshotai/Kimi-VL-A3B-Instruct revision-a unquantized 32768",
            )
        )
    )
    assert sourced.returncode == 0, sourced.stderr
    config = json.loads(
        (output_root / "kimi-vl-a3b-instruct" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["serving_engine"]["replica_mode"] == "independent-processes"

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        '{"question_id":"q1"}\n{"question_id":"q2"}\n', encoding="utf-8"
    )
    diagnostics = output_root / "kimi-vl-a3b-instruct" / "minds_eye.smoke.diagnostics.jsonl"
    diagnostics_bytes = (
        b'{"question_id":"q1","output":"<answer>A"}\n'
        b'{"question_id":"q2","output":"The choice is D"}\n'
    )
    diagnostics.write_bytes(diagnostics_bytes)
    admission = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "GPU_IDS=0,1,3",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=3",
                "CONCURRENCY=3",
                "SERVING_REPLICA_MODE=independent",
                "ALLOW_SMOKE_RAW_OUTPUT_FALLBACK=1",
                "FORCE=0",
                "ensure_run_config kimi-vl-a3b-instruct moonshotai/Kimi-VL-A3B-Instruct revision-a unquantized 32768",
                "SMOKE_SAMPLES=2",
                f"smoke_outputs_are_complete {shlex.quote(str(questions))} {shlex.quote(str(diagnostics))}",
            )
        )
    )
    assert admission.returncode == 0, admission.stderr
    assert diagnostics.read_bytes() == diagnostics_bytes
    migrated = json.loads(
        (output_root / "kimi-vl-a3b-instruct" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert migrated["unparseable_answers"]["smoke_admission"][
        "canonical_output_effect"
    ] == "none"
    assert migrated["artifact_migrations"][-1]["reason"] == (
        "provenance-preserving-smoke-raw-output-admission"
    )


def test_visual_suite_token_limit_follows_model_policy():
    result = _dry_run(
        MODELS="internvl35-8b,minicpm-v46,qwen3-vl-8b,qwen25-vl-7b",
        GPU_IDS="0",
        DYS_PROMPT_MODE="cot",
        MINDS_EYE_PROMPT_MODE="noncot",
    )

    assert result.returncode == 0, result.stderr
    assert "Track do_you_see_me  prompt=cot, temperature=1.0, top_p=0.95" in result.stdout
    assert "Track minds_eye      prompt=noncot, temperature=0.1, top_p=1.0" in result.stdout
    assert result.stdout.count("Reasoning profile: thinking; API max_tokens=8192") == 1
    assert result.stdout.count("Reasoning profile: nonthinking; API max_tokens=8192") == 3
    assert "Reasoning profile: nonthinking; API max_tokens=200" not in result.stdout


def test_visual_suite_internvl_token_limit_is_configurable_and_validated():
    result = _dry_run(
        MODELS="internvl35-8b",
        GPU_IDS="0",
        INTERNVL35_MAX_TOKENS="4096",
    )

    assert result.returncode == 0, result.stderr
    assert "Reasoning profile: thinking; API max_tokens=4096" in result.stdout

    invalid = _dry_run(
        MODELS="internvl35-8b",
        GPU_IDS="0",
        INTERNVL35_MAX_TOKENS="0",
    )
    assert invalid.returncode == 2
    assert "INTERNVL35_MAX_TOKENS must be positive" in invalid.stderr


def test_visual_suite_dry_run_honors_track_context_and_gpu_topology():
    result = _dry_run(
        MODELS="qwen35-9b",
        TRACKS="minds_eye",
        MAX_MODEL_LEN="16384",
        GPU_IDS="2,3",
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs: 2,3 (tensor parallel size 2, data parallel size 1" in result.stdout
    assert "Track minds_eye" in result.stdout
    assert "Track do_you_see_me" not in result.stdout
    assert "context=16384" in result.stdout
    assert "InternVL" not in result.stdout


def test_visual_suite_supports_four_way_data_parallel_internvl():
    result = _dry_run(
        MODELS="internvl35-8b",
        GPU_IDS="0,1,2,3",
        TENSOR_PARALLEL_SIZE="1",
        DATA_PARALLEL_SIZE="4",
        CONCURRENCY="4",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "GPUs: 0,1,2,3 (tensor parallel size 1, data parallel size 4, "
        "request concurrency 4)"
    ) in result.stdout
    assert "at most 1 active sequence(s) per data-parallel replica" in result.stdout
    assert "OpenGVLab/InternVL3_5-8B" in result.stdout


def test_visual_suite_derives_per_replica_admission_from_client_concurrency():
    result = _dry_run(
        MODELS="qwen3-vl-8b",
        GPU_IDS="0,1",
        TENSOR_PARALLEL_SIZE="1",
        DATA_PARALLEL_SIZE="2",
        CONCURRENCY="5",
    )

    assert result.returncode == 0, result.stderr
    assert "request concurrency 5)" in result.stdout
    assert "at most 3 active sequence(s) per data-parallel replica" in result.stdout
    assert "GPU allocation: utilization=0.84" in result.stdout

    script = SCRIPT.read_text(encoding="utf-8")
    assert '--max-num-seqs "$MAX_NUM_SEQS_PER_REPLICA"' in script
    assert 'command+=(--disable-custom-all-reduce)' in script
    assert (
        'RUN_CONFIG_DISABLE_CUSTOM_ALL_REDUCE="$DISABLE_CUSTOM_ALL_REDUCE"'
        in script
    )
    assert 'PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"' in script


def test_data_parallel_startup_prefetches_one_pinned_snapshot():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "prefetch_model_snapshot()" in script
    assert "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])" in script
    assert 'ignore_patterns=["*consolidated.pth"]' in script
    assert 'if (( DATA_PARALLEL_SIZE > 1 )) && [[ "$model_source" == "$model_id" ]]' in script
    assert 'prefetch_model_snapshot "$model_id" "$revision" "$model_cache"' in script
    assert 'HF_HUB_CACHE="$model_cache/hub"' in script
    assert 'TRANSFORMERS_CACHE="$model_cache/hub"' in script
    assert 'HF_XET_CACHE="$model_cache/xet"' in script


def test_glm_flash_explicitly_disables_thinking_in_its_pinned_template():
    result = _run_sourced("track_chat_kwargs glm46v-flash")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '{"enable_thinking":false}'


def test_qwen36_27b_uses_pinned_nonthinking_tp2_profile():
    script = SCRIPT.read_text(encoding="utf-8")
    assert (
        "qwen36-27b|Qwen/Qwen3.6-27B|"
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9|unquantized|32768"
    ) in script

    result = _dry_run(
        MODELS="qwen36-27b",
        TRACKS="minds_eye",
        GPU_IDS="1,2",
        TENSOR_PARALLEL_SIZE="2",
        DATA_PARALLEL_SIZE="1",
        CONCURRENCY="1",
    )

    assert result.returncode == 0, result.stderr
    assert "GPUs: 1,2 (tensor parallel size 2, data parallel size 1" in result.stdout
    assert "Qwen/Qwen3.6-27B" in result.stdout
    assert "Reasoning profile: nonthinking; API max_tokens=8192" in result.stdout

    chat_kwargs = _run_sourced("track_chat_kwargs qwen36-27b")
    assert chat_kwargs.returncode == 0, chat_kwargs.stderr
    assert chat_kwargs.stdout.strip() == '{"enable_thinking":false}'


def test_server_shutdown_waits_for_data_parallel_children_to_release_port():
    result = _run_sourced(
        "\n".join(
            (
                'SERVER_PID="424242"',
                'SERVER_OWNS_PROCESS_GROUP="1"',
                'attempts=0',
                'kill() { [[ "${1:-}" == "-0" ]] && return 1; return 0; }',
                'wait() { return 0; }',
                'sleep() { return 0; }',
                'port_is_available() { attempts=$((attempts + 1)); (( attempts >= 3 )); }',
                'stop_server',
                'printf "attempts=%s server_pid=%s\\n" "$attempts" "$SERVER_PID"',
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert "attempts=3 server_pid=" in result.stdout


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

    data_topology = _dry_run(
        GPU_IDS="0,1,2,3",
        TENSOR_PARALLEL_SIZE="1",
        DATA_PARALLEL_SIZE="2",
    )
    assert data_topology.returncode == 2
    assert "must match" in data_topology.stderr

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


def test_startup_failure_reports_exhausted_cache_filesystem(tmp_path):
    server_log = tmp_path / "vllm.log"
    server_log.write_text(
        "OSError: [Errno 28] No space left on device\n", encoding="utf-8"
    )
    result = _run_sourced(
        "\n".join(
            (
                f"SERVER_LOG={shlex.quote(str(server_log))}",
                "report_startup_failure",
            )
        )
    )

    assert result.returncode == 0
    assert "cache filesystem ran out of space" in result.stderr
    assert "set CACHE_ROOT to a larger mounted filesystem" in result.stderr


def test_smoke_gate_retries_only_invalid_responses(tmp_path):
    output_root = tmp_path / "outputs"
    questions = tmp_path / "questions.jsonl"
    questions.write_text("{}\n", encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    _make_executable(
        fake_python,
        """#!/usr/bin/env bash
set -eu
count_file="$FAKE_COUNT_FILE"
count=0
if [[ -f "$count_file" ]]; then count="$(cat "$count_file")"; fi
count=$((count + 1))
printf '%s' "$count" > "$count_file"
diagnostics=""
resume=0
while (( $# )); do
  case "$1" in
    --diagnostics) diagnostics="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) shift ;;
  esac
done
printf '{"question_id":"q1","output":"1"}\n' > "$diagnostics"
if (( count == 1 )); then exit 2; fi
(( resume == 1 )) || exit 9
""",
    )
    count_file = tmp_path / "calls"

    result = _run_sourced(
        "\n".join(
            (
                f"PYTHON_BIN={shlex.quote(str(fake_python))}",
                f"FAKE_COUNT_FILE={shlex.quote(str(count_file))}",
                "export FAKE_COUNT_FILE",
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                f"track_questions() {{ printf '%s\\n' {shlex.quote(str(questions))}; }}",
                "track_module() { printf '%s\\n' fake.module; }",
                "runner_base_args() { RUNNER_ARGS=(--model test/model); }",
                "MAX_EVAL_ATTEMPTS=3",
                "SMOKE_SAMPLES=20",
                "SMOKE_ONLY=1",
                "FORCE=1",
                "sleep() { :; }",
                "run_track test-model test/model do_you_see_me",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert count_file.read_text(encoding="utf-8") == "2"
    assert "retrying only those samples with seed 1" in result.stdout
    assert (
        output_root
        / "test-model"
        / "do_you_see_me.smoke.attempt-1.diagnostics.jsonl"
    ).is_file()


def test_saved_smoke_checkpoint_uses_text_extraction_before_visual_retry(tmp_path):
    output_root = tmp_path / "outputs"
    model_dir = output_root / "test-model"
    model_dir.mkdir(parents=True)
    questions = tmp_path / "questions.jsonl"
    questions.write_text("{}\n", encoding="utf-8")
    smoke = model_dir / "minds_eye.smoke.diagnostics.jsonl"
    smoke.write_text(
        '{"question_id":"q1","output":"Option C seems most likely"}\n',
        encoding="utf-8",
    )
    archived = model_dir / "minds_eye.smoke.attempt-1.diagnostics.jsonl"
    archived.write_text(smoke.read_text(encoding="utf-8"), encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    marker = tmp_path / "extractor-called"
    _make_executable(
        fake_python,
        """#!/usr/bin/env bash
set -eu
extract=0
limit=""
resume=0
source_diagnostics=""
while (( $# )); do
  case "$1" in
    --extract-unparseable-only) extract=1; shift ;;
    --limit) limit="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --extraction-source-diagnostics) source_diagnostics="$2"; shift 2 ;;
    *) shift ;;
  esac
done
(( extract == 1 )) || exit 31
[[ "$limit" == "20" ]] || exit 32
(( resume == 1 )) || exit 33
[[ "$source_diagnostics" == *"smoke.attempt-1.diagnostics.jsonl" ]] || exit 34
printf 'called' > "$FAKE_MARKER"
""",
    )

    result = _run_sourced(
        "\n".join(
            (
                f"PYTHON_BIN={shlex.quote(str(fake_python))}",
                f"FAKE_MARKER={shlex.quote(str(marker))}",
                "export FAKE_MARKER",
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                f"track_questions() {{ printf '%s\\n' {shlex.quote(str(questions))}; }}",
                "track_module() { printf '%s\\n' fake.module; }",
                "runner_base_args() { RUNNER_ARGS=(--model test/model); }",
                "SMOKE_SAMPLES=20",
                "SMOKE_ONLY=1",
                "FORCE=0",
                "run_track test-model test/model minds_eye",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "called"
    assert "Resuming the saved" in result.stdout
    assert "Running strict" not in result.stdout


def test_extraction_only_mode_skips_smoke_and_visual_requests(tmp_path):
    output_root = tmp_path / "outputs"
    model_dir = output_root / "test-model"
    model_dir.mkdir(parents=True)
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{}\n', encoding="utf-8")
    diagnostics = model_dir / "do_you_see_me.diagnostics.jsonl"
    diagnostics.write_text(
        '{"question_id":"q1","output":"The answer may be 4 or 6"}\n',
        encoding="utf-8",
    )
    archived = model_dir / "do_you_see_me.attempt-1.diagnostics.jsonl"
    archived.write_text(
        '{"question_id":"q1","output":"The answer is 6"}\n',
        encoding="utf-8",
    )
    fake_python = tmp_path / "fake-python"
    _make_executable(
        fake_python,
        """#!/usr/bin/env bash
set -eu
extract=0
limit=0
output=""
source_diagnostics=""
while (( $# )); do
  case "$1" in
    --extract-unparseable-only) extract=1; shift ;;
    --limit) limit=1; shift 2 ;;
    --out) output="$2"; shift 2 ;;
    --extraction-source-diagnostics) source_diagnostics="$2"; shift 2 ;;
    *) shift ;;
  esac
done
(( extract == 1 )) || exit 21
(( limit == 0 )) || exit 22
[[ "$source_diagnostics" == *"attempt-1.diagnostics.jsonl" ]] || exit 23
printf '{"question_id":"q1","condition":"standard","answer":"6"}\n' > "$output"
""",
    )

    result = _run_sourced(
        "\n".join(
            (
                f"PYTHON_BIN={shlex.quote(str(fake_python))}",
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                f"track_questions() {{ printf '%s\\n' {shlex.quote(str(questions))}; }}",
                "track_module() { printf '%s\\n' fake.module; }",
                "runner_base_args() { RUNNER_ARGS=(--model test/model); }",
                "validate_submission() { [[ -s \"$1\" ]]; }",
                "EXTRACT_UNPARSEABLE_ONLY=1",
                "FORCE=0",
                "run_track test-model test/model do_you_see_me",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert "same served model (text only" in result.stdout
    assert "smoke test" not in result.stdout


def test_full_track_finalizes_exact_raw_output_after_recovery_fails(tmp_path):
    output_root = tmp_path / "outputs"
    model_dir = output_root / "test-model"
    model_dir.mkdir(parents=True)
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{}\n', encoding="utf-8")
    (model_dir / "minds_eye.smoke.diagnostics.jsonl").write_text(
        '{}\n', encoding="utf-8"
    )
    fake_python = tmp_path / "fake-python"
    count_file = tmp_path / "visual-calls"
    _make_executable(
        fake_python,
        """#!/usr/bin/env bash
set -eu
diagnostics=""
output=""
finalize=0
raw_fallback=0
while (( $# )); do
  case "$1" in
    --diagnostics) diagnostics="$2"; shift 2 ;;
    --out) output="$2"; shift 2 ;;
    --finalize-existing-diagnostics) finalize=1; shift ;;
    --raw-output-fallback) raw_fallback=1; shift ;;
    *) shift ;;
  esac
done
if (( finalize == 1 )); then
  (( raw_fallback == 1 )) || exit 31
  printf '{"question_id":"q1","condition":"standard","answer":"<|begin_of_box|><answer>G</answer>"}\n' > "$output"
  exit 0
fi
count=0
if [[ -f "$FAKE_COUNT_FILE" ]]; then count="$(cat "$FAKE_COUNT_FILE")"; fi
printf '%s' "$((count + 1))" > "$FAKE_COUNT_FILE"
printf '{"question_id":"q1","output":"<|begin_of_box|><answer>G</answer>"}\n' > "$diagnostics"
exit 2
""",
    )

    result = _run_sourced(
        "\n".join(
            (
                f"PYTHON_BIN={shlex.quote(str(fake_python))}",
                f"FAKE_COUNT_FILE={shlex.quote(str(count_file))}",
                "export FAKE_COUNT_FILE",
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                f"track_questions() {{ printf '%s\\n' {shlex.quote(str(questions))}; }}",
                "track_module() { printf '%s\\n' fake.module; }",
                "runner_base_args() { RUNNER_ARGS=(--model test/model); }",
                "run_smoke_answer_extraction() { return 0; }",
                "run_answer_extraction() { return 1; }",
                "validate_submission() { [[ -s \"$1\" ]]; }",
                "MAX_EVAL_ATTEMPTS=2",
                "FORCE=0",
                "sleep() { :; }",
                "run_track test-model test/model minds_eye",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert count_file.read_text(encoding="utf-8") == "2"
    submission = model_dir / "minds_eye_submission.jsonl"
    assert json.loads(submission.read_text(encoding="utf-8"))["answer"] == (
        "<|begin_of_box|><answer>G</answer>"
    )
    assert (model_dir / "minds_eye.attempt-1.diagnostics.jsonl").is_file()
    assert (model_dir / "minds_eye.attempt-2.diagnostics.jsonl").is_file()
    assert "exact faulty model outputs" in result.stdout


def test_model_attempts_all_selected_tracks_before_reporting_failure(tmp_path):
    calls = tmp_path / "track-calls"
    result = _run_sourced(
        "\n".join(
            (
                f"TRACK_CALLS={shlex.quote(str(calls))}",
                "export TRACK_CALLS",
                f"OUTPUT_ROOT={shlex.quote(str(tmp_path / 'outputs'))}",
                "MAX_MODEL_LEN=auto",
                "SMOKE_ONLY=0",
                "FORCE=0",
                "ensure_run_config() { :; }",
                "model_outputs_complete() { return 1; }",
                "start_server() { return 0; }",
                "model_request_name() { printf '%s\\n' test/request; }",
                "selected_tracks() { printf '%s\\n' do_you_see_me minds_eye; }",
                "run_track() { printf '%s\\n' \"$3\" >> \"$TRACK_CALLS\"; [[ \"$3\" == minds_eye ]]; }",
                "stop_server() { :; }",
                "delete_model_cache() { :; }",
                "if run_model test-model test/model revision-a unquantized 32768; then exit 91; fi",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "do_you_see_me",
        "minds_eye",
    ]


def test_raw_output_finalization_policy_upgrade_preserves_diagnostics(tmp_path):
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
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["unparseable_answers"] = {
        "policy": "deterministic-smoke-and-full-visual-retries-then-text-extraction-v4"
    }
    old_config["answer_extraction"]["local_parser"] = (
        "strict-local-final-answer-parser-v3-number-words-and-explicit-odd-figure"
    )
    old_config["source_hashes"]["runner"] = {
        "visual_pipeline": "strict-visual-pipeline-hash",
        "vllm_runner": "strict-vllm-runner-hash",
    }
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
    diagnostics_bytes = b'{"question_id":"q1","output":"G"}\n'
    diagnostics.write_bytes(diagnostics_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "exact raw-output finalization policy" in migrate.stdout
    assert diagnostics.read_bytes() == diagnostics_bytes
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["unparseable_answers"]["final_fallback"] == {
        "applies_after": ["full_visual_retries", "same_model_text_extraction"],
        "eligible_records": "complete-nonerror-nonempty-output",
        "source_field": "diagnostics.output",
        "transformation": "none",
    }
    latest = migrated["artifact_migrations"][-1]
    assert latest["reason"] == (
        "provenance-preserving-exact-raw-output-finalization-upgrade"
    )
    assert latest["previous_runner_source_hashes"] == {
        "visual_pipeline": "strict-visual-pipeline-hash",
        "vllm_runner": "strict-vllm-runner-hash",
    }
    assert latest["previous_local_parser"].startswith(
        "strict-local-final-answer-parser-v3"
    )
    assert latest["current_local_parser"].startswith(
        "strict-local-final-answer-parser-v5"
    )


def test_schema_v7_extraction_upgrade_preserves_existing_diagnostics(tmp_path):
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
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["schema_version"] = 7
    old_config["answer_extraction"] = "strict-local-final-answer-parser"
    old_config["unparseable_answers"] = {
        "policy": "retry-with-deterministic-seed-sequence-then-fail-v1"
    }
    old_config["pipeline_revision"] = (
        "unquantized-bf16-model-generation-final-answer-caps-v7"
    )
    old_config["source_hashes"]["runner"] = {
        "visual_pipeline": "old-visual-pipeline-hash",
        "vllm_runner": "old-vllm-runner-hash",
    }
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = model_dir / "do_you_see_me.diagnostics.jsonl"
    original_diagnostics = b'{"question_id":"q1","output":"raw response"}\n'
    diagnostics.write_bytes(original_diagnostics)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "diagnostics were preserved" in migrate.stdout
    assert diagnostics.read_bytes() == original_diagnostics
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 10
    assert migrated["artifact_migrations"] == [
        {
            "reason": "provenance-preserving-answer-extraction-upgrade",
            "from_schema_version": 7,
            "from_pipeline_revision": (
                "unquantized-bf16-model-generation-final-answer-caps-v7"
            ),
            "previous_answer_extraction": "strict-local-final-answer-parser",
            "previous_unparseable_answers": {
                "policy": "retry-with-deterministic-seed-sequence-then-fail-v1"
            },
            "previous_runner_source_hashes": {
                "visual_pipeline": "old-visual-pipeline-hash",
                "vllm_runner": "old-vllm-runner-hash",
            },
            "to_schema_version": 10,
            "to_pipeline_revision": (
                "unquantized-bf16-smoke-and-full-text-extraction-v10"
            ),
        }
    ]


def test_schema_v8_archived_source_upgrade_preserves_judged_diagnostics(tmp_path):
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
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["schema_version"] = 8
    old_config["pipeline_revision"] = (
        "unquantized-bf16-same-model-text-extraction-v8"
    )
    old_config["unparseable_answers"] = {
        "policy": "deterministic-visual-retries-then-same-model-text-extraction-v2"
    }
    for field in (
        "source_order",
        "source_deduplication",
        "source_provenance",
        "judge_attempt_history",
    ):
        old_config["answer_extraction"]["fallback"].pop(field)
    old_config["source_hashes"]["runner"] = {
        "visual_pipeline": "schema-8-visual-pipeline-hash",
        "vllm_runner": "schema-8-vllm-runner-hash",
    }
    prior_migration = {
        "reason": "provenance-preserving-answer-extraction-upgrade",
        "from_schema_version": 7,
        "to_schema_version": 8,
    }
    old_config["artifact_migrations"] = [prior_migration]
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = model_dir / "do_you_see_me.diagnostics.jsonl"
    judged_diagnostics = (
        b'{"question_id":"q1","output":"raw response","extractor_output":'
        b'"<answer>6</answer>","extracted_answer":"6"}\n'
    )
    diagnostics.write_bytes(judged_diagnostics)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "schema 8 to 10" in migrate.stdout
    assert diagnostics.read_bytes() == judged_diagnostics
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 10
    assert migrated["artifact_migrations"][0] == prior_migration
    latest = migrated["artifact_migrations"][1]
    assert latest["from_schema_version"] == 8
    assert latest["to_schema_version"] == 10
    assert latest["previous_runner_source_hashes"] == {
        "visual_pipeline": "schema-8-visual-pipeline-hash",
        "vllm_runner": "schema-8-vllm-runner-hash",
    }


def test_schema_v9_smoke_recovery_upgrade_preserves_smoke_diagnostics(tmp_path):
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
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["schema_version"] = 9
    old_config["pipeline_revision"] = (
        "unquantized-bf16-archived-text-extraction-v9"
    )
    old_config["unparseable_answers"] = {
        "policy": (
            "deterministic-visual-retries-then-current-and-archived-"
            "text-extraction-v3"
        )
    }
    old_config["answer_extraction"]["fallback"].pop("applies_after")
    old_config["source_hashes"]["runner"]["vllm_runner"] = (
        "schema-9-vllm-runner-hash"
    )
    prior_migrations = [
        {"from_schema_version": 7, "to_schema_version": 8},
        {"from_schema_version": 8, "to_schema_version": 9},
    ]
    old_config["artifact_migrations"] = prior_migrations
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    smoke = model_dir / "minds_eye.smoke.diagnostics.jsonl"
    smoke_bytes = b'{"question_id":"q1","output":"unfinished reasoning"}\n'
    smoke.write_bytes(smoke_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "schema 9 to 10" in migrate.stdout
    assert smoke.read_bytes() == smoke_bytes
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 10
    assert migrated["artifact_migrations"][:2] == prior_migrations
    latest = migrated["artifact_migrations"][2]
    assert latest["from_schema_version"] == 9
    assert latest["to_schema_version"] == 10
    assert latest["previous_runner_source_hashes"]["vllm_runner"] == (
        "schema-9-vllm-runner-hash"
    )


def test_local_parser_upgrade_preserves_existing_diagnostics(tmp_path):
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
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["answer_extraction"]["local_parser"] = (
        "strict-local-final-answer-parser-v4-innermost-glm-box"
    )
    old_config["source_hashes"]["runner"]["visual_pipeline"] = (
        "old-visual-pipeline-hash"
    )
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
    original_diagnostics = b'{"question_id":"q1","output":"Figure D does not adhere."}\n'
    diagnostics.write_bytes(original_diagnostics)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "strict local-parser upgrade" in migrate.stdout
    assert diagnostics.read_bytes() == original_diagnostics
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    latest = migrated["artifact_migrations"][-1]
    assert latest["reason"] == "provenance-preserving-local-parser-upgrade"
    assert latest["previous_local_parser"].startswith(
        "strict-local-final-answer-parser-v4"
    )
    assert latest["current_local_parser"].startswith(
        "strict-local-final-answer-parser-v5"
    )
    assert latest["previous_visual_pipeline_sha256"] == (
        "old-visual-pipeline-hash"
    )
    assert latest["current_visual_pipeline_sha256"] == migrated["source_hashes"][
        "runner"
    ]["visual_pipeline"]


def test_force_replaces_stale_artifacts_with_schema_v10_fingerprint(tmp_path):
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
    assert run_config["schema_version"] == 10
    assert run_config["reasoning_profile"] == "nonthinking"
    assert run_config["weight_loading"] == "unquantized"
    assert run_config["compute_dtype"] == "bfloat16"
    assert run_config["kv_cache_dtype"] == "bfloat16"
    assert run_config["tensor_parallel_size"] == 1
    assert run_config["data_parallel_size"] == 1
    assert run_config["request_concurrency"] == 1
    assert run_config["serving_engine"]["max_num_seqs_per_replica"] == 1
    assert run_config["serving_engine"]["gpu_memory_utilization"] == 0.84
    assert run_config["serving_engine"]["cuda_allocator_config"] == (
        "expandable_segments:True"
    )
    assert run_config["generation"]["do_you_see_me"]["max_tokens"] == 200
    assert run_config["generation"]["do_you_see_me"]["max_tokens_policy"] == (
        "explicit-total-completion-cap"
    )
    assert run_config["generation"]["do_you_see_me"][
        "final_answer_max_tokens"
    ] == 200
    assert run_config["generation"]["do_you_see_me"][
        "final_answer_token_enforcement"
    ] == "total-completion-cap"
    assert run_config["generation"]["do_you_see_me"]["stop_sequences"] == []
    assert not run_config["generation"]["do_you_see_me"][
        "include_stop_str_in_output"
    ]
    assert run_config["generation"]["minds_eye"]["prompt_mode"] == "cot"
    assert run_config["generation"]["minds_eye"]["max_tokens"] == 200
    assert run_config["generation"]["minds_eye"]["max_tokens_policy"] == (
        "explicit-total-completion-cap"
    )
    assert run_config["generation"]["minds_eye"]["stop_sequences"] == [
        "</answer>"
    ]
    assert run_config["generation"]["minds_eye"][
        "include_stop_str_in_output"
    ]
    assert run_config["image_preprocessing"].startswith("original-bytes")
    extraction = run_config["answer_extraction"]
    assert extraction["local_parser"].startswith("strict-local-final-answer-parser-v5")
    assert extraction["fallback"] == {
        "method": "same-served-model-text-only-v1",
        "model": "test/model",
        "input_fields": ["question", "answer_type", "candidate_response"],
        "image_supplied": False,
        "ground_truth_supplied": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 200,
        "stop_sequences": ["</answer>"],
        "include_stop_str_in_output": True,
        "support_validation": "answer-must-be-stated-in-candidate-response",
        "source_order": (
            "current-then-archived-attempts-in-ascending-attempt-number"
        ),
        "source_deduplication": "candidate-response-utf8-sha256",
        "source_provenance": [
            "diagnostics_filename",
            "candidate-response-utf8-sha256",
        ],
        "judge_attempt_history": "preserved",
        "applies_after": ["smoke_visual_retries", "full_visual_retries"],
    }
    assert run_config["unparseable_answers"] == {
        "policy": "deterministic-retries-text-extraction-then-exact-raw-output-v5",
        "final_fallback": {
            "applies_after": ["full_visual_retries", "same_model_text_extraction"],
            "eligible_records": "complete-nonerror-nonempty-output",
            "source_field": "diagnostics.output",
            "transformation": "none",
        },
    }
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


def test_serving_resource_upgrade_preserves_existing_artifacts(tmp_path):
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
    config_path = model_dir / ".run_config.json"
    legacy = json.loads(config_path.read_text(encoding="utf-8"))
    for key in (
        "max_num_seqs_per_replica",
        "gpu_memory_utilization",
        "cuda_allocator_config",
    ):
        legacy["serving_engine"].pop(key)
    config_path.write_text(
        json.dumps(legacy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostics = model_dir / "minds_eye.smoke.diagnostics.jsonl"
    diagnostics_bytes = b'{"question_id":"q1","output":"raw response"}\n'
    diagnostics.write_bytes(diagnostics_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config test-model test/model revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "Recorded bounded serving resources" in migrate.stdout
    assert diagnostics.read_bytes() == diagnostics_bytes
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["serving_engine"]["max_num_seqs_per_replica"] == 1
    assert migrated["serving_engine"]["gpu_memory_utilization"] == 0.84
    assert migrated["artifact_migrations"][-1] == {
        "reason": "provenance-preserving-serving-resource-upgrade",
        "schema_version": 10,
        "previous": {
            "max_num_seqs_per_replica": 1,
            "gpu_memory_utilization": 0.88,
            "cuda_allocator_config": "default",
        },
        "current": {
            "max_num_seqs_per_replica": 1,
            "gpu_memory_utilization": 0.84,
            "cuda_allocator_config": "expandable_segments:True",
        },
    }


def test_qwen25_completion_cap_recovery_archives_prior_diagnostics(tmp_path):
    output_root = tmp_path / "outputs"
    initialize = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config qwen25-vl-7b Qwen/Qwen2.5-VL-7B-Instruct revision-a unquantized 32768",
            )
        )
    )
    assert initialize.returncode == 0, initialize.stderr

    model_dir = output_root / "qwen25-vl-7b"
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["tensor_parallel_size"] = 2
    old_config["data_parallel_size"] = 2
    old_config["request_concurrency"] = 2
    old_config["serving_engine"]["gpu_memory_utilization"] = 0.84
    for track in ("do_you_see_me", "minds_eye"):
        generation = old_config["generation"][track]
        generation["max_tokens"] = 200
        generation["max_tokens_policy"] = "explicit-total-completion-cap"
        generation["final_answer_token_enforcement"] = "total-completion-cap"
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
    diagnostics_bytes = (
        b'{"question_id":"q1","output":"complete"}\n'
        b'{"question_id":"q2","output":"truncated","extractor_error":"unsupported"}\n'
    )
    diagnostics.write_bytes(diagnostics_bytes)
    attempt = model_dir / "minds_eye.attempt-1.diagnostics.jsonl"
    attempt_bytes = b'{"question_id":"q2","output":"first truncation"}\n'
    attempt.write_bytes(attempt_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "GPU_MEMORY_UTILIZATION=0.65",
                "FORCE=0",
                "ensure_run_config qwen25-vl-7b Qwen/Qwen2.5-VL-7B-Instruct revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "only unresolved records will be regenerated" in migrate.stdout
    assert diagnostics.read_bytes() == diagnostics_bytes
    diagnostics_archive = (
        model_dir / "minds_eye.pre-completion-cap-recovery.diagnostics.jsonl"
    )
    assert diagnostics_archive.read_bytes() == diagnostics_bytes
    assert not attempt.exists()
    attempt_archive = (
        model_dir
        / "minds_eye.pre-completion-cap-recovery.attempt-1.diagnostics.jsonl"
    )
    assert attempt_archive.read_bytes() == attempt_bytes

    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    latest = migrated["artifact_migrations"][-1]
    assert latest["reason"] == "provenance-preserving-completion-cap-recovery"
    assert latest["scope"] == "resume-valid-records-and-regenerate-only-unresolved"
    assert latest["baseline_diagnostics"]["row_count"] == 2
    assert latest["baseline_diagnostics"]["extractor_error_count"] == 1
    assert latest["previous_generation"]["max_tokens"] == 200
    assert latest["current_generation"]["max_tokens"] == 8192
    assert latest["previous_serving"]["tensor_parallel_size"] == 2
    assert latest["current_serving"] == {
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "request_concurrency": 1,
        "max_num_seqs_per_replica": 1,
        "gpu_memory_utilization": 0.65,
        "cuda_allocator_config": "expandable_segments:True",
    }


def test_gemma_minds_eye_completion_cap_recovery_preserves_completed_dys(tmp_path):
    output_root = tmp_path / "outputs"
    initialize = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config gemma3-12b-it google/gemma-3-12b-it revision-a unquantized 32768",
            )
        )
    )
    assert initialize.returncode == 0, initialize.stderr

    model_dir = output_root / "gemma3-12b-it"
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    generation = old_config["generation"]["minds_eye"]
    generation["max_tokens"] = 200
    generation["max_tokens_policy"] = "explicit-total-completion-cap"
    generation["final_answer_token_enforcement"] = "total-completion-cap"
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dys_diagnostics = model_dir / "do_you_see_me.diagnostics.jsonl"
    dys_submission = model_dir / "do_you_see_me_submission.jsonl"
    dys_diagnostics_bytes = b"".join(
        json.dumps({"question_id": f"q{index}", "output": "1"}).encode() + b"\n"
        for index in range(4500)
    )
    dys_submission_bytes = b"".join(
        json.dumps(
            {"question_id": f"q{index}", "condition": "standard", "answer": "1"}
        ).encode()
        + b"\n"
        for index in range(4500)
    )
    dys_diagnostics.write_bytes(dys_diagnostics_bytes)
    dys_submission.write_bytes(dys_submission_bytes)
    smoke = model_dir / "minds_eye.smoke.diagnostics.jsonl"
    smoke_bytes = b'{"question_id":"m1","output":"truncated"}\n'
    smoke.write_bytes(smoke_bytes)
    attempt = model_dir / "minds_eye.smoke.attempt-1.diagnostics.jsonl"
    attempt_bytes = b'{"question_id":"m1","output":"first truncation"}\n'
    attempt.write_bytes(attempt_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=0",
                "ensure_run_config gemma3-12b-it google/gemma-3-12b-it revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "completed DYS artifacts were preserved" in migrate.stdout
    assert dys_diagnostics.read_bytes() == dys_diagnostics_bytes
    assert dys_submission.read_bytes() == dys_submission_bytes
    assert not smoke.exists()
    assert not attempt.exists()
    assert (
        model_dir
        / "minds_eye.pre-completion-cap-recovery.smoke.diagnostics.jsonl"
    ).read_bytes() == smoke_bytes
    assert (
        model_dir
        / "minds_eye.pre-completion-cap-recovery.smoke.attempt-1.diagnostics.jsonl"
    ).read_bytes() == attempt_bytes

    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["generation"]["do_you_see_me"]["max_tokens"] == 200
    assert migrated["generation"]["minds_eye"]["max_tokens"] == 8192
    latest = migrated["artifact_migrations"][-1]
    assert latest["reason"] == (
        "provenance-preserving-gemma-minds-eye-completion-cap-recovery"
    )
    assert latest["scope"] == "preserve-completed-dys-and-rerun-minds-eye-smoke"
    assert latest["completed_do_you_see_me_artifacts"][
        "do_you_see_me.diagnostics.jsonl"
    ]["row_count"] == 4500
    assert len(latest["archived_smoke_diagnostics"]) == 2


def test_qwen3_topology_resume_archives_prior_checkpoint(tmp_path):
    output_root = tmp_path / "outputs"
    initialize = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=6",
                "CONCURRENCY=6",
                "FORCE=1",
                "ensure_run_config qwen3-vl-8b Qwen/Qwen3-VL-8B-Instruct revision-a unquantized 32768",
            )
        )
    )
    assert initialize.returncode == 0, initialize.stderr

    model_dir = output_root / "qwen3-vl-8b"
    diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
    diagnostics_bytes = b'{"question_id":"q1","output":"checkpoint"}\n'
    diagnostics.write_bytes(diagnostics_bytes)
    attempt = model_dir / "minds_eye.smoke.attempt-1.diagnostics.jsonl"
    attempt_bytes = b'{"question_id":"q1","output":"first smoke"}\n'
    attempt.write_bytes(attempt_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "FORCE=0",
                "ensure_run_config qwen3-vl-8b Qwen/Qwen3-VL-8B-Instruct revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "serving-topology resume" in migrate.stdout
    assert diagnostics.read_bytes() == diagnostics_bytes
    archive = model_dir / "minds_eye.pre-serving-topology-resume.diagnostics.jsonl"
    assert archive.read_bytes() == diagnostics_bytes
    assert not attempt.exists()
    attempt_archive = (
        model_dir
        / "minds_eye.pre-serving-topology-resume.smoke.attempt-1.diagnostics.jsonl"
    )
    assert attempt_archive.read_bytes() == attempt_bytes
    migrated = json.loads(
        (model_dir / ".run_config.json").read_text(encoding="utf-8")
    )
    latest = migrated["artifact_migrations"][-1]
    assert latest["reason"] == "provenance-preserving-serving-topology-resume"
    assert latest["baseline_diagnostics"]["minds_eye"]["row_count"] == 1
    assert latest["previous_serving"]["data_parallel_size"] == 6
    assert latest["current_serving"]["data_parallel_size"] == 1


def test_glm_completion_and_seed_recovery_archives_prior_attempts(tmp_path):
    output_root = tmp_path / "outputs"
    initialize = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "BASE_SEED=3",
                "ANSWER_EXTRACTOR_SEED=0",
                "FORCE=1",
                "ensure_run_config glm46v-flash zai-org/GLM-4.6V-Flash revision-a unquantized 32768",
            )
        )
    )
    assert initialize.returncode == 0, initialize.stderr

    model_dir = output_root / "glm46v-flash"
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["data_parallel_size"] = 4
    old_config["request_concurrency"] = 4
    old_config["generation"]["base_seed"] = 0
    for track in ("do_you_see_me", "minds_eye"):
        generation = old_config["generation"][track]
        generation["max_tokens"] = 200
        generation["max_tokens_policy"] = "explicit-total-completion-cap"
        generation["final_answer_token_enforcement"] = "total-completion-cap"
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = model_dir / "do_you_see_me.diagnostics.jsonl"
    diagnostics_bytes = (
        b'{"question_id":"q1","output":"complete"}\n'
        b'{"question_id":"q2","output":"empty","extractor_error":"unsupported"}\n'
    )
    diagnostics.write_bytes(diagnostics_bytes)
    attempt = model_dir / "do_you_see_me.attempt-1.diagnostics.jsonl"
    attempt_bytes = b'{"question_id":"q2","output":"first attempt"}\n'
    attempt.write_bytes(attempt_bytes)
    smoke = model_dir / "do_you_see_me.smoke.diagnostics.jsonl"
    smoke_bytes = b'{"question_id":"q1","output":"smoke"}\n'
    smoke.write_bytes(smoke_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "BASE_SEED=3",
                "ANSWER_EXTRACTOR_SEED=0",
                "FORCE=0",
                "ensure_run_config glm46v-flash zai-org/GLM-4.6V-Flash revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "GLM completion and seed recovery" in migrate.stdout
    assert diagnostics.read_bytes() == diagnostics_bytes
    archive = model_dir / "do_you_see_me.pre-completion-cap-recovery.diagnostics.jsonl"
    assert archive.read_bytes() == diagnostics_bytes
    assert not attempt.exists()
    assert not smoke.exists()
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    latest = migrated["artifact_migrations"][-1]
    assert latest["reason"] == "provenance-preserving-completion-and-seed-recovery"
    assert latest["baseline_diagnostics"]["row_count"] == 2
    assert latest["previous_base_seed"] == 0
    assert latest["current_base_seed"] == 3
    assert latest["previous_generation"]["max_tokens"] == 200
    assert latest["current_generation"]["max_tokens"] == 8192
    assert len(latest["archived_diagnostics"]) == 2

    resume = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "BASE_SEED=3",
                "ANSWER_EXTRACTOR_SEED=0",
                "FORCE=0",
                "ensure_run_config glm46v-flash zai-org/GLM-4.6V-Flash revision-a unquantized 32768",
            )
        )
    )
    assert resume.returncode == 0, resume.stderr


def test_glm_extended_format_retry_archives_failed_checkpoint(tmp_path):
    output_root = tmp_path / "outputs"
    initialize = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "BASE_SEED=3",
                "ANSWER_EXTRACTOR_SEED=0",
                "MAX_EVAL_ATTEMPTS=3",
                "FORCE=1",
                "ensure_run_config glm46v-flash zai-org/GLM-4.6V-Flash revision-a unquantized 32768",
            )
        )
    )
    assert initialize.returncode == 0, initialize.stderr

    model_dir = output_root / "glm46v-flash"
    config_path = model_dir / ".run_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config["answer_extraction"]["local_parser"] = (
        "strict-local-final-answer-parser-v3-number-words-and-explicit-odd-figure"
    )
    old_config["source_hashes"]["runner"]["visual_pipeline"] = (
        "pre-nested-box-parser-hash"
    )
    config_path.write_text(
        json.dumps(old_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diagnostics = model_dir / "minds_eye.diagnostics.jsonl"
    diagnostics_bytes = (
        b'{"question_id":"q1","output":"<|begin_of_box|>E<|end_of_box|>"}\n'
        b'{"question_id":"q2","output":"<answer>G</answer>"}\n'
    )
    diagnostics.write_bytes(diagnostics_bytes)
    attempts = []
    for attempt_number in range(1, 4):
        attempt = model_dir / f"minds_eye.attempt-{attempt_number}.diagnostics.jsonl"
        attempt_bytes = (
            f'{{"question_id":"q2","output":"attempt {attempt_number}"}}\n'.encode()
        )
        attempt.write_bytes(attempt_bytes)
        attempts.append((attempt, attempt_bytes))
    smoke = model_dir / "minds_eye.smoke.diagnostics.jsonl"
    smoke_bytes = b'{"question_id":"q1","output":"E"}\n'
    smoke.write_bytes(smoke_bytes)

    migrate = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "BASE_SEED=3",
                "ANSWER_EXTRACTOR_SEED=0",
                "MAX_EVAL_ATTEMPTS=6",
                "FORCE=0",
                "ensure_run_config glm46v-flash zai-org/GLM-4.6V-Flash revision-a unquantized 32768",
            )
        )
    )

    assert migrate.returncode == 0, migrate.stderr
    assert "extended GLM formatting recovery" in migrate.stdout
    assert diagnostics.read_bytes() == diagnostics_bytes
    baseline = model_dir / "minds_eye.pre-extended-format-retry.diagnostics.jsonl"
    assert baseline.read_bytes() == diagnostics_bytes
    for attempt, attempt_bytes in attempts:
        assert not attempt.exists()
        archive = model_dir / attempt.name.replace(
            "minds_eye.", "minds_eye.pre-extended-format-retry.", 1
        )
        assert archive.read_bytes() == attempt_bytes
    assert smoke.read_bytes() == smoke_bytes
    smoke_archive = (
        model_dir / "minds_eye.pre-extended-format-retry.smoke.diagnostics.jsonl"
    )
    assert smoke_archive.read_bytes() == smoke_bytes

    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    latest = migrated["artifact_migrations"][-1]
    assert latest["reason"] == "provenance-preserving-glm-extended-format-retry"
    assert latest["baseline_diagnostics"]["row_count"] == 2
    assert len(latest["archived_attempt_diagnostics"]) == 3
    assert latest["previous_format_retry_attempts"] == 3
    assert latest["current_format_retry_attempts"] == 6
    assert latest["previous_local_parser"].startswith(
        "strict-local-final-answer-parser-v3"
    )
    assert latest["current_local_parser"].startswith(
        "strict-local-final-answer-parser-v5"
    )

    resume = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "DATA_PARALLEL_SIZE=1",
                "CONCURRENCY=1",
                "BASE_SEED=3",
                "ANSWER_EXTRACTOR_SEED=0",
                "MAX_EVAL_ATTEMPTS=6",
                "FORCE=0",
                "ensure_run_config glm46v-flash zai-org/GLM-4.6V-Flash revision-a unquantized 32768",
            )
        )
    )
    assert resume.returncode == 0, resume.stderr


def test_phi_fingerprint_records_required_vision_adapter(tmp_path):
    output_root = tmp_path / "outputs"
    result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config phi4-multimodal microsoft/Phi-4-multimodal-instruct revision-a unquantized 32768",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    run_config = json.loads(
        (output_root / "phi4-multimodal" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_config["serving_engine"]["request_model"] == "vision"
    assert run_config["reasoning_profile"] == "nonthinking"
    assert run_config["adapter"] == {
        "name": "vision",
        "source": "vision-lora",
        "revision": "revision-a",
    }


def test_internvl_fingerprint_records_generation_and_answer_caps(tmp_path):
    output_root = tmp_path / "outputs"
    result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config internvl35-8b OpenGVLab/InternVL3_5-8B revision-a unquantized 32768",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    run_config = json.loads(
        (output_root / "internvl35-8b" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_config["reasoning_profile"] == "thinking"
    for track in ("do_you_see_me", "minds_eye"):
        generation = run_config["generation"][track]
        assert generation["max_tokens"] == 8192
        assert generation["max_tokens_policy"] == "explicit-model-completion-cap"
        assert generation["final_answer_max_tokens"] == 200
        assert generation["final_answer_token_enforcement"] == (
            "post-extraction-served-model-tokenizer"
        )


def test_qwen3_fingerprint_records_generation_and_answer_caps(tmp_path):
    output_root = tmp_path / "outputs"
    result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config qwen3-vl-8b Qwen/Qwen3-VL-8B-Instruct revision-a unquantized 32768",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    run_config = json.loads(
        (output_root / "qwen3-vl-8b" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_config["reasoning_profile"] == "nonthinking"
    for track in ("do_you_see_me", "minds_eye"):
        generation = run_config["generation"][track]
        assert generation["max_tokens"] == 8192
        assert generation["max_tokens_policy"] == "explicit-model-completion-cap"
        assert generation["final_answer_max_tokens"] == 200
        assert generation["final_answer_token_enforcement"] == (
            "post-extraction-served-model-tokenizer"
        )


def test_qwen25_fingerprint_records_generation_and_answer_caps(tmp_path):
    output_root = tmp_path / "outputs"
    result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config qwen25-vl-7b Qwen/Qwen2.5-VL-7B-Instruct revision-a unquantized 32768",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    run_config = json.loads(
        (output_root / "qwen25-vl-7b" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_config["reasoning_profile"] == "nonthinking"
    for track in ("do_you_see_me", "minds_eye"):
        generation = run_config["generation"][track]
        assert generation["max_tokens"] == 8192
        assert generation["max_tokens_policy"] == "explicit-model-completion-cap"
        assert generation["final_answer_max_tokens"] == 200
        assert generation["final_answer_token_enforcement"] == (
            "post-extraction-served-model-tokenizer"
        )


def test_phi_fingerprint_records_official_modelscope_snapshot(tmp_path):
    output_root = tmp_path / "outputs"
    result = _run_sourced(
        "\n".join(
            (
                'PYTHON_BIN="$(command -v python3)"',
                f"OUTPUT_ROOT={shlex.quote(str(output_root))}",
                f"PHI_OFFICIAL_SNAPSHOT_PATH={shlex.quote(str(tmp_path / 'snapshot'))}",
                "TENSOR_PARALLEL_SIZE=1",
                "FORCE=1",
                "ensure_run_config phi4-multimodal microsoft/Phi-4-multimodal-instruct revision-a unquantized 32768",
            )
        )
    )

    assert result.returncode == 0, result.stderr
    run_config = json.loads(
        (output_root / "phi4-multimodal" / ".run_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_config["checkpoint_source"] == {
        "provider": "modelscope-official-git",
        "repo_id": "microsoft/Phi-4-multimodal-instruct",
        "revision": "7641bf905e6965ee54166808d275266371e28343",
        "object_sha256": {
            "model-00001-of-00003.safetensors": "c46bb03332d82f6a3eaf85bd20af388dd4d4d68b198c2203c965c7381a466094",
            "model-00002-of-00003.safetensors": "b3e812c0c8acef4e7f5e34d6c9f77a7640ee4a2b93ea351921365ac62f19918d",
            "model-00003-of-00003.safetensors": "7be96b7339303752634b202d3f377bcf312a03046586eca6cea23347ace1e65a",
            "vision-lora/adapter_model.safetensors": "1620b16722edf701038bf66e3cd46412c7cc5458e58df89e9f92cedb71fcbde8",
        },
    }
    assert run_config["adapter"]["revision"] == run_config["checkpoint_source"][
        "revision"
    ]


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
        '{"question_id":"q1","condition":"standard","answer":"G"}\n',
        encoding="utf-8",
    )
    (model_dir / "do_you_see_me.diagnostics.jsonl").write_text(
        '{"question_id":"q1","answer_type":"mcq_letter","output":"G"}\n',
        encoding="utf-8",
    )
    attempt = model_dir / "do_you_see_me.attempt-1.diagnostics.jsonl"
    attempt.write_text(
        '{"question_id":"q1","output":"unparseable"}\n', encoding="utf-8"
    )
    smoke_attempt = model_dir / "do_you_see_me.smoke.attempt-1.diagnostics.jsonl"
    smoke_attempt.write_text(
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
    assert manifest["schema_version"] == 10
    assert manifest["reasoning_profile"] == "nonthinking"
    assert manifest["checkpoint_source"] == {
        "provider": "huggingface",
        "repo_id": "test/model",
        "revision": "revision-a",
        "object_sha256": {},
    }
    archived = manifest["tracks"]["do_you_see_me"]["failed_attempt_diagnostics"]
    assert manifest["tracks"]["do_you_see_me"][
        "exact_raw_output_fallback_question_ids"
    ] == ["q1"]
    assert archived == [
        {
            "file": attempt.name,
            "seed": 0,
            "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest(),
        }
    ]
    assert manifest["tracks"]["do_you_see_me"][
        "failed_smoke_attempt_diagnostics"
    ] == [
        {
            "file": smoke_attempt.name,
            "seed": 0,
            "sha256": hashlib.sha256(smoke_attempt.read_bytes()).hexdigest(),
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
            "MIN_FREE_DISK_RESERVE_GB": "1",
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
    assert "required 3 GiB (1 GiB host reserve + 1 GiB x 2 models)" in result.stdout
    assert "no workers were started" in result.stdout

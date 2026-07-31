from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "spatial_harness" / "run_eval.sh"
INSTALLER = SCRIPT.with_name("install_track3_env.sh")
TWO_GPU_QUEUE = SCRIPT.with_name("run_two_gpu_research_queue.sh")
DATASETS = (
    "BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,"
    "MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,"
    "OmniSpatial,SAT-Real"
)


def test_launcher_uses_v2_contract_and_separate_endpoints():
    script = SCRIPT.read_text(encoding="utf-8")
    assert f'DATASETS="{DATASETS}"' in script
    assert '--max-tokens-noncot "${MAX_TOKENS:-0}"' in script
    assert '--max-tokens-cot "${MAX_TOKENS:-0}"' in script
    assert "--modes main noimage noimgpp" in script
    assert '--endpoints "$VLM_ENDPOINTS"' in script
    assert '--endpoint "$JUDGE_ENDPOINT"' in script
    assert 'PAPER_JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"' in script
    assert 'PAPER_JUDGE_REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"' in script
    assert '--model-revision "$MODEL_REVISION"' in script
    assert "--verify-only" in script
    assert "--manifest" not in script


def test_environment_installer_verifies_editable_vlmevalkit_checkout():
    script = INSTALLER.read_text(encoding="utf-8")
    assert 'VLMEVALKIT_COMMIT="7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f"' in script
    assert 'actual_commit="$(git -C "$VLMEVALKIT_SOURCE" rev-parse HEAD)"' in script
    assert 'git -C "$VLMEVALKIT_SOURCE" status --porcelain' in script
    assert 'conda install -y --prefix "$ENV_PREFIX" --file' in script
    assert 'pip install --no-deps -e "$VLMEVALKIT_SOURCE"' in script
    assert "unset HF_HUB_ENABLE_HF_TRANSFER" in script
    assert 'export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"' in script
    assert "python -m pip check" not in script
    assert '"$ENV_PREFIX/bin/python" -m pip check' in script


def test_two_gpu_queue_is_pinned_and_keeps_judge_independent():
    script = TWO_GPU_QUEUE.read_text(encoding="utf-8")
    assert 'GPU_IDS="${GPU_IDS:-0,1}"' in script
    assert "--tensor-parallel-size 2" in script
    assert "--dtype bfloat16" in script
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"' in script
    assert 'JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-33792}"' in script
    assert '--server-max-model-len "$JUDGE_MAX_MODEL_LEN"' in script
    assert 'MAX_TOKENS="${MAX_TOKENS:-0}"' in script
    assert "--modes main noimage noimgpp" in script
    assert 'JUDGE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"' in script
    assert 'JUDGE_STARTUP_TIMEOUT_SECONDS="${JUDGE_STARTUP_TIMEOUT_SECONDS:-3600}"' in script
    assert 'REBUILD_CONTRACT="${REBUILD_CONTRACT:-0}"' in script
    assert "Reusing the existing verified public Track-3 contract" in script
    assert "wait_for_model_server" in script
    assert 'curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models"' in script
    assert 'run_judge "OpenGVLab__InternVL3_5-8B"' in script
    assert 'run_judge "Qwen__Qwen3.6-27B"' in script

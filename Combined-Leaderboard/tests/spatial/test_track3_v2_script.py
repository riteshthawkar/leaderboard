from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "spatial_harness" / "run_eval.sh"
INSTALLER = SCRIPT.with_name("install_track3_env.sh")
DATASETS = (
    "BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,"
    "MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,"
    "OmniSpatial,SAT-Real"
)


def test_launcher_uses_v2_contract_and_separate_endpoints():
    script = SCRIPT.read_text(encoding="utf-8")
    assert f'DATASETS="{DATASETS}"' in script
    assert "--max-tokens-noncot 16384" in script
    assert "--max-tokens-cot 16384" in script
    assert '--endpoints "$VLM_ENDPOINTS"' in script
    assert '--endpoint "$JUDGE_ENDPOINT"' in script
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
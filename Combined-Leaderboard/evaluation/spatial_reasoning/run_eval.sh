#!/usr/bin/env bash
# One-command official six-condition evaluation for a served vision-language model.
#
# ENGINE-AGNOSTIC: the endpoints are any OpenAI-compatible /v1 URLs — serve your model and the judge
# with whatever you like (vLLM, SGLang, TGI, llama.cpp server, a hosted API, ...).
#
# Usage:
#   ./run_eval.sh <LEADERBOARD_MODEL_NAME> <VL_ENDPOINTS> <JUDGE_ENDPOINT> [OUT_DIR]
# Example:
#   ./run_eval.sh my-vlm http://localhost:8000/v1 http://localhost:8100/v1
#   ./run_eval.sh my-vlm http://host:8000/v1,http://host:8001/v1 http://host:8100/v1 results/my-vlm
#
# Prereqs: run  `python prepare_data.py --lmudata ./LMUData`  once, and serve your VL model + the
# judge (Qwen3-30B-A3B-Instruct-2507) — see serve/serve_example.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 3 ]; then
  echo "usage: $0 <leaderboard_model_name> <vl_endpoints(comma-sep)> <judge_endpoint> [out_dir]"
  echo "  endpoints are any OpenAI-compatible /v1 URLs (vLLM / SGLang / TGI / ...)."
  exit 1
fi
LEADERBOARD_MODEL_NAME="$1"
VL_EP="$2"
JUDGE_EP="$3"
ENDPOINT_MODEL="${SPATIAL_ENDPOINT_MODEL:-$LEADERBOARD_MODEL_NAME}"
JUDGE_ENDPOINT_MODEL="${SPATIAL_JUDGE_ENDPOINT_MODEL:-judge}"
OUT="${4:-$HERE/results/$LEADERBOARD_MODEL_NAME}"
LMUDATA="${LMUDATA:-$HERE/LMUData}"
ABLATION_MANIFEST="${ABLATION_MANIFEST:-$HERE/ablation_manifest.json}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-$HERE/benchmark_manifest.json}"
# the paper's 13 benchmarks (uses VSR_MCQ / MMSIBench_wo_circular, not the raw variants); override via $DATASETS
DATASETS="${DATASETS:-BLINK,CV-Bench-2D,CV-Bench-3D,MMVP,RealWorldQA,VStarBench,MMSIBench_wo_circular,3DSRBench,VSR_MCQ,SpatialBench,MindCube,OmniSpatial,SAT-Real}"

[ -d "$LMUDATA" ] || { echo "No data at $LMUDATA — first run:  python $HERE/prepare_data.py --lmudata $LMUDATA"; exit 1; }
[ -f "$BENCHMARK_MANIFEST" ] || {
  echo "No official benchmark manifest at $BENCHMARK_MANIFEST"
  echo "Download it from the leaderboard, place it there, or set BENCHMARK_MANIFEST=/path/to/manifest.json."
  exit 1
}
[ -f "$ABLATION_MANIFEST" ] || { echo "No ablation manifest at $ABLATION_MANIFEST"; exit 1; }

echo "== [1/2] inference: '$LEADERBOARD_MODEL_NAME' via '$ENDPOINT_MODEL' (13 datasets, six required conditions) -> $OUT =="
python3 "$HERE/run_track3_vllm.py" --lmudata "$LMUDATA" --datasets "$DATASETS" \
    --ablation-manifest "$ABLATION_MANIFEST" --benchmark-manifest "$BENCHMARK_MANIFEST" \
    --endpoint-model "$ENDPOINT_MODEL" --leaderboard-model-name "$LEADERBOARD_MODEL_NAME" \
    --endpoints "$VL_EP" --out "$OUT"

echo "== [2/2] judge and create leaderboard submission artifacts =="
python3 "$HERE/judge_track3.py" --results-dir "$OUT" --endpoints "$JUDGE_EP" \
    --endpoint-model "$JUDGE_ENDPOINT_MODEL"

echo
echo "Done. Upload this single file to the leaderboard:"
echo "  $OUT/spatial_reasoning_submission.zip"
echo "The package contains submission.jsonl, run_manifest.json, and leaderboard.json."
echo "Do not modify or repackage it. Accepted evidence is retained and published for audit."

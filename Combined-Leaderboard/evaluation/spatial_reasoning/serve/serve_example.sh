#!/usr/bin/env bash
# Reference serving setup for Track-3 (engine-agnostic; any OpenAI-compatible server works).
# The eval code (run_track3_vllm.py / judge_track3.py) only needs a base URL — it does not care
# what serves it. One replica per GPU gives the best throughput; the client round-robins --endpoints.

# ---------------------------------------------------------------------------
# 1) VL model under test — one server per GPU (ports 8000, 8001, ...)
#    Give the CoT answer budget headroom: serve max-model-len 40960 so 16384 generated tokens fit
#    ALONGSIDE the image+question prompt. (The paper used a 32768 context, but 32768 generated +
#    prompt is rejected by strict servers — see KNOWN_ISSUES C-10.) Allow multiple images per prompt.
# ---------------------------------------------------------------------------
#   vllm serve <VL_MODEL_PATH> --served-model-name <NAME> --port 8000 \
#       --dtype bfloat16 --max-model-len 40960 --limit-mm-per-prompt '{"image": 8}'
#
#   # one command (inference + judge -> leaderboard.json):
#   ./run_eval.sh <NAME> http://localhost:8000/v1,http://localhost:8001/v1 http://localhost:8100/v1
#   # ...or call inference directly for finer control:
#   python run_track3_vllm.py --endpoint-model <SERVED_ID> --leaderboard-model-name <NAME> --lmudata LMUData \
#       --endpoints http://localhost:8000/v1,http://localhost:8001/v1 \
#       --modes main,noimgpp --pmodes noncot,cot --out results/<NAME>

# ---------------------------------------------------------------------------
# 2) Judge — the paper's Qwen3-30B-A3B-Instruct-2507 (text), ~2 GPUs.
# ---------------------------------------------------------------------------
#   vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --served-model-name judge --port 8100 \
#       --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 8192
#
#   python judge_track3.py --results-dir results/<NAME> --endpoints http://localhost:8100/v1

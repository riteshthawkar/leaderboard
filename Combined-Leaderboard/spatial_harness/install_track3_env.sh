#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${TRACK3_CONDA_ENV:-/share/data/drive_3/conda_envs/track3-v2}"
VLMEVALKIT_SOURCE="${VLMEVALKIT_SOURCE:-/share/data/drive_3/vendor/VLMEvalKit-7055d301}"
VLMEVALKIT_REPOSITORY="https://github.com/open-compass/VLMEvalKit.git"
VLMEVALKIT_COMMIT="7055d3010c38ccb5dcae1bc9535ca19c7fe5d79f"

unset HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y --prefix "$ENV_PREFIX" --file "$SCRIPT_DIR/conda-linux-64.lock"
else
  conda install -y --prefix "$ENV_PREFIX" --file "$SCRIPT_DIR/conda-linux-64.lock"
fi

if [[ ! -d "$VLMEVALKIT_SOURCE/.git" ]]; then
  [[ ! -e "$VLMEVALKIT_SOURCE" ]] || {
    printf 'VLMEvalKit source exists but is not a Git checkout: %s\n' "$VLMEVALKIT_SOURCE" >&2
    exit 2
  }
  git clone --filter=blob:none --no-checkout "$VLMEVALKIT_REPOSITORY" "$VLMEVALKIT_SOURCE"
  git -C "$VLMEVALKIT_SOURCE" fetch --depth 1 origin "$VLMEVALKIT_COMMIT"
  git -C "$VLMEVALKIT_SOURCE" checkout --detach "$VLMEVALKIT_COMMIT"
fi

actual_commit="$(git -C "$VLMEVALKIT_SOURCE" rev-parse HEAD)"
[[ "$actual_commit" == "$VLMEVALKIT_COMMIT" ]] || {
  printf 'VLMEvalKit checkout is %s; required %s\n' "$actual_commit" "$VLMEVALKIT_COMMIT" >&2
  exit 2
}
[[ -z "$(git -C "$VLMEVALKIT_SOURCE" status --porcelain)" ]] || {
  printf 'VLMEvalKit checkout has uncommitted changes: %s\n' "$VLMEVALKIT_SOURCE" >&2
  exit 2
}

"$ENV_PREFIX/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-track3-v2.lock.txt"
"$ENV_PREFIX/bin/python" -m pip install --no-deps -e "$VLMEVALKIT_SOURCE"
"$ENV_PREFIX/bin/python" -m pip check
"$ENV_PREFIX/bin/python" -c \
  "import decord, pyarrow, vlmeval; print('Track-3 environment ready:', vlmeval.__file__)"
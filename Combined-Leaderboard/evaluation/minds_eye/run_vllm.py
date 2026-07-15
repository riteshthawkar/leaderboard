"""Run Mind's Eye through one or more OpenAI-compatible VLM endpoints."""

from evaluation.common.vllm_runner import main
from evaluation.minds_eye.config import TRACK


if __name__ == "__main__":
    raise SystemExit(main(TRACK))

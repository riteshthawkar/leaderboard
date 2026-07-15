"""Run Do You See Me through one or more OpenAI-compatible VLM endpoints."""

from evaluation.common.vllm_runner import main
from evaluation.do_you_see_me.config import TRACK


if __name__ == "__main__":
    raise SystemExit(main(TRACK))

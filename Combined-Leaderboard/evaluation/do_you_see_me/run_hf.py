"""Run Do You See Me directly with Hugging Face transformers."""

from evaluation.common.hf_runner import main
from evaluation.do_you_see_me.config import TRACK


if __name__ == "__main__":
    raise SystemExit(main(TRACK))

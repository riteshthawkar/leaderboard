"""Run Mind's Eye directly with Hugging Face transformers."""

from evaluation.common.hf_runner import main
from evaluation.minds_eye.config import TRACK


if __name__ == "__main__":
    raise SystemExit(main(TRACK))

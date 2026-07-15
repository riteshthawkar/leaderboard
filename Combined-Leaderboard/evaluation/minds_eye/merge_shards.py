"""Merge Mind's Eye Hugging Face diagnostics into one submission."""

from evaluation.common.merge_shards import main
from evaluation.minds_eye.config import TRACK


if __name__ == "__main__":
    raise SystemExit(main(TRACK))

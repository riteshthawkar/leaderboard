"""Merge Do You See Me Hugging Face diagnostics into one submission."""

from evaluation.common.merge_shards import main
from evaluation.do_you_see_me.config import TRACK


if __name__ == "__main__":
    raise SystemExit(main(TRACK))

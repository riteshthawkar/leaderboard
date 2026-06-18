"""
VLMEvalKit data backend for the spatial (Task 3) harness.

Thin wrapper over ``vlmeval.dataset.build_dataset`` that:
  * downloads + loads a benchmark (handled entirely by VLMEvalKit),
  * normalises each row to a uniform sample dict, and
  * exposes the private answer (used only by build_ground_truth.py).

VLMEvalKit's dataset ``.data`` is a pandas DataFrame whose rows carry at least
``index`` and ``question``; MCQ benchmarks add option columns ``A``, ``B`` ...
and an ``answer`` column. ``dataset.dump_image(line)`` decodes the (base64 or
referenced) image to local PNG paths.

``vlmeval`` is imported lazily so the rest of the toolchain works without it.
"""

import sys

_OPTION_LETTERS = list("ABCDEFGH")


def make_sample_id(dataset_id, index):
    """Canonical, stable id shared by the harness and the ground-truth builder."""
    return f"{dataset_id}:{index}"


def build(vlmeval_name):
    """Build (downloading if needed) a VLMEvalKit dataset object."""
    from vlmeval.dataset import build_dataset
    ds = build_dataset(vlmeval_name)
    if ds is None:
        raise RuntimeError(
            f"VLMEvalKit could not build dataset '{vlmeval_name}'. "
            f"Check the name against vlmeval.dataset.SUPPORTED_DATASETS."
        )
    return ds


def _clean(value):
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def extract_options(row):
    """Pull MCQ options (A, B, C ...) from a row; empty dict for free-form items."""
    options = {}
    for letter in _OPTION_LETTERS:
        val = _clean(row.get(letter))
        if val:
            options[letter] = val
    return options


def iter_samples(dataset_id, vlmeval_name, limit=0, dataset_obj=None):
    """Yield uniform sample dicts for one benchmark.

    Each sample:
        {sample_id, dataset_id, index, question, options{A..}, answer, images[]}
    ``answer`` is the private ground truth (None if the dataset has none locally).
    """
    ds = dataset_obj or build(vlmeval_name)
    data = ds.data
    count = 0
    for i in range(len(data)):
        line = dict(data.iloc[i])
        index = line.get("index", i)
        try:
            images = ds.dump_image(line)
        except Exception as exc:  # noqa: BLE001 - keep going on a bad row
            print(f"  ! {vlmeval_name} row {index}: image decode failed: {exc}",
                  file=sys.stderr)
            images = []
        if isinstance(images, str):
            images = [images]

        answer = line.get("answer")
        yield {
            "sample_id": make_sample_id(dataset_id, index),
            "dataset_id": dataset_id,
            "index": index,
            "question": _clean(line.get("question")),
            "options": extract_options(line),
            "answer": (_clean(answer) or None),
            "images": list(images) if images else [],
        }
        count += 1
        if limit and count >= limit:
            break

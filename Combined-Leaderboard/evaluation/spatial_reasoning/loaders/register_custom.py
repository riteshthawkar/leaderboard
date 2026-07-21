#!/usr/bin/env python3
"""
register_custom.py — make VLMEvalKit load our local MCQ TSVs.

The custom-loader outputs (SpatialBench, MindCube, OmniSpatial, SAT-Real, VSR_MCQ) and every
No-Image++ variant (<name>_NOIMGPP) are plain VLMEvalKit MCQ TSVs sitting in $LMUData. This
module registers them so `--data <name>` works with the standard MCQ prompt + evaluator.

Use it by importing once before VLMEvalKit builds the dataset, e.g. add near the top of run.py:

    import sys; sys.path.insert(0, '/path/to/track3_spatial_cot/loaders')
    import register_custom; register_custom.register()

The LocalMCQDataset class below is version-independent (it just reads the TSV and reuses
ImageMCQDataset's build_prompt + evaluate). Only the registry hook in register() can differ
between VLMEvalKit versions — it tries the known ones and prints what it did.
"""
import os

LMUData = os.environ.get("LMUData", os.path.expanduser("~/LMUData"))


def local_tsv_names():
    if not os.path.isdir(LMUData):
        return []
    return sorted(f[:-4] for f in os.listdir(LMUData) if f.endswith(".tsv"))


def _make_local_class():
    import pandas as pd
    from vlmeval.dataset.image_mcq import ImageMCQDataset

    class LocalMCQDataset(ImageMCQDataset):
        """An MCQ dataset whose TSV already lives in $LMUData (no download)."""
        TYPE = "MCQ"

        def load_data(self, dataset):
            path = os.path.join(LMUData, f"{dataset}.tsv")
            assert os.path.exists(path), f"{path} not found — build it with prepare_custom.py / build_variants.py first"
            return pd.read_csv(path, sep="\t")

    return LocalMCQDataset


def register(names=None):
    """Point the given dataset names (default: every local TSV) at LocalMCQDataset."""
    names = names or local_tsv_names()
    cls = _make_local_class()
    hooked = None
    # try the registry objects used across VLMEvalKit versions
    try:
        import vlmeval.dataset as D
        for attr in ("DATASET_CLASSES", "SUPPORTED_DATASETS", "dataset_map"):
            reg = getattr(D, attr, None)
            if isinstance(reg, dict):
                for n in names:
                    reg[n] = cls
                hooked = attr
                break
    except Exception as e:
        print("register_custom: could not import vlmeval.dataset:", e)
    if hooked:
        print(f"register_custom: registered {len(names)} local datasets via {hooked}: {names}")
    else:
        print("register_custom: no known registry dict found. Add this class to your VLMEvalKit "
              "dataset registry manually (it follows the ImageMCQDataset interface):", names)
    return cls, names


if __name__ == "__main__":
    print("Local MCQ TSVs in", LMUData, ":")
    for n in local_tsv_names():
        print("  ", n)
    register()

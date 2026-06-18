"""
Maps our 13 spatial-benchmark ids (see backend/config.py SPATIAL_DATASETS) to
VLMEvalKit's own dataset-registry names, and validates them against whatever is
actually installed at runtime.

VLMEvalKit refers to datasets by its OWN registry names (e.g. ``BLINK``,
``MMVP``, ``RealWorldQA``, ``3DSRBench``), not by HuggingFace ids. The exact
string can vary between releases, so for every benchmark we list candidate names
in priority order and pick the first one that exists in the installed registry
(``vlmeval.dataset.SUPPORTED_DATASETS``). Anything we cannot resolve is reported
as missing and skipped -- we never fabricate a name.

A couple of benchmarks in the paper are not (currently) shipped by VLMEvalKit
(SpatialBench, SAT). They are listed here with empty candidates so the harness
reports them clearly instead of silently dropping them.

The concrete names below were verified against an installed VLMEvalKit registry
(``len(SUPPORTED_DATASETS) == 600``). Candidate lists keep older/alternate
spellings as fallbacks in case the registry changes between releases.
"""

# our dataset id -> ordered list of candidate VLMEvalKit registry names
DATASET_CANDIDATES = {
    "blink":       ["BLINK"],
    "cvbench2d":   ["CV-Bench-2D", "CV-Bench_2D", "CVBench_2D", "CVBench"],
    "mmvp":        ["MMVP"],
    "realworldqa": ["RealWorldQA"],
    "spatialbench": [],          # not shipped by VLMEvalKit
    "vsr":         ["VSR-zeroshot"],
    "vstar":       ["VStarBench", "VStar"],
    "3dsrbench":   ["3DSRBench"],
    "cvbench3d":   ["CV-Bench-3D", "CV-Bench_3D", "CVBench_3D"],
    "mindcube":    ["MindCubeBench_raw_qa", "MindCubeBench_tiny_raw_qa", "MindCube", "MindCubeBench"],
    "mmsibench":   ["MMSIBench_wo_circular", "MMSIBench_circular", "MMSI_Bench", "MMSIBench"],
    "omnispatial": ["OmniSpatialBench", "OmniSpatialBench_default", "OmniSpatial"],
    "satreal":     [],           # not shipped by VLMEvalKit
}


def installed_dataset_names():
    """Return the set of dataset names the installed VLMEvalKit supports."""
    from vlmeval.dataset import SUPPORTED_DATASETS
    return set(SUPPORTED_DATASETS)


def resolve(dataset_ids, installed=None):
    """Resolve our ids to concrete VLMEvalKit names.

    Returns ``(resolved, missing)`` where ``resolved`` is ``{our_id: vlm_name}``
    and ``missing`` is the list of ids that could not be resolved.
    """
    if installed is None:
        installed = installed_dataset_names()
    resolved, missing = {}, []
    for ds_id in dataset_ids:
        candidates = DATASET_CANDIDATES.get(ds_id, [])
        pick = next((c for c in candidates if c in installed), None)
        if pick:
            resolved[ds_id] = pick
        else:
            missing.append(ds_id)
    return resolved, missing

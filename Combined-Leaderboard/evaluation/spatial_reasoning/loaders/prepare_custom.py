#!/usr/bin/env python3
"""
prepare_custom.py — convert the 4 non-built-in spatial benchmarks to VLMEvalKit MCQ TSVs.

Pipeline per dataset:  download (verified HF source) -> parse raw records ->
to_intermediate() field-map -> common.py normalizes -> ~/LMUData/<NAME>.tsv

The upstream repos DO NOT document their internal keys uniformly, so INSPECT FIRST:

    python prepare_custom.py --dataset MindCube --inspect     # prints the real keys of record[0]

If the printed keys differ from the FIELD_MAP marked "# VERIFY" in that loader's
to_intermediate(), fix those 2-3 lines, then run for real:

    python prepare_custom.py --dataset MindCube               # writes ~/LMUData/MindCube.tsv

Verified sources:
  SpatialBench  RussRobin/SpatialBench  (GATED: accept terms on the HF page + `hf auth login`)
  MindCube      MLL-Lab/MindCube        data.zip  (multi-image)
  OmniSpatial   qizekun/OmniSpatial     OmniSpatial-test.zip
  SAT-Real      array/SAT               SAT_val.parquet / SAT_test.parquet  (~150 real images; --split)
"""
import argparse, glob, json, os, re, shutil, zipfile
from collections import Counter
try:
    from common import intermediate_to_row, rows_to_tsv
except ImportError:  # pragma: no cover - package import fallback
    from .common import intermediate_to_row, rows_to_tsv

LMUData = os.environ.get("LMUData", os.path.expanduser("~/LMUData"))
CACHE = os.path.join(LMUData, "_track3_src")


# ---------- download / file helpers ----------
def _snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, repo_type="dataset",
                             local_dir=os.path.join(CACHE, repo.replace("/", "_")))


def _file(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                           local_dir=os.path.join(CACHE, repo.replace("/", "_")))


def _unzip(path):
    dst = path + "_x"
    if not os.path.isdir(dst):
        temporary = dst + ".part"
        shutil.rmtree(temporary, ignore_errors=True)
        os.makedirs(temporary, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            corrupt_member = z.testzip()
            if corrupt_member:
                raise ValueError(f"Downloaded ZIP failed CRC validation: {corrupt_member}")
            destination = os.path.realpath(temporary)
            for member in z.infolist():
                target = os.path.realpath(os.path.join(destination, member.filename))
                if os.path.commonpath([destination, target]) != destination:
                    raise ValueError(f"Unsafe path in downloaded ZIP: {member.filename}")
            z.extractall(destination)
        os.replace(temporary, dst)
    return dst


def _find(root, patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"none of {patterns} under {root}")


def _read_json_any(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path) if l.strip()]
    data = json.load(open(path))
    if isinstance(data, list):
        return data
    for k in ("data", "questions", "annotations", "records"):
        if isinstance(data.get(k), list):
            return data[k]
    return list(data.values())


# ---------- loaders (one per dataset) ----------
class Loader:
    name = None
    index_base = 0            # integer answers: 0- or 1-based (declare per dataset!)
    expected_skips = {}
    def __init__(self, split=None):
        self.split = split
    def raw_records(self):    # -> list[dict]
        raise NotImplementedError
    def to_intermediate(self, r):  # -> {image, question, options, answer, category}
        raise NotImplementedError


class SpatialBench(Loader):
    name = "SpatialBench"
    index_base = 0
    TASKS = ["counting", "existence", "positional", "reach", "size"]
    expected_skips = {"not_mcq": 20, "unresolved_answer": 2}

    def raw_records(self):
        try:
            self.root = _snapshot("RussRobin/SpatialBench")
        except Exception as e:
            raise SystemExit("SpatialBench is GATED. Accept the terms at "
                             "https://huggingface.co/datasets/RussRobin/SpatialBench then run "
                             "`hf auth login`. Original error: %r" % e)
        recs = []
        for task in self.TASKS:
            try:
                jf = _find(self.root, (f"{task}.json",))
            except FileNotFoundError:
                continue
            for r in _read_json_any(jf):
                r["_task"], r["_dir"] = task, os.path.dirname(jf)
                recs.append(r)
        return recs

    def to_intermediate(self, r):
        img_name = str(r.get("image") or r.get("image_path") or r.get("img"))
        img = os.path.join(r["_dir"], img_name)           # image field already carries 'task/file.jpg'
        if not os.path.exists(img):
            img = os.path.join(r["_dir"], r["_task"], os.path.basename(img_name))
        if not os.path.exists(img):
            img = _find(r["_dir"], (os.path.basename(img_name),))
        q = r.get("question") or ""
        ans = r.get("answer") if "answer" in r else (r.get("gt") or r.get("label"))
        stem, opts = _split_paren_options(q)              # positional/reach/size: "(A) .. (B) .."
        if opts:                                          # answer is already the letter
            return dict(image=img, question=stem, options=opts, answer=ans, category=r["_task"])
        a = str(ans).strip().lower()
        if a in ("yes", "no", "y", "n", "true", "false"): # existence: Yes/No -> A/B MCQ
            return dict(image=img, question=q, options=["Yes", "No"],
                        answer=("A" if a in ("yes", "y", "true") else "B"), category=r["_task"])
        return dict(image=img, question=q, options=None,  # counting (numeric) -> not MCQ, skipped
                    answer=ans, category=r["_task"])


class MindCube(Loader):
    name = "MindCube"
    index_base = 0

    def raw_records(self):
        self.root = _unzip(_file("MLL-Lab/MindCube", "data.zip"))
        jf = _find(self.root, ("*tinybench*.jsonl", "*.jsonl", "*bench*.json", "*.json"))
        self._jf_dir = os.path.dirname(jf)
        return _read_json_any(jf)

    def to_intermediate(self, r):
        imgs = r.get("images") or r.get("image")
        if isinstance(imgs, str):
            imgs = [imgs]
        base = os.path.dirname(self._jf_dir)                     # .../data  (imgs under data/other_all_image/...)
        imgs = [p if os.path.isabs(p) else os.path.join(base, p) for p in imgs]
        stem, opts = _split_inline_options(r.get("question") or "")   # options are embedded in the stem
        cat = r.get("category")
        if isinstance(cat, list):
            cat = "|".join(str(x) for x in cat)
        return dict(image=imgs,                                   # list -> multi-image row
                    question=stem,
                    options=opts,                                 # {A:.., B:.., ...} parsed from the stem
                    answer=r.get("gt_answer"),                    # already a letter
                    category=cat or r.get("type"))


class OmniSpatial(Loader):
    name = "OmniSpatial"
    index_base = 0
    expected_skips = {"not_mcq": 229}

    def raw_records(self):
        self.root = _unzip(_file("qizekun/OmniSpatial", "OmniSpatial-test.zip"))
        jf = _find(self.root, ("*.jsonl", "*test*.json", "*.json"))
        self._jf_dir = os.path.dirname(jf)
        return _read_json_any(jf)

    def to_intermediate(self, r):
        first = str(r["id"]).split("_")[0]                        # image = {task_type}/{first}.png
        img = os.path.join(self._jf_dir, r["task_type"], first + ".png")
        return dict(image=img,
                    question=r.get("question"),
                    options=r.get("options"),                     # proper list
                    answer=r.get("answer"),                       # 0-based int -> index_base=0
                    category=r.get("task_type"))


class SATReal(Loader):
    name = "SAT-Real"
    # SAT exposes the correct option text, so integer indexing is not used.
    index_base = 0

    def raw_records(self):
        import pandas as pd
        split = self.split or "test"    # SAT-Real = SAT_test.parquet (exactly 150 real-image questions)
        pf = _file("array/SAT", f"SAT_{split}.parquet")
        df = pd.read_parquet(pf)
        print(f"[SAT-Real] SAT_{split}.parquet -> {len(df)} rows "
              f"({'≈150 as expected' if 100 < len(df) < 300 else 'NOT ~150 — try --split test'})")
        return df.to_dict("records")

    def to_intermediate(self, r):
        ib = r.get("image_bytes")                     # ndarray of per-frame JPEG bytes (1-2 frames)
        if hasattr(ib, "tolist") and not isinstance(ib, (bytes, bytearray)):
            ib = list(ib)
        if not isinstance(ib, (list, tuple)):
            ib = [ib]
        opts = r.get("answers")
        if hasattr(opts, "tolist"):
            opts = list(opts)
        return dict(image=list(ib),                   # multi-frame -> common.to_b64 emits a JSON list
                    question=r.get("question"),
                    options=opts,
                    answer=r.get("correct_answer"),   # answer text -> matched to option text
                    category=r.get("question_type") or "sat")


def _split_inline_options(q):
    """Split 'stem ... A. x B. y C. z' -> (stem, {A:x, B:y, ...}); returns (q, None) if <2 markers."""
    marks = list(re.finditer(r'(?:^|\s)([A-H])\.\s+', q))
    if len(marks) < 2:
        return q.strip(), None
    stem = q[:marks[0].start()].strip()
    opts = {}
    for i, m in enumerate(marks):
        start, end = m.end(), (marks[i + 1].start() if i + 1 < len(marks) else len(q))
        opts[m.group(1)] = q[start:end].strip()
    return stem, opts


def _split_paren_options(q):
    """Split 'stem ... (A) x (B) y' -> (stem, {A:x, B:y}); (q, None) if <2 markers.
    Strips a trailing 'Answer with the option's letter ...' instruction from the stem."""
    marks = list(re.finditer(r'\(([A-H])\)\s*', q))
    if len(marks) < 2:
        return q.strip(), None
    stem = q[:marks[0].start()].strip()
    stem = re.sub(r"\s*Answer with the option'?s letter[^.\n]*\.?\s*$", "", stem, flags=re.I).strip()
    opts = {}
    for i, m in enumerate(marks):
        start, end = m.end(), (marks[i + 1].start() if i + 1 < len(marks) else len(q))
        opts[m.group(1)] = q[start:end].strip().rstrip(".")
    return stem, opts


def _resolve(root, jf_dir, rel):
    """Resolve an image path that may be relative to the archive root or the json's folder."""
    rel = str(rel)
    for base in (jf_dir, root):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    try:
        return _find(root, (os.path.basename(rel),))
    except FileNotFoundError:
        return os.path.join(root, rel)                   # let common.to_b64 raise a clear error


REGISTRY = {c.name: c for c in (SpatialBench, MindCube, OmniSpatial, SATReal)}


def _trunc(v, n=140):
    s = repr(v)
    return s if len(s) <= n else s[:n] + f" …<{type(v).__name__}, {len(s)} chars>"


def convert_records(loader, records, limit=0):
    """Convert records while enforcing the dataset's intentional exclusion contract."""
    rows, skip = [], Counter()
    for i, record in enumerate(records):
        if limit and i >= limit:
            break
        try:
            intermediate = loader.to_intermediate(record)
            row, error = intermediate_to_row(i, intermediate, loader.index_base)
        except Exception as exc:
            raise RuntimeError(
                f"[{loader.name}] failed to convert source record {i}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if error:
            skip[error] += 1
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError(
            f"[{loader.name}] produced 0 rows. Run --inspect and fix the field map. "
            f"skips={dict(skip)}"
        )
    if not limit and dict(skip) != loader.expected_skips:
        raise RuntimeError(
            f"[{loader.name}] exclusion counts changed: got {dict(skip)}, "
            f"expected {loader.expected_skips}. Inspect the upstream dataset and update "
            "the pinned contract deliberately before generating an official bundle."
        )
    return rows, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(REGISTRY))
    ap.add_argument("--inspect", action="store_true", help="print raw keys of record[0] and exit")
    ap.add_argument("--split", default=None, help="SAT-Real: which parquet split (val/test)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (debug)")
    a = ap.parse_args()

    loader = REGISTRY[a.dataset](split=a.split)
    recs = loader.raw_records()
    print(f"[{a.dataset}] loaded {len(recs)} raw records")
    if not recs:
        raise SystemExit(f"[{a.dataset}] source returned no records")

    if a.inspect:
        r0 = recs[0]
        keys = r0.keys() if isinstance(r0, dict) else range(len(r0))
        print("\n--- record[0] fields (map these in to_intermediate) ---")
        for k in keys:
            print(f"  {k!r:30s} = {_trunc(r0[k])}")
        print("\nEdit the '# VERIFY' lines in this loader's to_intermediate() if names differ, then re-run without --inspect.")
        return

    rows, skip = convert_records(loader, recs, limit=a.limit)
    n, path = rows_to_tsv(rows, os.path.join(LMUData, a.dataset + ".tsv"))
    print(f"[{a.dataset}] wrote {n} MCQ rows -> {path}   skipped={dict(skip)}")
    print(f"   -> use  --data {a.dataset}  in config.yaml / make_manifest / run_track3")


if __name__ == "__main__":
    main()

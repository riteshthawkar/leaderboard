#!/usr/bin/env python3
"""
prepare_data.py — ONE command to fetch & build all 13 spatial benchmarks as uniform MCQ TSVs.

    python prepare_data.py --lmudata ./LMUData [--hf-token hf_xxx | env HF_TOKEN] [--skip SpatialBench]

What it does, in order:
  1. Downloads the 8 VLMEvalKit-hosted TSVs over verified HTTPS and checks published MD5 hashes.
  2. Recasts VSR (Yes/No) -> VSR_MCQ.
  3. Builds the 4 custom datasets from their original sources via loaders/prepare_custom.py
     (SpatialBench is gated: accept terms at https://huggingface.co/datasets/RussRobin/SpatialBench
      and pass a HF token with read access to gated repos).
  4. Verifies everything: row counts, MCQ columns, image decode (incl. multi-image rows).

Requirements: pip install pandas pillow huggingface_hub pyarrow
(Deliberately NOT `pip install vlmevalkit` — not needed for data prep, and its pinned deps
 fail on some platforms, e.g. decord on arm64 macs.)
"""
import argparse, ast, base64, hashlib, io, json, os, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- the 8 hosted TSVs (URLs + md5s as published in VLMEvalKit's source) ----
HOSTED = [
    # name                  url                                                                              md5 (None = not published)
    ("BLINK",               "https://opencompass.openxlab.space/utils/VLMEval/BLINK.tsv",                    "d5e8af148b10ac69f535ff7b23f3f989"),
    ("MMVP",                "https://opencompass.openxlab.space/utils/VLMEval/MMVP.tsv",                     None),
    ("RealWorldQA",         "https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv",              "4de008f55dc4fd008ca9e15321dc44b7"),
    ("CV-Bench-2D",         "https://opencompass.openxlab.space/utils/VLMEval/CV-Bench-2D.tsv",              None),
    ("CV-Bench-3D",         "https://opencompass.openxlab.space/utils/VLMEval/CV-Bench-3D.tsv",              None),
    ("3DSRBench",           "https://opencompass.openxlab.space/utils/VLMEval/3DSRBench.tsv",                "610516a0b4710595545b7613c60524e8"),
    ("VStarBench",          "https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv", None),
    ("MMSIBench_wo_circular","https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MMSIBench_wo_circular.tsv", None),
    ("VSR",                 "https://huggingface.co/datasets/ignoreandfly/vsr_zeroshot_tsv/resolve/main/vsr_zeroshot_dataset_yn_strict.tsv", None),
]
CUSTOM = ["SAT-Real", "OmniSpatial", "MindCube", "SpatialBench"]   # built by loaders/prepare_custom.py
EXPECTED_ROWS = {  # sanity floor: warn if a build lands far from these
    "BLINK": 1901, "MMVP": 300, "RealWorldQA": 765, "CV-Bench-2D": 1438, "CV-Bench-3D": 1200,
    "3DSRBench": 11686, "VStarBench": 191, "MMSIBench_wo_circular": 1000, "VSR_MCQ": 1222,
    "SAT-Real": 150, "OmniSpatial": 1304, "MindCube": 1050, "SpatialBench": 152,
}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, url, md5, lmu):
    out = os.path.join(lmu, f"{name}.tsv")
    if os.path.exists(out) and (md5 is None or md5_file(out) == md5):
        print(f"  [skip] {name}: already present" + ("" if md5 is None else " (md5 OK)"))
        return
    print(f"  [get ] {name} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "spatial-eval/1.0"})
    part = out + ".part"
    try:
        with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(part, out)
    finally:
        if os.path.exists(part):
            os.remove(part)
    if md5 and md5_file(out) != md5:
        sys.exit(f"  [FAIL] {name}: md5 mismatch — corrupted download, delete and retry")
    print(f"  [ok  ] {name}: {os.path.getsize(out)//(1<<20)} MB" + ("" if md5 is None else "  md5 VERIFIED"))


def image_cells(cell):
    s = str(cell).strip()
    if s.startswith("["):
        for parse in (json.loads, ast.literal_eval):
            try:
                v = parse(s)
                if isinstance(v, list):
                    return [str(x) for x in v]
            except Exception:
                pass
    return [s]


def verify(lmu, names):
    import pandas as pd
    from PIL import Image
    print("\n== verification ==")
    bad = 0
    for n in names:
        p = os.path.join(lmu, f"{n}.tsv")
        if not os.path.exists(p):
            print(f"  [MISS] {n}"); bad += 1; continue
        df = pd.read_csv(p, sep="\t")
        rows = len(df)
        opts = [c for c in "ABCDEFGH" if c in df.columns]
        # dedup-aware image decode on first 3 rows (resolve index-references, parse multi-image lists)
        big = {str(r["index"]): str(r["image"]) for _, r in df.head(500).iterrows() if len(str(r["image"])) > 64}
        ok = True
        for _, r in df.head(3).iterrows():
            cell = str(r["image"])
            if len(cell) < 32 and cell in big:
                cell = big[cell]
            for b in image_cells(cell):
                try:
                    Image.open(io.BytesIO(base64.b64decode(b)))
                except Exception:
                    ok = False
        exp = EXPECTED_ROWS.get(n)
        drift = exp and abs(rows - exp) > max(3, exp // 20)
        flag = "OK " if (ok and not drift) else ("DRIFT" if drift else "IMG!")
        if flag != "OK ":
            bad += 1
        print(f"  [{flag}] {n:24s} rows={rows:6d} (expected~{exp})  opts={opts[:6]}  img-decode={'ok' if ok else 'FAIL'}")
    print("verification " + ("PASSED" if bad == 0 else f"finished with {bad} issue(s) — see above"))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmudata", default=os.path.join(HERE, "LMUData"))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--skip", default="", help="comma list to skip (e.g. SpatialBench if no gated access)")
    a = ap.parse_args()
    lmu = os.path.abspath(a.lmudata)
    os.makedirs(lmu, exist_ok=True)
    skip = set(x.strip() for x in a.skip.split(",") if x.strip())
    if a.hf_token:
        os.environ["HF_TOKEN"] = a.hf_token.strip()

    print("== 1/4 hosted TSVs ==")
    for name, url, md5 in HOSTED:
        if name in skip:
            print(f"  [skip] {name} (--skip)"); continue
        download(name, url, md5, lmu)

    print("\n== 2/4 VSR -> VSR_MCQ ==")
    if "VSR_MCQ" not in skip:
        sys.path.insert(0, os.path.join(HERE, "loaders"))
        import vsr_to_mcq
        vsr_to_mcq.recast(os.path.join(lmu, "VSR.tsv"), os.path.join(lmu, "VSR_MCQ.tsv"))

    print("\n== 3/4 custom datasets (original sources) ==")
    env = dict(os.environ, LMUData=lmu)
    for ds in CUSTOM:
        if ds in skip:
            print(f"  [skip] {ds}"); continue
        if os.path.exists(os.path.join(lmu, f"{ds}.tsv")):
            print(f"  [skip] {ds}: already built"); continue
        print(f"  [build] {ds}")
        r = subprocess.run([sys.executable, os.path.join(HERE, "loaders", "prepare_custom.py"),
                            "--dataset", ds], env=env)
        if r.returncode != 0:
            hint = (" (gated — accept terms on its HF page and pass --hf-token)" if ds == "SpatialBench" else "")
            print(f"  [FAIL] {ds}{hint} — re-run later or add to --skip")

    print("\n== 4/4 verify ==")
    names = [n for n, _, _ in HOSTED if n != "VSR" and n not in skip] + ["VSR_MCQ"] + [d for d in CUSTOM if d not in skip]
    bad = verify(lmu, names)
    print(f"\nDone. LMUData = {lmu}")
    print("Next: serve your model + the judge (see serve/serve_example.sh), then:")
    print(f"  python run_track3_vllm.py --lmudata {lmu} --endpoint-model <SERVED_ID> --leaderboard-model-name <NAME> --endpoints <URLS> --out results/<NAME>")
    print(f"  python judge_track3.py --results-dir results/<NAME> --endpoints <JUDGE_URL>")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""occgateRAW_FT (finetuned HGNet): facemesh 좌표계 기준 (정상 facemesh 원본, 가림만 hgnet 변환).

occgateGT 와 반대 — 좌표계 기준을 hgnet→facemesh 로 뒤집음:
  정상 region (occ=0) → facemesh 좌표 원본 그대로 (변환 X → iris 정밀도 손실 없음, mediapipe 0.61 수준)
  가림 region (occ=1) → hgnet 을 facemesh 좌표계로 Umeyama 변환 (복원 좌표)
정상 frame 이 다수라 facemesh 좌표계 기준이 손실 최소. 출력: _hgnet478_occgateRAW.npz
"""
import json, importlib.util as ilu
import numpy as np
from pathlib import Path

SPLIT = "/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json"
LISTS = ["train_clean_masked_1to1", "val_clean_masked_1to1", "test_clean_paired", "test_masked"]
s = ilu.spec_from_file_location("fr7", "/data/shared/scuppy/Gaze_image_model/src/data/face_regions7.py")
fr7 = ilu.module_from_spec(s); s.loader.exec_module(fr7)
FR = fr7.FACE_REGIONS_7
REGION_PTS = {
    "left_eye": sorted(set(FR["left_eye"]) | set(range(468, 473))),
    "right_eye": sorted(set(FR["right_eye"]) | set(range(473, 478))),
    "mouth": sorted(set(FR["mouth"])),
}
GATE = sorted(set().union(*[set(v) for v in REGION_PTS.values()]))
FIT = np.array([i for i in range(478) if i not in set(GATE)])  # 안정점(코/윤곽)으로 hgnet→facemesh fit


def umeyama2d(src, dst):
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    if len(src) < 8:
        return None
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / max(var_s, 1e-9)
    return scale, R, mu_d - scale * (R @ mu_s)


def build_one(face_path, occ_labels, dry=False):
    hg_path = Path(str(face_path).replace("_facemesh.npz", "_hgnet478_ft.npz"))
    out_path = Path(str(hg_path).replace("_hgnet478_ft.npz", "_hgnet478_occgateRAW_ft.npz"))
    zm = np.load(face_path, allow_pickle=True)            # facemesh (full-frame, 기준)
    mp = zm["landmarks"].astype(np.float32).copy()        # ← 기준: 정상 facemesh 원본
    mdet = zm["detected"].astype(bool) if "detected" in zm else np.ones(len(mp), bool)
    has_occ = any(int(occ_labels.get(r,0))==1 for r in ["left_eye","right_eye","mouth"])
    if has_occ and hg_path.exists():
        zh = np.load(hg_path, allow_pickle=True); hg = zh["landmarks"].astype(np.float32)
        hdet = zh["detected"].astype(bool) if "detected" in zh else np.ones(len(hg), bool)
    else:
        hg = mp; hdet = mdet  # clean: facemesh raw (hg 미사용)
    meta = dict(zm["meta"].item()) if "meta" in zm else {}
    g = {r: int(occ_labels.get(r, 0)) for r in ["left_eye", "right_eye", "mouth"]}
    out = mp.copy()                                       # 정상 region = facemesh 원본 (변환 X)
    n_gate = 0
    if any(v == 1 for v in g.values()):                   # 가림 있는 frame 만 hgnet 변환
        T = min(len(mp), len(hg))
        for fi in range(T):
            if not (mdet[fi] and hdet[fi]):
                continue
            a, b = hg[fi, :, :2], mp[fi, :, :2]            # hgnet → facemesh 방향
            if a.shape[0] < 478 or b.shape[0] < 478:
                continue
            res = umeyama2d(a[FIT], b[FIT])
            if res is None:
                continue
            scale, R, t = res
            for rname, pts in REGION_PTS.items():
                if g[rname] == 1:                          # 가림 region → hgnet 을 facemesh 좌표계로
                    out[fi, pts, :2] = (scale * (R @ a[pts].T)).T + t
            n_gate += 1
    meta["occgateRAW"] = dict(occ_labels=g, gated_frames=n_gate, base="facemesh_raw")
    if not dry:
        np.savez(out_path, landmarks=out, detected=mdet, meta=meta)
    return out_path, n_gate, len(mp)


def main():
    import sys, shutil, time
    dry = "--dry" in sys.argv
    d = json.load(open(SPLIT)); items = d["items"]
    seen = {}
    for L in LISTS:
        for it in items[L]:
            seen[it["face_path"]] = it
    samples = list(seen.items())
    print(f"samples={len(samples)} GATE={len(GATE)} FIT={len(FIT)}  (정상=facemesh raw, 가림=hgnet→facemesh)", flush=True)
    if dry:
        for variant in ("clean", "masked"):
            fp, it = next((k, v) for k, v in samples if v["variant"] == variant)
            _, ng, T = build_one(fp, it["occ_labels"], dry=True)
            print(f"[{variant}] occ={it['occ_labels']} gated_frames={ng}/{T}  {Path(fp).name[:45]}", flush=True)
        return
    t0 = time.time(); ok = fail = 0
    for i, (fp, it) in enumerate(samples, 1):
        try:
            build_one(fp, it["occ_labels"]); ok += 1
        except Exception as e:
            pass  # ft: fallback uses facemesh
            # fallback: facemesh 원본을 그대로 (정상 좌표)
            try:
                zm = np.load(fp, allow_pickle=True)
                op = str(fp).replace("_facemesh.npz", "_hgnet478_occgateRAW_ft.npz")
                np.savez(op, landmarks=zm["landmarks"], detected=zm["detected"], meta={})
                fail += 1
            except Exception as e2:
                print(f"  !! {Path(fp).name}: {e}/{e2}", flush=True)
        if i % 50 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)} ({time.time()-t0:.0f}s) ok={ok} fallback={fail}", flush=True)
    print(f"done ok={ok} fallback={fail}", flush=True)


if __name__ == "__main__":
    main()

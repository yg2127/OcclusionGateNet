#!/usr/bin/env python3
"""모델2 oracle: occ_labels GT gating 좌표 blend cache.

각 sample 의 manifest occ_labels(left_eye/right_eye/mouth = 0/1) 로 region별 좌표 선택:
  occ=1 (가림)  → HGNet478 좌표 (robust)
  occ=0 (clear) → facemesh 좌표 (정밀)
  nose/나머지   → facemesh (clear 가정)
좌표계: facemesh(full-frame) → HGNet(crop112) Umeyama(non-gate 점 fit) 변환 후 치환.
출력: 각 _hgnet478.npz 옆 _hgnet478_occgateGT.npz
저장: 기존 안 건드림 (새 suffix). 모델2 는 npz_swap.to=_hgnet478_occgateGT.npz 로 학습.
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
FIT = np.array([i for i in range(478) if i not in set(GATE)])  # gating 안 받는 안정점(코/윤곽)


def umeyama2d(src, dst):
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    if len(src) < 8: return None
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / max(var_s, 1e-9)
    return scale, R, mu_d - scale * (R @ mu_s)


def build_one(face_path, occ_labels, dry=False):
    hg_path = Path(str(face_path).replace("_facemesh.npz", "_hgnet478.npz"))
    out_path = Path(str(hg_path).replace("_hgnet478.npz", "_hgnet478_occgateGT.npz"))
    zh = np.load(hg_path, allow_pickle=True)
    hg = zh["landmarks"].astype(np.float32).copy()
    det = zh["detected"].astype(bool) if "detected" in zh else np.ones(len(hg), bool)
    meta = dict(zh["meta"].item()) if "meta" in zh else {}
    zm = np.load(face_path, allow_pickle=True)
    mp = zm["landmarks"].astype(np.float32)
    mdet = zm["detected"].astype(bool) if "detected" in zm else np.ones(len(mp), bool)
    T = min(len(hg), len(mp))
    g = {"left_eye": int(occ_labels.get("left_eye", 0)),
         "right_eye": int(occ_labels.get("right_eye", 0)),
         "mouth": int(occ_labels.get("mouth", 0))}
    n = 0
    for fi in range(T):
        if not (det[fi] and mdet[fi]): continue
        a, b = mp[fi, :, :2], hg[fi, :, :2]
        if a.shape[0] < 478 or b.shape[0] < 478: continue
        res = umeyama2d(a[FIT], b[FIT])
        if res is None: continue
        scale, R, t = res
        for rname, pts in REGION_PTS.items():
            if g[rname] == 1:  # 가림 → HGNet 유지 (이미 hg)
                continue
            # clear → facemesh 좌표를 hgnet 좌표계로 변환해 치환
            hg[fi, pts, :2] = (scale * (R @ a[pts].T)).T + t
        n += 1
    meta["occgateGT"] = dict(occ_labels=g, blended_frames=n)
    if not dry:
        np.savez(out_path, landmarks=hg, detected=det, meta=meta)
    return out_path, n, T


def main():
    import sys, shutil, time
    dry = "--dry" in sys.argv
    d = json.load(open(SPLIT)); items = d["items"]
    seen = {}
    for L in LISTS:
        for it in items[L]:
            seen[it["face_path"]] = it
    samples = list(seen.items())
    print(f"samples={len(samples)} GATE_pts={len(GATE)} FIT_pts={len(FIT)}", flush=True)
    if dry:
        for variant in ("clean", "masked"):
            fp, it = next((k, v) for k, v in samples if v["variant"] == variant)
            _, n, T = build_one(fp, it["occ_labels"], dry=True)
            print(f"[{variant}] occ_labels={it['occ_labels']} blended={n}/{T}  {Path(fp).name[:45]}", flush=True)
        return
    t0 = time.time(); ok = fail = 0
    for i, (fp, it) in enumerate(samples, 1):
        try:
            build_one(fp, it["occ_labels"]); ok += 1
        except Exception as e:
            hg = Path(str(fp).replace("_facemesh.npz", "_hgnet478.npz"))
            try:
                shutil.copy(hg, str(hg).replace("_hgnet478.npz", "_hgnet478_occgateGT.npz")); fail += 1
            except Exception as e2:
                print(f"  !! {Path(fp).name}: {e}/{e2}", flush=True)
        if i % 50 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)} ({time.time()-t0:.0f}s) ok={ok} fallback={fail}", flush=True)
    print(f"done ok={ok} fallback={fail}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""HGNet fixedmask finetune — 8 appearance 가림 robust 화.

배경: phase3a HGNet 은 zero-paint(검정) 6 variant 만 학습 → checker/stripe/noise 등
      fixedmask appearance 에서 가린부위 NME 폭증(진단 노트북). messenger 좌표는 맞지만
      가린부위 복원이 OOD. → 같은 messenger 방식으로 8 appearance 를 추가 학습.

방식: DMDHeatmapDataset 상속. clean crop 로드 후 region(le/re/mouth/both_eyes)에
      making_subset_diversity.make_pattern 8 appearance 를 on-the-fly 렌더링.
      GT = clean mediapipe 좌표 (변경 X) → 가린부위 좌표 복원 압력.
      ORFormer frozen, HGNet 만 finetune. 기존 phase3a/best.pt init.
      subject-disjoint(hyi fixed split). 저장: /data/shared/scuppy/yg/hgnet_fixedmask_ft/ (기존 안 건드림).
"""
import sys, csv, time, json, copy, random, importlib.util as ilu
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path("/home/yg/fusion/pretrain_v4")
for p in ["configs", "src/data", "src"]:
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, "/data/shared/orformer/vendor")
from default import get_cfg
from dataset_dmd import DMDHeatmapDataset
from augmentation import phase3_train_transform, normalize_transform
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer
from models.StackedHGNet import IntergrationStackedHGNet

# make_pattern (fixedmask appearance 생성기) import
mp_spec = ilu.spec_from_file_location("msd", "/data/shared/scuppy/external_scripts/hyi_masking/making_subset_diversity.py")
msd = ilu.module_from_spec(mp_spec); mp_spec.loader.exec_module(msd)
make_pattern = msd.make_pattern

# region landmark index (112 coord 기준 동일 인덱스)
fr_spec = ilu.spec_from_file_location("fr7", "/data/shared/scuppy/Gaze_image_model/src/data/face_regions7.py")
fr7 = ilu.module_from_spec(fr_spec); fr_spec.loader.exec_module(fr7); FR = fr7.FACE_REGIONS_7
LE = np.array(sorted(set(FR["left_eye"])  | set(range(468, 473))))
RE = np.array(sorted(set(FR["right_eye"]) | set(range(473, 478))))
MO = np.array(sorted(set(FR["mouth"])))
REGION_IDX = {"left_eye": LE, "right_eye": RE, "mouth": MO, "both_eyes": np.union1d(LE, RE)}
REGION_NAMES = ["left_eye", "right_eye", "mouth", "both_eyes"]
APPS = ["solid", "soft_solid", "blur_patch", "smooth_noise", "soft_noise", "noise", "checker", "stripe"]

SPLIT = json.load(open("/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json"))
SUBJ = SPLIT["subjects"]


def clip_subject(rel: str):
    parts = Path(rel).parts
    for i, p in enumerate(parts):
        if p == "dmd" and i + 2 < len(parts):
            return f"{parts[i+1]}_{parts[i+2]}"
    return None


class FixedmaskFTDataset(DMDHeatmapDataset):
    """clean crop + on-the-fly fixedmask appearance occlusion. GT=clean 좌표."""
    def __init__(self, cfg, role, occ_prob=0.6, frame_stride=5, max_clips=None, seed=0):
        self.role = role            # "train" | "val"
        self.occ_prob = occ_prob
        self._ft_rng = random.Random(seed + (0 if role == "train" else 999))
        super().__init__(cfg, "DMD", "all",
                         augmentation_transform=(phase3_train_transform() if role == "train" else None),
                         normalize_transform=normalize_transform(),
                         edge_type="ADNet", ratio=4, max_clips=max_clips, mix_prob=0.0, rng_seed=seed)
        # subject-disjoint 필터: train role→train subj, val role→val+test subj
        want = set(SUBJ["train"]) if role == "train" else (set(SUBJ["val"]) | set(SUBJ["test"]))
        self.database = [r for r in self.database if clip_subject(r["rel"]) in want]
        if frame_stride > 1:
            self.database = self.database[::frame_stride]

    def _select_variant(self, idx):
        return "normal"     # 항상 clean crop 로드 (occlusion 은 우리가 렌더)

    def _apply_occ(self, face112, lm112):
        region = self._ft_rng.choice(REGION_NAMES)
        app = self._ft_rng.choice(APPS)
        idx = REGION_IDX[region]
        pts = lm112[idx]
        ok = np.isfinite(pts).all(1)
        if ok.sum() < 4:
            return face112, np.ones(478, np.float32)
        hull = cv2.convexHull(pts[ok].astype(np.int32))
        mask = np.zeros((112, 112), np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        pad = self._ft_rng.choice([3, 5, 7])
        mask = cv2.dilate(mask, np.ones((pad, pad), np.uint8), 1)
        x, y, w, h = cv2.boundingRect(mask)
        if w < 3 or h < 3:
            return face112, np.ones(478, np.float32)
        src = face112[y:y+h, x:x+w]
        pat = make_pattern(h, w, app, self._ft_rng, src_gray=src)
        rmask = mask[y:y+h, x:x+w] > 0
        face112 = face112.copy()
        face112[y:y+h, x:x+w][rmask] = pat[rmask]
        manifest = np.ones(478, np.float32)
        manifest[idx] = 0.0
        return face112, manifest

    def _load_face_and_landmarks(self, rec, variant):
        # 항상 normal crop + clean 좌표 로드
        face, lm112, manifest = super()._load_face_and_landmarks(rec, "normal")
        if self.role == "train" and self._ft_rng.random() < self.occ_prob:
            face, manifest = self._apply_occ(face, lm112)
        return face, lm112, manifest


class NME_DMD(nn.Module):
    def __init__(self, al, ar): super().__init__(); self.al, self.ar = al, ar
    def forward(self, pred, gt):
        norm = torch.linalg.vector_norm(gt[:, self.al] - gt[:, self.ar], dim=1)[:, None]
        return torch.mean(torch.linalg.vector_norm(pred - gt, 2, dim=2) / norm, dim=1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="/data/shared/scuppy/yg/hgnet_fixedmask_ft")
    ap.add_argument("--init-hgnet", default=str(ROOT / "artifacts/phase3a_hgnet_478/best.pt"))
    ap.add_argument("--orformer", default=str(ROOT / "artifacts/phase2_orformer_fixed/best.pt"))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--occ-prob", type=float, default=0.6)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = Path(args.save_dir); sd.mkdir(parents=True, exist_ok=True)
    cfg = get_cfg(); cfg.DMD.GT_SOURCE = "mediapipe"; cfg.DMD_68.GT_SOURCE = "mediapipe"; cfg.freeze()
    dsc = cfg.DMD
    mc = args.max_clips if args.max_clips > 0 else None

    tr = FixedmaskFTDataset(cfg, "train", args.occ_prob, args.frame_stride, mc)
    va = FixedmaskFTDataset(cfg, "val", 0.0, args.frame_stride, mc)
    print(f"[data] train={len(tr)} val={len(va)} (subject-disjoint, stride={args.frame_stride}, occ_prob={args.occ_prob})", flush=True)

    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=True)
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=max(1, args.workers // 2),
                    pin_memory=True, persistent_workers=True)

    # models
    vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
    orf = VQVAE(h_dim=128, res_h_dim=32, output_dim=dsc.NUM_EDGE, n_res_layers=2, n_embeddings=2048,
                embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(dev).eval()
    orf.load_state_dict(torch.load(args.orformer, map_location=dev, weights_only=False).get("model_state_dict"), strict=False)
    for p in orf.parameters(): p.requires_grad = False
    hg = IntergrationStackedHGNet(classes_num=[dsc.NUM_POINT, dsc.NUM_EDGE, dsc.NUM_POINT],
                                  edge_info=[list(x) for x in dsc.EDGE_INFO], nstack=4).to(dev)
    st = torch.load(args.init_hgnet, map_location=dev, weights_only=False)
    hg.load_state_dict(st["hgnet_state_dict"] if "hgnet_state_dict" in st else st, strict=True)
    print(f"[init] hgnet from {args.init_hgnet}", flush=True)

    opt = torch.optim.Adam(hg.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch, eta_min=1e-7)
    crit = NME_DMD(*dsc.NME_ANCHOR).to(dev)

    with open(sd / "metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_nme", "val_nme", "lr", "sec"])
    best = float("inf"); noimp = 0
    for ep in range(args.epoch):
        hg.train(); orf.eval(); s_nme = 0; nb = 0; t0 = time.time()
        for step, batch in enumerate(tl):
            input_t, res_in, _, meta, _, _ = batch
            input_t = input_t.to(dev, non_blocking=True); res_in = res_in.to(dev, non_blocking=True)
            ge = meta["Edge_Heatmaps"].to(dev, non_blocking=True)
            gp = meta["Point_Heatmaps"].to(dev, non_blocking=True)
            glm = meta["Landmarks"].to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.no_grad():
                _, ref, *_ = orf(res_in)
            y, _ = hg(input_t, reference_heatmaps=ref)
            loss = 0
            for si in range(4):
                pl, pe, pp = y[3*si], y[3*si+1], y[3*si+2]
                nme_l = crit(pl, glm).sum()
                loss = loss + nme_l + args.alpha * (torch.mean((pe-ge)**2, 1).sum() + torch.mean((pp-gp)**2, 1).sum())
                s_nme += float(nme_l.detach()) / 4
            loss.backward(); opt.step(); nb += input_t.shape[0]
            if step % 100 == 0:
                print(f"  ep{ep} {step}/{len(tl)} nme={float(nme_l)/input_t.shape[0]:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        sched.step()
        tn = s_nme / max(nb, 1) * 100
        # val
        hg.eval(); errs = []
        with torch.no_grad():
            for batch in vl:
                input_t, res_in, _, meta, _, _ = batch
                input_t = input_t.to(dev, non_blocking=True); res_in = res_in.to(dev, non_blocking=True)
                glm = meta["Landmarks"].to(dev, non_blocking=True)
                _, ref, *_ = orf(res_in)
                _, lmk = hg(input_t, reference_heatmaps=ref)
                errs.append(float(crit(lmk, glm).mean()))
        vn = float(np.mean(errs)) * 100 if errs else 0.0
        sec = time.time() - t0
        print(f"[ep {ep}] train_nme={tn:.4f} val_nme={vn:.4f} lr={opt.param_groups[0]['lr']:.2e} {sec:.0f}s", flush=True)
        with open(sd / "metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([ep, tn, vn, opt.param_groups[0]['lr'], sec])
        if vn < best:
            best = vn; noimp = 0
            torch.save({"hgnet_state_dict": hg.state_dict(), "epoch": ep, "best_nme": best,
                        "note": "fixedmask 8-appearance finetune, init phase3a_478, subject-disjoint"}, sd / "best.pt")
            print(f"  [best] ep{ep} val_nme={best:.4f}", flush=True)
        else:
            noimp += 1
            if noimp >= args.patience:
                print(f"[early stop] no improve {noimp}", flush=True); break
    print(f"DONE best_val_nme={best:.4f}", flush=True)


if __name__ == "__main__":
    main()

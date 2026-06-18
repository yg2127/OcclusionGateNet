#!/usr/bin/env python3
"""occ CNN (TinyRegionCNN) 재학습 — 메모리 안전판.

OOM 수정: frame 을 미리 (N,128,128) uint8 단일 배열로 추출해 RAM 상주(≈0.3GB).
  → npz 반복 로드/worker cache 폭주 없음 (이전 _cache 버전이 465GB OOM 유발).
입력: face_crops_112(_occluded) face crop (1ch,128), 라벨: variant→(le,re,mo) occlusion.
split: hyi fixed split subject-disjoint. 저장: /data/shared/scuppy/yg/occ_cnn_v1/ (기존 안 건드림)
"""
import sys, json, glob, random, importlib.util as ilu
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TCODE = "/data/shared/scuppy/external_scripts/hyi_masking/Step3_full_dir/3_task_train.py"
spec = ilu.spec_from_file_location("t3", TCODE); t3 = ilu.module_from_spec(spec); spec.loader.exec_module(t3)
TinyRegionCNN = t3.TinyRegionCNN
IMG = 128
OUT = Path("/data/shared/scuppy/yg/occ_cnn_v1"); OUT.mkdir(parents=True, exist_ok=True)
RN = "/data/shared/DMD_landmarks/face_crops_112"
RO = "/data/shared/DMD_landmarks/face_crops_112_occluded"
VAR_LABEL = {
    "normal": (0, 0, 0),
    "sunglasses_both_100": (1, 1, 0), "sunglasses_left_100": (1, 0, 0), "sunglasses_right_100": (0, 1, 0),
    "lower_face_without_nose_100": (0, 0, 1), "left_face_half_100": (1, 0, 1), "right_face_half_100": (0, 1, 1),
}
SPLIT = json.load(open("/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json"))
SUBJ = SPLIT["subjects"]
FRAMES_PER_CLIP = 25


def clip_subject(relpath):
    parts = Path(relpath).parts
    for i, p in enumerate(parts):
        if p == "dmd" and i + 2 < len(parts):
            return f"{parts[i+1]}_{parts[i+2]}"
    return None


def extract_arrays():
    """clip 별 npz 1회 로드 → picks frame 만 (128,128) uint8 로 추출. 메모리=추출분만."""
    data = {"train": [[], []], "val": [[], []], "test": [[], []]}
    for variant, lab in VAR_LABEL.items():
        root = RN if variant == "normal" else f"{RO}/{variant}"
        for npz in glob.glob(f"{root}/**/*_crops112.npz", recursive=True):
            rel = npz[len(root) + 1:]
            subj = clip_subject(rel)
            split = "train" if subj in SUBJ["train"] else "val" if subj in SUBJ["val"] else "test" if subj in SUBJ["test"] else None
            if split is None:
                continue
            try:
                with np.load(npz, allow_pickle=True) as d:
                    det = d["detected"].astype(bool)
                    valid = np.where(det)[0]
                    if len(valid) == 0:
                        continue
                    picks = valid[np.linspace(0, len(valid) - 1, min(FRAMES_PER_CLIP, len(valid)), dtype=int)]
                    imgs = d["images"][picks]  # (k,112,112) — picks 만 메모리에
                for im in imgs:
                    data[split][0].append(cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA))
                    data[split][1].append(lab)
            except Exception:
                continue
    out = {}
    for s in data:
        X = np.asarray(data[s][0], dtype=np.uint8) if data[s][0] else np.zeros((0, IMG, IMG), np.uint8)
        y = np.asarray(data[s][1], dtype=np.float32) if data[s][1] else np.zeros((0, 3), np.float32)
        out[s] = (X, y)
    return out


class ArrDS(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        g = self.X[i].astype(np.float32) / 255.0
        if self.train and random.random() < 0.5:
            g = np.clip(g * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)
        g = (g - 0.5) / 0.5
        return torch.from_numpy(g).unsqueeze(0), torch.from_numpy(self.y[i])


def main():
    random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("extracting frames...", flush=True)
    D = extract_arrays()
    for s in D:
        X, y = D[s]
        print(f"  {s}: {len(X)} frames, {X.nbytes/1e9:.2f}GB" + (f", occ(le,re,mo)={y.mean(0).round(3)}" if len(y) else ""), flush=True)

    Xtr, ytr = D["train"]
    tl = DataLoader(ArrDS(Xtr, ytr, True), batch_size=128, shuffle=True, num_workers=2, pin_memory=(dev != "cpu"))
    vl = DataLoader(ArrDS(*D["val"], False), batch_size=256, shuffle=False, num_workers=2)

    model = TinyRegionCNN(3).to(dev)
    pos = ytr.mean(0); pw = torch.tensor((1 - pos) / np.maximum(pos, 1e-3), dtype=torch.float32).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, 8, 0.5)
    from sklearn.metrics import f1_score

    def evaluate(loader):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                ps.append(torch.sigmoid(model(x.to(dev))).cpu().numpy()); ys.append(y.numpy())
        y = np.concatenate(ys); p = np.concatenate(ps); pred = (p > 0.5).astype(int)
        f1 = [f1_score(y[:, k], pred[:, k], zero_division=0) for k in range(3)]
        return float(np.mean(f1)), f1

    best = -1
    for ep in range(20):
        model.train(); tot = 0
        for x, y in tl:
            opt.zero_grad(); loss = crit(model(x.to(dev)), y.to(dev)); loss.backward(); opt.step(); tot += float(loss) * len(x)
        sched.step()
        mf1, f1 = evaluate(vl)
        log = {"epoch": ep, "train_loss": tot / max(len(Xtr), 1), "val_macro_f1": mf1, "le_f1": f1[0], "re_f1": f1[1], "mo_f1": f1[2]}
        print(f"[{ep}] loss={log['train_loss']:.4f} val_macroF1={mf1:.4f} (le {f1[0]:.3f} re {f1[1]:.3f} mo {f1[2]:.3f})", flush=True)
        with open(OUT / "metrics.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
        if mf1 > best:
            best = mf1
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "val_macro_f1": mf1,
                        "label_names": ["left_eye", "right_eye", "mouth"], "image_size": IMG, "var_label": VAR_LABEL,
                        "note": "face_crops_112 domain, mem-safe extract"}, OUT / "best.pt")
            print(f"  [best] ep{ep} macroF1={mf1:.4f}", flush=True)
    # test
    tmf1, tf1 = evaluate(DataLoader(ArrDS(*D["test"], False), batch_size=256, num_workers=2))
    print(f"DONE best_val_macroF1={best:.4f} | TEST macroF1={tmf1:.4f} (le {tf1[0]:.3f} re {tf1[1]:.3f} mo {tf1[2]:.3f})", flush=True)


if __name__ == "__main__":
    main()

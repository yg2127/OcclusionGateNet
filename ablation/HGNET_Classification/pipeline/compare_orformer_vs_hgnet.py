#!/usr/bin/env python3
"""ORFormer+HGNet vs HGNet 단독 — fixedmask NME 비교.

질문: ORFormer reference heatmap 이 HGNet landmark 복원에 실제로 기여하나?
세 모드 (같은 frame, 같은 HGNet weight):
  full   : reference_heatmaps = ORFormer edge heatmap (정상 경로)
  zero   : reference_heatmaps = zeros (conv fusion 은 유지, ORFormer 신호만 0)
  none   : reference_heatmaps = None (conv 까지 skip, 순수 HGNet stack)
appearance × region 별 NME. CPU 고정 (model4 GPU 학습과 충돌 회피).
"""
import sys, re, importlib.util as ilu, numpy as np, torch, cv2, glob, json
from pathlib import Path

torch.set_num_threads(4)
DEV = "cpu"  # model4 GPU 학습중 → CPU 고정
R = Path("/home/yg/fusion/pretrain_v4")
for p in ["configs", "src/data", "src"]:
    sys.path.insert(0, str(R / p))
sys.path.insert(0, "/data/shared/orformer/vendor")
from default import get_cfg
from heatmap_gen import denorm_points
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer
from models.StackedHGNet import IntergrationStackedHGNet
import torchvision.transforms as T
import models.quantizer as _q
_q.device = torch.device(DEV)
NORM = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
cfg = get_cfg(); cfg.DMD.GT_SOURCE = "mediapipe"; ds = cfg.DMD

s = ilu.spec_from_file_location("fr7", "/data/shared/scuppy/Gaze_image_model/src/data/face_regions7.py")
fr7 = ilu.module_from_spec(s); s.loader.exec_module(fr7)
FR = fr7.FACE_REGIONS_7
LE = sorted(set(FR["left_eye"]) | set(range(468, 473)))
RE = sorted(set(FR["right_eye"]) | set(range(473, 478)))
MO = sorted(set(FR["mouth"]))
EYE = np.array(sorted(set(LE) | set(RE)))
REG = {"left_eye": np.array(LE), "right_eye": np.array(RE), "mouth": np.array(MO), "eye_all": EYE, "all": np.arange(478)}

vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
orf = VQVAE(h_dim=128, res_h_dim=32, output_dim=ds.NUM_EDGE, n_res_layers=2, n_embeddings=2048,
            embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(DEV).eval()
orf.load_state_dict(torch.load(str(R / "artifacts/phase2_orformer_fixed/best.pt"), map_location=DEV, weights_only=False).get("model_state_dict"), strict=False)
hg = IntergrationStackedHGNet(classes_num=[ds.NUM_POINT, ds.NUM_EDGE, ds.NUM_POINT],
                              edge_info=[list(x) for x in ds.EDGE_INFO], nstack=4).to(DEV).eval()
hg.load_state_dict(torch.load(str(R / "artifacts/phase3a_hgnet_478/best.pt"), map_location=DEV, weights_only=False)["hgnet_state_dict"], strict=True)
print(f"loaded on {DEV}", flush=True)


@torch.no_grad()
def predict(crop112, mode):
    rgb = np.stack([cv2.resize(crop112, (256, 256))] * 3, -1)
    res = cv2.resize(rgb, (64, 64))
    _, ref, *_ = orf(NORM(res).unsqueeze(0).to(DEV))
    if mode == "full":
        rh = ref
    elif mode == "zero":
        rh = torch.zeros_like(ref)
    else:  # none
        rh = None
    _, lm = hg(NORM(rgb).unsqueeze(0).to(DEV), reference_heatmaps=rh)
    return denorm_points(lm, 64, 64)[0].cpu().numpy() * (112 / 64)


def square_crop_and_gt(frame, bb, gt_full, pad=0.1, sz=112):
    x1, y1, x2, y2 = bb; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    ax, ay = cx - s / 2, cy - s / 2
    h, w = frame.shape[:2]
    a, b = max(0, int(ax)), max(0, int(ay)); a2, b2 = min(w, int(cx + s / 2)), min(h, int(cy + s / 2))
    c = frame[b:b2, a:a2]
    if c.size == 0: return None, None
    if c.ndim == 3: c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(c, (sz, sz))
    gt112 = (gt_full - np.array([ax, ay])) * (sz / s)
    return crop, gt112


def nme(pred, gt, idx):
    al, ar = ds.NME_ANCHOR
    d = max(np.linalg.norm(gt[al] - gt[ar]), 1e-6)
    return np.linalg.norm(pred[idx] - gt[idx], axis=1).mean() / d * 100


VID = "/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask/videos"
BB = "/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask_yolo_face_facemesh/yolo_face"
FM = "/data/shared/DMD_landmarks/facemesh"
APPS = ["solid", "soft_solid", "blur_patch", "smooth_noise", "soft_noise", "noise", "checker", "stripe"]
MODES = ["full", "zero", "none"]

agg = {m: {r: [] for r in REG} for m in MODES}     # 전체
per_app = {a: {m: [] for m in MODES} for a in APPS}  # appearance별 eye_all
n_frames = 0
for app in APPS:
    vids = glob.glob(f"{VID}/*/{app}/*.mp4")[:3]
    for vid in vids:
        name = Path(vid).stem
        m = re.search(r'(g[A-Z]_\d+_s\d+_[0-9T;:+-]+)_ir_face', name)
        if not m: continue
        region = Path(vid).parent.parent.name
        bbf = f"{BB}/{region}/{app}/{name}_face5pt.npz"
        cf = glob.glob(f"{FM}/**/{m.group(1)}_ir_face_facemesh.npz", recursive=True)
        if not Path(bbf).exists() or not cf: continue
        zb = np.load(bbf, allow_pickle=True); bbox = zb["bbox"]; det = zb["detected"].astype(bool)
        zc = np.load(cf[0], allow_pickle=True); clean_lm = zc["landmarks"]; cdet = zc["detected"].astype(bool)
        cap = cv2.VideoCapture(vid)
        valid = np.where(det & cdet[:len(det)])[0]
        if len(valid) == 0: cap.release(); continue
        for fi in valid[np.linspace(0, len(valid) - 1, 5, dtype=int)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
            if not ok: continue
            gt_full = clean_lm[fi][:, :2]
            crop, gt112 = square_crop_and_gt(fr, bbox[fi], gt_full)
            if crop is None or not np.isfinite(gt112).all(): continue
            for mode in MODES:
                pred = predict(crop, mode)
                for r, idx in REG.items():
                    agg[mode][r].append(nme(pred, gt112, idx))
                per_app[app][mode].append(nme(pred, gt112, EYE))
            n_frames += 1
        cap.release()
    done = {m: (np.mean(per_app[app][m]) if per_app[app][m] else float("nan")) for m in MODES}
    print(f"  [{app:<11}] eye_NME full={done['full']:.2f} zero={done['zero']:.2f} none={done['none']:.2f}", flush=True)

print(f"\n=== 전체 평균 NME ({n_frames} frames) ===")
print(f"{'region':<11}{'full(orf+hg)':>14}{'zero(no-orf)':>14}{'none(hg단독)':>14}{'Δ(none-full)':>14}")
out = {}
for r in REG:
    vals = {m: float(np.mean(agg[m][r])) for m in MODES}
    out[r] = vals
    d = vals["none"] - vals["full"]
    print(f"{r:<11}{vals['full']:>14.2f}{vals['zero']:>14.2f}{vals['none']:>14.2f}{d:>+14.2f}")
print("\n해석: Δ(none-full) ≈ 0 이면 ORFormer reference 기여 미미 = '별 차이 없음' 확정.")
print("       Δ 가 크면(eye/iris 에서) ORFormer 가 가림 복원에 실제 기여.")
Path("/data/shared/scuppy/yg/occ_cnn_v1/orformer_vs_hgnet_nme.json").write_text(
    json.dumps({"overall": out, "per_app": {a: {m: float(np.mean(per_app[a][m])) if per_app[a][m] else None for m in MODES} for a in APPS}, "n_frames": n_frames}, indent=2))
print("saved: orformer_vs_hgnet_nme.json")

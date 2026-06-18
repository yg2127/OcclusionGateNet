#!/usr/bin/env python3
"""검증: ORFormer+HGNet 이 hyi fixedmask(8 appearance)에서 landmark 복원 잘 하나.

배경: pretrain_v4 의 HGNet 은 검은마스킹(solid)으로만 occlusion 학습 → noise/checker 등
      새 마스킹은 본 적 없음(OOD). 이게 무너지면 hgnet478 cache 가 masked 에서 부정확 →
      gaze masked 전제 붕괴.
방법: fixedmask 영상 crop → HGNet pred vs clean mediapipe(정답) NME. appearance별 비교.
GPU 있으면 cuda.
"""
import sys, re, importlib.util as ilu, numpy as np, torch, cv2, glob
from pathlib import Path

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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_q.device = torch.device(DEV)
NORM = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
cfg = get_cfg(); cfg.DMD.GT_SOURCE = "mediapipe"; ds = cfg.DMD

s = ilu.spec_from_file_location("fr7", "/data/shared/scuppy/Gaze_image_model/src/data/face_regions7.py")
fr7 = ilu.module_from_spec(s); s.loader.exec_module(fr7)
EYE = np.array(sorted(set(fr7.FACE_REGIONS_7["left_eye"]) | set(fr7.FACE_REGIONS_7["right_eye"]) | set(range(468, 478))))
ALL = np.arange(478)

vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
orf = VQVAE(h_dim=128, res_h_dim=32, output_dim=ds.NUM_EDGE, n_res_layers=2, n_embeddings=2048,
            embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(DEV).eval()
orf.load_state_dict(torch.load(str(R / "artifacts/phase2_orformer_fixed/best.pt"), map_location=DEV, weights_only=False).get("model_state_dict"), strict=False)
hg = IntergrationStackedHGNet(classes_num=[ds.NUM_POINT, ds.NUM_EDGE, ds.NUM_POINT],
                              edge_info=[list(x) for x in ds.EDGE_INFO], nstack=4).to(DEV).eval()
hg.load_state_dict(torch.load(str(R / "artifacts/phase3a_hgnet_478/best.pt"), map_location=DEV, weights_only=False)["hgnet_state_dict"], strict=True)
print(f"loaded on {DEV}", flush=True)


@torch.no_grad()
def hgnet_lm(crop112):
    rgb = np.stack([cv2.resize(crop112, (256, 256))] * 3, -1)
    res = cv2.resize(rgb, (64, 64))
    _, ref, *_ = orf(NORM(res).unsqueeze(0).to(DEV))
    _, lm = hg(NORM(rgb).unsqueeze(0).to(DEV), reference_heatmaps=ref)
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
    gt112 = (gt_full - np.array([ax, ay])) * (sz / s)  # clean facemesh → crop112 좌표
    return crop, gt112


def nme(pred, gt, idx):
    al, ar = ds.NME_ANCHOR
    d = max(np.linalg.norm(gt[al] - gt[ar]), 1e-6)
    return np.linalg.norm(pred[idx] - gt[idx], axis=1).mean() / d * 100


VID = "/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask/videos"
BB = "/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask_yolo_face_facemesh/yolo_face"
FM = "/data/shared/DMD_landmarks/facemesh"
APPS = ["solid", "soft_solid", "blur_patch", "smooth_noise", "soft_noise", "noise", "checker", "stripe"]

print(f"\n{'appearance':<14}{'eye_NME':>9}{'all_NME':>9}{'n_frame':>9}   (검은마스킹=solid/soft_solid, 나머지 OOD)")
print("-" * 60)
results = {}
for app in APPS:
    eye_nmes, all_nmes = [], []
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
            pred = hgnet_lm(crop)
            eye_nmes.append(nme(pred, gt112, EYE)); all_nmes.append(nme(pred, gt112, ALL))
        cap.release()
    if eye_nmes:
        results[app] = (float(np.mean(eye_nmes)), float(np.mean(all_nmes)), len(eye_nmes))
        print(f"{app:<14}{np.mean(eye_nmes):>9.2f}{np.mean(all_nmes):>9.2f}{len(eye_nmes):>9}", flush=True)
    else:
        print(f"{app:<14}{'(no data)':>9}", flush=True)

print("\n해석: solid/soft_solid(학습도메인) 대비 noise/checker/stripe(OOD) NME 가 크게 높으면")
print("  → HGNet 이 새 마스킹에서 landmark 복원 실패 = hgnet478 cache masked 부정확 = gaze masked 전제 붕괴")
import json
Path("/data/shared/scuppy/yg/occ_cnn_v1/hgnet_fixedmask_nme.json").write_text(json.dumps(results, indent=2))
EOF

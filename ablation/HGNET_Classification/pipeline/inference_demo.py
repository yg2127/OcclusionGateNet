#!/usr/bin/env python3
"""Landmark + occlusion inference 데모 — 번들 모델(models/) 사용.

face crop 이미지 → ORFormer+HGNet 으로 478 landmark 복원 + occ CNN region 가림 예측.
이 저장소의 핵심 기여(가림 강건 좌표 복원)를 단독 실행 가능하게 보여준다.

요구: PYTHONPATH 에 ORFormer vendor 추가 (외부)
  export PYTHONPATH=/data/shared/orformer/vendor:$PYTHONPATH

사용:
  python pipeline/inference_demo.py --image path/to/face.png
  python pipeline/inference_demo.py --image face.png --occ   # occ CNN 가림 예측까지
출력: 좌표 npy + (옵션) 오버레이 png
"""
import sys, argparse, importlib.util as ilu
from pathlib import Path
import numpy as np, torch, cv2

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "models"
LANDMARK = REPO / "landmark"
for p in ["src", "src/data", "configs"]:
    sys.path.insert(0, str(LANDMARK / p))
# ORFormer vendor (외부 — README 참조)
for cand in ["/data/shared/orformer/vendor"]:
    if Path(cand).exists():
        sys.path.insert(0, cand)

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


def load_landmark_models(hgnet_ckpt=None):
    vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
    orf = VQVAE(h_dim=128, res_h_dim=32, output_dim=ds.NUM_EDGE, n_res_layers=2, n_embeddings=2048,
                embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(DEV).eval()
    orf.load_state_dict(torch.load(CKPT / "orformer/best.pt", map_location=DEV, weights_only=False)["model_state_dict"], strict=False)
    hg = IntergrationStackedHGNet(classes_num=[ds.NUM_POINT, ds.NUM_EDGE, ds.NUM_POINT],
                                  edge_info=[list(x) for x in ds.EDGE_INFO], nstack=4).to(DEV).eval()
    ck = torch.load(hgnet_ckpt or (CKPT / "hgnet_phase3a_v3/best.pt"), map_location=DEV, weights_only=False)
    hg.load_state_dict(ck["hgnet_state_dict"] if "hgnet_state_dict" in ck else ck, strict=True)
    return orf, hg


@torch.no_grad()
def infer_landmarks(crop_gray_112, orf, hg):
    """crop_gray_112: (112,112) uint8 grayscale → (478,2) in 112-crop coord."""
    rgb = np.stack([cv2.resize(crop_gray_112, (256, 256))] * 3, -1)
    res = cv2.resize(rgb, (64, 64))
    _, ref, *_ = orf(NORM(res).unsqueeze(0).to(DEV))
    _, lm = hg(NORM(rgb).unsqueeze(0).to(DEV), reference_heatmaps=ref)
    return denorm_points(lm, 64, 64)[0].cpu().numpy() * (112 / 64)


def load_occ_cnn():
    """region occlusion CNN (TinyRegionCNN) — models/occ_cnn_retrain_mine/best.pt (※ model4 occ_pred 와 무관)."""
    code = "/data/shared/scuppy/external_scripts/hyi_masking/Step3_full_dir/3_task_train.py"
    if not Path(code).exists():
        return None
    spec = ilu.spec_from_file_location("t3", code); t3 = ilu.module_from_spec(spec); spec.loader.exec_module(t3)
    m = t3.TinyRegionCNN(3).to(DEV).eval()
    ck = torch.load(CKPT / "occ_cnn_retrain_mine/best.pt", map_location=DEV, weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); return m


@torch.no_grad()
def infer_occ(crop_gray, occ_model, sz=128):
    g = cv2.resize(crop_gray, (sz, sz)).astype(np.float32) / 255.0
    g = (g - 0.5) / 0.5
    x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(DEV)
    p = torch.sigmoid(occ_model(x))[0].cpu().numpy()
    return dict(left_eye=float(p[0]), right_eye=float(p[1]), mouth=float(p[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="face crop 이미지 (grayscale 권장)")
    ap.add_argument("--hgnet-ckpt", default=None, help="기본 phase3a, finetune 모델 경로 지정 가능")
    ap.add_argument("--occ", action="store_true", help="occ CNN region 가림 예측")
    ap.add_argument("--out", default="landmarks.npy")
    ap.add_argument("--overlay", default=None, help="오버레이 png 저장 경로")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"이미지 로드 실패: {args.image}")
    crop = cv2.resize(img, (112, 112))
    orf, hg = load_landmark_models(args.hgnet_ckpt)
    lm = infer_landmarks(crop, orf, hg)
    np.save(args.out, lm)
    print(f"[landmark] {lm.shape} → {args.out}  (device={DEV})")

    if args.occ:
        om = load_occ_cnn()
        if om is None:
            print("[occ] occ CNN 코드(TinyRegionCNN) 경로 없음 — 스킵")
        else:
            print("[occ] region 가림 prob:", infer_occ(img, om))

    if args.overlay:
        vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        for x, y in lm.astype(int):
            cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv2.imwrite(args.overlay, vis); print(f"[overlay] → {args.overlay}")


if __name__ == "__main__":
    main()

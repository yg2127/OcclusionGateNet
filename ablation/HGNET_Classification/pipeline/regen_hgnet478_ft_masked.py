"""finetuned HGNet 으로 masked 샘플의 hgnet478 좌표 재생성 → *_hgnet478_ft.npz.

기존 _hgnet478.npz(v3 base) 안 건드림(새 suffix). occgateRAW 는 가림 부위만 hgnet 쓰므로
masked variant 만 재생성. clean 은 facemesh raw 라 불필요.
init: hgnet_fixedmask_ft/best.pt (fixedmask 8 appearance finetune).
"""
import json, sys, torch, cv2, numpy as np, time, argparse
from pathlib import Path

P4 = Path("/home/yg/fusion/pretrain_v4")
sys.path[:0] = [str(P4/"src"), str(P4/"src/data"), str(P4/"configs"), "/data/shared/orformer/vendor"]
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer
from models.StackedHGNet import IntergrationStackedHGNet
from heatmap_gen import denorm_points
from default import get_cfg
import torchvision.transforms as T

ap = argparse.ArgumentParser()
ap.add_argument("--hgnet-ckpt", default="/data/shared/scuppy/yg/hgnet_fixedmask_ft/best.pt")
ap.add_argument("--suffix", default="_hgnet478_ft.npz")
args = ap.parse_args()

cfg = get_cfg(); ds_cfg = cfg.DMD
DEV = torch.device("cuda")
NORM = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
ORF_CKPT = P4/"artifacts/phase2_orformer_fixed/best.pt"
SPLIT = json.load(open("/data/shared/scuppy/hyi/fixed_splits/dms_clean_masked_fixed_items_v1.json"))

vit = ORFormer(image_size=16, patch_size=1, num_classes=2048, dim=256, depth=3, heads=8, mlp_dim=512, channels=256)
orf = VQVAE(h_dim=128, res_h_dim=32, output_dim=ds_cfg.NUM_EDGE, n_res_layers=2, n_embeddings=2048,
            embedding_dim=256, code_dim=256, beta=0.25, vit=vit).to(DEV).eval()
orf.load_state_dict(torch.load(str(ORF_CKPT), map_location=DEV, weights_only=False)["model_state_dict"], strict=False)
hg = IntergrationStackedHGNet(classes_num=[ds_cfg.NUM_POINT, ds_cfg.NUM_EDGE, ds_cfg.NUM_POINT],
                              edge_info=[list(x) for x in ds_cfg.EDGE_INFO], nstack=4).to(DEV).eval()
st = torch.load(args.hgnet_ckpt, map_location=DEV, weights_only=False)
hg.load_state_dict(st["hgnet_state_dict"] if "hgnet_state_dict" in st else st, strict=True)
print(f"[hgnet] finetuned ckpt {args.hgnet_ckpt} (best_nme={st.get('best_nme')})", flush=True)


@torch.no_grad()
def infer(arr, bs=256):
    T_ = len(arr); out = np.zeros((T_, 478, 2), np.float32)
    for s in range(0, T_, bs):
        e = min(s+bs, T_); il, rl = [], []
        for i in range(s, e):
            rgb = np.stack([cv2.resize(arr[i], (256,256))]*3, -1)
            il.append(NORM(rgb)); rl.append(NORM(cv2.resize(rgb, (64,64))))
        inp = torch.stack(il).to(DEV); res = torch.stack(rl).to(DEV)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, ref, *_ = orf(res); _, lm = hg(inp, reference_heatmaps=ref)
        out[s:e] = denorm_points(lm.float(), 64, 64).cpu().numpy() * (112/64)
    return out


# masked face_paths 만
masked = {}
for L in ["train_clean_masked_1to1","val_clean_masked_1to1","test_clean_paired","test_masked"]:
    for it in SPLIT["items"][L]:
        if it.get("variant") == "masked":
            masked[it["face_path"]] = it
print(f"masked samples: {len(masked)}", flush=True)
done = 0
for fp_s in masked:
    fp = Path(fp_s)
    crop = fp.parent / fp.name.replace("_facemesh.npz", "_crops112.npz")
    if not crop.exists():
        print(f"  SKIP no crop {crop.name}"); continue
    out = fp.parent / fp.name.replace("_facemesh.npz", args.suffix)
    if out.exists(): done += 1; continue
    with np.load(crop) as d: arr = d["images"]
    ts = time.time(); lm = infer(arr)
    landmarks = np.concatenate([lm, np.zeros((*lm.shape[:-1],1), lm.dtype)], -1)
    meta = {"source": "hgnet_fixedmask_ft", "ckpt": args.hgnet_ckpt, "n_frames": int(len(arr)), "hyi_split_face_path": str(fp)}
    np.savez_compressed(out, landmarks=landmarks, detected=np.ones(len(arr), bool), meta=meta)
    done += 1
    print(f"  [{done}/{len(masked)}] {out.name} T={len(arr)} ({time.time()-ts:.1f}s)", flush=True)
print(f"DONE regen {done}/{len(masked)}", flush=True)

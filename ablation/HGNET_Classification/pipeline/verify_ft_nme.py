"""finetune 검증: old phase3a HGNet vs fixedmask-finetuned HGNet — 가린부위 NME 비교.

같은 fixedmask 영상/프레임에 두 HGNet 추론 → 가린 region NME. 개선되면 checker/stripe/both_eyes ↓.
CPU 가능(샘플 적음). model5 진행 전 gate.
"""
import sys, re, glob, importlib.util as ilu, numpy as np, torch, cv2, json
from pathlib import Path
import collections

torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = Path("/home/yg/fusion/pretrain_v4")
for p in ["configs","src/data","src"]: sys.path.insert(0,str(R/p))
sys.path.insert(0,"/data/shared/orformer/vendor")
from default import get_cfg
from heatmap_gen import denorm_points
from models.VQVAE import VQVAE
from models.simple_vit import ORFormer
from models.StackedHGNet import IntergrationStackedHGNet
import torchvision.transforms as T
import models.quantizer as _q
_q.device = torch.device(DEV)
NORM=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
cfg=get_cfg(); cfg.DMD.GT_SOURCE="mediapipe"; ds=cfg.DMD

s=ilu.spec_from_file_location("fr7","/data/shared/scuppy/Gaze_image_model/src/data/face_regions7.py")
fr7=ilu.module_from_spec(s); s.loader.exec_module(fr7); FR=fr7.FACE_REGIONS_7
LE=np.array(sorted(set(FR["left_eye"])|set(range(468,473)))); RE=np.array(sorted(set(FR["right_eye"])|set(range(473,478))))
MO=np.array(sorted(set(FR["mouth"]))); IRIS=np.arange(468,478)
ROCC={"left_eye":LE,"right_eye":RE,"mouth":MO,"both_eyes":np.union1d(LE,RE)}

vit=ORFormer(image_size=16,patch_size=1,num_classes=2048,dim=256,depth=3,heads=8,mlp_dim=512,channels=256)
orf=VQVAE(h_dim=128,res_h_dim=32,output_dim=ds.NUM_EDGE,n_res_layers=2,n_embeddings=2048,embedding_dim=256,code_dim=256,beta=0.25,vit=vit).to(DEV).eval()
orf.load_state_dict(torch.load(str(R/"artifacts/phase2_orformer_fixed/best.pt"),map_location=DEV,weights_only=False)["model_state_dict"],strict=False)

def load_hg(ckpt):
    hg=IntergrationStackedHGNet(classes_num=[ds.NUM_POINT,ds.NUM_EDGE,ds.NUM_POINT],edge_info=[list(x) for x in ds.EDGE_INFO],nstack=4).to(DEV).eval()
    st=torch.load(ckpt,map_location=DEV,weights_only=False)
    hg.load_state_dict(st["hgnet_state_dict"] if "hgnet_state_dict" in st else st,strict=True)
    return hg
OLD=load_hg(str(R/"artifacts/phase3a_hgnet_478/best.pt"))
FT =load_hg("/data/shared/scuppy/yg/hgnet_fixedmask_ft/best.pt")
print("loaded old + ft",flush=True)

@torch.no_grad()
def pred(crop,hg):
    rgb=np.stack([cv2.resize(crop,(256,256))]*3,-1); res=cv2.resize(rgb,(64,64))
    _,ref,*_=orf(NORM(res).unsqueeze(0).to(DEV)); _,lm=hg(NORM(rgb).unsqueeze(0).to(DEV),reference_heatmaps=ref)
    return denorm_points(lm,64,64)[0].cpu().numpy()*(112/64)
def crop_gt(frame,bb,gt,pad=0.1,sz=112):
    x1,y1,x2,y2=bb; cx,cy=(x1+x2)/2,(y1+y2)/2; s=max(x2-x1,y2-y1)*(1+2*pad); ax,ay=cx-s/2,cy-s/2
    h,w=frame.shape[:2]; a,b=max(0,int(ax)),max(0,int(ay)); a2,b2=min(w,int(cx+s/2)),min(h,int(cy+s/2))
    c=frame[b:b2,a:a2]
    if c.size==0: return None,None
    if c.ndim==3: c=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
    return cv2.resize(c,(sz,sz)),(gt-np.array([ax,ay]))*(sz/s)
def nme(p,g,idx):
    al,ar=ds.NME_ANCHOR; d=max(np.linalg.norm(g[al]-g[ar]),1e-6)
    return np.linalg.norm(p[idx]-g[idx],axis=1).mean()/d*100

VID="/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask/videos"
BB="/data/shared/Occlusion_subset_dataset/region_occlusion_video_dataset_v3_original_fixedmask_yolo_face_facemesh/yolo_face"
FM="/data/shared/DMD_landmarks/facemesh"
APPS=["solid","soft_solid","blur_patch","smooth_noise","soft_noise","noise","checker","stripe"]
REGIONS=["left_eye","right_eye","mouth","both_eyes"]
old_t=collections.defaultdict(list); ft_t=collections.defaultdict(list)
for region in REGIONS:
    for app in APPS:
        for vid in sorted(glob.glob(f"{VID}/{region}/{app}/*.mp4"))[:2]:
            name=Path(vid).stem; m=re.search(r'(g[A-Z]_\d+_s\d+_[0-9T;:+-]+)_ir_face',name)
            if not m: continue
            bbf=f"{BB}/{region}/{app}/{name}_face5pt.npz"; cf=glob.glob(f"{FM}/**/{m.group(1)}_ir_face_facemesh.npz",recursive=True)
            if not Path(bbf).exists() or not cf: continue
            zb=np.load(bbf,allow_pickle=True); bbox=zb["bbox"]; det=zb["detected"].astype(bool)
            zc=np.load(cf[0],allow_pickle=True); clean=zc["landmarks"]; cdet=zc["detected"].astype(bool)
            cap=cv2.VideoCapture(vid); valid=np.where(det&cdet[:len(det)])[0]
            if len(valid)==0: cap.release(); continue
            for fi in valid[np.linspace(0,len(valid)-1,4,dtype=int)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ok,fr=cap.read()
                if not ok: continue
                crop,gt=crop_gt(fr,bbox[fi],clean[fi][:,:2])
                if crop is None or not np.isfinite(gt).all(): continue
                idx=ROCC[region]
                old_t[region].append(nme(pred(crop,OLD),gt,idx)); ft_t[region].append(nme(pred(crop,FT),gt,idx))
            cap.release()
print(f"\n=== 가린-부위 NME: OLD(phase3a) vs FT(fixedmask finetune) ===")
print(f"{'region':<11}{'OLD':>8}{'FT':>8}{'Δ':>8}")
for region in REGIONS:
    o=np.mean(old_t[region]) if old_t[region] else float('nan'); f=np.mean(ft_t[region]) if ft_t[region] else float('nan')
    print(f"{region:<11}{o:>8.2f}{f:>8.2f}{f-o:>+8.2f}")
allo=[v for r in REGIONS for v in old_t[r]]; allf=[v for r in REGIONS for v in ft_t[r]]
print(f"{'ALL':<11}{np.mean(allo):>8.2f}{np.mean(allf):>8.2f}{np.mean(allf)-np.mean(allo):>+8.2f}")
print("\nΔ<0 = finetune 개선. both_eyes/checker/stripe 큰 개선 기대.")
json.dump({"old":{r:float(np.mean(old_t[r])) if old_t[r] else None for r in REGIONS},
           "ft":{r:float(np.mean(ft_t[r])) if ft_t[r] else None for r in REGIONS}},
          open("/data/shared/scuppy/yg/occ_cnn_v1/ft_nme_compare.json","w"),indent=2)

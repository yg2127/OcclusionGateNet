#!/usr/bin/env python3
"""model4 gaze 분류 사례 시각화 노트북 생성기.

각 test window 에 대해 model4 gaze pred vs GT → 성공/실패 × clean/masked 버킷.
face branch 입력값(occgateRAW 478 landmark)을 입력 영상 프레임 위에 오버레이해 다수 예시 표시.
- clean window: clean 영상 프레임 / masked window: fixedmask 가림 영상 프레임 (meta.source_video)
- iris(468~477)·눈·입 region 색상 구분 → 가림 시 어디가 무너지나 가시화
"""
import nbformat as nbf

REPO = "/data/shared/scuppy/yg/Ablation/AblationB"
RUN = "/data/shared/scuppy/yg/Ablation/AblationB/results/model4_occgateRAW_taskGated_occCNN_seed42"

nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def co(s): C.append(nbf.v4.new_code_cell(s))

md("""# model4 Gaze 분류 사례 시각화 — face branch landmark + 입력 이미지

model4(occgateRAW, gaze clean 0.600 / masked 0.546) 의 gaze head 가
**어떤 입력(landmark)에서 맞고 틀리나**를 입력 영상과 함께 본다.

- face branch 입력 = occgateRAW 478 landmark (정상=facemesh 원본 / 가림=hgnet→facemesh)
- clean window → clean 영상 프레임 / masked window → fixedmask 가림 영상 프레임
- region 색: iris(빨강)·눈(주황)·입(노랑)·기타(흐림) → 가림 시 어느 좌표가 무너지나
- gaze zone 9-class: left_mirror/left/front/center_mirror/front_right/right_mirror/right/infotainment/steering_wheel""")

co('''import sys, json, os, glob
from pathlib import Path
import numpy as np, torch, cv2
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

REPO = "''' + REPO + '''"; RUN = "''' + RUN + '''"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/configs")
from src.training.builders import build_clip_splits, build_model
from src.data.dataset import MemoryMultitaskDataset, preload_multitask_windows
from src.training.loops import IGNORE_LABEL
from torch.utils.data import DataLoader

import logging
log = logging.getLogger("gazeviz"); log.addHandler(logging.StreamHandler()); log.setLevel(logging.WARNING)
DEV = "cpu"   # finetune GPU 점유 회피 (classifier 작음)
cfg = json.loads((Path(RUN) / "config.json").read_text())
GAZE_ZONES = ["left_mirror","left","front","center_mirror","front_right","right_mirror","right","infotainment","steering_wheel"]
sw = cfg["face"].get("npz_swap", {})
print("loaded cfg, fusion:", cfg["model"]["fusion"]["kind"], "face npz_swap:", sw.get("to"))''')

co("""# test clips (gaze source 만) + clip_id→spec
_, _, test_clips = build_clip_splits(cfg, Path("/tmp/_gaze_viz_splits"), log)
print("test splits:", {k: len(v) for k,v in test_clips.items()})
gaze_clips = {k: [c for c in v if c.source == "gaze"] for k,v in test_clips.items()}
print("gaze-only:", {k: len(v) for k,v in gaze_clips.items()})
spec_by_id = {}
for v in gaze_clips.values():
    for c in v: spec_by_id[c.clip_id] = c""")

co("""# window preload (gaze clip)
fc = cfg["face"]
common = dict(
    window_size=cfg["window"]["size"], window_stride=cfg["window"]["stride"],
    max_windows_per_clip=cfg["window"]["max_per_clip"],
    pose_min_valid_frames=cfg["window"]["pose_min_valid_frames"],
    pose_min_valid_ratio=cfg["window"]["pose_min_valid_ratio"],
    pose_min_valid_joint_ratio=cfg["window"]["pose_min_valid_joint_ratio"],
    face_min_detected_ratio=cfg["window"]["face_min_detected_ratio"],
    joint_conf_thres=cfg["pose"]["joint_conf_thres"],
    face_mode=fc["mode"], face_use_z=fc.get("use_z", True),
    face_use_detected_channel=fc.get("use_detected_channel", True),
    face_use_det_score_channel=fc.get("use_det_score_channel", True),
    face_bbox_det_thres=fc.get("bbox_det_thres", 0.25),
    occ_cfg=cfg.get("occ", {}), face_npz_swap=cfg.get("face", {}).get("npz_swap"), logger=log)
items = {k: preload_multitask_windows(v, desc=f"preload {k}", **common) for k,v in gaze_clips.items()}
print("windows:", {k: len(v) for k,v in items.items()})""")

co("""# model4 추론 → per-window gaze pred (split_items 순서 보존)
model, _ = build_model(cfg, DEV)
ck = torch.load(Path(RUN)/"best.pt", map_location=DEV, weights_only=False)
model.load_state_dict(ck["model_state_dict"]); model.eval()

records = []   # dict: split, clip_id, window_idx, num_win, gt, pred, correct, occ
for split, wins in items.items():
    loader = DataLoader(MemoryMultitaskDataset(wins), batch_size=128, shuffle=False, num_workers=2)
    preds = []
    with torch.no_grad():
        for batch in loader:
            xb=batch["x_body"].to(DEV); xf=batch["x_face"].to(DEV)
            xo=batch.get("x_occ"); xo=xo.to(DEV) if xo is not None else None
            lg = model(xb, xf, x_occ=xo)["gaze"]
            preds.extend(lg.argmax(1).cpu().numpy().tolist())
    for w, p in zip(wins, preds):
        if w.y_gaze_fine == IGNORE_LABEL: continue
        records.append(dict(split=split, clip_id=w.clip_id, window_idx=w.window_idx,
                            num_win=w.num_windows_in_clip, gt=int(w.y_gaze_fine), pred=int(p),
                            correct=(int(p)==int(w.y_gaze_fine)), occ=np.asarray(w.x_occ)))
print("gaze windows:", len(records))
for split in items:
    rs=[r for r in records if r["split"]==split]
    acc=np.mean([r["correct"] for r in rs]) if rs else 0
    print(f"  {split}: n={len(rs)} window_acc={acc:.3f}")""")

md("""## 1. 혼동행렬 & zone별 정확도 (clean vs masked)""")

co("""from sklearn.metrics import confusion_matrix, f1_score
fig, axes = plt.subplots(1, len(items), figsize=(7*len(items),6))
if len(items)==1: axes=[axes]
for ax,(split,_) in zip(axes, items.items()):
    rs=[r for r in records if r["split"]==split]
    if not rs: continue
    y=[r["gt"] for r in rs]; p=[r["pred"] for r in rs]
    cm=confusion_matrix(y,p,labels=range(9))
    cmn=cm/cm.sum(1,keepdims=True).clip(min=1)
    im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(GAZE_ZONES,rotation=90,fontsize=7); ax.set_yticklabels(GAZE_ZONES,fontsize=7)
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    mf1=f1_score(y,p,labels=range(9),average="macro",zero_division=0)
    ax.set_title(f"{split}  macroF1={mf1:.3f}  acc={np.mean([r['correct'] for r in rs]):.3f}")
    for i in range(9):
        for j in range(9):
            if cmn[i,j]>0.02: ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if cmn[i,j]>0.5 else "black")
plt.tight_layout(); plt.show()""")

md("""## 2. landmark + 입력영상 오버레이 렌더러
occgateRAW 478 landmark(face branch 입력)를 해당 영상 프레임 위에. region 색으로 가림 영향 가시화.""")

co("""LE=np.array(sorted(set(range(33,134))|set(range(468,473))))  # 근사 eye 영역
# 정확 region: face_regions7
import importlib.util as ilu
s=ilu.spec_from_file_location("fr7","/data/shared/scuppy/Gaze_image_model/src/data/face_regions7.py")
fr7=ilu.module_from_spec(s); s.loader.exec_module(fr7); FR=fr7.FACE_REGIONS_7
LE=np.array(sorted(set(FR["left_eye"])|set(range(468,473))))
RE=np.array(sorted(set(FR["right_eye"])|set(range(473,478))))
MO=np.array(sorted(set(FR["mouth"]))); IRIS=np.arange(468,478)
EYE=np.union1d(LE,RE)
OTHER=np.setdiff1d(np.arange(478), np.concatenate([EYE,MO]))

def resolve_video(face_npz):
    z=np.load(face_npz,allow_pickle=True); sv=z["meta"].item().get("source_video","")
    if sv and os.path.isabs(sv) and os.path.exists(sv): return sv
    # clean: basename → DMD 경로 복원
    cand=face_npz.replace("/DMD_landmarks/facemesh/","/DMD/").replace("_facemesh.npz",".mp4")
    return cand if os.path.exists(cand) else None

_vid_cache={}
def get_frame(video, idx):
    cap=_vid_cache.get(video)
    if cap is None:
        cap=cv2.VideoCapture(video); _vid_cache[video]=cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok,fr=cap.read()
    return fr if ok else None

def _render(ax, rec):
    spec=spec_by_id[rec["clip_id"]]
    raw=spec.face_npz.replace("_facemesh.npz","_hgnet478_occgateRAW.npz")
    if not os.path.exists(raw): ax.axis("off"); ax.set_title("no npz",fontsize=7); return
    lm=np.load(raw,allow_pickle=True)["landmarks"]
    frac=(rec["window_idx"]+0.5)/max(rec["num_win"],1)
    fidx=int(spec.face_start + frac*(spec.face_end-spec.face_start))
    fidx=min(max(fidx,0), len(lm)-1)
    pts=lm[fidx][:,:2]
    good=np.isfinite(pts).all(1)
    if good.sum()<8: ax.axis("off"); ax.set_title("no landmark",fontsize=7); return
    video=resolve_video(spec.face_npz); frame=get_frame(video,fidx) if video else None
    x0,y0=np.nanmin(pts[good],0); x1,y1=np.nanmax(pts[good],0)
    pad=0.35*max(x1-x0,y1-y0,1.0); x0,y0,x1,y1=x0-pad,y0-pad,x1+pad,y1+pad
    if frame is not None:
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        H,W=g.shape; xa,ya,xb,yb=int(max(0,x0)),int(max(0,y0)),int(min(W,x1)),int(min(H,y1))
        if xb>xa and yb>ya: ax.imshow(g[ya:yb,xa:xb],cmap="gray",extent=[xa,xb,yb,ya])
    for idx,c,sz in [(OTHER,"#7777ff",2),(EYE,"orange",6),(MO,"yellow",6),(IRIS,"red",16)]:
        m=np.isfinite(pts[idx]).all(1)
        ax.scatter(pts[idx][m,0],pts[idx][m,1],s=sz,c=c,edgecolors="none")
    ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.axis("off")
    mark="O" if rec["correct"] else "X"
    occ=rec["occ"]; occs=f"le{occ[0]:.1f} re{occ[1]:.1f} mo{occ[3]:.1f}" if len(occ)>=4 else ""
    ax.set_title(f"[{mark}] gt={GAZE_ZONES[rec['gt']]}\\npred={GAZE_ZONES[rec['pred']]}\\nocc {occs}",
                 fontsize=7, color=("green" if rec["correct"] else "red"))

def render(ax, rec):
    try: _render(ax, rec)
    except Exception as e:
        ax.axis("off"); ax.set_title("err",fontsize=6)

def grid(recs, title, ncol=6, nrow=3):
    recs=recs[:ncol*nrow]
    if not recs: print("(no samples)",title); return
    fig,axes=plt.subplots(nrow,ncol,figsize=(2.5*ncol,3*nrow))
    for ax,rec in zip(np.array(axes).ravel(), recs): render(ax,rec)
    for ax in np.array(axes).ravel()[len(recs):]: ax.axis("off")
    fig.suptitle(title,y=1.01,fontsize=12); plt.tight_layout(); plt.show()""")

md("""## 3. 성공 / 실패 사례 (clean) — 다수 예시
iris(빨강)가 선명하고 위치가 시선 방향과 맞으면 정답. 틀린 경우 iris/눈 좌표가 애매한지 관찰.""")

co("""import random; random.seed(0)
def bucket(split, correct):
    rs=[r for r in records if r["split"]==split and r["correct"]==correct]
    random.shuffle(rs); return rs
clean_key=[k for k in items if "clean" in k][0] if any("clean" in k for k in items) else list(items)[0]
grid(bucket(clean_key,True),  "CLEAN  CORRECT (gt=pred)")
grid(bucket(clean_key,False), "CLEAN  WRONG (gt!=pred)")""")

md("""## 4. 성공 / 실패 사례 (masked) — 가림 영상 위 landmark
가림 region 의 landmark(occgateRAW=hgnet 복원)가 시선과 맞는지. iris 가림 시 틀리는 패턴 확인.""")

co("""mask_keys=[k for k in items if "mask" in k]
if mask_keys:
    mk=mask_keys[0]
    grid(bucket(mk,True),  "MASKED  CORRECT")
    grid(bucket(mk,False), "MASKED  WRONG")
else:
    print("masked split 없음")""")

md("""## 5. 대표 혼동쌍 (자주 틀리는 gt→pred) 사례
혼동행렬에서 큰 off-diagonal 을 골라 실제 입력을 본다.""")

co("""pairs=Counter((r["gt"],r["pred"]) for r in records if not r["correct"])
top=[p for p in pairs.most_common(4)]
for (gt,pr),cnt in top:
    rs=[r for r in records if r["gt"]==gt and r["pred"]==pr]
    random.shuffle(rs)
    grid(rs, f"CONFUSION x{cnt}:  true={GAZE_ZONES[gt]}  ->  pred={GAZE_ZONES[pr]}", ncol=6, nrow=1)""")

md("""## 6. 해석 가이드
- **clean 오답에서 iris 좌표가 시선과 어긋나면** → region_pool(478→10)이 iris 미세정보를 버려 좌표만으로 zone 구분 한계 (eye-image 필요 근거).
- **masked 오답이 iris 가림 window 에 몰리면** → hgnet 복원 좌표가 시선 정보를 못 살림 = masked 병목, fixedmask finetune 대상.
- **clean·masked 모두 특정 zone(center_mirror↔front 등) 혼동** → zone 경계 모호성(라벨/기하 한계), 모델 무관.
- occ 벡터(le/re/mo)와 오답 상관 → occlusion gating 이 실제로 도움/방해 되는지 진단.""")

nb["cells"]=C
out="/data/shared/scuppy/yg/occ_cnn_v1/model4_gaze_cases.ipynb"
nbf.write(nb, out); print("wrote", out, "cells", len(C))

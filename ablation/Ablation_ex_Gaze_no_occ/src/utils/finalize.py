"""학습 종료 후 results/ 에 multi-head REPORT.md + 핵심 파일 복사.

REPORT.md 구성:
1. TL;DR 요약
2. 실험 설정
3. 데이터 파이프라인 통계
4. 학습 진행 요약 (first/best/last epoch)
5. Test per-head metric + per-class table
6. 약점 top-3 (각 head)
7. 주요 혼동 쌍 (confusion pairs)
8. 시스템·성능 (params, latency, env)
9. 산출물 경로
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from constants.gaze_zones import GAZE_ZONES
from src.utils.io import load_json


_CURATED_FILES = (
    "summary.json",
    "config.json",
    "metrics.csv",
    "history.json",
    "clip_manifest.json",
    "split_info.json",
    "train.log",
    "best.pt",
    "test_action_clip_confusion.csv",
    "test_gaze_clip_confusion.csv",
    "test_hands_clip_confusion.csv",
    "test_talk_clip_confusion.csv",
    "test_gaze_binary_on_distraction.csv",   # v1.2 신설
)

ACTION_LABELS = [
    "safe_drive", "texting_right", "texting_left",
    "phonecall_right", "phonecall_left",
    "radio", "drinking",
    "reach_side", "reach_backseat",
    "hair_and_makeup", "talking_to_passenger",
]

HANDS_LABELS = ["both", "only_left", "only_right", "none"]
TALK_LABELS = ["not_talking", "talking"]

HEAD_LABEL_NAMES = {
    "action": ACTION_LABELS,
    "gaze":   GAZE_ZONES,
    "hands":  HANDS_LABELS,
    "talk":   TALK_LABELS,
}


# ---------- per-class metrics ----------
def per_class_metrics(cm: np.ndarray, labels: list[str]) -> pd.DataFrame:
    eps = 1e-12
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1).astype(np.float64) - tp
    fp = cm.sum(axis=0).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.int64)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    for i in range(len(labels)):
        if tp[i] + fn[i] == 0: rec[i] = 0; f1[i] = 0
        if tp[i] + fp[i] == 0: prec[i] = 0
    return pd.DataFrame({
        "label": labels, "support": support,
        "precision": prec, "recall (acc)": rec, "f1": f1,
    })


def _md_table(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    df2 = df.copy()
    for c in ("precision", "recall (acc)", "f1"):
        if c in df2.columns:
            df2[c] = df2[c].map(fmt.format)
    cols = list(df2.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df2.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _load_cm(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    return df.values.astype(np.int64)


# ---------- confusion pair 분석 ----------
def _notable_confusions(cm: np.ndarray, labels: list[str], top_k: int = 5) -> list[tuple[str, str, int, float]]:
    """off-diagonal top-k: (true_label, pred_label, count, rate_of_class_support)."""
    n = cm.shape[0]
    support = cm.sum(axis=1)
    pairs: list[tuple[str, str, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i, j])
            if c == 0:
                continue
            rate = c / max(int(support[i]), 1)
            pairs.append((labels[i], labels[j], c, rate))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]


# ---------- training curve summary ----------
def _training_curve_summary(save_root: Path) -> dict | None:
    """metrics.csv 에서 epoch 0, best, last 행 추출 + 주요 값 요약."""
    mpath = save_root / "metrics.csv"
    if not mpath.exists():
        return None
    try:
        df = pd.read_csv(mpath, encoding="utf-8-sig")
    except Exception:
        return None
    if len(df) == 0:
        return None
    # best by val_weighted_score
    if "val_weighted_score" in df.columns:
        best_row = df.loc[df["val_weighted_score"].idxmax()]
    else:
        best_row = df.iloc[-1]
    first_row = df.iloc[0]
    last_row = df.iloc[-1]
    return {
        "n_epochs_run": len(df),
        "first": first_row.to_dict(),
        "best": best_row.to_dict(),
        "last": last_row.to_dict(),
    }


def _format_training_curve_block(tc: dict) -> str:
    lines = [f"- **총 epoch 수**: {tc['n_epochs_run']}"]
    for label, row in (("first (ep1)", tc["first"]), ("best", tc["best"]), ("last", tc["last"])):
        ep = int(row.get("epoch", -1))
        train_l = row.get("train_loss", 0)
        val_l = row.get("val_loss", 0)
        score = row.get("val_weighted_score", 0)
        act_c = row.get("val_action_c_f1", 0)
        gaze_c = row.get("val_gaze_c_f1", 0)
        hands_c = row.get("val_hands_c_f1", 0)
        talk_c = row.get("val_talk_c_f1", 0)
        lr = row.get("lr", 0)
        lines.append(
            f"- **{label}** (ep{ep}, lr={lr:.2e}): "
            f"train_loss={train_l:.3f}, val_loss={val_l:.3f}, score={score:.3f} "
            f"| action_c_f1={act_c:.3f} gaze={gaze_c:.3f} hands={hands_c:.3f} talk={talk_c:.3f}"
        )
    return "\n".join(lines)


# ---------- data pipeline stats ----------
def _data_pipeline_block(save_root: Path, summary: dict) -> str:
    """log 파싱 (옵션) + summary.json 에서 데이터 통계."""
    lines = []
    lines.append(f"- **클립 수**: train {summary.get('n_train_clips','?')} / val {summary.get('n_val_clips','?')} / test {summary.get('n_test_clips','?')}")
    lines.append(f"- **윈도우 수**: train {summary.get('n_train_windows','?')} / val {summary.get('n_val_windows','?')} / test {summary.get('n_test_windows','?')}")

    # clip_manifest.json 에서 source 별 breakdown (샘플링)
    cm_path = save_root / "clip_manifest.json"
    if cm_path.exists():
        try:
            clips = load_json(cm_path)
            sources = Counter(c.get("source") for c in clips)
            lines.append(f"- **데이터 소스 분포** (total clips {len(clips)}): distraction={sources.get('distraction',0)}, gaze={sources.get('gaze',0)}")
        except Exception:
            pass

    # split_info.json 에서 subject 수
    split_path = save_root / "split_info.json"
    if split_path.exists():
        try:
            s = load_json(split_path)
            subj = s.get("subjects", {})
            lines.append(
                f"- **Subject split**: "
                f"train {len(subj.get('train',[]))} / val {len(subj.get('val',[]))} / test {len(subj.get('test',[]))} "
                f"(subject-disjoint)"
            )
        except Exception:
            pass
    return "\n".join(lines)


def _class_distribution_block(save_root: Path) -> str:
    """clip_manifest.json 에서 head 별 라벨 분포 집계."""
    cm_path = save_root / "clip_manifest.json"
    if not cm_path.exists():
        return ""
    try:
        clips = load_json(cm_path)
    except Exception:
        return ""

    action_cnt = Counter()
    gaze_cnt = Counter()
    hands_cnt = Counter()
    talk_cnt = Counter()
    for c in clips:
        lbl = c.get("labels", {})
        if lbl.get("action") is not None: action_cnt[lbl["action"]] += 1
        if lbl.get("gaze_fine") is not None: gaze_cnt[lbl["gaze_fine"]] += 1
        if lbl.get("hands") is not None: hands_cnt[lbl["hands"]] += 1
        if lbl.get("talk") is not None: talk_cnt[lbl["talk"]] += 1

    lines = ["### 전체 clip 기준 라벨 분포\n"]
    lines.append("| head | distribution (id: count) |")
    lines.append("|---|---|")
    lines.append(f"| action (11) | {', '.join(f'{i}:{action_cnt.get(i,0)}' for i in range(11))} |")
    lines.append(f"| gaze fine (10) | {', '.join(f'{i}:{gaze_cnt.get(i,0)}' for i in range(10))} |")
    lines.append(f"| hands (4) | {', '.join(f'{i}:{hands_cnt.get(i,0)}' for i in range(4))} |")
    lines.append(f"| talk (2) | {', '.join(f'{i}:{talk_cnt.get(i,0)}' for i in range(2))} |")
    return "\n".join(lines)


# ---------- hw/env block ----------
def _env_block() -> str:
    lines = []
    try:
        import torch
        lines.append(f"- **torch**: {torch.__version__} (cuda available: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            lines.append(f"- **GPU**: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    lines.append(f"- **Python**: {sys.version.split()[0]} on {platform.system()} {platform.release()} ({platform.machine()})")
    try:
        nv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=3)
        if nv.returncode == 0 and nv.stdout.strip():
            lines.append(f"- **NVIDIA driver**: {nv.stdout.strip().splitlines()[0]}")
    except Exception:
        pass
    return "\n".join(lines)


# ---------- main report generator ----------
def derive_results_name(save_root: Path) -> str:
    name = save_root.name
    if name.startswith("baseline_"):
        name = name[len("baseline_"):]
    return name


def generate_report_text(save_root: Path, title: str) -> str:
    summary = load_json(save_root / "summary.json")
    cfg = load_json(save_root / "config.json")

    lines: list[str] = [f"# {title}\n"]

    # ========== 1. TL;DR ==========
    lines.append("## 1. TL;DR\n")
    lines.append(f"- **best epoch**: {summary.get('best_epoch')}")
    lines.append(f"- **best weighted score**: {summary.get('best_score', 0):.4f}")
    per = summary.get("test_per_head", {})
    test_scores = "  ".join(
        f"{h}={per.get(h,{}).get('clip_f1_macro',0):.3f}"
        for h in ("action", "gaze", "hands", "talk")
    )
    lines.append(f"- **test clip f1 (per head)**: {test_scores}")
    lines.append(f"- **모델 params**: {summary.get('model_params',0)/1e6:.3f} M")
    lines.append(f"- **latency**: {summary.get('inference_ms_per_window',0):.2f} ms/window")
    lines.append(f"- **face_mode**: `{summary.get('face_mode')}`, temporal: `{cfg.get('model',{}).get('temporal',{}).get('kind')}`")
    lines.append("")

    # ========== 2. 실험 설정 ==========
    lines.append("## 2. 실험 설정\n")
    fc = cfg.get("face", {}); pc = cfg.get("pose", {}); mc = cfg.get("model", {})
    tc_cfg = cfg.get("train", {}); lc = cfg.get("loss", {})
    lines.append(f"- **face.mode**: `{fc.get('mode')}`  (num_regions={fc.get('num_regions')}, use_z={fc.get('use_z')})")
    lines.append(f"- **pose**: COCO17 xy/bone/vel/conf (conf_thres={pc.get('joint_conf_thres')})")
    lines.append(f"- **model**: pose_mid={mc.get('pose_mid_channels')}, face_mid={mc.get('face_mid_channels')}, "
                 f"fused={mc.get('fused_channels')}, temporal={mc.get('temporal', {}).get('kind')}")
    lines.append(f"- **train**: batch={tc_cfg.get('batch_size')} lr={tc_cfg.get('lr')} wd={tc_cfg.get('weight_decay')} patience={tc_cfg.get('patience')}")
    lines.append(f"- **loss α**: action={lc.get('alpha_action')} gaze={lc.get('alpha_gaze')} "
                 f"hands={lc.get('alpha_hands')} talk={lc.get('alpha_talk')}, gaze_weak_weight={lc.get('gaze_weak_weight')}")
    sw = cfg.get("best_score_weights", {})
    lines.append(f"- **best score weights**: {sw}")
    lines.append("")

    # ========== 3. 데이터 파이프라인 ==========
    lines.append("## 3. 데이터 파이프라인\n")
    lines.append(_data_pipeline_block(save_root, summary))
    lines.append("")
    dist = _class_distribution_block(save_root)
    if dist:
        lines.append(dist)
        lines.append("")

    # ========== 4. 학습 진행 요약 ==========
    lines.append("## 4. 학습 진행 요약\n")
    tc = _training_curve_summary(save_root)
    if tc is not None:
        lines.append(_format_training_curve_block(tc))
    else:
        lines.append("_metrics.csv 없음_")
    lines.append("")

    # ========== 5. Test per-head summary + per-class ==========
    lines.append("## 5. Test per-head metrics\n")
    lines.append("| head | window_f1 | window_acc | clip_f1 | clip_acc |")
    lines.append("|---|---|---|---|---|")
    for h in ("action", "gaze", "hands", "talk"):
        ph = summary["test_per_head"].get(h, {})
        lines.append(
            f"| {h} | {ph.get('window_f1_macro',0):.4f} | {ph.get('window_acc',0):.4f} "
            f"| {ph.get('clip_f1_macro',0):.4f} | {ph.get('clip_acc',0):.4f} |"
        )
    lines.append("")

    # v1.2 신설: gaze head → front binary 로 distraction 전이 평가
    gbd = summary.get("test_gaze_binary_on_distraction")
    if gbd:
        lines.append("### 5.⁂ Gaze head 의 distraction 전이 평가 (front vs not-front)\n")
        lines.append("> gaze head 는 gaze s6 로만 학습됐지만 'front' zone 예측을 binary 로 축소해 "
                     "distraction 의 `looking_road` / `not_looking_road` 정답과 비교한 전이 성능.\n")
        lines.append("| level | acc | f1_macro | support |")
        lines.append("|---|---|---|---|")
        lines.append(f"| window | {gbd.get('window_acc',0):.4f} | {gbd.get('window_f1_macro',0):.4f} | {gbd.get('n_windows',0)} |")
        lines.append(f"| clip   | {gbd.get('clip_acc',0):.4f} | {gbd.get('clip_f1_macro',0):.4f} | (front={gbd.get('support_front',0)}, not_front={gbd.get('support_not_front',0)}) |")
        lines.append("")

    # ========== 6. Per-class tables + 약점 ==========
    for head in ("action", "gaze", "hands", "talk"):
        cm_path = save_root / f"test_{head}_clip_confusion.csv"
        cm = _load_cm(cm_path)
        if cm is None:
            continue
        labels = HEAD_LABEL_NAMES[head]
        df = per_class_metrics(cm, labels)
        macro = float(df["f1"].mean())
        lines.append(f"### 6.{'abcd'['action gaze hands talk'.split().index(head)]} Test `{head}` (macro f1 = {macro:.4f})\n")
        lines.append(_md_table(df))
        lines.append("")

        weak = df.sort_values("f1").head(3)
        lines.append(f"⚠️ `{head}` 약점 top-3:")
        for _, r in weak.iterrows():
            lines.append(f"- `{r['label']}` — f1={float(r['f1']):.3f}, recall={float(r['recall (acc)']):.3f}, "
                         f"precision={float(r['precision']):.3f}, support={int(r['support'])}")
        lines.append("")

    # ========== 7. 주요 혼동 쌍 ==========
    lines.append("## 7. 주요 혼동 쌍 (off-diagonal top-5)\n")
    for head, top_k in (("action", 5), ("gaze", 5), ("hands", 3)):
        cm_path = save_root / f"test_{head}_clip_confusion.csv"
        cm = _load_cm(cm_path)
        if cm is None:
            continue
        labels = HEAD_LABEL_NAMES[head]
        pairs = _notable_confusions(cm, labels, top_k=top_k)
        if not pairs:
            continue
        lines.append(f"### {head}\n")
        lines.append("| true → pred | count | rate of support |")
        lines.append("|---|---|---|")
        for true_lbl, pred_lbl, cnt, rate in pairs:
            lines.append(f"| `{true_lbl}` → `{pred_lbl}` | {cnt} | {rate:.2%} |")
        lines.append("")

    # ========== 8. 시스템·성능 ==========
    lines.append("## 8. 시스템 / 성능\n")
    lines.append(f"- **model_params**: {summary.get('model_params',0)/1e6:.3f} M")
    lines.append(f"- **inference_ms_per_window**: {summary.get('inference_ms_per_window',0):.2f} ms")
    lines.append(f"- **n_train_windows**: {summary.get('n_train_windows','?')}")
    lines.append(f"- **face_mode**: {summary.get('face_mode')}")
    lines.append(_env_block())
    lines.append("")

    # ========== 9. 산출물 경로 ==========
    lines.append("## 9. 산출물 경로\n")
    lines.append(f"- 원본 아티팩트: `{save_root}/`")
    lines.append(f"- 체크포인트: `{save_root}/best.pt`, `{save_root}/last.pt`")
    lines.append(f"- 로그: `{save_root}/train.log`")
    lines.append(f"- 학습 curve: `{save_root}/metrics.csv`, `{save_root}/history.json`")
    lines.append(f"- Epoch 별 val confusion: `{save_root}/val_*_clip_confusion_epoch_*.csv`")
    lines.append(f"- Test confusion: `{save_root}/test_*_clip_confusion.csv`")
    lines.append("")
    return "\n".join(lines)


def finalize_results(
    save_root: Path | str,
    results_root: Path | str,
    results_name: str | None = None,
    title: str | None = None,
    logger=None,
) -> Path:
    save_root = Path(save_root)
    results_root = Path(results_root)
    if results_name is None:
        results_name = derive_results_name(save_root)
    if title is None:
        title = f"Experiment: {results_name}"

    out_dir = results_root / results_name
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in _CURATED_FILES:
        src = save_root / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)

    summary_path = save_root / "summary.json"
    if summary_path.exists():
        try:
            s = load_json(summary_path)
            be = int(s.get("best_epoch") or 0)
            if be > 0:
                for head in ("action", "gaze", "hands", "talk"):
                    src = save_root / f"val_{head}_clip_confusion_epoch_{be:03d}.csv"
                    if src.exists():
                        shutil.copy2(src, out_dir / src.name)
                        copied.append(src.name)
        except Exception:
            pass

    try:
        md = generate_report_text(save_root, title)
        (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
        copied.append("REPORT.md")
    except Exception as e:
        if logger is not None:
            logger.warning(f"[finalize] REPORT.md 생성 실패: {e}")

    if logger is not None:
        logger.info(f"[finalize] results → {out_dir}  ({len(copied)} files)")
    return out_dir

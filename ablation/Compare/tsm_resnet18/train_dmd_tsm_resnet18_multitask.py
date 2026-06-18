#!/usr/bin/env python3
"""TSM-ResNet18 DMD multitask comparison baseline.

This baseline follows the DMD driver-monitoring comparison direction:

- video input rather than landmark restoration
- DMD-style multi-view streams: body IR, face IR, hands IR
- current project label heads: action / gaze / hands / talk
- final clean-vs-masked split JSON support
- TSM-style temporal shift on top of a 2D ResNet-18 backbone

For masked samples, the face stream is read from the fixed split JSON's
`masked_video_path`; body and hands streams remain the original DMD IR videos.
Baselines intentionally ignore `occ_label_vector`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from tsm_cache_utils import maybe_cache_dataset

try:
    import torchvision
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for torchvision.models.resnet18") from exc


IGNORE_LABEL = -100

ACTION_CLASSES = [
    "safe_drive",
    "texting_right",
    "texting_left",
    "phonecall_right",
    "phonecall_left",
    "radio",
    "drinking",
    "reach_side",
    "reach_backseat",
    "hair_and_makeup",
    "talking_to_passenger",
]
GAZE_CLASSES = [
    "left_mirror",
    "left",
    "front",
    "center_mirror",
    "front_right",
    "right_mirror",
    "right",
    "infotainment",
    "steering_wheel",
]
HANDS_CLASSES = ["both", "only_left", "only_right", "none"]
TALK_CLASSES = ["not_talking", "talking"]
HEADS = ("action", "gaze", "hands", "talk")
NUM_CLASSES = {
    "action": len(ACTION_CLASSES),
    "gaze": len(GAZE_CLASSES),
    "hands": len(HANDS_CLASSES),
    "talk": len(TALK_CLASSES),
}
LABEL_KEY = {
    "action": "y_action",
    "gaze": "y_gaze",
    "hands": "y_hands",
    "talk": "y_talk",
}


def normalize_task_weights(raw: dict[str, Any] | None, default: float = 1.0) -> dict[str, float]:
    raw = raw or {}
    out: dict[str, float] = {}
    for head in HEADS:
        if head in raw:
            out[head] = float(raw[head])
        elif f"alpha_{head}" in raw:
            out[head] = float(raw[f"alpha_{head}"])
        else:
            out[head] = float(default)
    return out


def save_drop_table(drop: dict[str, Any], path: str | Path) -> None:
    rows = []
    for head in HEADS:
        for metric, values in drop.get(head, {}).items():
            rows.append({
                "task": head,
                "metric": metric,
                "clean": values.get("clean"),
                "masked": values.get("masked"),
                "drop_abs": values.get("drop_abs"),
                "drop_rel": values.get("drop_rel"),
            })
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

PREFIX_RE = re.compile(r"^(?P<group>g[A-Z])_(?P<subject>\d+)_(?P<session>s\d+)_(?P<timestamp>.+)$")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_prefix(prefix: str) -> dict[str, str]:
    match = PREFIX_RE.match(prefix)
    if not match:
        raise ValueError(f"Cannot parse DMD prefix: {prefix}")
    return match.groupdict()


def find_video_file(video_dir: Path, prefix: str, suffix: str) -> Path | None:
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        path = video_dir / f"{prefix}_{suffix}{ext}"
        if path.exists():
            return path
    return None


def resolve_original_video_paths(record: dict[str, Any], dmd_root: str | Path) -> dict[str, str]:
    meta = parse_prefix(record["video_prefix"])
    video_dir = Path(dmd_root) / record["source"] / "dmd" / meta["group"] / meta["subject"] / meta["session"]
    prefix = record["video_prefix"]
    return {
        "body_ir": str(find_video_file(video_dir, prefix, "ir_body") or ""),
        "face_ir": str(find_video_file(video_dir, prefix, "ir_face") or ""),
        "hands_ir": str(find_video_file(video_dir, prefix, "ir_hands") or ""),
    }


def _valid_label(value: Any, n_classes: int) -> int:
    if value is None:
        return IGNORE_LABEL
    value = int(value)
    return value if 0 <= value < n_classes else IGNORE_LABEL


def labels_from_record(record: dict[str, Any]) -> dict[str, int]:
    labels = record.get("labels") or {}
    return {
        "y_action": _valid_label(labels.get("action"), NUM_CLASSES["action"]),
        "y_gaze": _valid_label(labels.get("gaze_fine"), NUM_CLASSES["gaze"]),
        "y_hands": _valid_label(labels.get("hands"), NUM_CLASSES["hands"]),
        "y_talk": _valid_label(labels.get("talk"), NUM_CLASSES["talk"]),
    }


def relative_face_key(path: str | Path) -> str:
    text = str(path)
    if text.endswith(".npz"):
        text = text[:-4]
    for token in ("distraction/dmd/", "gaze/dmd/"):
        idx = text.find(token)
        if idx >= 0:
            return text[idx:]
    return text


def clone_for_fixed_item(base: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out["clip_id"] = (
        f"{item.get('variant', 'unknown')}::{item.get('mask_region', 'unknown')}::"
        f"{item.get('mask_appearance', 'unknown')}::{base['clip_id']}"
    )
    out["face_npz"] = item["face_path"]
    out["face5pt_npz"] = item["face5pt_path"]
    out["fixed_sample_id"] = item.get("sample_id")
    out["fixed_sample_key"] = item.get("sample_key")
    out["variant"] = item.get("variant")
    out["mask_region"] = item.get("mask_region")
    out["mask_appearance"] = item.get("mask_appearance")
    out["masked_video_path"] = item.get("masked_video_path") or ""
    return out


def load_clip_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_lookup(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        lookup.setdefault(relative_face_key(rec["face_npz"]), []).append(rec)
    return lookup


def build_fixed_split_records(cfg: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    split_cfg = cfg["fixed_split"]
    fixed = json.loads(Path(split_cfg["json_path"]).read_text(encoding="utf-8"))
    protocol_name = split_cfg.get("protocol", "clean_masked_augmentation_baseline")
    protocol = fixed["protocols"][protocol_name]

    base_records = load_clip_records(cfg["paths"]["clip_manifest_path"])
    lookup = build_lookup(base_records)

    fallback_added = {}
    for fallback_path in split_cfg.get("fallback_clip_manifest_paths", []) or []:
        added_keys = 0
        added_records = 0
        fallback_lookup = build_lookup(load_clip_records(fallback_path))
        for key, records in fallback_lookup.items():
            if key in lookup:
                continue
            lookup[key] = records
            added_keys += 1
            added_records += len(records)
        fallback_added[str(fallback_path)] = {"added_keys": added_keys, "added_records": added_records}

    def expand(item_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        out = []
        missing = []
        for item in fixed["items"][item_key]:
            matches = lookup.get(item["sample_key"], [])
            if not matches:
                missing.append(item["sample_key"])
                continue
            out.extend(clone_for_fixed_item(record, item) for record in matches)
        return out, {
            "item_key": item_key,
            "n_items": len(fixed["items"][item_key]),
            "n_records": len(out),
            "n_missing_items": len(missing),
            "missing_items": missing[:20],
        }

    train, train_stats = expand(protocol["train"])
    val, val_stats = expand(protocol["val"])
    test_clean, test_clean_stats = expand(protocol["test_clean"])
    test_masked, test_masked_stats = expand(protocol["test_masked"])

    split_info = {
        "mode": "fixed_split_json",
        "json_path": split_cfg["json_path"],
        "split_name": fixed.get("split_name"),
        "version": fixed.get("version"),
        "protocol": protocol_name,
        "protocol_definition": protocol,
        "fallback_added": fallback_added,
        "stats": {
            "train": train_stats,
            "val": val_stats,
            "test_clean": test_clean_stats,
            "test_masked": test_masked_stats,
        },
        "notes": fixed.get("notes", []),
    }
    return {"train": train, "val": val, "test_clean": test_clean, "test_masked": test_masked}, split_info


def maybe_limit(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if not limit or limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    keep = sorted(idx[:limit])
    return [records[i] for i in keep]


def segment_centers(start: int, end: int, num_segments: int, train: bool) -> list[int]:
    if end < start:
        return [max(start, 0)] * num_segments
    edges = np.linspace(start, end + 1, num_segments + 1)
    centers = []
    for i in range(num_segments):
        lo = int(math.floor(edges[i]))
        hi = max(lo, int(math.ceil(edges[i + 1])) - 1)
        centers.append(random.randint(lo, hi) if train and hi > lo else (lo + hi) // 2)
    return centers


def snippet_indices(start: int, end: int, num_segments: int, frames_per_segment: int, train: bool) -> list[list[int]]:
    centers = segment_centers(start, end, num_segments, train)
    half = frames_per_segment // 2
    snippets = []
    for center in centers:
        first = center - half
        idxs = [min(max(first + j, start), end) for j in range(frames_per_segment)]
        snippets.append(idxs)
    return snippets


def _zero_video(num_segments: int, channels: int, frames_per_segment: int, image_size: int) -> torch.Tensor:
    return torch.zeros((num_segments, channels, frames_per_segment, image_size, image_size), dtype=torch.float32)


def read_ir_snippets(path: str, snippets: list[list[int]], image_size: int) -> tuple[torch.Tensor, bool]:
    if not path:
        return _zero_video(len(snippets), 3, len(snippets[0]), image_size), False
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return _zero_video(len(snippets), 3, len(snippets[0]), image_size), False

    out = []
    ok_any = False
    for idxs in snippets:
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(idx), 0))
            ok, frame = cap.read()
            if not ok or frame is None:
                frames.append(np.zeros((image_size, image_size), dtype=np.float32))
                continue
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            ok_any = True
        arr = np.stack(frames, axis=0)
        arr = (arr - 0.5) / 0.5
        arr = np.repeat(arr[:, None, :, :], 3, axis=1)
        out.append(arr)
    cap.release()
    # [S, T, C, H, W] -> [S, C, T, H, W]
    return torch.from_numpy(np.stack(out, axis=0)).permute(0, 2, 1, 3, 4).contiguous(), ok_any


@dataclass
class VideoSample:
    record: dict[str, Any]
    paths: dict[str, str]


class DMDFixedVideoDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        dmd_root: str | Path,
        num_segments: int,
        frames_per_segment: int,
        image_size: int,
        train: bool,
    ):
        self.num_segments = int(num_segments)
        self.frames_per_segment = int(frames_per_segment)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.samples = []
        for rec in records:
            paths = resolve_original_video_paths(rec, dmd_root)
            if rec.get("variant") == "masked" and rec.get("masked_video_path"):
                paths["face_ir"] = rec["masked_video_path"]
            self.samples.append(VideoSample(record=rec, paths=paths))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        rec = sample.record
        body_start = int(rec.get("body_start", rec.get("mosaic_start", 0)))
        body_end = int(rec.get("body_end", rec.get("mosaic_end", body_start)))
        face_start = int(rec.get("face_start", body_start))
        face_end = int(rec.get("face_end", body_end))

        body_snips = snippet_indices(body_start, body_end, self.num_segments, self.frames_per_segment, self.train)
        face_snips = snippet_indices(face_start, face_end, self.num_segments, self.frames_per_segment, self.train)

        x_body, has_body = read_ir_snippets(sample.paths.get("body_ir", ""), body_snips, self.image_size)
        x_face, has_face = read_ir_snippets(sample.paths.get("face_ir", ""), face_snips, self.image_size)
        x_hands, has_hands = read_ir_snippets(sample.paths.get("hands_ir", ""), body_snips, self.image_size)

        labels = labels_from_record(rec)
        out: dict[str, Any] = {
            "x_body": x_body,
            "x_face": x_face,
            "x_hands": x_hands,
            "mask_body": torch.tensor(1.0 if has_body else 0.0, dtype=torch.float32),
            "mask_face": torch.tensor(1.0 if has_face else 0.0, dtype=torch.float32),
            "mask_hands": torch.tensor(1.0 if has_hands else 0.0, dtype=torch.float32),
            "clip_id": rec["clip_id"],
            "subject_key": rec["subject_key"],
            "source": rec["source"],
            "variant": rec.get("variant", "unknown"),
            "mask_region": rec.get("mask_region", "unknown"),
        }
        for key, value in labels.items():
            out[key] = torch.tensor(value, dtype=torch.long)
        return out



class TemporalShift(nn.Module):
    """Temporal Shift Module.

    Input is flattened as [B * T, C, H, W]. The module reshapes it to
    [B, T, C, H, W], shifts a small fraction of channels along the temporal
    dimension, then flattens back. This gives a 2D CNN lightweight temporal
    modeling without 3D convolutions.
    """

    def __init__(self, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.n_segment = int(n_segment)
        self.fold_div = int(fold_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        if self.n_segment <= 1 or nt % self.n_segment != 0:
            return x

        n_batch = nt // self.n_segment
        fold = c // self.fold_div
        if fold <= 0:
            return x

        x = x.view(n_batch, self.n_segment, c, h, w)
        out = torch.zeros_like(x)

        # shift 1/fold_div channels backward and another 1/fold_div forward
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(nt, c, h, w)


def _insert_tsm_into_resnet(model: nn.Module, n_segment: int, fold_div: int = 8, layers: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")) -> None:
    """Wrap residual blocks with TemporalShift.

    This keeps the ResNet block implementation intact and prepends a shift
    operation before each residual block.
    """
    for layer_name in layers:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        for block_name, block in list(layer._modules.items()):
            layer._modules[block_name] = nn.Sequential(
                TemporalShift(n_segment=n_segment, fold_div=fold_div),
                block,
            )


class TSMResNet18Backbone(nn.Module):
    def __init__(
        self,
        n_segment: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        fold_div: int = 8,
        shift_layers: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
    ):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            model = torchvision.models.resnet18(weights=weights)
        except TypeError:
            model = torchvision.models.resnet18(pretrained=pretrained)

        _insert_tsm_into_resnet(model, n_segment=n_segment, fold_div=fold_div, layers=shift_layers)

        self.out_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        self.model = model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> features: [B, T, D]
        bsz, steps, c, h, w = x.shape
        feat = self.model(x.reshape(bsz * steps, c, h, w))
        feat = self.dropout(feat)
        return feat.view(bsz, steps, -1)


class TSMViewEncoder(nn.Module):
    def __init__(self, feat_dim: int, view_dim: int, dropout: float):
        super().__init__()
        self.out_dim = int(view_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, view_dim),
            nn.LayerNorm(view_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # TSM already mixes temporal information inside the backbone.
        # Use simple segment consensus / temporal average for the view feature.
        return self.proj(seq.mean(dim=1))


class MultitaskHeads(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({head: nn.Linear(in_dim, NUM_CLASSES[head]) for head in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.dropout(x)
        return {head: layer(z) for head, layer in self.heads.items()}


class DMDTSMResNet18Multitask(nn.Module):
    def __init__(
        self,
        total_frames: int,
        pretrained: bool = False,
        share_backbone: bool = True,
        feat_dropout: float = 0.0,
        view_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        tsm_fold_div: int = 8,
    ):
        super().__init__()
        self.share_backbone = bool(share_backbone)
        self.total_frames = int(total_frames)

        self.body_backbone = TSMResNet18Backbone(
            n_segment=self.total_frames,
            pretrained=pretrained,
            dropout=feat_dropout,
            fold_div=tsm_fold_div,
        )
        if self.share_backbone:
            self.face_backbone = self.body_backbone
            self.hands_backbone = self.body_backbone
        else:
            self.face_backbone = TSMResNet18Backbone(
                n_segment=self.total_frames,
                pretrained=pretrained,
                dropout=feat_dropout,
                fold_div=tsm_fold_div,
            )
            self.hands_backbone = TSMResNet18Backbone(
                n_segment=self.total_frames,
                pretrained=pretrained,
                dropout=feat_dropout,
                fold_div=tsm_fold_div,
            )

        feat_dim = self.body_backbone.out_dim
        self.body_encoder = TSMViewEncoder(feat_dim, view_dim, dropout)
        self.face_encoder = TSMViewEncoder(feat_dim, view_dim, dropout)
        self.hands_encoder = TSMViewEncoder(feat_dim, view_dim, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(view_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.heads = MultitaskHeads(fusion_dim, dropout)

    @staticmethod
    def _merge_snippets(x: torch.Tensor) -> torch.Tensor:
        # Dataset returns [B, S, C, T, H, W].
        # TSM expects one temporal axis: [B, S*T, C, H, W].
        bsz, segs, c, t, h, w = x.shape
        return x.permute(0, 1, 3, 2, 4, 5).reshape(bsz, segs * t, c, h, w)

    def _encode_view(self, x: torch.Tensor, backbone: nn.Module, encoder: nn.Module) -> torch.Tensor:
        frames = self._merge_snippets(x)
        feat_seq = backbone(frames)
        return encoder(feat_seq)

    def forward(self, x_body: torch.Tensor, x_face: torch.Tensor, x_hands: torch.Tensor) -> dict[str, torch.Tensor]:
        body = self._encode_view(x_body, self.body_backbone, self.body_encoder)
        face = self._encode_view(x_face, self.face_backbone, self.face_encoder)
        hands = self._encode_view(x_hands, self.hands_backbone, self.hands_encoder)
        fused = self.fusion(torch.cat([body, face, hands], dim=1))
        return self.heads(fused)


class MultitaskCriterion(nn.Module):
    def __init__(self, loss_weights: dict[str, float], class_weights: dict[str, torch.Tensor | None]):
        super().__init__()
        self.loss_weights = loss_weights
        self.criteria = nn.ModuleDict({
            head: nn.CrossEntropyLoss(weight=class_weights.get(head), ignore_index=IGNORE_LABEL)
            for head in HEADS
        })

    def forward(self, logits: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        total = logits["action"].new_zeros(())
        parts = {}
        for head in HEADS:
            target = batch[LABEL_KEY[head]]
            if (target != IGNORE_LABEL).sum() == 0:
                loss = logits[head].new_zeros(())
            else:
                loss = self.criteria[head](logits[head], target)
            parts[head] = float(loss.detach().cpu())
            total = total + float(self.loss_weights.get(head, 1.0)) * loss
        return total, parts


def class_weights(records: list[dict[str, Any]], head: str) -> torch.Tensor:
    counts = np.zeros((NUM_CLASSES[head],), dtype=np.float32)
    key = LABEL_KEY[head]
    for rec in records:
        y = labels_from_record(rec)[key]
        if y != IGNORE_LABEL:
            counts[y] += 1.0
    if counts.sum() <= 0:
        return torch.ones((NUM_CLASSES[head],), dtype=torch.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.sum() * NUM_CLASSES[head]
    return torch.tensor(inv, dtype=torch.float32)


def masked_metrics(targets: list[int], preds: list[int], probs: list[list[float]] | None = None) -> dict[str, float | int]:
    t = np.asarray(targets)
    p = np.asarray(preds)
    mask = t != IGNORE_LABEL
    if int(mask.sum()) == 0:
        return {"n": 0, "acc": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    t = t[mask]
    p = p[mask]
    return {
        "n": int(mask.sum()),
        "acc": float(accuracy_score(t, p)),
        "precision_macro": float(precision_score(t, p, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(t, p, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(t, p, average="macro", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultitaskCriterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    total_epochs: int,
    amp: bool,
    grad_clip: float,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_n = 0
    loss_state = {head: 0.0 for head in HEADS}
    state = {head: {"targets": [], "preds": []} for head in HEADS}
    scaler = torch.cuda.amp.GradScaler(enabled=train and amp and device.type == "cuda")

    pbar = tqdm(loader, desc=f"[{epoch}/{total_epochs}] {'train' if train else 'eval'}", ncols=120, ascii=True)
    for batch in pbar:
        x_body = batch["x_body"].to(device, non_blocking=True)
        x_face = batch["x_face"].to(device, non_blocking=True)
        x_hands = batch["x_hands"].to(device, non_blocking=True)
        for key in ("y_action", "y_gaze", "y_hands", "y_talk"):
            batch[key] = batch[key].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(x_body, x_face, x_hands)
                loss, parts = criterion(logits, batch)
            if train:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        bsz = int(x_body.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        total_n += bsz
        for head in HEADS:
            loss_state[head] += parts[head] * bsz
            state[head]["preds"].extend(logits[head].argmax(dim=1).detach().cpu().numpy().tolist())
            state[head]["targets"].extend(batch[LABEL_KEY[head]].detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{total_loss / max(total_n, 1):.4f}")

    return {
        "loss": total_loss / max(total_n, 1),
        "loss_parts": {head: loss_state[head] / max(total_n, 1) for head in HEADS},
        "heads": {head: masked_metrics(state[head]["targets"], state[head]["preds"]) for head in HEADS},
    }


def weighted_score(result: dict[str, Any], weights: dict[str, float]) -> float:
    return float(sum(float(weights.get(head, 0.0)) * result["heads"][head]["f1_macro"] for head in HEADS))


def save_predictions(model: nn.Module, loader: DataLoader, device: torch.device, path: Path, amp: bool) -> None:
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"save {path.name}", ncols=120, ascii=True, leave=False):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(
                    batch["x_body"].to(device),
                    batch["x_face"].to(device),
                    batch["x_hands"].to(device),
                )
            probs = {head: torch.softmax(logits[head], dim=1).cpu().numpy() for head in HEADS}
            preds = {head: probs[head].argmax(axis=1) for head in HEADS}
            for i in range(len(batch["clip_id"])):
                row = {
                    "clip_id": batch["clip_id"][i],
                    "subject_key": batch["subject_key"][i],
                    "source": batch["source"][i],
                    "variant": batch["variant"][i],
                    "mask_region": batch["mask_region"][i],
                }
                for head in HEADS:
                    row[f"{head}_target"] = int(batch[LABEL_KEY[head]][i])
                    row[f"{head}_pred"] = int(preds[head][i])
                    for cls_idx, prob in enumerate(probs[head][i].tolist()):
                        row[f"{head}_p{cls_idx}"] = float(prob)
                rows.append(row)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_loader(dataset: Dataset, cfg: dict[str, Any], train: bool, device: torch.device) -> DataLoader:
    num_workers = int(cfg["train"]["num_workers"] if train else min(int(cfg["train"]["num_workers"]), 2))
    kwargs: dict[str, Any] = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "shuffle": train,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 1))
        kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", False))
    return DataLoader(dataset, **kwargs)


def compute_drop(clean: dict[str, Any], masked: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for head in HEADS:
        out[head] = {}
        for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
            clean_value = float(clean["heads"][head].get(metric, 0.0))
            masked_value = float(masked["heads"][head].get(metric, 0.0))
            out[head][metric] = {
                "clean": clean_value,
                "masked": masked_value,
                "drop_abs": clean_value - masked_value,
                "drop_rel": (clean_value - masked_value) / clean_value if abs(clean_value) > 1e-12 else None,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tsm_resnet18_seed42_gaze045_light.yaml"))
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    cv2.setNumThreads(0)
    torch.set_num_threads(int(cfg["train"].get("torch_num_threads", 2)))

    save_root = Path(cfg["paths"]["save_root"]).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    save_json(cfg, save_root / "config.json")

    device_arg = str(cfg.get("device", "cuda"))
    device = torch.device(device_arg if device_arg == "cpu" or torch.cuda.is_available() else "cpu")
    records_by_split, split_info = build_fixed_split_records(cfg)
    save_json(split_info, save_root / "split_info.json")
    save_json(split_info, save_root / "fixed_manifest_split_info.json")

    debug = cfg.get("debug", {}) or {}
    records_by_split["train"] = maybe_limit(records_by_split["train"], args.max_train or int(debug.get("max_train", 0)), int(cfg["seed"]))
    records_by_split["val"] = maybe_limit(records_by_split["val"], args.max_val or int(debug.get("max_val", 0)), int(cfg["seed"]) + 1)
    records_by_split["test_clean"] = maybe_limit(records_by_split["test_clean"], args.max_test or int(debug.get("max_test", 0)), int(cfg["seed"]) + 2)
    records_by_split["test_masked"] = maybe_limit(records_by_split["test_masked"], args.max_test or int(debug.get("max_test", 0)), int(cfg["seed"]) + 3)

    data_cfg = cfg["data"]
    ds_kwargs = {
        "dmd_root": cfg["paths"]["dmd_root"],
        "num_segments": int(data_cfg["num_segments"]),
        "frames_per_segment": int(data_cfg["frames_per_segment"]),
        "image_size": int(data_cfg["image_size"]),
    }
    train_ds = DMDFixedVideoDataset(records_by_split["train"], train=True, **ds_kwargs)
    val_ds = DMDFixedVideoDataset(records_by_split["val"], train=False, **ds_kwargs)
    test_clean_ds = DMDFixedVideoDataset(records_by_split["test_clean"], train=False, **ds_kwargs)
    test_masked_ds = DMDFixedVideoDataset(records_by_split["test_masked"], train=False, **ds_kwargs)

    # ---------- optional sample cache ----------
    # Video decoding is the bottleneck for TSM. If cache.enabled=true,
    # each dataset item is decoded once and saved as .pt, then reused.
    train_ds = maybe_cache_dataset(train_ds, "train", cfg)
    val_ds = maybe_cache_dataset(val_ds, "val", cfg)
    test_clean_ds = maybe_cache_dataset(test_clean_ds, "test_clean", cfg)
    test_masked_ds = maybe_cache_dataset(test_masked_ds, "test_masked", cfg)



    train_loader = make_loader(train_ds, cfg, train=True, device=device)
    val_loader = make_loader(val_ds, cfg, train=False, device=device)
    test_clean_loader = make_loader(test_clean_ds, cfg, train=False, device=device)
    test_masked_loader = make_loader(test_masked_ds, cfg, train=False, device=device)

    model_cfg = cfg["model"]
    total_frames = int(data_cfg["num_segments"]) * int(data_cfg["frames_per_segment"])
    model = DMDTSMResNet18Multitask(
        total_frames=total_frames,
        pretrained=bool(model_cfg.get("pretrained", False)),
        share_backbone=bool(model_cfg.get("share_backbone", True)),
        feat_dropout=float(model_cfg.get("feat_dropout", 0.0)),
        view_dim=int(model_cfg.get("view_dim", 256)),
        fusion_dim=int(model_cfg.get("fusion_dim", 512)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        tsm_fold_div=int(model_cfg.get("tsm_fold_div", 8)),
    ).to(device)

    use_cw = cfg["train"].get("use_class_weight", {})
    cw = {
        head: class_weights(records_by_split["train"], head).to(device) if bool(use_cw.get(head, True)) else None
        for head in HEADS
    }
    loss_weights = normalize_task_weights(cfg.get("loss"), default=1.0)
    best_score_weights = normalize_task_weights(cfg.get("best_score_weights"), default=0.0)
    criterion = MultitaskCriterion(loss_weights, cw).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    amp = bool(cfg["train"].get("amp", True))
    grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))

    run_meta = {
        "device": str(device),
        "save_root": str(save_root),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test_clean": len(test_clean_ds),
        "n_test_masked": len(test_masked_ds),
        "model_params": sum(p.numel() for p in model.parameters()),
        "model_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "split_stats": split_info["stats"],
    }
    print(json.dumps(run_meta, ensure_ascii=False, indent=2), flush=True)
    save_json(run_meta, save_root / "run_meta.json")

    history = []
    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_out = run_epoch(model, train_loader, criterion, device, optimizer, epoch, epochs, amp, grad_clip)
        val_out = run_epoch(model, val_loader, criterion, device, None, epoch, epochs, amp, grad_clip)
        score = weighted_score(val_out, best_score_weights)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "val_score": score,
            "seconds": time.time() - t0,
        }
        for head in HEADS:
            for key, value in val_out["heads"][head].items():
                row[f"val_{head}_{key}"] = value
        history.append(row)
        with (save_root / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        save_json(history, save_root / "history.json")
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg},
            save_root / "last.pt",
        )
        print(f"[{epoch}/{epochs}] train_loss={train_out['loss']:.4f} val_loss={val_out['loss']:.4f} score={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "config": cfg},
                save_root / "best.pt",
            )
            print(f"[best] epoch={epoch} score={score:.4f}", flush=True)
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[early stop] no improvement for {patience} epochs", flush=True)
            break

    best_path = save_root / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    test_clean = run_epoch(model, test_clean_loader, criterion, device, None, best_epoch, epochs, amp, grad_clip)
    test_masked = run_epoch(model, test_masked_loader, criterion, device, None, best_epoch, epochs, amp, grad_clip)
    summary = {
        "baseline_name": "dmd_tsm_resnet18_ir_multiview_multitask",
        "best_epoch": best_epoch,
        "best_score": best_score,
        "test_clean": test_clean,
        "test_masked": test_masked,
        "masked_drop": compute_drop(test_clean, test_masked),
        "label_heads": {
            "action": ACTION_CLASSES,
            "gaze": GAZE_CLASSES,
            "hands": HANDS_CLASSES,
            "talk": TALK_CLASSES,
        },
        **run_meta,
    }
    save_json(summary, save_root / "summary.json")
    save_json(summary["masked_drop"], save_root / "test_clean_vs_masked_drop.json")
    save_drop_table(summary["masked_drop"], save_root / "test_clean_vs_masked_drop.csv")
    if bool(cfg["eval"].get("save_predictions", True)):
        save_predictions(model, test_clean_loader, device, save_root / "test_clean_predictions.csv", amp)
        save_predictions(model, test_masked_loader, device, save_root / "test_masked_predictions.csv", amp)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

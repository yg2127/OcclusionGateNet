"""Multi-task masked loss epoch loop + per-head metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from src.training.aggregation import aggregate_clip_level


from constants.gaze_zones import FRONT_ZONE_ID as _FRONT_ZONE_ID

IGNORE_LABEL = -100
HEAD_NAMES = ("gaze",)


def make_class_weights(labels: list[int], num_classes: int, ignore_label: int = IGNORE_LABEL) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.float32)
    for y in labels:
        if int(y) == ignore_label:
            continue
        counts[int(y)] += 1.0
    if counts.sum() == 0:
        return torch.ones((num_classes,), dtype=torch.float32)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


class MultitaskCriterion(nn.Module):
    """head 별 CE loss (ignore_index 포함) + 가중합.

    - gaze_fine · gaze_weak 는 모두 gaze head 로 감. 두 loss 를 가중합해 gaze head 를 지도.
    """
    def __init__(
        self,
        alpha_action: float,
        alpha_gaze: float,
        alpha_hands: float,
        alpha_talk: float,
        gaze_weak_weight: float,
        action_class_weights: torch.Tensor | None = None,
        gaze_class_weights: torch.Tensor | None = None,
        hands_class_weights: torch.Tensor | None = None,
        talk_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha_action = alpha_action
        self.alpha_gaze = alpha_gaze
        self.alpha_hands = alpha_hands
        self.alpha_talk = alpha_talk
        self.gaze_weak_weight = gaze_weak_weight

        self.ce_action = nn.CrossEntropyLoss(weight=action_class_weights, ignore_index=IGNORE_LABEL)
        self.ce_gaze_fine = nn.CrossEntropyLoss(weight=gaze_class_weights, ignore_index=IGNORE_LABEL)
        self.ce_hands = nn.CrossEntropyLoss(weight=hands_class_weights, ignore_index=IGNORE_LABEL)
        self.ce_talk = nn.CrossEntropyLoss(weight=talk_class_weights, ignore_index=IGNORE_LABEL)

    def _gaze_weak_loss(self, gaze_logits: torch.Tensor, y_weak: torch.Tensor) -> torch.Tensor:
        """gaze head 10-class logits → front vs not-front binary 로 변환해 BCE.

        mapping: front (zone 2) vs 나머지 9.
        """
        from constants.gaze_zones import FRONT_ZONE_ID
        mask = y_weak != IGNORE_LABEL
        if mask.sum() == 0:
            return gaze_logits.new_zeros(())
        # front 확률 = softmax[FRONT] over 10
        p = F.softmax(gaze_logits[mask], dim=1)
        p_front = p[:, FRONT_ZONE_ID]
        y = y_weak[mask].float()
        # binary CE (target: 1 = front, 0 = not-front)
        eps = 1e-6
        bce = -(y * torch.log(p_front + eps) + (1 - y) * torch.log(1 - p_front + eps))
        return bce.mean()

    def forward(self, logits: dict[str, torch.Tensor], batch: dict) -> dict:
        # Gaze-only: action / hands / talk losses are not computed and do not train.
        y_gaze = batch["y_gaze_fine"]
        y_weak = batch["y_gaze_weak"]

        def _safe_ce(ce, logit, tgt):
            if (tgt != IGNORE_LABEL).sum() == 0:
                return logit.new_zeros(())
            return ce(logit, tgt)

        l_gaze_f = _safe_ce(self.ce_gaze_fine, logits["gaze"], y_gaze)
        l_gaze_w = self._gaze_weak_loss(logits["gaze"], y_weak)
        total = self.alpha_gaze * (l_gaze_f + self.gaze_weak_weight * l_gaze_w)

        return {
            "total": total,
            "gaze_fine": l_gaze_f.detach(),
            "gaze_weak": l_gaze_w.detach(),
        }


def _masked_window_metric(targets, preds):
    """ignore label 제외 후 acc / macro f1."""
    t = np.asarray(targets); p = np.asarray(preds)
    mask = t != IGNORE_LABEL
    if mask.sum() == 0:
        return 0.0, 0.0
    t = t[mask]; p = p[mask]
    return (float(accuracy_score(t, p)),
            float(f1_score(t, p, average="macro", zero_division=0)))


def run_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: MultitaskCriterion,
    device: str,
    train: bool,
    grad_clip_norm: float | None,
    epoch_idx: int | None = None,
    total_epochs: int | None = None,
    agg_mode: str = "topk_mean",
    topk: int = 3,
    ablation_cfg: dict | None = None,
) -> dict:
    model.train() if train else model.eval()
    phase = "Train" if train else "Eval"

    if ablation_cfg is None:
        ablation_cfg = {}

    # per-head 누적
    running_loss = 0.0
    running_n = 0
    head_state = {h: {"preds": [], "targets": [], "probs": [], "clip_ids": []}
                  for h in HEAD_NAMES}
    # gaze head 의 "front vs not-front" binary 평가용 — distraction 샘플에서 y_gaze_weak 로 비교
    gaze_weak_state = {"targets": [], "preds_binary": [], "clip_ids": []}

    desc = f"[Epoch {epoch_idx}/{total_epochs}] {phase}" if epoch_idx is not None else phase
    pbar = tqdm(loader, desc=desc, leave=True, ncols=120, ascii=True, mininterval=0.3)

    for batch in pbar:
        xb = batch["x_body"].to(device, non_blocking=True)
        xf = batch["x_face"].to(device, non_blocking=True)
        xocc = batch.get("x_occ", None)
        if xocc is not None:
            xocc = xocc.to(device, non_blocking=True)

        if ablation_cfg.get("zero_pose", False):
            xb = torch.zeros_like(xb)

        if ablation_cfg.get("zero_face", False):
            xf = torch.zeros_like(xf)

        # targets to device
        for k in ("y_action", "y_gaze_fine", "y_gaze_weak", "y_hands", "y_talk"):
            batch[k] = batch[k].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(xb, xf, x_occ=xocc)
            losses = criterion(logits, batch)
            total_loss = losses["total"]
            if train:
                total_loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        bs = xb.size(0)
        running_loss += float(total_loss.item()) * bs
        running_n += bs

        # per-head preds / probs 수집
        with torch.no_grad():
            for head, logit_name, tgt_key in (
                ("gaze", "gaze", "y_gaze_fine"),
            ):
                prob = torch.softmax(logits[logit_name], dim=1)
                pred = prob.argmax(dim=1)
                head_state[head]["probs"].extend(prob.detach().cpu().numpy().tolist())
                head_state[head]["preds"].extend(pred.detach().cpu().numpy().tolist())
                head_state[head]["targets"].extend(batch[tgt_key].detach().cpu().numpy().tolist())
                head_state[head]["clip_ids"].extend(list(batch["clip_id"]))

            # gaze head 를 "front vs not-front" binary 로 축소한 예측. y_gaze_weak 타깃과 비교.
            prob_gaze = torch.softmax(logits["gaze"], dim=1)
            gaze_argmax = prob_gaze.argmax(dim=1)
            gaze_weak_state["targets"].extend(batch["y_gaze_weak"].detach().cpu().numpy().tolist())
            gaze_weak_state["preds_binary"].extend((gaze_argmax == _FRONT_ZONE_ID).to(torch.long).cpu().numpy().tolist())
            gaze_weak_state["clip_ids"].extend(list(batch["clip_id"]))

        pbar.set_postfix(loss=f"{running_loss / max(running_n,1):.4f}")

    avg_loss = running_loss / max(running_n, 1)
    result = {"loss": avg_loss, "heads": {}}

    for head in HEAD_NAMES:
        s = head_state[head]
        acc_w, f1_w = _masked_window_metric(s["targets"], s["preds"])
        head_result = {"window_acc": acc_w, "window_f1_macro": f1_w}
        if not train:
            # clip-level aggregation
            probs_np = [np.asarray(p, dtype=np.float32) for p in s["probs"]]
            clip_res = aggregate_clip_level(
                targets=s["targets"], probs=probs_np, clip_ids=s["clip_ids"],
                agg_mode=agg_mode, topk=topk, ignore_label=IGNORE_LABEL,
            )
            head_result["clip_acc"] = clip_res["acc"]
            head_result["clip_f1_macro"] = clip_res["f1_macro"]
            head_result["clip_targets"] = clip_res["targets"]
            head_result["clip_preds"] = clip_res["preds"]
            head_result["clip_ids"] = clip_res["clip_ids"]
        head_result["window_targets"] = s["targets"]
        head_result["window_preds"] = s["preds"]
        result["heads"][head] = head_result

    # ----- gaze head → front binary (distraction y_gaze_weak 기반 전이 평가) -----
    if not train:
        t = np.array(gaze_weak_state["targets"])
        p = np.array(gaze_weak_state["preds_binary"])
        ids = gaze_weak_state["clip_ids"]
        mask = t != IGNORE_LABEL
        n = int(mask.sum())
        entry = {"n": n}
        if n > 0:
            t_ = t[mask]; p_ = p[mask]
            entry["window_acc"] = float(accuracy_score(t_, p_))
            entry["window_f1_macro"] = float(f1_score(t_, p_, average="macro", zero_division=0))
            # clip-level (majority vote)
            from collections import defaultdict
            clip_preds = defaultdict(list)
            clip_tgts = {}
            for tt, pp, cc in zip(t, p, ids):
                if int(tt) == IGNORE_LABEL:
                    continue
                clip_preds[cc].append(int(pp))
                if cc in clip_tgts:
                    if clip_tgts[cc] != int(tt):
                        # 같은 clip 안에 label 다름 (이론상 drop 해야)
                        pass
                else:
                    clip_tgts[cc] = int(tt)
            clip_t, clip_p = [], []
            for c, preds in clip_preds.items():
                clip_t.append(clip_tgts[c])
                # majority vote
                clip_p.append(1 if sum(preds) > len(preds) / 2 else 0)
            if clip_t:
                entry["clip_acc"] = float(accuracy_score(clip_t, clip_p))
                entry["clip_f1_macro"] = float(f1_score(clip_t, clip_p, average="macro", zero_division=0))
                entry["clip_targets"] = clip_t
                entry["clip_preds"] = clip_p
            # 분포 기록
            entry["support_front"] = int((t_ == 1).sum())
            entry["support_not_front"] = int((t_ == 0).sum())
        result["gaze_binary_on_distraction"] = entry

    return result


def weighted_score(val_out: dict, weights: dict[str, float]) -> float:
    """각 head clip_f1_macro 가중합. best checkpoint 기준."""
    total = 0.0
    for head, w in weights.items():
        if head in val_out["heads"] and "clip_f1_macro" in val_out["heads"][head]:
            total += w * val_out["heads"][head]["clip_f1_macro"]
    return total

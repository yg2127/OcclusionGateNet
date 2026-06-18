# Auto-split from Pasted code(257).py
from __future__ import annotations

import copy
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path


def _as_path_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def _get_field(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_or_replace(obj, updates: dict):
    """
    dataclass / plain object / dict 모두 대응.
    discover_all()이 만든 원본 video 객체의 annotation/label metadata를 보존하면서
    manifest의 clean/masked face 경로만 교체한다.
    """
    clean_updates = {k: v for k, v in updates.items() if v is not None}

    if isinstance(obj, dict):
        out = copy.deepcopy(obj)
        out.update(clean_updates)
        return out

    if is_dataclass(obj):
        valid = {f.name for f in fields(obj)}
        safe_updates = {k: v for k, v in clean_updates.items() if k in valid}
        out = replace(obj, **safe_updates)

        # dataclass 필드가 아닌 부가 metadata는 가능한 경우에만 붙인다.
        for k, v in clean_updates.items():
            if k not in valid:
                try:
                    setattr(out, k, v)
                except Exception:
                    pass
        return out

    out = copy.copy(obj)
    for k, v in clean_updates.items():
        try:
            setattr(out, k, v)
        except Exception:
            pass
    return out


def _load_fixed_items_manifest(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fixed_items_json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_manifest_items(
    manifest: dict,
    split: str,
    variants: list[str] | tuple[str, ...],
) -> list[dict]:
    """
    manifest["items"] 안에서 split/variant 기준으로 item 수집.

    문제:
      explicit key(train_clean_all 등)만 부분적으로 존재하면
      clean만 collected되고 early return되어 masked가 누락될 수 있음.

    해결:
      variant별 explicit key를 먼저 찾고,
      못 찾은 variant는 전체 scan fallback으로 보완한다.
    """
    import json

    items_root = manifest.get("items", {})
    wanted_variants = list(variants)

    collected: list[dict] = []
    seen_ids = set()

    def add_item(item: dict):
        if not isinstance(item, dict):
            return

        uid = (
            item.get("sample_id")
            or item.get("sample_key")
            or item.get("clip_id")
            or item.get("face_npz")
            or json.dumps(item, sort_keys=True, ensure_ascii=False)
        )

        if uid in seen_ids:
            return

        seen_ids.add(uid)
        collected.append(item)

    # 1) variant별 explicit key 우선 수집
    found_variant = set()

    for variant in wanted_variants:
        candidate_keys = [
            f"{split}_{variant}_all",
            f"{split}_{variant}",
            f"{split}_{variant}_items",
        ]

        for key in candidate_keys:
            value = items_root.get(key)
            if isinstance(value, list):
                for item in value:
                    add_item(item)
                found_variant.add(variant)
                break

    # 2) explicit key로 못 찾은 variant는 fallback scan
    missing_variants = set(wanted_variants) - found_variant

    if missing_variants:
        for _, value in items_root.items():
            if not isinstance(value, list):
                continue

            for item in value:
                if not isinstance(item, dict):
                    continue

                if item.get("split") == split and item.get("variant") in missing_variants:
                    add_item(item)

    return collected


def _infer_prefix_from_manifest_item(item: dict) -> str:
    """
    manifest sample_key 예:
      distraction/dmd/gA/1/s1/gA_1_s1_2019-03-08T09;31;15+01;00_ir_face_facemesh

    반환 prefix:
      gA_1_s1_2019-03-08T09;31;15+01;00

    frame_shifts.json, discover_all()의 VideoRecord.prefix는
    _ir_face, _facemesh가 붙지 않은 원본 video prefix를 사용한다.
    """
    sample_key = item.get("sample_key") or item.get("face_path") or ""

    stem = Path(str(sample_key)).stem

    # 반드시 긴 suffix부터 제거해야 함.
    suffixes = (
        "_ir_face_facemesh",
        "_ir_face_face5pt",
        "_ir_body_skeleton",
        "_face_facemesh",
        "_face_face5pt",
        "_facemesh",
        "_face5pt",
    )

    for suf in suffixes:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    # 혹시 위 suffix 제거 후에도 _ir_face가 남는 경우를 한 번 더 방어
    if stem.endswith("_ir_face"):
        stem = stem[: -len("_ir_face")]

    if stem.endswith("_face"):
        stem = stem[: -len("_face")]

    return stem


def _get_video_prefix(video) -> str | None:
    """
    discover_all() 결과 video 객체에서 manifest prefix와 매칭 가능한 prefix를 추출한다.
    """
    for name in ("prefix", "video_prefix"):
        value = _get_field(video, name, None)
        if value:
            return str(value)

    # fallback: face_path / face5pt_path에서 prefix 추론
    for name in ("face_path", "face5pt_path"):
        path = _get_field(video, name, None)
        if not path:
            continue

        stem = Path(str(path)).stem
        for suf in ("_facemesh", "_face5pt"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
        if stem:
            return stem

    return None


def _build_video_index_by_prefix(videos: list) -> dict[str, object]:
    """
    discover_all() 결과를 prefix 기준으로 index.
    manifest item과 매칭할 때는 sample_key에서 추출한 prefix를 사용한다.
    """
    index: dict[str, object] = {}

    for v in videos:
        prefix = _get_video_prefix(v)
        if prefix:
            index[prefix] = v

    return index


def _manifest_item_to_video_from_template(item: dict, template_video):
    """
    discover_all()이 만든 원본 VideoRecord를 deepcopy하고,
    manifest의 clean/masked face npz 경로만 실제 사용 필드에 주입한다.

    중요:
      copy.copy()를 쓰면 template_video.extras dict가 clean/masked 객체 사이에서
      공유될 수 있다. 그러면 masked variant가 clean 객체의 extras까지 덮어써서
      clip_id가 전부 __masked__로 생성되고 duplicate가 발생한다.

    따라서 deepcopy + extras 새 dict 생성이 필요하다.

    보존:
      - body_npz
      - ann_path
      - hands_ann_path
      - source
      - 기존 action/gaze/hands/talk annotation metadata

    교체:
      - prefix
      - face_npz
      - face5pt_npz
      - variant/split metadata
    """
    out = copy.deepcopy(template_video)

    prefix = _infer_prefix_from_manifest_item(item)
    out.prefix = prefix

    # clip_builder.py가 실제로 사용하는 필드.
    if item.get("face_path"):
        out.face_npz = Path(item["face_path"])

    if item.get("face5pt_path"):
        out.face5pt_npz = Path(item["face5pt_path"])

    # template의 body_npz / ann_path / hands_ann_path는 deepcopy된 값을 그대로 유지한다.
    if item.get("subject_key"):
        out.subject_key = item["subject_key"]

    if item.get("source"):
        out.source = item["source"]

    if item.get("group"):
        out.group = item["group"]

    if item.get("subject") is not None:
        try:
            out.subject = int(item["subject"])
        except Exception:
            pass

    if item.get("session"):
        out.session = item["session"]

    # variant/split은 extras뿐 아니라 attribute로도 둔다.
    # clip_builder 또는 로그 함수가 어느 쪽을 보더라도 동일하게 동작하게 하기 위함.
    try:
        out.variant = item.get("variant")
    except Exception:
        pass

    try:
        out.split = item.get("split")
    except Exception:
        pass

    # deepcopy 이후에도 extras를 명시적으로 새 dict로 분리한다.
    old_extras = getattr(out, "extras", None)
    if isinstance(old_extras, dict):
        new_extras = dict(old_extras)
    else:
        new_extras = {}

    new_extras.update({
        "sample_id": item.get("sample_id"),
        "sample_key": item.get("sample_key"),
        "variant": item.get("variant"),
        "split": item.get("split"),
        "mask_region": item.get("mask_region"),
        "mask_appearance": item.get("mask_appearance"),
        "occ_labels": item.get("occ_labels"),
        "occ_label_vector": item.get("occ_label_vector"),
        "masked_video_path": item.get("masked_video_path"),
        "manifest_face_path": item.get("face_path"),
        "manifest_face5pt_path": item.get("face5pt_path"),
    })

    out.extras = new_extras

    return out


def _variant_from_video(v) -> str:
    extras = getattr(v, "extras", None) or {}
    if isinstance(extras, dict) and extras.get("variant"):
        return str(extras["variant"])
    value = getattr(v, "variant", None)
    return str(value) if value else "unknown"


def _variant_from_clip_id(clip_id: str) -> str:
    if "__clean__" in clip_id:
        return "clean"
    if "__masked__" in clip_id:
        return "masked"
    if "__orig__" in clip_id:
        return "orig"
    return "unknown"


def _log_video_variant_counts(name: str, videos: list, logger) -> None:
    from collections import Counter

    cnt = Counter(_variant_from_video(v) for v in videos)
    total = sum(cnt.values())
    ratio = {k: round(v / total, 4) if total else 0.0 for k, v in cnt.items()}
    logger.info(f"[variant videos] {name}: count={dict(cnt)} ratio={ratio} total={total}")


def _log_clip_variant_counts(name: str, clips: list, logger) -> None:
    from collections import Counter

    cnt = Counter(_variant_from_clip_id(getattr(c, "clip_id", "")) for c in clips)
    total = sum(cnt.values())
    ratio = {k: round(v / total, 4) if total else 0.0 for k, v in cnt.items()}
    logger.info(f"[variant clips] {name}: count={dict(cnt)} ratio={ratio} total={total}")


def _log_duplicate_clip_ids(name: str, clips: list, logger) -> None:
    from collections import Counter

    cnt = Counter(getattr(c, "clip_id", "") for c in clips)
    dup = {k: v for k, v in cnt.items() if v > 1}
    logger.info(
        f"[debug clip_id] {name}: total={len(clips)} unique={len(cnt)} duplicate_ids={len(dup)}"
    )
    if dup:
        logger.warning(f"[debug clip_id] {name}: duplicate examples={list(dup.items())[:10]}")


def _log_window_variant_counts(name: str, items: list, logger) -> None:
    """
    preload 이후 실제 학습에 들어가는 window 기준 clean/masked 수를 확인한다.
    clip_builder.py에서 clip_id에 __clean__ / __masked__가 들어가야 정확히 집계된다.
    """
    from collections import Counter

    cnt = Counter()
    for it in items:
        clip_id = getattr(it, "clip_id", "")
        cnt[_variant_from_clip_id(clip_id)] += 1

    total = sum(cnt.values())
    ratio = {k: round(v / total, 4) if total else 0.0 for k, v in cnt.items()}
    logger.info(f"[variant windows] {name}: count={dict(cnt)} ratio={ratio} total={total}")


def build_manifest_split_videos(
    *,
    manifest: dict,
    discovered_videos: list,
    clean_face_root: str | Path | None = None,
    train_variants: list[str],
    val_variants: list[str],
    test_variants: list[str],
    logger,
):
    """
    fixed manifest 기준으로 train/val/test videos 생성.

    direct SimpleNamespace를 새로 만들지 않고, discover_all()의 원본 video 객체를
    prefix로 찾아 template로 사용한다. 그래야 build_all_clips가 필요한
    annotation/label metadata가 보존된다.
    """
    del clean_face_root  # prefix 매칭 방식에서는 사용하지 않는다.

    video_index = _build_video_index_by_prefix(discovered_videos)

    logger.info(f"[fixed manifest/template] discovered videos={len(discovered_videos)}")
    logger.info(f"[fixed manifest/template] discovered templates={len(video_index)}")

    if video_index:
        logger.info(
            f"[fixed manifest/template] template prefix examples="
            f"{list(video_index.keys())[:5]}"
        )

    def convert_split(split: str, variants: list[str]) -> list:
        specs = _collect_manifest_items(manifest, split, variants)

        out = []
        missing_template = []
        missing_path = []

        for item in specs:
            if not item.get("face_path") or not item.get("face5pt_path"):
                missing_path.append(item.get("sample_id") or item.get("sample_key"))
                continue

            prefix = _infer_prefix_from_manifest_item(item)
            template = video_index.get(prefix)

            if template is None:
                missing_template.append(prefix)
                continue

            out.append(_manifest_item_to_video_from_template(item, template))

        logger.info(
            f"[fixed manifest/template] {split}: "
            f"specs={len(specs)} videos={len(out)} "
            f"missing_template={len(missing_template)} "
            f"missing_path={len(missing_path)} "
            f"variants={variants}"
        )

        if missing_template:
            logger.warning(
                f"[fixed manifest/template] {split} missing_template examples="
                f"{missing_template[:10]}"
            )

        if missing_path:
            logger.warning(
                f"[fixed manifest/template] {split} missing_path examples="
                f"{missing_path[:10]}"
            )

        if out:
            ex = out[0]
            logger.info(
                f"[fixed manifest/template] {split} example: "
                f"variant={_variant_from_video(ex)} "
                f"source={getattr(ex, 'source', None)} "
                f"subject_key={getattr(ex, 'subject_key', None)} "
                f"session={getattr(ex, 'session', None)} "
                f"prefix={getattr(ex, 'prefix', None)} "
                f"face_npz={getattr(ex, 'face_npz', None)} "
                f"face5pt_npz={getattr(ex, 'face5pt_npz', None)}"
            )

        return out

    train_videos = convert_split("train", train_variants)
    val_videos = convert_split("val", val_variants)
    test_clean_videos = convert_split("test", ["clean"])
    test_masked_videos = convert_split("test", ["masked"])

    if not train_videos:
        raise RuntimeError("[fixed manifest] train_videos is empty")
    if not val_videos:
        raise RuntimeError("[fixed manifest] val_videos is empty")
    if not test_clean_videos:
        raise RuntimeError("[fixed manifest] test_clean_videos is empty")
    if not test_masked_videos:
        raise RuntimeError("[fixed manifest] test_masked_videos is empty")

    return train_videos, val_videos, test_clean_videos, test_masked_videos

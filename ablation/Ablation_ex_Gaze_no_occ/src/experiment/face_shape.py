# Auto-split from Pasted code(257).py
from __future__ import annotations




def resolve_face_shape(face_cfg: dict, face_mode: str) -> tuple[int, int, str]:
    """Resolve face input channel and model face node count.

    Important:
        For facemesh_full + region_pool:
            - loader input V remains 478
            - model face graph V becomes cfg.face.num_regions, usually 7
            - latency dummy input should still use loader V=478 because model pools internally

    Return:
        face_in_ch:
            channel count of x_face
        num_face_regions:
            number of face nodes after optional model-side pooling
        face_encoder:
            "full" or "region_pool"
    """
    face_encoder = face_cfg.get("encoder", "full")

    if face_encoder not in ("full", "region_pool"):
        raise ValueError(
            f"Unknown face.encoder={face_encoder}. Use 'full' or 'region_pool'."
        )

    if face_mode in ("facemesh", "facemesh_full"):
        face_in_ch = (3 if face_cfg.get("use_z", True) else 2) + (
            1 if face_cfg.get("use_detected_channel", True) else 0
        )

        if face_mode == "facemesh_full":
            if face_encoder == "region_pool":
                num_face_regions = int(face_cfg.get("num_regions", 10))
            else:
                num_face_regions = int(face_cfg.get("num_landmarks", 478))
        else:
            num_face_regions = int(face_cfg.get("num_regions", 10))

    else:
        face_in_ch = (
            2
            + (1 if face_cfg.get("use_detected_channel", True) else 0)
            + (1 if face_cfg.get("use_det_score_channel", True) else 0)
        )
        num_face_regions = int(face_cfg.get("num_regions", 5))

    return face_in_ch, num_face_regions, face_encoder


def resolve_loader_face_v(face_cfg: dict, face_mode: str) -> int:
    """Resolve raw face node count emitted by preload/dataset.

    For facemesh_full, the dataset emits 478 landmarks even if the model later
    pools them to 7 regions.
    """
    if face_mode == "facemesh_full":
        return int(face_cfg.get("num_landmarks", 478))
    if face_mode == "facemesh":
        return int(face_cfg.get("num_regions", 10))
    return int(face_cfg.get("num_regions", 5))

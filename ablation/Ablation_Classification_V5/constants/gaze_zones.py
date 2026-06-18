"""DMD gaze_zone 라벨 ↔ id 매핑.

v1.2: `not_valid` 는 annotator 판정불가 태그라 학습·평가에서 전면 제외.
결과적으로 9-class (front 중심 유효 zone 들).
"""

# v1.2 — 9 zones (not_valid 제외)
GAZE_ZONES = [
    "left_mirror",      # 0
    "left",             # 1
    "front",            # 2
    "center_mirror",    # 3
    "front_right",      # 4
    "right_mirror",     # 5
    "right",            # 6
    "infotainment",     # 7
    "steering_wheel",   # 8
]

ZONE_TO_ID = {z: i for i, z in enumerate(GAZE_ZONES)}
ID_TO_ZONE = {i: z for z, i in ZONE_TO_ID.items()}

NUM_GAZE_ZONES = len(GAZE_ZONES)     # 9
FRONT_ZONE_ID = ZONE_TO_ID["front"]  # 2


def raw_label_to_id(raw: str) -> int | None:
    """`gaze_zone/front` 또는 `front` 형태 모두 허용. 미등록 라벨 (not_valid 등) 은 None."""
    if raw.startswith("gaze_zone/"):
        raw = raw[len("gaze_zone/"):]
    return ZONE_TO_ID.get(raw)


# Distraction weak supervision (binary) — v1.2 에서 loss 가중치 0 이지만 evaluation 용으로 보존
GAZE_WEAK_NUM = 2
GAZE_WEAK_FRONT = 1      # looking_road
GAZE_WEAK_NOT_FRONT = 0  # not_looking_road

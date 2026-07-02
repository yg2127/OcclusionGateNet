from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import csv

# =========================
# 설정
# =========================
ROOT = Path("/data/shared/Occlusion_subset_dataset/region_occlusion_cnn_dataset_v2_facecrop_256/images")
OUT_DIR = ROOT.parent / "sample_grids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_GRID = OUT_DIR / "occlusion_region_appearance_grid.png"
OUT_MANIFEST = OUT_DIR / "occlusion_region_appearance_grid_samples.csv"

REGIONS = [
    "clean",
    "both_eyes",
    "left_eye",
    "right_eye",
    "mouth",
    "full_occlusion",
]

APPEARANCES = [
    "clean",
    "blur_patch",
    "checker",
    "noise",
    "smooth_noise",
    "soft_noise",
    "soft_solid",
    "solid",
    "stripe",
]

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

CELL_W = 180
CELL_H = 210
IMG_SIZE = 150
HEADER_H = 45
LEFT_W = 130
BG = (255, 255, 255)
GRID = (210, 210, 210)
TEXT = (20, 20, 20)
MISSING = (245, 245, 245)


def get_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = get_font(16)
FONT_SMALL = get_font(12)
FONT_BOLD = get_font(18)


def find_images(path: Path):
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in EXTS])


def pick_sample(region: str, appearance: str):
    # clean은 보통 images/clean 안에 있거나 images/clean/clean 안에 있을 수 있어서 둘 다 대응
    if region == "clean":
        if appearance != "clean":
            return None

        candidates = []
        candidates += find_images(ROOT / "clean" / "clean")
        candidates += find_images(ROOT / "clean")

        # 중복 제거
        candidates = sorted(set(candidates))
        return random.choice(candidates) if candidates else None

    # 일반 occlusion region
    candidates = find_images(ROOT / region / appearance)
    return random.choice(candidates) if candidates else None


def load_thumb(path: Path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (250, 250, 250))
    x = (IMG_SIZE - img.width) // 2
    y = (IMG_SIZE - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_center(draw, box, text, font, fill=TEXT):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2), text, font=font, fill=fill)


# =========================
# grid 생성
# =========================
rows = REGIONS
cols = APPEARANCES

W = LEFT_W + CELL_W * len(cols)
H = HEADER_H + CELL_H * len(rows)

grid_img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(grid_img)

# title/header
draw.rectangle([0, 0, W, HEADER_H], fill=(235, 240, 245))
draw.text((10, 12), "Occlusion subset samples", font=FONT_BOLD, fill=TEXT)

# column headers
for j, app in enumerate(cols):
    x = LEFT_W + j * CELL_W
    draw.rectangle([x, 0, x + CELL_W, HEADER_H], outline=GRID, fill=(235, 240, 245))
    draw_center(draw, (x, 0, x + CELL_W, HEADER_H), app, FONT_SMALL)

records = []

for i, region in enumerate(rows):
    y = HEADER_H + i * CELL_H

    # row header
    draw.rectangle([0, y, LEFT_W, y + CELL_H], outline=GRID, fill=(240, 240, 240))
    draw_center(draw, (0, y, LEFT_W, y + CELL_H), region, FONT_BOLD)

    for j, app in enumerate(cols):
        x = LEFT_W + j * CELL_W
        draw.rectangle([x, y, x + CELL_W, y + CELL_H], outline=GRID, fill=BG)

        sample = pick_sample(region, app)

        if sample is None:
            draw.rectangle([x + 8, y + 8, x + CELL_W - 8, y + CELL_H - 8], fill=MISSING, outline=(230, 230, 230))
            draw_center(draw, (x, y, x + CELL_W, y + CELL_H), "-", FONT_BOLD, fill=(150, 150, 150))
            records.append({
                "region": region,
                "appearance": app,
                "sample_path": "",
                "status": "missing",
            })
            continue

        thumb = load_thumb(sample)
        px = x + (CELL_W - IMG_SIZE) // 2
        py = y + 12
        grid_img.paste(thumb, (px, py))

        # label
        draw_center(draw, (x, y + IMG_SIZE + 16, x + CELL_W, y + CELL_H - 4), app, FONT_SMALL)

        records.append({
            "region": region,
            "appearance": app,
            "sample_path": str(sample),
            "status": "ok",
        })

# 저장
grid_img.save(OUT_GRID, quality=95)

with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["region", "appearance", "sample_path", "status"])
    writer.writeheader()
    writer.writerows(records)

print("saved grid:", OUT_GRID)
print("saved manifest:", OUT_MANIFEST)
print("image size:", grid_img.size)

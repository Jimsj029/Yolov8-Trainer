"""
YOLOv8n-seg training script for stitch segmentation.
- Converts COCO annotations to YOLO format
- Splits train set 80/20 into train/val
- Trains yolov8n-seg with imgsz=256
"""

import json
import os
import shutil
import random
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / "train"
ANNO_FILE  = TRAIN_DIR / "_annotations.coco.json"
YOLO_DIR   = BASE_DIR / "yolo_dataset"

IMAGES_TRAIN = YOLO_DIR / "images" / "train"
IMAGES_VAL   = YOLO_DIR / "images" / "val"
LABELS_TRAIN = YOLO_DIR / "labels" / "train"
LABELS_VAL   = YOLO_DIR / "labels" / "val"

SEED = 42
VAL_RATIO = 0.2

# ─── Load COCO annotations ───────────────────────────────────────────────────
with open(ANNO_FILE, "r") as f:
    coco = json.load(f)

# Build lookup: image_id -> image info
images = {img["id"]: img for img in coco["images"]}

# Build category id -> zero-based class index
# Skip id=0 (background placeholder Roboflow sometimes adds)
real_cats = [c for c in coco["categories"] if c["supercategory"] != "none"]
cat_id_to_idx = {c["id"]: i for i, c in enumerate(real_cats)}
class_names = [c["name"] for c in real_cats]
print(f"Classes ({len(class_names)}): {class_names}")

# Group annotations by image id
from collections import defaultdict
ann_by_image = defaultdict(list)
for ann in coco["annotations"]:
    ann_by_image[ann["image_id"]].append(ann)

# ─── Train / Val split ───────────────────────────────────────────────────────
all_ids = list(images.keys())
random.seed(SEED)
random.shuffle(all_ids)
split = int(len(all_ids) * (1 - VAL_RATIO))
train_ids = set(all_ids[:split])
val_ids   = set(all_ids[split:])
print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")

# ─── Create output directories (clear any old data first) ───────────────────
for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# ─── Convert and copy ────────────────────────────────────────────────────────
def coco_seg_to_yolo(segmentation, img_w, img_h):
    """Convert COCO polygon [x1,y1,x2,y2,...] to normalised YOLO polygon."""
    pts = segmentation[0]  # take first polygon
    norm = []
    for i in range(0, len(pts), 2):
        x = max(0.0, min(1.0, pts[i]   / img_w))
        y = max(0.0, min(1.0, pts[i+1] / img_h))
        norm.extend([x, y])
    return norm

skipped = 0
for img_id, img_info in images.items():
    src_img = TRAIN_DIR / img_info["file_name"]
    if not src_img.exists():
        skipped += 1
        continue

    is_train = img_id in train_ids
    dst_imgs = IMAGES_TRAIN if is_train else IMAGES_VAL
    dst_lbls = LABELS_TRAIN if is_train else LABELS_VAL

    # Copy image
    shutil.copy2(src_img, dst_imgs / img_info["file_name"])

    # Write segmentation label file
    label_path = dst_lbls / (Path(img_info["file_name"]).stem + ".txt")
    anns = ann_by_image.get(img_id, [])
    with open(label_path, "w") as lf:
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_idx:
                continue
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list) or len(seg[0]) < 6:
                continue
            cls = cat_id_to_idx[cat_id]
            norm_pts = coco_seg_to_yolo(seg, img_info["width"], img_info["height"])
            pts_str = " ".join(f"{p:.6f}" for p in norm_pts)
            lf.write(f"{cls} {pts_str}\n")

if skipped:
    print(f"Warning: {skipped} images not found in {TRAIN_DIR}")

# ─── Write dataset.yaml ──────────────────────────────────────────────────────
yaml_path = YOLO_DIR / "dataset.yaml"
yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
with open(yaml_path, "w") as yf:
    yf.write(yaml_content)
print(f"dataset.yaml written to {yaml_path}")

# ─── Train YOLOv8n-seg ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics not installed. Run:  pip install ultralytics"
    )

model = YOLO('yolov8n-seg.pt')

results = model.train(
    data=str(yaml_path),
    imgsz=640,
    epochs=200,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    max_det=100,
    workers=8,
    augment=True,
    mosaic=0.5,
    mixup=0.1,
    fliplr=0.5,
    flipud=0.0,
    patience=50,
    scale=0.3,
    translate=0.1,
    degrees=10,
    shear=0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    name="segmentation",
    project=str(BASE_DIR / "runs"),
    exist_ok=True,
)

print("\nYOLOv8n-seg training complete!")
print(f"Segmentation results saved to: {BASE_DIR / 'runs' / 'segmentation'}")

import os
import json
import random

SRC_JSON    = r'c:\new github repo\last hope stitch zoomed\train\_annotations.coco.json'
IMG_SRC     = r'c:\new github repo\last hope stitch zoomed\train'
TRAIN_LABEL = r'c:\new github repo\last hope stitch zoomed\yolo_dataset\labels\train'
VAL_LABEL   = r'c:\new github repo\last hope stitch zoomed\yolo_dataset\labels\val'

# ---- load COCO JSON -------------------------------------------------------
with open(SRC_JSON) as f:
    data = json.load(f)

# image_id -> (filename, width, height)
id_to_img = {img['id']: img for img in data['images']}

# Build category id -> zero-based class index
# Skip id=0 (background placeholder Roboflow sometimes adds)
real_cats = [c for c in data['categories'] if c['supercategory'] != 'none']
cat_id_to_idx = {c['id']: i for i, c in enumerate(real_cats)}
class_names = [c['name'] for c in real_cats]
print(f"Classes ({len(class_names)}): {class_names}")

# image_id -> list of YOLO lines (segmentation)
id_to_anns = {}
for ann in data['annotations']:
    iid = ann['image_id']
    img = id_to_img[iid]
    iw, ih = img['width'], img['height']

    cat_id = ann['category_id']
    if cat_id not in cat_id_to_idx:
        continue
    cls = cat_id_to_idx[cat_id]

    # Segmentation: COCO polygons
    seg = ann.get('segmentation')
    if seg and isinstance(seg, list) and len(seg[0]) >= 6:
        norm_poly = []
        for i in range(0, len(seg[0]), 2):
            x = max(0.0, min(1.0, seg[0][i]   / iw))
            y = max(0.0, min(1.0, seg[0][i+1] / ih))
            norm_poly.extend([x, y])
        # YOLOv8 segmentation format: class x1 y1 x2 y2 ...
        line = f"{cls} " + " ".join(f"{p:.6f}" for p in norm_poly)
        id_to_anns.setdefault(iid, []).append(line)
    else:
        # If no segmentation, skip
        continue

# ---- reproduce the exact same 80/20 split used for images -----------------
all_images = [f for f in os.listdir(IMG_SRC) if f.lower().endswith('.jpg')]
random.seed(42)
random.shuffle(all_images)

split      = int(0.8 * len(all_images))
train_imgs = set(all_images[:split])
val_imgs   = set(all_images[split:])

# filename -> image_id
fname_to_id = {img['file_name']: img['id'] for img in data['images']}

# ---- clear old labels -----------------------------------------------------
for folder in (TRAIN_LABEL, VAL_LABEL):
    for f in os.listdir(folder):
        if f.endswith('.txt'):
            os.remove(os.path.join(folder, f))

# ---- write new labels -----------------------------------------------------
written_train = written_val = skipped = 0

for fname, iid in fname_to_id.items():
    lines = id_to_anns.get(iid, [])
    stem  = os.path.splitext(fname)[0]
    txt   = stem + '.txt'

    if fname in train_imgs:
        dest = os.path.join(TRAIN_LABEL, txt)
        written_train += 1
    elif fname in val_imgs:
        dest = os.path.join(VAL_LABEL, txt)
        written_val += 1
    else:
        skipped += 1
        continue

    with open(dest, 'w') as out:
        out.write('\n'.join(lines))

print(f"Segmentation labels written — train: {written_train}, val: {written_val}, skipped: {skipped}")

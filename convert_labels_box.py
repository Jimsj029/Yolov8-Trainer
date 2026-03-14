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

# image_id -> list of YOLO lines
id_to_anns = {}
for ann in data['annotations']:
    iid = ann['image_id']
    img = id_to_img[iid]
    iw, ih = img['width'], img['height']

    x_min, y_min, bw, bh = [float(v) for v in ann['bbox']]
    x_center = (x_min + bw / 2) / iw
    y_center  = (y_min + bh / 2) / ih
    bw_norm   = bw / iw
    bh_norm   = bh / ih

    # single class -> always 0
    line = f"0 {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}"
    id_to_anns.setdefault(iid, []).append(line)

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

print(f"Labels written — train: {written_train}, val: {written_val}, skipped: {skipped}")

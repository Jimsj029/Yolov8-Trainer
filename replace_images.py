import os
import shutil
import random

src = r'c:\new github repo\last hope stitch zoomed\train'
train_dst = r'c:\new github repo\last hope stitch zoomed\yolo_dataset\images\train'
val_dst   = r'c:\new github repo\last hope stitch zoomed\yolo_dataset\images\val'

train_label_dst = r'c:\new github repo\last hope stitch zoomed\yolo_dataset\labels\train'
val_label_dst   = r'c:\new github repo\last hope stitch zoomed\yolo_dataset\labels\val'

images = [f for f in os.listdir(src) if f.lower().endswith('.jpg')]
random.seed(42)
random.shuffle(images)

# ---- clear old images ----
for folder in (train_dst, val_dst):
    for f in os.listdir(folder):
        if f.lower().endswith('.jpg'):
            os.remove(os.path.join(folder, f))

# ---- clear old labels ----
for folder in (train_label_dst, val_label_dst):
    for f in os.listdir(folder):
        if f.lower().endswith('.txt'):
            os.remove(os.path.join(folder, f))

split = int(0.8 * len(images))
train_imgs = images[:split]
val_imgs   = images[split:]

for f in train_imgs:
    shutil.copy2(os.path.join(src, f), os.path.join(train_dst, f))

for f in val_imgs:
    shutil.copy2(os.path.join(src, f), os.path.join(val_dst, f))

print(f'Done. Copied {len(train_imgs)} images to train, {len(val_imgs)} images to val.')

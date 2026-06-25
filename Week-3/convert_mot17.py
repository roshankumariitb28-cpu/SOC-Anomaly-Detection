import os
from PIL import Image

MOT_ROOT = r"Dataset\MOT17\train"

OUT_IMAGES = r"Dataset\processed\train\images"
OUT_LABELS = r"Dataset\processed\train\labels"

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN"
]

for seq in SEQUENCES:

    seq_path = os.path.join(MOT_ROOT, seq)

    gt_file = os.path.join(seq_path, "gt", "gt.txt")
    img_dir = os.path.join(seq_path, "img1")

    labels = {}

    with open(gt_file, "r") as f:

        for line in f:

            data = line.strip().split(",")

            frame = int(data[0])
            cls = int(data[7])

            # keep pedestrians
            if cls != 1:
                continue

            x = float(data[2])
            y = float(data[3])
            w = float(data[4])
            h = float(data[5])

            img_name = f"{frame:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path)
            W, H = img.size

            xc = (x + w/2) / W
            yc = (y + h/2) / H
            wn = w / W
            hn = h / H

            labels.setdefault(img_name, []).append(
                f"0 {xc} {yc} {wn} {hn}"
            )

    for img_name, anns in labels.items():

        src = os.path.join(img_dir, img_name)

        dst = os.path.join(
            OUT_IMAGES,
            f"{seq}_{img_name}"
        )

        with open(
            os.path.join(
                OUT_LABELS,
                f"{seq}_{img_name.replace('.jpg','.txt')}"
            ),
            "w"
        ) as f:
            f.write("\n".join(anns))

        import shutil
        shutil.copy(src, dst)

print("DONE")
print("Hello MOT17")
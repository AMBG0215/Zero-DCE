"""Split images/ and labels/ into train/ (80%) and valid/ (20%)."""
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR / "images"
LABELS_DIR = SCRIPT_DIR / "labels"
TRAIN_RATIO = 0.8
SEED = 42


def main():
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError("Need yolo_dataset/images/ and yolo_dataset/labels/")

    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg files in {IMAGES_DIR}")

    random.seed(SEED)
    random.shuffle(images)
    n_train = int(len(images) * TRAIN_RATIO)
    train_imgs = set(images[:n_train])
    valid_imgs = images[n_train:]

    for split, split_images in (("train", train_imgs), ("valid", valid_imgs)):
        img_out = SCRIPT_DIR / split / "images"
        lbl_out = SCRIPT_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            stem = img_path.stem
            lbl_path = LABELS_DIR / f"{stem}.txt"
            if not lbl_path.exists():
                print(f"Warning: missing label for {img_path.name}")
                continue
            shutil.move(str(img_path), str(img_out / img_path.name))
            shutil.move(str(lbl_path), str(lbl_out / f"{stem}.txt"))

    for d in (IMAGES_DIR, LABELS_DIR):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()

    train_n = len(list((SCRIPT_DIR / "train" / "images").glob("*.jpg")))
    valid_n = len(list((SCRIPT_DIR / "valid" / "images").glob("*.jpg")))
    print(f"Done. train: {train_n} images, valid: {valid_n} images")


if __name__ == "__main__":
    main()

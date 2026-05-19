"""
Train YOLO11 on the enhanced dataset.
Run from this folder: python train_yolo11.py
"""
from pathlib import Path

from ultralytics import YOLO


EPOCHS = 100
MODEL = "yolo11m.pt"
BATCH = 8
IMAGE_SIZE = 640
PATIENCE = 50 


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_DIR / "data.yaml"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing {DATA_YAML}")

    train_images = SCRIPT_DIR / "train" / "images"
    train_labels = SCRIPT_DIR / "train" / "labels"
    valid_images = SCRIPT_DIR / "valid" / "images"
    valid_labels = SCRIPT_DIR / "valid" / "labels"

    def count_files(folder: Path, pattern: str) -> int:
        return len(list(folder.glob(pattern))) if folder.is_dir() else 0

    n_train_img = count_files(train_images, "*.jpg")
    n_train_lbl = count_files(train_labels, "*.txt")
    n_valid_img = count_files(valid_images, "*.jpg")
    n_valid_lbl = count_files(valid_labels, "*.txt")

    if n_train_img == 0 or n_valid_img == 0:
        flat_images = count_files(SCRIPT_DIR / "images", "*.jpg")
        raise FileNotFoundError(
            "No images in train/images or valid/images.\n"
            f"  train/images: {n_train_img} jpg, train/labels: {n_train_lbl} txt\n"
            f"  valid/images: {n_valid_img} jpg, valid/labels: {n_valid_lbl} txt\n"
            f"  (top-level images/: {flat_images} jpg — empty after split is normal)\n\n"
            "Your workflow: Roboflow split + labels, enhanced images from Zero-DCE.\n"
            "  Fix: edit paths in setup_enhanced_dataset.py and run it,\n"
            "  or copy Roboflow train/valid labels + enhanced images into train/ and valid/."
        )

    model = YOLO(MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        project=str(SCRIPT_DIR / "runs"),
        name="detect",
    )

    print("\nDone. Best weights:")
    print(SCRIPT_DIR / "runs" / "detect" / "weights" / "best.pt")


if __name__ == "__main__":
    main()

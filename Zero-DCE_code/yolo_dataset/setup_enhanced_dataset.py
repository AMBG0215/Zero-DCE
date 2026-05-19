"""
Build yolo_dataset from Roboflow labels + Zero-DCE enhanced images.

Run: python setup_enhanced_dataset.py
"""
import random
import shutil
from pathlib import Path

ROBOFLOW_EXPORT = Path(r"C:\Users\ADMIN\Downloads\Annotations.yolov11 (1)")
ENHANCED_IMAGES = Path(
    r"C:\Users\ADMIN\Desktop\PROJECT\Zero-Dce1\Zero-DCE_code\data\result\DICM"
)

SCRIPT_DIR = Path(__file__).resolve().parent
VAL_RATIO = 0.2
SEED = 42


def find_enhanced(enhanced: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = enhanced / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_pair(lbl: Path, enhanced: Path, dst_img: Path, dst_lbl: Path) -> bool:
    img = find_enhanced(enhanced, lbl.stem)
    if img is None:
        print(f"Warning: no enhanced image for {lbl.name}")
        return False
    shutil.copy2(img, dst_img / img.name)
    shutil.copy2(lbl, dst_lbl / lbl.name)
    return True


def clear_split(out_root: Path, split: str) -> None:
    for sub in ("images", "labels"):
        d = out_root / split / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)


def copy_roboflow_split(export: Path, enhanced: Path, out_root: Path, split: str) -> int:
    src = export / split
    if not src.is_dir() and split == "valid":
        src = export / "val"
    if not src.is_dir():
        return -1

    src_labels = src / "labels"
    if not src_labels.is_dir():
        raise FileNotFoundError(f"No labels in {src}")

    dst_img = out_root / split / "images"
    dst_lbl = out_root / split / "labels"
    clear_split(out_root, split)

    count = 0
    for lbl in sorted(src_labels.glob("*.txt")):
        if write_pair(lbl, enhanced, dst_img, dst_lbl):
            count += 1
    return count


def split_train_into_train_valid(
    export: Path, enhanced: Path, out_root: Path
) -> tuple[int, int]:
    """Roboflow export has only train/ — split labels 80/20 for YOLO val."""
    src_labels = sorted((export / "train" / "labels").glob("*.txt"))
    random.seed(SEED)
    random.shuffle(src_labels)
    n_val = max(1, int(len(src_labels) * VAL_RATIO))
    val_labels = set(src_labels[:n_val])
    train_labels = src_labels[n_val:]

    clear_split(out_root, "train")
    clear_split(out_root, "valid")

    train_n = val_n = 0
    for lbl in src_labels:
        split = "valid" if lbl in val_labels else "train"
        dst_img = out_root / split / "images"
        dst_lbl = out_root / split / "labels"
        if write_pair(lbl, enhanced, dst_img, dst_lbl):
            if split == "train":
                train_n += 1
            else:
                val_n += 1
    return train_n, val_n


def main():
    if not ROBOFLOW_EXPORT.is_dir():
        raise FileNotFoundError(f"Roboflow folder not found: {ROBOFLOW_EXPORT}")
    if not ENHANCED_IMAGES.is_dir():
        raise FileNotFoundError(f"Enhanced images not found: {ENHANCED_IMAGES}")

    valid_n = copy_roboflow_split(ROBOFLOW_EXPORT, ENHANCED_IMAGES, SCRIPT_DIR, "valid")
    train_n = copy_roboflow_split(ROBOFLOW_EXPORT, ENHANCED_IMAGES, SCRIPT_DIR, "train")

    if valid_n < 0 or train_n < 0:
        print(
            "Roboflow export has no valid/ folder — splitting train labels 80/20 "
            f"(enhanced images from {ENHANCED_IMAGES.name})."
        )
        train_n, valid_n = split_train_into_train_valid(
            ROBOFLOW_EXPORT, ENHANCED_IMAGES, SCRIPT_DIR
        )

    print("Done.")
    print(f"  train: {train_n} pairs (enhanced image + Roboflow label)")
    print(f"  valid: {valid_n} pairs")
    print("\nNext: python train_yolo11.py")


if __name__ == "__main__":
    main()

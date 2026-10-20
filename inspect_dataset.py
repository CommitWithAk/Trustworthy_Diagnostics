
from pathlib import Path

DATASET_ROOT = Path(r"D:\Downloads\dataset\brisc2025")

folders = {
    "Classification train": DATASET_ROOT / "classification_task" / "train",
    "Classification test": DATASET_ROOT / "classification_task" / "test",
    "Segmentation train images": DATASET_ROOT / "segmentation_task" / "train" / "images",
    "Segmentation train masks": DATASET_ROOT / "segmentation_task" / "train" / "masks",
    "Segmentation test images": DATASET_ROOT / "segmentation_task" / "test" / "images",
    "Segmentation test masks": DATASET_ROOT / "segmentation_task" / "test" / "masks",
}

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"Dataset root not found: {DATASET_ROOT}\n"
        "Update DATASET_ROOT to your extracted dataset location."
    )

for name, folder in folders.items():
    print(f"\n--- {name} ---")
    print("Path:", folder)
    print("Exists:", folder.exists())

    if not folder.exists():
        continue

    if name.startswith("Classification"):
        for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            count = sum(1 for p in class_dir.rglob("*") if p.is_file())
            print(f"{class_dir.name}: {count} files")
    else:
        files = [p for p in folder.rglob("*") if p.is_file()]
        print("Total files:", len(files))
        print("Examples:", [p.name for p in files[:5]])

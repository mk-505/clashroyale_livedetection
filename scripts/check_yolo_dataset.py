#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a dictionary")
    return data


def resolve_image(base: Path, item: str) -> Path:
    p = Path(item.strip())
    if p.is_absolute():
        return p
    return (base / p).resolve()


def ultralytics_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            break
    p = Path(*parts)
    return p.with_suffix(".txt")


def same_dir_label_path(img_path: Path) -> Path:
    return img_path.with_suffix(".txt")


def has_valid_yolo_rows(label_path: Path) -> bool:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    for line in lines:
        row = line.strip()
        if not row:
            continue
        parts = row.split()
        if len(parts) != 5:
            continue
        try:
            int(parts[0])
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
            return True
        except ValueError:
            continue
    return False


def read_split_entries(dataset_root: Path, split_value: str) -> list[str]:
    split_value = str(split_value).strip()
    split_path = (dataset_root / split_value).resolve()
    if split_path.exists() and split_path.is_file():
        lines = split_path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]
    if any(split_value.lower().endswith(ext) for ext in IMAGE_EXTS):
        return [split_value]
    raise FileNotFoundError(
        f"Split reference '{split_value}' is neither a file list nor an image path"
    )


def analyze_split(dataset_root: Path, split_name: str, split_value: str, sample: int) -> None:
    entries = read_split_entries(dataset_root, split_value)
    image_paths = [resolve_image(dataset_root, line) for line in entries]
    image_exists = [p for p in image_paths if p.exists()]
    image_missing = [p for p in image_paths if not p.exists()]

    auto_labels = [ultralytics_label_path(p) for p in image_paths]
    same_labels = [same_dir_label_path(p) for p in image_paths]

    auto_exists = [p for p in auto_labels if p.exists()]
    same_exists = [p for p in same_labels if p.exists()]
    auto_valid = [p for p in auto_exists if has_valid_yolo_rows(p)]
    same_valid = [p for p in same_exists if has_valid_yolo_rows(p)]

    print(f"\n=== Split: {split_name} ===")
    print(f"Entries: {len(entries)}")
    print(f"Images found: {len(image_exists)} | missing: {len(image_missing)}")
    print(f"Labels (Ultralytics auto images->labels): {len(auto_exists)} existing, {len(auto_valid)} with valid rows")
    print(f"Labels (same image directory): {len(same_exists)} existing, {len(same_valid)} with valid rows")

    if image_missing:
        print("Sample missing images:")
        for p in image_missing[:sample]:
            print(f"  - {p}")

    missing_auto = [p for p in auto_labels if not p.exists()]
    if missing_auto:
        print("Sample missing auto-mapped labels:")
        for p in missing_auto[:sample]:
            print(f"  - {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check YOLO dataset image/label layout.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to Ultralytics data YAML (e.g. images/part2/ClashRoyale_detection.yaml).",
    )
    parser.add_argument("--sample", type=int, default=5, help="Sample count per warning section.")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    cfg = load_yaml(data_yaml)

    if "path" not in cfg:
        raise KeyError(f"'path' missing in {data_yaml}")
    dataset_root = Path(str(cfg["path"])).resolve()
    print(f"Data YAML: {data_yaml}")
    print(f"Dataset root: {dataset_root}")

    for split_name in ("train", "val", "test"):
        split_value = cfg.get(split_name)
        if split_value in (None, "", "null"):
            continue
        analyze_split(dataset_root, split_name, str(split_value), args.sample)


if __name__ == "__main__":
    main()

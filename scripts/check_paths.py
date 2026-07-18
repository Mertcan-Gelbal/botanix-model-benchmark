#!/usr/bin/env python3
"""Check that the dataset directory layout the notebooks expect is present.

The benchmark notebooks read the dataset from ``./data`` (relative to the
notebooks directory) with ``train/``, ``val/`` and ``test/`` subfolders. This
script verifies that layout and prints per-split class/image counts so a run is
not started against an empty or half-downloaded dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_data_root() -> Path | None:
    """Look for a data directory in the usual locations."""
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "data",
        here.parent / "data",
        here.parent / "notebooks" / "data",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def summarize_split(split_dir: Path) -> tuple[int, int]:
    """Return (class_count, image_count) for a split directory."""
    classes = [d for d in split_dir.iterdir() if d.is_dir()]
    images = sum(
        1
        for cls in classes
        for f in cls.iterdir()
        if f.suffix.lower() in IMAGE_SUFFIXES
    )
    return len(classes), images


def main() -> int:
    data_root = find_data_root()
    if data_root is None:
        print("FAIL: no ./data directory found.")
        print("Download the dataset first (see README 'Setup').")
        return 1

    print(f"Data root: {data_root}")
    ok = True
    class_sets = {}
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"  MISS {split}/ (expected at {split_dir})")
            ok = False
            continue
        n_classes, n_images = summarize_split(split_dir)
        class_sets[split] = {d.name for d in split_dir.iterdir() if d.is_dir()}
        print(f"  ok   {split}/  classes={n_classes}  images={n_images}")

    if len(class_sets) == len(SPLITS):
        base = class_sets["train"]
        for split in ("val", "test"):
            if class_sets[split] != base:
                print(f"  WARN class set of {split}/ differs from train/")

    if not ok:
        print("\nFAIL: dataset layout incomplete.")
        return 1

    print("\nOK: dataset layout looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate the Python environment before running the Botanix benchmark notebooks.

Checks that required packages import, reports PyTorch/CUDA availability, and
warns (without failing) on anything optional. Exit code is non-zero only when a
hard-required package is missing, so it is safe to run in CI.
"""
from __future__ import annotations

import importlib
import sys

REQUIRED = ["torch", "torchvision", "timm", "sklearn", "numpy", "pandas", "PIL"]
OPTIONAL = ["matplotlib", "seaborn", "kagglehub", "kaggle"]


def check(names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
            print(f"  ok   {name}")
        except ImportError:
            print(f"  MISS {name}")
            missing.append(name)
    return missing


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")

    print("\nRequired packages:")
    missing_required = check(REQUIRED)

    print("\nOptional packages:")
    check(OPTIONAL)

    print("\nPyTorch / CUDA:")
    try:
        import torch

        print(f"  torch: {torch.__version__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  torch not importable")

    if missing_required:
        print(f"\nFAIL: missing required packages: {', '.join(missing_required)}")
        print("Install them with: pip install -r requirements.txt")
        return 1

    print("\nOK: environment satisfies benchmark requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

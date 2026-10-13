#!/usr/bin/env python3
"""Report required Python packages without importing model weights or data."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys


PACKAGES = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "pandas": "pandas", "Pillow": "Pillow", "matplotlib": "matplotlib", "hydra-core": "hydra-core"}


def main() -> int:
    results = {}
    missing = []
    for module, distribution in PACKAGES.items():
        try:
            version = importlib.metadata.version(distribution)
            results[module] = {"available": importlib.util.find_spec(module) is not None, "version": version}
        except importlib.metadata.PackageNotFoundError:
            results[module] = {"available": False, "version": None}
            missing.append(distribution)
    try:
        results["sam2"] = {"available": importlib.util.find_spec("sam2") is not None, "version": None}
    except (ImportError, ValueError):
        results["sam2"] = {"available": False, "version": None}
    print(json.dumps({"python": sys.version.split()[0], "packages": results, "missing": missing}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

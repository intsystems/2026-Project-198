"""Auto-import utility for factory/registry pattern."""

import importlib
from pathlib import Path


def import_modules(modules_dir: str, package: str) -> None:
    """Auto-import all .py files in a directory to trigger @register decorators."""
    for f in sorted(Path(modules_dir).glob("*.py")):
        if f.name.startswith("_"):
            continue
        module_name = f.stem
        importlib.import_module(f"{package}.{module_name}")

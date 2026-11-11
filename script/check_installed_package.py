"""Check that the installed distribution, not the checkout, provides TORAI."""
import importlib
import os
from pathlib import Path
import sys

workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()

import RCAEval

installed_root = Path(RCAEval.__file__).resolve().parent
source_root = (workspace / "RCAEval").resolve()
if installed_root == source_root:
    raise RuntimeError(f"RCAEval was imported from the checkout: {installed_root}")

modules = ["RCAEval.e2e.torai", "RCAEval.utility"]
if sys.version_info >= (3, 10):
    modules.append("RCAEval.logparser")

for name in modules:
    module = importlib.import_module(name)
    module_path = Path(module.__file__).resolve()
    if installed_root not in module_path.parents:
        raise RuntimeError(f"{name} resolved outside the installed package: {module_path}")

print(f"Installed RCAEval package verified at {installed_root}")

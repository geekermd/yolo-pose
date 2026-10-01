"""Guard against relative-import mistakes (a module using `..` from the root)."""

import importlib
import os
import pkgutil

import pytest

import yolopose

PACKAGE_ROOT = os.path.dirname(yolopose.__file__)


def _all_modules():
    names = ["yolopose"]
    for info in pkgutil.walk_packages(yolopose.__path__, prefix="yolopose."):
        names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name):
    """Every module must import cleanly (no 'beyond top-level package')."""
    importlib.import_module(name)


def test_no_double_dot_imports_at_package_root():
    """Files directly inside the package must only use single-dot imports."""
    offenders = []
    for entry in os.listdir(PACKAGE_ROOT):
        if not entry.endswith(".py"):
            continue
        path = os.path.join(PACKAGE_ROOT, entry)
        with open(path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                stripped = line.strip()
                if stripped.startswith("from ..") or stripped.startswith("import .."):
                    offenders.append(f"{entry}:{number}: {stripped}")
    assert not offenders, "double-dot imports at package root: " + "; ".join(offenders)


def test_cli_and_demo_entry_points_exist():
    from yolopose.cli import COMMANDS
    assert "demo" in COMMANDS
    assert "face" in COMMANDS
    from yolopose.demo import DemoState, png_bytes
    assert callable(DemoState)
    assert callable(png_bytes)

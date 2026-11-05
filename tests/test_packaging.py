from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("localmelo_setup", ROOT / "setup.py")
assert spec and spec.loader
setup_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_script)


def test_package_list_is_every_package_folder_but_the_tests() -> None:
    """The wheel's packages are read off the tree, whatever the clone is called."""
    roots = [
        folder
        for folder in ROOT.iterdir()
        if (folder / "__init__.py").is_file() and folder.name != "tests"
    ]
    expected = {"localmelo"} | {
        ".".join(["localmelo", *init.parent.relative_to(ROOT).parts])
        for root in roots
        for init in root.rglob("__init__.py")
    }
    packages = setup_script.package_list()
    assert len(packages) == len(set(packages))
    assert set(packages) == expected
    assert "localmelo.melo.agent" in packages
    assert "localmelo.support.gateway" in packages
    assert not any(name.startswith("localmelo.tests") for name in packages)

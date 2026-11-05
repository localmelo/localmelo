"""Build layout: the folder that holds this file IS the package `localmelo`.

setuptools cannot list the packages of such a layout by itself, and a search of the
parent folder only works when the clone is called `localmelo`. So the list is read off
this tree: every folder with an `__init__.py`, whatever the clone is called.
"""

from __future__ import annotations

from collections.abc import Container
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOT_SHIPPED = {"tests"}  # package folders at the top of the clone that stay out


def find(folder: Path, name: str, skip: Container[str] = ()) -> list[str]:
    """*name* and every package below *folder* that is reached through packages."""
    found = [name]
    for child in sorted(folder.iterdir()):
        if child.name not in skip and (child / "__init__.py").is_file():
            found += find(child, f"{name}.{child.name}")
    return found


def package_list() -> list[str]:
    return find(ROOT, "localmelo", NOT_SHIPPED)


if __name__ == "__main__":
    from setuptools import setup  # a build has it; the tests import this file without

    setup(packages=package_list(), package_dir={"localmelo": "."})

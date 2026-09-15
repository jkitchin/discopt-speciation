"""MkDocs hook: copy the example notebooks into the docs tree before each build.

mkdocs-jupyter only renders files under ``docs/``, but the notebooks live in
``examples/``. Files are copied only when changed, so ``mkdocs serve`` does not
rebuild in a loop.
"""
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent


def on_pre_build(config, **kwargs):
    dest = ROOT / "docs" / "examples"
    dest.mkdir(exist_ok=True)
    for nb in sorted((ROOT / "examples").glob("*.ipynb")):
        out = dest / nb.name
        if not out.exists() or out.read_bytes() != nb.read_bytes():
            shutil.copyfile(nb, out)

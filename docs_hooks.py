"""MkDocs hooks.

``on_pre_build`` copies the example notebooks into the docs tree:
mkdocs-jupyter only renders files under ``docs/``, but the notebooks live in
``examples/``. Files are copied only when changed, so ``mkdocs serve`` does not
rebuild in a loop.

``on_post_build`` puts the Python side of the web app (``docs/app/``) into the
site, under ``app/py/``:

- ``discopt-<version>.zip``: the pure-Python sources of discopt, taken from its
  PyPI wheel at the version pinned in ``uv.lock`` (the compiled extension is
  left out; the solve path does not use it);
- ``discopt-speciation.zip``: this package, with its databases;
- the pounce WebAssembly wheel for Pyodide, from the pounce documentation site;
- ``manifest.json``, which the app's worker reads.

Downloads are cached in ``.cache/webapp/``.
"""
import json
import pathlib
import re
import shutil
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "webapp"
POUNCE_WHEELS = "https://kitchingroup.cheme.cmu.edu/pounce/demo/python/wheels/"
BINARY = (".so", ".pyd", ".dylib", ".dll")


def on_pre_build(config, **kwargs):
    dest = ROOT / "docs" / "examples"
    dest.mkdir(exist_ok=True)
    for nb in sorted((ROOT / "examples").glob("*.ipynb")):
        out = dest / nb.name
        if not out.exists() or out.read_bytes() != nb.read_bytes():
            shutil.copyfile(nb, out)


def on_post_build(config, **kwargs):
    build_webapp_bundle(pathlib.Path(config["site_dir"]) / "app" / "py")


def _fetch(url: str, dest: pathlib.Path) -> pathlib.Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        dest.write_bytes(data)
    return dest


def _locked_version(name: str) -> str:
    lock = (ROOT / "uv.lock").read_text()
    m = re.search(rf'\[\[package\]\]\nname = "{re.escape(name)}"\nversion = "([^"]+)"', lock)
    if not m:
        raise RuntimeError(f"{name} is not in uv.lock")
    return m.group(1)


def _zip_py(members, out: pathlib.Path):
    """Write ``(archive name, bytes)`` pairs to a zip with a fixed timestamp, so rebuilds are identical."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(members):
            z.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)


def build_webapp_bundle(out: pathlib.Path):
    out.mkdir(parents=True, exist_ok=True)

    # discopt: the .py files of its wheel (any platform; the Python sources are the same)
    version = _locked_version("discopt")
    info = json.loads(_fetch(f"https://pypi.org/pypi/discopt/{version}/json",
                             CACHE / f"discopt-{version}.json").read_text())
    wheel = min((u for u in info["urls"] if u["packagetype"] == "bdist_wheel"), key=lambda u: u["filename"])
    whl = _fetch(wheel["url"], CACHE / wheel["filename"])
    with zipfile.ZipFile(whl) as z:
        members = [(n, z.read(n)) for n in z.namelist()
                   if n.startswith("discopt/") and n.endswith(".py") and not n.startswith("discopt/speciation/")]
    discopt_zip = f"discopt-{version}.zip"
    _zip_py(members, out / discopt_zip)

    # this package: sources and databases (not the Claude Code skill)
    src = ROOT / "src"
    pkg = src / "discopt" / "speciation"
    members = [(p.relative_to(src).as_posix(), p.read_bytes()) for p in pkg.rglob("*")
               if p.is_file() and p.suffix in (".py", ".dat", ".yaml", ".json")
               and "__pycache__" not in p.parts and "skill" not in p.relative_to(pkg).parts]
    _zip_py(members, out / "discopt-speciation.zip")

    # pounce, compiled for Pyodide
    pm = json.loads(urllib.request.urlopen(POUNCE_WHEELS + "pounce-wheel.json", timeout=60).read())
    pounce = _fetch(POUNCE_WHEELS + pm["wheel"], CACHE / pm["wheel"])
    shutil.copyfile(pounce, out / pm["wheel"])

    (out / "manifest.json").write_text(json.dumps({
        "pyodide_version": pm["pyodide_version"],
        "pounce_wheel": pm["wheel"],
        "pounce_version": pm["version"],
        "discopt_version": version,
        "archives": [discopt_zip, "discopt-speciation.zip"],
    }, indent=2))

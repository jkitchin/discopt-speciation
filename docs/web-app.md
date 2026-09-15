# Web app

[**Open the web app →**](app/index.html)

The web app runs `discopt.speciation` in your browser. You can:

- pick a bundled database;
- choose elements or valence states, and the minerals and gases that may form;
- set amounts, temperature, pH, pe, fixed activities, a charge-balance ion and the activity model.

It then shows results in five tabs:

| tab | shows |
|-----|-------|
| Speciation | molalities, activities and activity coefficients; saturation indices; the phases present; the text summary |
| Sweep | species, solids, saturation indices and pH along pH, pe, temperature, an amount, a partial pressure or a fixed activity (CSV export) |
| Pourbaix / predominance | the dominant species or solid of an element on a grid of two variables, with the water stability lines and an Eh axis |
| log K(T) | dissolution constants of the selected phases against temperature |
| Python | the same calculation as a Python script |

**Sharing.** The address bar holds the whole setup, so a copied link reproduces it.

## How it works

Nothing is computed on a server. The page loads, in a Web Worker:

- [Pyodide](https://pyodide.org) 0.28.3 (CPython compiled to WebAssembly), with numpy, scipy and pydantic;
- the WebAssembly build of the [pounce](https://github.com/jkitchin/pounce) NLP solver;
- the Python sources of [discopt](https://github.com/jkitchin/discopt) and of this package, with its databases.

**Same algorithm as the package.** The equilibrium calculation is the one the
Python package does:
- speciation;
- the complementarity homotopy;
- the active-set polish, solved with discopt and pounce.

**Warm starts.** Sweeps and diagrams start each point from the previous
solution.

**First load.** The first visit downloads about 30 MB, which the browser then
caches. A single equilibrium takes a fraction of a second, and a 25 × 25
Pourbaix grid takes 10–30 s.

**Limitations.** Everything the Python API offers beyond the form (your own
reactions, solid solutions, exchangers, surfaces, fitting, optimization) needs
the Python package; see the [Quick start](index.md#quick-start).

**Stopping.** A long run can be stopped. The stop restarts Python, which takes
a few seconds.

## Building

The app's sources are in `docs/app/`:

- `index.html`, `app.js`, `app.css`: the page;
- `worker.js`: loads Python;
- `web_backend.py`: the JSON functions the page calls.

At `mkdocs build`, a hook in `docs_hooks.py` writes `site/app/py/`. It contains:
- a zip of discopt's Python sources, at the version in `uv.lock`;
- a zip of this package;
- the pounce wheel;
- a manifest.

Downloads are cached in `.cache/webapp/`.

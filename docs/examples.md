# Examples

Six Jupyter notebooks introduce the package, from a first equilibrium
calculation to rare-earth and thorium chemistry in brines. They are written as
tutorials for new users and are meant to be read in order; 01 is the
prerequisite for all the others.

Each notebook is shown here with its stored outputs. To run one yourself,
download it from GitHub and install the package with the `plot` extra
(`pip install "discopt-speciation[plot] @ git+https://github.com/jkitchin/discopt-speciation"`).

| # | notebook | what you will learn | source |
|---|----------|---------------------|--------|
| 01 | [Getting started](examples/01-getting-started.ipynb) | declaring species, reactions and solids; reading a `Result`; the operator, string and YAML forms | [`.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/01-getting-started.ipynb) |
| 02 | [Validation](examples/02-validation.ipynb) | the Laliwala et al. (2025) cases, and five independent ways to check an answer: residuals, PHREEQC, alternative algorithms, invariance, Gibbs energy | [`.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/02-validation.ipynb) |
| 03 | [Databases and your own data](examples/03-databases-and-your-own-data.ipynb) | loading the bundled PHREEQC databases, overriding and adding data, writing a small database, YAML specs, choosing an activity model | [`.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/03-databases-and-your-own-data.ipynb) |
| 04 | [Fitting thermodynamic data](examples/04-fitting-thermodynamic-data.ipynb) | estimating `logK`, `dH` and activity parameters from solubility and pH data, with standard errors, correlations and profile likelihood | [`.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/04-fitting-thermodynamic-data.ipynb) |
| 05 | [Diagrams](examples/05-diagrams.ipynb) | speciation sweeps, solubility curves, Pourbaix and Ellingham diagrams | [`.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/05-diagrams.ipynb) |
| 06 | [Rare earths and thorium](examples/06-rare-earths-and-thorium.ipynb) | ThO₂ polymorphs, rare-earth phosphates, Th in brines with SIT, temperature effects, and adding minerals missing from the databases | [`.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/06-rare-earths-and-thorium.ipynb) |

**Rebuilding the notebooks.** The notebooks are generated and executed by
[`examples/build_notebooks.py`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/build_notebooks.py):

```bash
uv run python examples/build_notebooks.py        # all notebooks
uv run python examples/build_notebooks.py 02     # just one
```

Edit the text in that script rather than in the `.ipynb` files, so a rebuild
keeps your changes.

**More examples.** The guides also contain short runnable examples:
[Diagrams](diagrams.md) (with figures), [Fitting](fitting.md),
[Stability](stability.md) and [Optimization](optimization.md).

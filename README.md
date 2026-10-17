<h1 align="center">
  <img src="docs/assets/logo.png" alt="discopt-speciation" width="480">
</h1>

[![tests](https://github.com/jkitchin/discopt-speciation/actions/workflows/tests.yml/badge.svg)](https://github.com/jkitchin/discopt-speciation/actions/workflows/tests.yml)
[![docs](https://github.com/jkitchin/discopt-speciation/actions/workflows/docs.yml/badge.svg)](https://github.com/jkitchin/discopt-speciation/actions/workflows/docs.yml)
[![documentation](https://img.shields.io/badge/documentation-online-blue)](https://kitchingroup.cheme.cmu.edu/discopt-speciation/)
[![web app](https://img.shields.io/badge/web%20app-try%20in%20browser-orange)](https://kitchingroup.cheme.cmu.edu/discopt-speciation/app/)

Chemical equilibrium for aqueous systems with solids, gases, solid solutions,
ion exchangers and sorbing surfaces, as a discopt plugin (`import discopt.speciation`).

**Solving.** You declare species and reactions with their equilibrium
constants, or load a PHREEQC database. `discopt.speciation` then finds the state
that satisfies mass action, conservation, and phase complementarity (every
candidate phase is either present and saturated, or absent and
undersaturated). It solves this with discopt's NLP solvers.

**Uniqueness.** The answer does not depend on how reactions are written or on
which inert or metastable phases are listed.

## Features

- **Temperature and pressure.** $\log K(T,P)$ from van 't Hoff with
  $\Delta H$, $\Delta C_p$ and $\Delta V$, or from PHREEQC analytic
  expressions. Above 1 atm, species molar volumes (PHREEQC `-Vm`) give the
  pressure dependence, and water properties follow T and P.
- **Activity models.**
  - ideal, Debye–Hückel, Davies
  - extended Debye–Hückel, Truesdell-Jones (PHREEQC), B-dot, LLNL B-dot
    with temperature tables ([`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat))
  - SIT (with the ThermoChimie ε table from [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat)), Pitzer (with E-θ and
    MacInnes scaling)
  - water activity consistent with each model
- **Phases.**
  - pure solids, with metastable polymorphs correctly excluded, or kept via
    `suppress=`
  - ideal solid solutions
  - gas phases at fixed pressure or volume, ideal or Peng–Robinson (with
    PHREEQC's critical constants, or your own)
  - fixed partial pressures and activities (pH, pe)
- **Surfaces and exchange.** Gaines–Thomas ion exchange; diffuse double layer
  surface complexation.
- **Solution setup.** Redox, and charge balance on a chosen ion.
- **Three solution methods:**
  - complementarity homotopy with an active-set polish (default)
  - Gibbs energy minimization
  - exhaustive assemblage enumeration
- **Bundled PHREEQC databases.** [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat), [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat),
  [`wateq4f_PWN.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat), [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) (rare earths, many elements, to 300 °C) and
  [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) (ThermoChimie: thorium, actinides, SIT), with a parser for any
  PHREEQC-format file.
- **Your own data.** Supply it through the Python DSL, strict YAML/JSON
  specifications, or overrides and additions to a database.
- **Parameter fitting.** Fit `logK`, `ΔH` and activity parameters to
  measurements, with linearized and profile-likelihood confidence intervals
  and predictions with uncertainty.
- **Diagrams.** Sweeps, predominance and Pourbaix diagrams, Ellingham plots.
- **Validation.** Checked against PHREEQC; see [docs/validation.md](docs/validation.md).

## Install

From GitHub:

```bash
pip install "discopt-speciation @ git+https://github.com/jkitchin/discopt-speciation"
pip install "discopt-speciation[plot] @ git+https://github.com/jkitchin/discopt-speciation"  # matplotlib, for diagrams
```

From a clone, for development:

```bash
pip install -e ".[plot]"    # installs discopt from PyPI
```

Python 3.10 or later is required.

## Quick start

**No install:** try the [web app](https://kitchingroup.cheme.cmu.edu/discopt-speciation/app/).
It runs the package in your browser (WebAssembly, via Pyodide and pounce). It
computes speciation, sweeps, Pourbaix diagrams and log K(T).

```python
from discopt.speciation import PhreeqcDatabase

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Ca", "C(4)"], phases=["Calcite", "Aragonite", "CO2(g)"])
r = s.equilibrate({"Calcite": 1.0}, fixed={"CO2(g)": -3.5})
print(r.present, round(r.pH, 3))        # ['Calcite'] 8.279
```

Gases at high pressure (Peng–Robinson CO₂ and H₂O(g), log K at the gas
pressure from species molar volumes; see [docs/gases-and-pressure.md](docs/gases-and-pressure.md)):

```python
s = db.system(["C(4)"], gas_phase={"members": ["CO2(g)", "H2O(g)"], "pressure": 100.0})
r = s.equilibrate({"CO2(g)": 50.0}, T=323.15)
print(r.pressure, round(r.molality["CO2"], 3), round(r.fugacity_coefficient["CO2(g)"], 3))
# 100.0 1.041 0.614
```

Your own reactions:

```python
from discopt.speciation import System

s = System("gypsum", activity="davies")
Ca, SO4, CaSO4 = s.aqueous("Ca+2"), s.aqueous("SO4-2"), s.aqueous("CaSO4")
s.reaction(Ca + SO4 >> CaSO4, logK=2.30, dH=5500.0)
s.reaction(s.solid("Gypsum") >> Ca + SO4, logK=-4.58, dH=-460.0, name="Gypsum")
print(s.equilibrate({"Gypsum": 1.0}, T=313.15).summary())
```

Fitting constants to data:

```python
from discopt.speciation import fit

s.aqueous("Na+")                         # every input must be a declared species
obs = [fit.Observation({"amounts": {"Gypsum": 1.0, "Na+": 2 * c, "SO4-2": c}},
                       fit.total("Ca+2", "CaSO4"), y, sigma=0.02 * y) for c, y in data]
res = fit.fit_equilibrium(s, obs, [fit.logK(s, "Gypsum"), fit.logK(s, "Ca+2 + SO4-2 = CaSO4")],
                          absolute_sigma=True)
print(res.summary())                     # values, standard errors, 95% intervals, correlations
res.profile("logK Gypsum").plot()        # profile likelihood
```

## Claude Code skill

The package includes a [Claude Code](https://claude.com/claude-code) skill.
Install it and you can ask equilibrium questions in plain language.

```bash
discopt-speciation-install-skill            # ~/.claude (all projects)
discopt-speciation-install-skill --project  # ./.claude (this repository)
discopt-speciation-install-skill --force    # overwrite after upgrading
```

The installer adds three things:
- the `/discopt-speciation` skill;
- five tested example scripts, which the skill copies from;
- a `speciation-expert` agent for diagnosing difficult cases.

See [docs/agent-skill.md](docs/agent-skill.md).

## Documentation

The documentation is published at
<https://kitchingroup.cheme.cmu.edu/discopt-speciation/>. Its sources are:

| | |
|-|-|
| [docs/concepts.md](docs/concepts.md) | equilibrium conditions, methods, units |
| [docs/databases.md](docs/databases.md) | bundled PHREEQC databases |
| [docs/custom-data.md](docs/custom-data.md) | your own species, constants, activity parameters, YAML, database files |
| [docs/activity-models.md](docs/activity-models.md) | activity models and their parameters |
| [docs/gases-and-pressure.md](docs/gases-and-pressure.md) | Peng–Robinson gases and pressure-dependent log K |
| [docs/stability.md](docs/stability.md) | certified global phase stability |
| [docs/fitting.md](docs/fitting.md) | parameter estimation with confidence intervals |
| [docs/optimization.md](docs/optimization.md) | equilibrium-constrained design and worst-case bounds |
| [docs/diagrams.md](docs/diagrams.md) | sweeps, predominance, Pourbaix, Ellingham |
| [docs/agent-skill.md](docs/agent-skill.md) | the Claude Code skill, agent and example scripts |
| [docs/examples.md](docs/examples.md) | guide to the example notebooks |
| [docs/validation.md](docs/validation.md) | comparison with PHREEQC |
| [docs/references.md](docs/references.md) | citations for the databases, models and methods |

Notebooks in [examples/](examples/):

| notebook | topic |
|----------|-------|
| 01 | getting started |
| 02 | the Laliwala et al. (2025) mass-action examples, with evidence of correctness |
| 03 | databases and your own data |
| 04 | fitting with confidence intervals |
| 05 | diagrams |
| 06 | rare earths and thorium |

Rebuild the notebooks with `python examples/build_notebooks.py [03 04 ...]`.

## Tests

```bash
python -m pytest -m "not slow"   # fast suite
python -m pytest                 # everything, including slow solves
```

## Data attribution

**Bundled databases.** [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat), [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat) and [`wateq4f_PWN.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat) are
distributed with PHREEQC by the U.S. Geological Survey, and are in the public
domain. They were copied from phreeqc-python (Apache-2.0).

**Citation.** When you use them, cite:

- Parkhurst, D.L., and Appelo, C.A.J., 2013, *Description of input and examples
  for PHREEQC version 3*, USGS Techniques and Methods 6-A43.
- The original sources of each database, as listed in
  [src/discopt/speciation/data/README.md](src/discopt/speciation/data/README.md).
- The activity model you used.

**Other databases.** [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) (Delany and Lundeen 1991) and [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat)
(ThermoChimie; Giffaut et al. 2014) are the copies distributed with PHREEQC 3.
ThermoChimie data are subject to the ThermoChimie terms of use,
<https://www.thermochimie-tdb.com>.

**References.** [docs/references.md](docs/references.md) lists every database,
model, method and paper the package uses, with DOIs and links. The validation
cases come from Laliwala et al. (2025), *Systems and Control
Transactions* 4, 2140–2145, <https://doi.org/10.69997/sct.132742>. The solver is
[discopt](https://github.com/jkitchin/discopt).

## License

MIT; see [LICENSE](LICENSE). The bundled databases keep their own terms
(see [Data attribution](#data-attribution)).

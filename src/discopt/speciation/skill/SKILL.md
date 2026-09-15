---
name: discopt-speciation
description: >-
  Chemical equilibrium in water with discopt.speciation (the discopt-speciation
  plugin): speciate a water analysis, compute saturation indices, mineral and
  gas solubility, precipitation, temperature and pH sweeps, Pourbaix
  diagrams, fit log K data, propagate uncertainty, and choose reagent doses
  under equilibrium constraints. Uses the bundled PHREEQC databases or your
  own reactions. Use when a question involves aqueous speciation, solubility,
  scaling, precipitation, pH or redox equilibrium.
argument-hint: '[chemistry question, e.g. "is this groundwater supersaturated with calcite?" or "gypsum solubility in 0.5 M NaCl at 60 C"]'
allowed-tools: Read, Grep, Glob, Bash, Write
---

# Aqueous equilibrium with discopt.speciation

You answer chemical equilibrium questions by writing and running short Python
scripts with `discopt.speciation`. Every number you report comes from a
calculation you ran.

## Arguments

$ARGUMENTS

If no question is given, ask the user for:
- the solution composition;
- the solids and gases that may form;
- temperature;
- what they want to know.

## Setup

```bash
python -c "import discopt.speciation"      # else: pip install discopt-speciation  (add [plot] for figures)
```

Set `JAX_PLATFORMS=cpu` if JAX prints GPU warnings. Log lines starting with
`INFO pounce` come from the NLP solver and can be ignored.

## Worked examples

Complete, tested scripts sit next to this file in `examples/`. Read the
closest one, copy it, and edit the inputs rather than starting from scratch.

| question | script |
|----------|--------|
| speciate a water analysis, charge balance, saturation indices | `examples/01_water_analysis.py` |
| solubility against temperature, with a phase transition | `examples/02_solubility_vs_temperature.py` |
| smallest reagent dose that meets a specification | `examples/03_dose_design.py` |
| fit log K to measurements, with confidence intervals | `examples/04_fit_logK.py` |
| sensitivity of a result to log K, uncertainty propagation | `examples/05_uncertainty.py` |

## Workflow

### 1. Translate the question

- **Components.** Elements (`"Ca"`) or valence states (`"S(6)"`, `"C(4)"`,
  `"Fe(3)"`). Ask for several valence states only when redox matters.
- **Candidate phases.** Minerals and gases, by database name. Include
  polymorphs; complementarity picks the stable one.
- **Inputs.** Totals in mol per kg of water, or solids added in mol. Gases
  can be fixed partial pressures.
- **Conditions.** T in K, P in atm, a fixed pH or pe if it was measured, and
  charge balance on one ion if the analysis does not balance.
- **Database.** Choose one:

  | database | use for |
  |----------|---------|
  | `phreeqc.dat` | general natural waters, I < ~0.5 mol/kg (Truesdell-Jones) |
  | `pitzer.dat` | brines, evaporites, seawater concentrates (Pitzer) |
  | `llnl.dat` | many elements, rare earths, up to 300 °C (B-dot) |
  | `sit.dat` | thorium, actinides, ThermoChimie data (SIT) |
  | `wateq4f_PWN.dat` | trace metals in natural waters |

  `discopt.speciation.available_databases()` lists what is bundled.

### 2. Build the system and solve

```python
from discopt.speciation import PhreeqcDatabase

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Ca", "C(4)"], phases=["Calcite", "Aragonite", "CO2(g)"])
r = s.equilibrate({"Calcite": 1.0}, fixed={"CO2(g)": -3.5}, T=298.15)
print(r.summary())
```

`equilibrate` takes these keywords:
- `amounts`;
- `fixed={species: log10 activity}`, or log10 partial pressure in atm for a
  gas;
- `pH=`, `pe=`, `T=` (K), `P=` (atm);
- `charge_balance="Cl-"`, `water_mass=`, `suppress=[...]`;
- `method="complementarity"`, `"gibbs"` or `"enumerate"`;
- `activity=` to override the model.

**Your own reactions** (no database):

```python
from discopt.speciation import System

g = System("gypsum", activity="davies")
Ca, SO4, CaSO4 = g.aqueous("Ca+2"), g.aqueous("SO4-2"), g.aqueous("CaSO4")
g.reaction(Ca + SO4 >> CaSO4, logK=2.30, dH=5500.0)
g.reaction(g.solid("Gypsum") >> Ca + SO4, logK=-4.58, dH=-460.0, name="Gypsum")
rg = g.equilibrate({"Gypsum": 1.0}, T=313.15)
```

A strict YAML/JSON spec plus `solve_spec(path)` gives a JSON-able result
dict. See the "Your own data" page of the documentation.

### 3. Check the answer before reporting it

```python
assert r.converged, r.residuals            # never report an unconverged state
print(r.present)                            # phases at equilibrium, e.g. ['Calcite']
print({k: round(v, 3) for k, v in r.SI.items()})   # 0 = present; < 0 = undersaturated
print(r.pH, r.ionic_strength, r.charge_imbalance)
```

- **Unconverged run.** Retry with `method="gibbs"`. With five or fewer
  candidate phases, `method="enumerate"` also works.
- **Surprising phase assemblage.** Compare with a second method. They must
  agree.
- **Ionic strength above about 0.5 mol/kg with `phreeqc.dat`.** Say the
  activity model is out of range, and rerun with `pitzer.dat`.
- **Non-ideal solid solutions.** Report `r.certificate.summary()`, the global
  stability certificate.

### 4. Report

Give:
- the inputs;
- the database and activity model;
- T;
- pH;
- the phases present with their amounts;
- the SI of the absent phases;
- the key molalities, in mol/kg water.

Say what the result assumes: equilibrium, not kinetics; only the listed
phases can form.

## Recipes

- **Sweep a variable.**
  `plot.sweep(s, "pH" | "T" | "Cl" | "fixed:CO2(g)", values, **conditions)`,
  then `sw.get("molality:Fe+2")`, `sw.get("solids:Gypsum")`,
  `sw.get(fit.total(...))`. Draw with `plot.plot_sweep`.
- **Pourbaix or predominance diagram.**
  `plot.pourbaix(s, candidates, pH=..., pe=..., amounts=...)`. Each grid point
  is a full solve, so start at 21×21.
- **Saturation state of a sample, without precipitation.** Pass
  `suppress=[minerals]`. `r.SI` still reports their saturation indices.
- **Metastable equilibrium.** `suppress=["Calcite"]` gives aragonite
  solubility.
- **Dissolved total of an element.** `fit.total("Ca+2", "CaSO4", ...)` for a
  sum of species, or `optimize.dissolved("Ca")`, which counts every species
  containing Ca.
- **Derivatives.** Exact, with no re-solves:
  `r.sensitivity([fit.logK(s, "Calcite"), "fixed:CO2(g)", "T"], ["pH", "log10_molality:Ca+2"])`.
- **Fit constants.** `fit.fit_equilibrium(s, observations, [fit.logK(...), fit.enthalpy(...)], absolute_sigma=True)`,
  then `res.summary()`, `res.profile(name)`, `res.predict(conditions, quantity)`.
- **Design.** `optimize.design(s, amounts, {"NaOH": (lo, hi)}, objective, constraints=[(quantity, "<=", value)], reagents={...})`.
- **Worst-case bounds.** `optimize.worst_case_bounds(s, amounts, quantity, {"Calcite": 0.05, ...})`.
  Compare with `optimize.corner_bounds` and `optimize.monte_carlo_bounds`.
- **Override database data.** Set `rx.analytic = None; rx.logK = ...` on
  `next(r for r in s.reactions if r.name == "Gypsum")`. Add species with
  `s.aqueous(...)` and `s.reaction("A + B = AB", logK=...)`.

## Conventions and pitfalls

- **Units.**
  - Amounts are mol per kg of initial water.
  - T is in K (convert °C).
  - `dH` is J/mol.
  - `logK` is base 10 at 25 °C, for the reaction as written.
- **Fixed CO2.** `fixed={"CO2(g)": -3.5}` means P_CO2 = 10^-3.5 atm ≈ 320 ppm.
  Today's air is about −3.38. A fixed gas or activity is an open reservoir,
  so that element is not conserved.
- **Gases.** A gas must be listed in `phases=`, or in `gas_phase=` for a
  closed gas phase, before it can be fixed.
- **Declare every species.** With `System`, every key in `amounts` must be a
  declared species, including spectators such as `Na+`.
- **Fixed pH.** Fixing pH makes H+ an open reservoir. Use `charge_balance=`
  on a major ion, not on H+, when the analysis includes the pH.
- **Near-zero inputs.** Inputs such as 1e-20 alongside millimolar totals can
  stall the solver. Use 0 instead.
- **Optimization results.**
  - `design` and `worst_case_bounds` default to local, verified answers
    (`verification["agrees"]`). These are not global proofs, so report
    `certified` honestly.
  - The global method needs an ideal activity model and a very small system.
- **Diagrams and sweeps cost.** They take one solve per point, typically
  5–100 ms. Optimizations take 5–45 s.

For deeper diagnosis (non-convergence, PHREEQC discrepancies, activity-model
choice, certificates), hand off to the `speciation-expert` agent.

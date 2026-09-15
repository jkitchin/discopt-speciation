# Claude Code skill

discopt-speciation ships an agent skill for [Claude Code](https://claude.com/claude-code).
With it installed, you can ask equilibrium questions in plain language and
Claude answers them the way you would. It writes a short `discopt.speciation`
script, runs it, checks convergence and saturation indices, and reports the
numbers along with the assumptions behind them.

## What is installed

| file | installed to | role |
|------|--------------|------|
| [`skill/SKILL.md`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/skill/SKILL.md) | `.claude/skills/discopt-speciation/SKILL.md` | the `/discopt-speciation` skill: workflow, API recipes, units, pitfalls |
| [`skill/examples/*.py`](https://github.com/jkitchin/discopt-speciation/tree/main/src/discopt/speciation/skill/examples) | `.claude/skills/discopt-speciation/examples/` | five tested scripts the skill copies from |
| [`skill/agents/speciation-expert.md`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/skill/agents/speciation-expert.md) | `.claude/agents/speciation-expert.md` | a subagent for diagnosis: non-convergence, database and activity-model choice, PHREEQC discrepancies, certificates |

The skill points to the examples by relative path, so Claude reads the
closest example and edits it instead of writing a script from memory.

## Install

```bash
pip install discopt-speciation
discopt-speciation-install-skill            # for your user: ~/.claude
discopt-speciation-install-skill --project  # for one repository: ./.claude (commit it)
discopt-speciation-install-skill --force    # overwrite after upgrading the package
```

The installer leaves existing files alone unless you pass `--force`, so any
local edits you make to the skill are safe. Restart Claude Code, or start a
new session, to pick up the skill.

## Using it

Call it directly with a question:

```text
/discopt-speciation is this groundwater supersaturated with calcite? Ca 2 mM, Mg 0.8 mM, Na 3 mM, Cl 3 mM, SO4 1 mM, alkalinity 4.5 mM, pH 7.6
```

You can also just ask. Claude loads the skill when a question is about
speciation, solubility, scaling or precipitation:

- "How does gypsum solubility change from 25 to 95 °C, and where does anhydrite take over?"
- "What is the smallest NaOH dose that brings dissolved aluminium below 1 µM in 1 mM AlCl3?"
- "Fit log K for gypsum and CaSO4(aq) to these solubility data and give confidence intervals."
- "How uncertain is the calcite solubility at P_CO2 = 10^-3.5 atm, given ±0.02 in log K?"
- "Draw a Pourbaix diagram for iron at 1 µM."

For a result that looks wrong, ask for the `speciation-expert` agent ("use the
speciation-expert agent to find out why this does not converge").

The skill instructs Claude to:

- give units, the database and activity model, and the phases present with
  the saturation indices of those absent;
- never report an unconverged solve;
- say when the activity model is out of range (I above about 0.5 mol/kg with
  `phreeqc.dat`);
- report `certified=False` optimization results as local, not global,
  answers.

## Example scripts

Each script runs in a few seconds and is exercised by
[`tests/test_skill.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_skill.py).
The outputs below come from those runs.

### 1. Water analysis and saturation indices

[`01_water_analysis.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/skill/examples/01_water_analysis.py)
speciates a measured analysis, like a PHREEQC `SOLUTION` block. It uses
[`phreeqc.dat`](databases.md), fixes pH at 7.6, and charge-balances on Cl⁻.

The minerals are passed in `suppress=`, so nothing precipitates. The result
describes the sample as measured, and `r.SI` still reports each mineral's
saturation index.

```python
s = db.system(["Ca", "Mg", "Na", "Cl", "S(6)", "C(4)"], phases=minerals)
r = s.equilibrate(analysis, pH=7.6, charge_balance="Cl-", suppress=minerals)
```

```text
pH 7.60   ionic strength 0.01148 mol/kg
Cl adjusted by -7.173e-04 mol/kg for electroneutrality
saturation indices:
  Calcite    +0.399
  Aragonite  +0.255
  Dolomite   +0.528
  Gypsum     -1.618
  Halite     -6.832
main species (mol/kg):
  HCO3-      4.169e-03
  Na+        2.986e-03
  Cl-        2.283e-03
  Ca+2       1.811e-03
  ...
```

The sample is supersaturated with the carbonates and undersaturated with
gypsum and halite.

### 2. Solubility against temperature

[`02_solubility_vs_temperature.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/skill/examples/02_solubility_vs_temperature.py)
dissolves gypsum in 0.1 mol/kg NaCl from 25 to 95 °C, with both gypsum and
anhydrite as candidate phases. Complementarity selects the stable one at each
temperature, so the phase change shows up without any special handling.

The dissolved calcium, in mmol/kg, is:

| T (°C) | 25 | 35 | 45 | 55 | 65 | 75 | 85 | 95 |
|--------|----|----|----|----|----|----|----|----|
| dissolved Ca | 21.41 | 21.87 | 22.00 | 21.52 | 18.25 | 15.44 | 13.03 | 11.00 |
| stable phase | Gypsum | Gypsum | Gypsum | Anhydrite | Anhydrite | Anhydrite | Anhydrite | Anhydrite |

See [Diagrams](diagrams.md#solubility-curves) for the full curve.

### 3. Reagent dose design

[`03_dose_design.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/skill/examples/03_dose_design.py)
finds the smallest NaOH dose that leaves at most 1 µmol/kg of dissolved
aluminium in 1 mmol/kg AlCl₃, with gibbsite as the precipitate. It calls
[`optimize.design`](optimization.md):

```python
res = O.design(s, {"Al": 1e-3, "Cl": 3e-3}, {"NaOH": (0.0, 6e-3)}, objective="NaOH",
               constraints=[(O.dissolved("Al"), "<=", 1e-6)],
               reagents={"NaOH": {"Na": 1, "OH-": 1}})
```

```text
status: local_optimum   certified: False
NaOH dose: 2.9818 mmol/kg
constraint: {'quantity': 'dissolved[Al]', 'op': '<=', 'value': 1e-06, 'achieved': 9.99999999292397e-07, 'satisfied': True}
pH at the optimum: 4.799
verified by equilibrate: True
```

The answer is a verified local optimum, not a certified global one. The
[Optimization guide](optimization.md) explains why.

### 4. Fitting log K to data

[`04_fit_logK.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/skill/examples/04_fit_logK.py)
fits log K for gypsum and CaSO₄(aq) to gypsum solubility in Na₂SO₄ solutions
(synthetic data with 2% noise). It also predicts a new condition, with a
standard error.

```text
chi2 = 3.27393, dof = 4, reduced chi2 = 0.8185 (sigma treated as absolute)
parameter                                value     std err   95% CI (linearized)
logK Gypsum                            -4.5844      0.0235   [-4.6305, -4.5383]
logK CaSO4                              2.3132      0.0465   [2.222, 2.4043]
correlation:
  logK Gypsum                     1.000 -0.987
  logK CaSO4                     -0.987  1.000
predicted dissolved Ca in 0.15 mol/kg Na2SO4: 7.442 ± 0.189 mmol/kg
```

The strong negative correlation shows that solubility data alone barely
separate the two constants. See [Fitting](fitting.md).

### 5. Sensitivity and uncertainty

[`05_uncertainty.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/skill/examples/05_uncertainty.py)
solves calcite in equilibrium with P_CO₂ = 10⁻³·⁵ atm, then estimates how
uncertain the result is in three steps:

1. It takes exact derivatives with respect to three log K values from
   `Result.sensitivity`.
2. It combines them into a linear 1σ estimate.
3. It checks that estimate against the extreme corners of a ±2σ box.

```text
pH 8.279, dissolved Ca 0.4934 mmol/kg
d(dissolved Ca, mmol/kg) / d(log K):
  Calcite  +0.4106
  CaHCO3+  +0.0076
  CaCO3    +0.0128
linear 1σ: ± 0.0082 mmol/kg
±2σ corners: [0.4755, 0.5126] mmol/kg  (an inner estimate, not a bound)
```

Almost all of the uncertainty comes from the calcite solubility product.

## Keeping the skill correct

`tests/test_skill.py` guards the skill against drifting away from the API. It
checks that:

- the installer writes every file and respects `--force`;
- the frontmatter is valid;
- every example that SKILL.md mentions exists;
- the Python blocks in SKILL.md run;
- each example script runs.

When you change the API, the tests fail until the skill is updated too.

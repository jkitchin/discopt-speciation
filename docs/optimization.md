# Optimization

`discopt.speciation.optimize` puts the equilibrium conditions (mass action,
conservation, complementarity; see [Concepts](concepts.md)) into one discopt
model and promotes some inputs to variables. Two problems use it:

- **design**: choose doses, pH or other inputs to optimize a quantity at
  equilibrium, subject to specifications;
- **worst-case bounds**: the smallest and largest value of a quantity when
  thermodynamic data lie anywhere in a box.

## What can vary

| key | parameter | enters the model as |
|-----|-----------|---------------------|
| a reagent name in `reagents`, or a species | input amount (mol) | $b = b_0 + A\,e\,d$ (linear in the dose) |
| a reaction name, or a phase / product species name | $\log_{10} K$ at 25 °C | a shift of $\log_{10}K(T)$ by the same amount |
| `"pH"`, `"pe"`, or `O.LogActivity("CO2(g)")` for a species also given in `fixed=` | fixed log activity | the fixed value |

A range is `(lo, hi)`. For a $\log_{10}K$ a single number `u` also works and
means nominal $\pm u$. Amounts, pH and fixed activities need `(lo, hi)`.
`O.Amount(name, composition)` and `O.LogK(reaction)` can be used as keys
instead of strings.

## Quantities

`dissolved(el)`, `log10_dissolved(el)`, `precipitated(el)`,
`dissolved_fraction(el)`, `precipitated_fraction(el)`, `molality(sp)`,
`log10_molality(sp)`, `log10_activity(sp)`, `pH()`, `solid(name)`,
`saturation_index(name)`, `parameter(label)`.

An element is counted in every species and phase through its formula, or
through its reaction for names without a formula, such as `"Gibbsite"`. A dict
`{"M+": 1, "ML2-": 1}` gives explicit weights. A string objective that names a
parameter means `parameter(...)`.

## Design

```python
from discopt.speciation import PhreeqcDatabase, optimize as O

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Al", "Na", "Cl"], phases=["Gibbsite"], activity="ideal")
r = O.design(s, {"Al": 1e-3, "Cl": 3e-3}, {"NaOH": (0, 6e-3)}, objective="NaOH",
             constraints=[(O.dissolved("Al"), "<=", 1e-6)],
             reagents={"NaOH": {"Na": 1, "OH-": 1}})
r.status, r.decisions, r.verification["agrees"]
# ('local_optimum', {'NaOH': 0.0029818}, True)
```

The call returns an `OptimizationResult` with these fields:

- `status`, `certified`, `decisions`, `objective`;
- `bound` (proven only), `conditional_bound` (valid only under `assumptions`), `gap`;
- `constraints`, a list with each specification and its achieved value;
- `result`, a `Result` from `System.equilibrate` at the optimum;
- `verification`, `assumptions`, `wall_time`, `info`.

## Worst-case bounds

```python
b = O.worst_case_bounds(s, amounts, O.log10_dissolved("Th"),
                        {"ThO2(fresh)": 0.6, "Th(CO3)5-6": 0.3, ...}, pH=8)
b.lower, b.upper, b.argmin, b.argmax, b.certified
```

`b.minimize` and `b.maximize` are the two `OptimizationResult`s.

Two baselines use the same keys: `corner_bounds` (all $2^n$ corners) and
`monte_carlo_bounds` (uniform samples). Both are *inner* estimates. They can
miss an interior extreme, or a corner when $n$ is large. The tests contain
examples of each.

## Methods

**`method="local"`** (the default for `design`) uses multi-start. Each start
runs four steps:

1. an ordinary `equilibrate` at a parameter point;
2. a smoothed Fischer–Burmeister homotopy with the objective and constraints;
3. one smooth NLP per phase assemblage (present phases saturated with
   $N \ge 0$; absent phases with $N = 0$ and $\mathrm{SI} \le 0$);
4. phase flips at a boundary while they improve the objective.

The answer is a KKT point of one assemblage. It is not a proof: other starts
can do better, and `no_feasible_point_found` does not prove infeasibility.

**`method="global"`** (the default for `worst_case_bounds`) runs four steps:

1. Complementarity becomes a disjunction (`discopt.mpec.reformulate_gdp`).
2. Each saturation index is lifted into a variable with an interval enclosure.
3. The disjunction is handed to discopt's spatial branch and bound, warm-started
   from the local answer.
4. Variable bounds are tightened first by linear programs over the
   conservation laws.

**Verification.** Every answer, local or global, is recomputed with
`System.equilibrate` at the optimal parameters. `verification["agrees"]`
requires two things: the objective matches, and the assemblage matches (phases
with non-negligible amounts only).

## What "certified" means

`certified=True` requires all three of the following:

1. discopt returned status `optimal` with a certified gap within `gap_tolerance`;
2. every variable bound in the model is rigorous:
   - upper bounds on amounts come from the LP $\max n_j$ s.t. $A n = b(d)$, $n \ge 0$;
   - bounds on water mass, phase amounts and ionic strength follow from those;
   - saturation-index enclosures come from interval arithmetic on mass action;
3. the model is the one being claimed: an ideal activity model, no surfaces,
   and no non-ideal solid solutions.

Then `bound` is a proven bound on the model's optimum.

**Log-molality lower bounds.** No finite lower bound on $\log_{10} m$ is
rigorous. The default is −200, which makes relaxations useless. Passing
`log_span=12` sets $\log_{10} m \ge \text{ub} - 12$. This helps branch and
bound, but it is an **assumption**: it is listed in `assumptions`, and the
result is never certified. Its dual bound is reported as `conditional_bound`.

**Scope of the proof.** A proof concerns the model at the solver's feasibility
tolerance. If `verification["agrees"]` is false, `info["warning"]` says so.

## Limits

**Global.** Certification is realistic only for very small models, a handful of
species with one phase. It is not realistic for database systems:

- On the 5-species toy system, discopt's bound stayed at the $\log_{10} m$
  floor for 60–90 s (6–18 thousand nodes), with gap ≈ 1–1.8. This held even
  with ±0.5 bounds around the known solution.
- Proving infeasibility is also out of reach. For the Al/Gibbsite example
  with the NaOH dose limited to ≤ 1e-3 mol, no dose meets the target. The
  global run stopped at its 60 s limit (about 2100 nodes) and reported
  `no_feasible_point_found`, not `infeasible`.
- The saturation-index enclosures that make the big-M reformulation exact
  are rigorous but wide. For Gibbsite the enclosure is about [−593, 269],
  because nothing bounds $\log_{10} m$ from below.
- Conservation alone can let water convert entirely to H⁺, so the minimum
  water mass is only its floor. Upper bounds on $\log_{10} m$ are then loose
  (for example 3.0 for H⁺ in a millimolar system).
- Most results are therefore `local_optimum` or `feasible` with a
  `conditional_bound`.

**Local.** A local call costs a few `equilibrate` calls plus a few NLPs:

- 5–15 s for the Al/NaOH and Th/Eu design examples;
- a bounds call is two such optimizations.

**Activity models.** The global path raises `NotImplementedError` for
non-ideal activity models (no rigorous enclosures of $\gamma$), surfaces and
non-ideal solid solutions. The local path supports everything that
`equilibrate` supports.

**Baselines.** Corners are limited to `max_dim=12` parameters by default.
Monte Carlo reports its sample count; neither is a bound.

## Assumptions behind the bounds

- The thermodynamic model is exact inside the box: the uncertainty is in
  $\log_{10}K$ at 25 °C, and the temperature dependence shifts rigidly.
- The parameters are independent and range over the whole box. Correlated
  data, such as a common reference reaction, make the box conservative.
- Only the listed phases can form. A bound says nothing about solids that
  were not given to `System`.
- Any `log_span` or other non-rigorous bound is listed in `assumptions`.

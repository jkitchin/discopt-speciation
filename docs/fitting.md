# Fitting thermodynamic data to measurements

`discopt.speciation.fit` estimates constants such as `logK`, `dH` and activity
parameters from equilibrium measurements, and reports how well the data
determine them.

## The idea

**Each observation.** An observation is a set of experimental conditions
(input amounts, pH, pCO₂, T, ...) plus a measured quantity with a standard
deviation. For each one:

- the model value is a **full equilibrium calculation**, so data points on
  either side of a solubility boundary are handled correctly;
- the solid is present in some experiments and absent in others without any
  special treatment.

**Objective.** The fit minimizes
$\chi^2(\theta)=\sum_k \left(\dfrac{y_k^{\text{model}}(\theta)-y_k}{\sigma_k}\right)^2$
with bounded trust-region least squares (`scipy.optimize.least_squares`).

**Sensitivities.** Sensitivities $J_{kj} = \partial y_k/\partial\theta_j$ are
exact derivatives at each converged equilibrium (implicit function theorem,
see [below](#sensitivities)), so they cost no extra solves.

## Example: calcite solubility and CaHCO₃⁺ stability from pH and dissolved Ca

Calcite is equilibrated with water at eight CO₂ partial pressures. At each
one, pH and total dissolved Ca are measured:

| log₁₀ P(CO₂) (atm) | pH | dissolved Ca (mol/kg) |
|---:|---:|---:|
| −4.0 | 8.616 | 3.494 × 10⁻⁴ |
| −3.5 | 8.291 | 4.837 × 10⁻⁴ |
| −3.0 | 7.938 | 7.311 × 10⁻⁴ |
| −2.5 | 7.633 | 1.103 × 10⁻³ |
| −2.0 | 7.315 | 1.670 × 10⁻³ |
| −1.5 | 6.977 | 2.466 × 10⁻³ |
| −1.0 | 6.635 | 3.954 × 10⁻³ |
| −0.5 | 6.324 | 6.050 × 10⁻³ |

These data are synthetic, so the right answer is known. They were computed
with the [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat)
values ($\log K_{\text{Calcite}} = -8.480$, $\log K_{\text{CaHCO}_3^+} = 11.435$),
then random noise was added: σ = 0.01 in pH and 2 % in Ca (seed 2024).

```python
from discopt.speciation import PhreeqcDatabase, fit

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])

measurements = [            # (log10 P_CO2, pH, dissolved Ca in mol/kg)
    (-4.0, 8.616, 3.494e-4),
    (-3.5, 8.291, 4.837e-4),
    (-3.0, 7.938, 7.311e-4),
    (-2.5, 7.633, 1.103e-3),
    (-2.0, 7.315, 1.670e-3),
    (-1.5, 6.977, 2.466e-3),
    (-1.0, 6.635, 3.954e-3),
    (-0.5, 6.324, 6.050e-3),
]

dissolved_ca = fit.total("Ca+2", "CaHCO3+", "CaCO3", "CaOH+")
obs = []
for log_pco2, pH, ca in measurements:
    cond = {"amounts": {"Calcite": 1.0}, "fixed": {"CO2(g)": log_pco2}}
    obs.append(fit.Observation(cond, "pH", pH, sigma=0.01))
    obs.append(fit.Observation(cond, dissolved_ca, ca, sigma=0.02 * ca))

params = [
    fit.logK(s, "Calcite", lower=-10, upper=-6),
    fit.logK(s, "CaHCO3+", lower=8, upper=14),
]
res = fit.fit_equilibrium(s, obs, params, absolute_sigma=True)
print(res.summary())
```

```text
fit: `ftol` termination condition is satisfied. (success=True, 6 model evaluations)
chi2 = 15.0712, dof = 14, reduced chi2 = 1.077 (sigma treated as absolute)
parameter                                value     std err   95% CI (linearized)
logK Calcite                           -8.4711      0.0095   [-8.4897, -8.4525]
logK CaHCO3+                            11.471        0.15   [11.178, 11.764]
correlation:
  logK Calcite                    1.000 -0.734
  logK CaHCO3+                   -0.734  1.000
```

Both 95 % intervals contain the true values, and reduced χ² ≈ 1 shows that the
assumed σ match the added noise. `logK Calcite` is determined 16 times more
precisely than `logK CaHCO3+`: CaHCO₃⁺ is a minor species at these conditions,
from 0.7 % of dissolved Ca at the lowest P(CO₂) to 8 % at the highest.

On return the system holds the best-fit values, so it can be used directly for
predictions.

## Parameters

| helper | fits | notes |
|--------|------|-------|
| `fit.logK(system, reaction)` | $\log_{10}K$ at 25 °C | For `-analytic` reactions, $A_1$ is shifted so the T-dependence is kept. |
| `fit.enthalpy(system, reaction)` | $\Delta H$ in kJ/mol | Needs data at several temperatures. |
| `fit.activity_parameter(system, species, "a0")` | per-species activity parameter | |
| `fit.Parameter(name, set, get, lower, upper)` | anything | E.g. a Pitzer coefficient, a site density or a surface area. |

**Choosing the reaction.** `reaction` is a `Reaction`, its `name` (database
reactions are named after the species or phase) or its equation string.

A custom parameter, for example a Pitzer $\beta^{(0)}$:

```python
pp = s.activity.params                                  # PitzerParameters
key = ("Cl-", "Na+")
def set_b0(v):
    c = list(pp.data["B0"][key]); c[0] = v; pp.data["B0"][key] = tuple(c)
b0 = fit.Parameter("B0 NaCl", set_b0, lambda: pp.data["B0"][key][0], lower=0.0, upper=0.2)
```

## Observations

`fit.Observation(conditions, quantity, value, sigma=1.0, label=None)`.

**Conditions.** `conditions` holds the keyword arguments of
`System.equilibrate`, with the input amounts under `"amounts"`.

**Quantities.** `quantity` is one of:

- `"pH"`, `"pe"`, `"ionic_strength"` or `"water_mass"`;
- `"molality:Ca+2"`, `"log10_molality:Ca+2"`, `"log10_activity:H2O"` or `"gamma:Na+"`;
- `"SI:Gypsum"`, `"solids:Calcite"` (moles, 0 when absent), `"partial_pressure:CO2(g)"`,
  `"log10_partial_pressure:CO2(g)"`, `"exchange:CaX2"` or `"surface:Hfo_wOZn+"`;
- `fit.total("Ca+2", "CaSO4", ...)`, a sum of molalities with optional weights;
- any callable `f(result) -> float`, e.g. a titration alkalinity or a
  conductivity model.

**Weighting.** Choose `sigma` in the units of the quantity. A 2 % relative
error is `sigma=0.02*value`. For concentrations spanning decades, fitting
`log10_molality` with constant `sigma` is usually better than fitting
molalities.

## Sensitivities

`result.sensitivity(parameters, quantities)` returns derivatives of an
equilibrium state without re-solving:

```python
r = s.equilibrate({"Calcite": 1.0}, fixed={"CO2(g)": -3.5})
S = r.sensitivity([fit.logK(s, "Calcite"), "fixed:CO2(g)", "total:Calcite"],
                  ["pH", "log10_molality:Ca+2", dissolved_ca])
S.jacobian           # quantities x parameters
S.ok, S.issues       # False / reasons when a derivative is not well defined
```

**Parameters.** A parameter is either a `fit.Parameter` or one of these specs:

- `"total:<amounts key>"`;
- `"fixed:<species>"`;
- `"pH"`, `"pe"` or `"T"`.

**Quantities.** Quantities are the same as for observations.

**How they are computed.** $dx/d\theta = -F_x^{-1}F_\theta$ at the converged
state (see [concepts](concepts.md#sensitivities)).

- $F_x$ is exact (JAX).
- $F_\theta$ differences the residuals of networks rebuilt at $\theta\pm h$,
  with no solves. The residuals are affine in `logK`, `dH`, fixed activities,
  pH and input totals, so these derivatives are exact to rounding.
- Nonlinear parameters (ion sizes, Pitzer coefficients, T, exchange and surface
  site totals) use a noise-free Richardson difference. `S.linear` says which
  parameters those were.

Because nothing is re-solved, derivatives stay accurate for trace species and
for parameters of very different sizes.

**When a derivative is not defined.** `S.ok` is false in these cases:

- **Saturation boundary.** A phase can be exactly at saturation with zero
  amount. The state then has a kink, and `S.boundary` names the phase.
  `S.left` and `S.right` hold the one-sided derivatives, and `S.jacobian` is
  NaN where they differ.
- **Singular or ill-conditioned $F_x$.** An example is two polymorphs with
  (nearly) the same `logK`, both present.

**In fitting.** `fit_equilibrium(..., jacobian="ift")` is the default, and
`predict` and `profile` use the same derivatives.

- If an observation's sensitivity is not `ok`, it falls back to central
  differences of re-solves (step `fd_step`). The fallbacks are listed in
  `res.jacobian_fallbacks`.
- `jacobian="fd"` restores pure finite differences.
- `res.n_solves` counts all equilibrium calculations.

## Uncertainty

### Linearized (Fisher information)

$\mathrm{Cov}(\hat\theta) = s^2 (J_w^\top J_w)^{-1}$ with $J_w = J/\sigma$. The
factor $s^2$ depends on the weighting:

- **`absolute_sigma=True`:** $s^2=1$. Use this when the σ are real
  measurement standard deviations. Intervals use the normal quantile.
  $\chi^2/\text{dof}$ should then be about 1; much larger means the model or
  the σ are wrong.
- **`absolute_sigma=False`** (default): $s^2=\chi^2/(n-p)$, and intervals use
  Student-t. This suits σ that are only relative weights.

`res.standard_errors`, `res.confidence_intervals`, `res.covariance` and
`res.correlation` hold the results. `res.predict(conditions, quantity)`
returns a prediction with its delta-method standard error.

### Profile likelihood

**When to use it.** Linearized intervals assume the model is linear in the
parameters near the optimum. They can mislead when:

- a parameter is weakly determined, for example a complex that is minor at
  every condition measured;
- a bound is active;
- data sit close to a phase boundary.

**How it works.** `res.profile(name)` fixes one parameter on a grid and
re-optimizes the others. It returns the $\chi^2$ profile and the [interval](references.md#venzon1988) where
$\chi^2 \le \chi^2_{\min}(1 + t^2/\text{dof})$, or
$\chi^2 \le \chi^2_{\min} + \chi^2_{1,\alpha}$ with absolute σ.

```python
pr = res.profile("logK CaHCO3+", n=21, span=4)
pr.interval          # (lower, upper); ±inf if the profile never crosses: not identifiable in that direction
pr.plot()
```

**Reading the result.**
- A profile that stays flat means the data [cannot bound that parameter](references.md#raue2009).
- Strongly asymmetric intervals mean the linearized ones are unreliable.

### Identifiability warnings

**Ill-conditioned sensitivities.** When the weighted sensitivity matrix is
ill-conditioned (condition number > 1e10), `fit_equilibrium` warns. Typical
causes:

- two parameters that only ever appear in combination, such as the `logK` of
  two species that are never distinguished by the data;
- a species that is negligible at all measured conditions.

**Strong correlation.** Otherwise, `fit_equilibrium` warns when two parameters
are correlated beyond |r| = 0.99. The fit is still valid, but each parameter's
individual interval is wide; a combination of them is well determined.

**What to do.**
- Fix one of the parameters, or add experiments that separate them.
- Look at `res.correlation` for values near ±1.
- For systematic design of informative experiments, see the `discopt-doe`
  plugin.

## Practical advice

- **Start from database values** and give physically sensible bounds.
  $\log K$ values are well scaled; `enthalpy` is in kJ/mol so its steps are
  comparable.
- **Temperature data:** fit `logK` and `dH` together only with data spanning at
  least ~30 K. Otherwise they are nearly collinear.
- **Cost:** each residual is an equilibrium solve, 5–100 ms depending on size.
  A fit makes about $n$ solves per iteration, plus a small Jacobian cost per
  observation. With `jacobian="fd"` it makes about $(2p+1)\times n$. A profile
  repeats a small fit at each grid point. Keep `n` modest while exploring.
- **Validate on held-out data:** remove a few observations, refit, and compare
  predictions with `res.predict`.

# Providing your own data and parameters

Every constant used in a calculation is a plain attribute you can set. There
are five ways to supply data, from quickest to most structured:

1. **Python DSL:** declare species and reactions directly.
2. **YAML or JSON specification:** the same content as a file, validated
   strictly.
3. **Database plus changes:** start from a [PHREEQC database](databases.md), then add or
   override.
4. **Your own PHREEQC-format file:** reuse existing `.dat` data.
5. **Fitting:** estimate constants from measurements (see [fitting.md](fitting.md)).

## 1. Species and reactions in Python

```python
from discopt.speciation import System

s = System("my system", activity="truesdell-jones")
Ca  = s.aqueous("Ca+2", a0=5.0, b=0.165)       # charge from the name; ion size a0 (Å) and b
SO4 = s.aqueous("SO4-2", a0=5.0, b=-0.04)
CaSO4 = s.aqueous("CaSO4")                     # neutral: log γ = b I, b defaults to 0.1
H2O = s.solvent("H2O")                         # water as a reacting species (optional)
gyp = s.solid("Gypsum")

s.reaction(Ca + SO4 >> CaSO4, logK=2.25, dH=5544.0, name="CaSO4 pair")          # J/mol
s.reaction("Gypsum = Ca+2 + SO4-2 + 2 H2O", name="Gypsum",
           analytic=[68.2401, 0.0, -3221.51, -25.0627])                            # PHREEQC A1..A6
```

**Reaction forms.** A reaction can be written with operators, as an equation
string, or as a dict of net coefficients (`{"Gypsum": -1, "Ca+2": 1, ...}`).

**Constraints.**
- Every solid and gas needs exactly one reaction linking it to non-phase
  species.
- Reactions must be linearly independent.
- A dependent reaction is rejected, because its `logK` would be an independent
  number that the thermodynamics does not allow.

### Temperature and pressure dependence

$\log K(T, P)$ uses the first form that is present:

| data | formula |
|------|---------|
| `analytic=[A1..A6]` | $A_1 + A_2T + A_3/T + A_4\log_{10}T + A_5/T^2 + A_6T^2$ |
| `logK`, `dH`, `dCp` | $\log K_r - \frac{\Delta H}{R\ln 10}(\frac1T-\frac1{T_r}) + \frac{\Delta C_p}{R\ln10}(\frac{T_r}{T}-1+\ln\frac{T}{T_r})$ |
| `dV` | adds $-\Delta V (P - P_r)/(RT\ln 10)$ |

Tabulated data: fit `analytic` coefficients to your $\log K(T)$ table with
`numpy.polyfit`-style least squares, or fit `logK` and `dH` directly with
[fitting.md](fitting.md).

### Other phases

```python
s.gas("CO2(g)")
s.reaction("CO2(g) = CO2", logK=-1.468, dH=-19983.0)
s.gas_phase(["CO2(g)"], pressure=1.0)              # or volume=1.0 (L); or fix log10 P in equilibrate

s.exchange_site("X")                                # returns the virtual master X-
s.exchange("NaX"); s.exchange("CaX2")               # sites per formula from the name
s.reaction("Na+ + X- = NaX", logK=0.0)
s.reaction("Ca+2 + 2 X- = CaX2", logK=0.8)

s.surface_type("Hfo", ["Hfo_w", "Hfo_s"], area=600, mass=0.09)   # diffuse double layer
s.surface("Hfo_wOH"); s.surface("Hfo_wOH2+"); ...

s.solid_solution("(Ca,Sr)CO3", ["Calcite", "Strontianite"])       # ideal
s.solid_solution("Arag-Stront", ["Aragonite", "Strontianite"], guggenheim=[3.43, -1.82])  # non-ideal: stability.md
s.electron()                                         # e- for redox; then equilibrate(..., pe=4)
```

## 2. YAML / JSON specification

The same content as a strictly validated document (unknown keys are errors):

```yaml
name: gypsum
activity: truesdell-jones
temperature: 298.15
species:
  - {name: Ca+2, a0: 5.0, b: 0.165}
  - {name: SO4-2, a0: 5.0, b: -0.04}
  - {name: CaSO4}
  - {name: Gypsum, phase: solid}
reactions:
  - {equation: "Ca+2 + SO4-2 = CaSO4", logK: 2.25, dH: 5544}
  - {equation: "Gypsum = Ca+2 + SO4-2", logK: -4.58, name: Gypsum}
problem:
  amounts: {Gypsum: 1.0}
```

```python
from discopt.speciation import from_yaml, solve_spec
s = from_yaml("gypsum.yaml")      # a System
out = solve_spec("gypsum.yaml")   # JSON-able result dict (agents, scripts)
s.to_spec()                       # any System back to a spec dict
```

**Sections.** The spec also takes these sections:

- `exchange_sites`
- `surfaces`
- `solid_solutions`
- `gas_phases`
- `masters`: input aliases such as `{Ca: Ca+2}`
- species `phase`: `solvent`, `gas`, `exchange`, `surface` or `electron`
- species `params`: extra activity parameters
- reaction fields `analytic`, `dCp` and `dV`

**Activity models.** An activity model is given as a name or a dict, for
example:

- `activity: {model: davies, A: 0.509}`
- `activity: {model: sit, eps: [[Na+, Cl-, 0.03]]}`

## 3. Start from a database, then add or override

```python
db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Ca", "S(6)", "Na", "Cl"], phases=["Gypsum"])

gyp = next(r for r in s.reactions if r.name == "Gypsum")
gyp.analytic = None; gyp.logK = -4.60; gyp.dH = -460.0     # your own value

s.aqueous("CaCl+", a0=5.0, b=0.0)        # a species phreeqc.dat lacks (check "CaCl+" in s.species first)
s.reaction("Ca+2 + Cl- = CaCl+", logK=0.40, name="CaCl+")

s["Ca+2"].params["a0"] = 5.2             # change an ion size
```

**Override rules.**
- Setting `analytic = None` is required when you override a reaction that has
  analytic coefficients, since those take precedence over `logK`.
- Parameter changes take effect at the next `equilibrate`.

In YAML:

```yaml
database: {file: phreeqc.dat, components: [Ca, "S(6)"], phases: [Gypsum]}
overrides:
  Gypsum: {logK: -4.60, dH: -460}      # clears analytic unless you give one
species:
  - {name: MyComplex}
reactions:
  - {equation: "Ca+2 + SO4-2 = MyComplex", logK: 0.3, name: MyComplex}
```

## 4. Your own PHREEQC-format database

Any file in [PHREEQC](references.md#parkhurst2013) format works. You can write a small one with only the
species you need:

```text
SOLUTION_MASTER_SPECIES
H       H+      -1.0    H       1.008
H(1)    H+      -1.0    0
E       e-      0       0.0     0
O       H2O     0       O       16.0
O(-2)   H2O     0       0
Ca      Ca+2    0       Ca      40.08
S(6)    SO4-2   0       SO4     32.064
SOLUTION_SPECIES
H+ = H+
    log_k 0
e- = e-
    log_k 0
H2O = H2O
    log_k 0
Ca+2 = Ca+2
    log_k 0
    -gamma 5.0 0.165
SO4-2 = SO4-2
    log_k 0
    -gamma 5.0 -0.04
H2O = OH- + H+
    log_k -14.0
    delta_h 13.362 kcal
Ca+2 + SO4-2 = CaSO4
    log_k 2.25
    delta_h 1.325 kcal
PHASES
Gypsum
    CaSO4:2H2O = Ca+2 + SO4-2 + 2 H2O
    log_k -4.58
    delta_h -0.109 kcal
END
```

```python
db = PhreeqcDatabase.from_file("mine.dat")    # or PhreeqcDatabase(text)
s = db.system(["Ca", "S(6)"], phases=["Gypsum"])
```

**Units.** `delta_h` is read in kJ/mol unless followed by `kcal` (or `J`).

**Pitzer data.** Add a `PITZER` block to your file, or build the parameters in
Python:

```python
from discopt.speciation import Pitzer, PitzerParameters

p = PitzerParameters()
p.add("B0", ["Na+", "Cl-"], [0.0765, -777.03, -4.4706, 0.008946, -3.3158e-6])  # a0..a5 (T dependence)
p.add("B1", ["Na+", "Cl-"], 0.2664)
p.add("C0", ["Na+", "Cl-"], 0.00127)
p.add("THETA", ["Na+", "K+"], -0.012)
p.add("PSI", ["Na+", "K+", "Cl-"], -0.0018)
s = System("brine", activity=Pitzer(p))
```

**Temperature form.** The Pitzer coefficient temperature form is PHREEQC's:
$P(T) = a_0 + a_1(1/T - 1/T_r) + a_2\ln(T/T_r) + a_3(T-T_r) + a_4(T^2-T_r^2) + a_5(1/T^2-1/T_r^2)$.

**Other models.** SIT takes `SIT(eps={("Na+", "Cl-"): 0.03})`. B-dot takes
per-species `a0` and a common `bdot`. See [activity-models.md](activity-models.md).

## Checking your data

- **Structure:** `s.validate()` checks for undeclared species, reactions
  without a constant, and solids or gases that do not have exactly one
  reaction. Linearly dependent reactions are rejected when equilibrating.
- **Result state:** `r.residuals` reports the mass-action, balance and
  complementarity residuals of the result; `r.converged` is true when all are
  below 1e-8.
- **Sensitivity:** perturb a constant and re-solve. `fit.FitResult.jacobian`
  holds these sensitivities for a set of observations.

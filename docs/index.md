# discopt-speciation documentation

`discopt.speciation` computes chemical equilibrium in aqueous systems with solids,
gases, solid solutions, ion exchangers and sorbing surfaces. You declare species
and reactions with their equilibrium constants. The plugin finds a state where:

- mass action holds for every reaction;
- conservation holds for every element, charge and site total;
- complementarity holds for every phase: it is either present and saturated,
  or absent and undersaturated.

The equations are solved with discopt's NLP solvers.

| guide | contents |
|-------|----------|
| [Concepts](concepts.md) | the equilibrium conditions, the three solution methods, why complementarity matters |
| [Databases](databases.md) | the bundled PHREEQC databases, what is read, choosing components and phases |
| [Your own data](custom-data.md) | species, reactions, `logK`/`dH`/`analytic`, activity parameters, YAML specs, custom `.dat` files, overriding database values |
| [Activity models](activity-models.md) | ideal, Davies, Truesdell-Jones, B-dot, LLNL, SIT, Pitzer: parameters and validity |
| [Stability](stability.md) | non-ideal solid solutions, miscibility gaps, certificates of global stability |
| [Fitting](fitting.md) | estimating `logK`, `dH` and activity parameters from measurements, with confidence intervals; exact implicit-function sensitivities |
| [Optimization](optimization.md) | choosing inputs to reach an equilibrium target, and guaranteed worst-case bounds over uncertain `logK` |
| [Diagrams](diagrams.md) | sweeps, predominance and Pourbaix diagrams, Ellingham plots |
| [Validation](validation.md) | what has been checked against PHREEQC and how |
| [References](references.md) | citations and links for databases, models, methods and papers |

Example notebooks are in [`examples/`](https://github.com/jkitchin/discopt-speciation/tree/main/examples):

| notebook | topic |
|----------|-------|
| [`01-getting-started.ipynb`](examples/01-getting-started.ipynb) | declaring a system, the equilibrium conditions, the DSL and YAML forms |
| [`02-validation.ipynb`](examples/02-validation.ipynb) | the Laliwala, Amusat and Torres (2025) cases, with five independent kinds of evidence of correctness |
| [`03-databases-and-your-own-data.ipynb`](examples/03-databases-and-your-own-data.ipynb) | PHREEQC databases, adding and overriding data, custom databases |
| [`04-fitting-thermodynamic-data.ipynb`](examples/04-fitting-thermodynamic-data.ipynb) | fitting constants to solubility and pH data, profile likelihood |
| [`05-diagrams.ipynb`](examples/05-diagrams.ipynb) | speciation, Pourbaix and Ellingham diagrams |
| [`06-rare-earths-and-thorium.ipynb`](examples/06-rare-earths-and-thorium.ipynb) | ThO₂ polymorphs, rare-earth phosphates, Th in brines with SIT, temperature, and ore minerals missing from the databases |

## Quick start

**No install:** try the [web app](app/index.html). It runs the package in your
browser (WebAssembly), with speciation, sweeps, Pourbaix diagrams and log K(T).
See [Web app](web-app.md).

```python
from discopt.speciation import PhreeqcDatabase

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Ca", "C(4)"], phases=["Calcite", "Aragonite", "CO2(g)"])
r = s.equilibrate({"Calcite": 1.0}, fixed={"CO2(g)": -3.5})   # log10 P_CO2 (atm)
print(r.summary())
```

This puts 1 mol of calcite in 1 kg of pure water that is open to the air.
`fixed` holds a gas at a constant $\log_{10}$ partial pressure in atm (an
aqueous species at a constant $\log_{10}$ activity). CO₂ is exchanged with an
unlimited reservoir, so carbon is not a conserved total. So −3.5 means
$P_{\mathrm{CO_2}} = 10^{-3.5} \approx 3.2\times10^{-4}$ atm, about 320 ppm.
That is the conventional atmospheric value used in the PHREEQC examples.
Today's air is closer to 420 ppm, or −3.38.

```text
converged=True method=complementarity I=0.001463 pH=8.2792 charge_imbalance=7.58e-19 water=0.999991 kg
aqueous                   molality     log10 a
H+                    5.473980e-09     -8.2792
Ca+2                  4.828344e-04     -3.3894
CO3-2                 9.607816e-06     -5.0904
OH-                   2.009773e-06     -5.7155
HCO3-                 9.494024e-04     -3.0408
CO2                   1.075691e-05     -4.9682
(CO2)2                2.124615e-12    -11.6726
CaOH+                 1.343994e-08     -7.8902
CaCO3                 5.562977e-06     -5.2545
CaHCO3+               4.940139e-06     -5.3244
solid                       amount          SI
Calcite               9.995067e-01     -0.0000
Aragonite             0.000000e+00     -0.1438
gas                         amount     log10 P
CO2(g)                       fixed     -3.5000
residuals: balance=7.5e-19, mass_action=1.8e-15, complementarity=0.0e+00, ionic_strength=1.2e-14
```

Aragonite is metastable here. Calcite is present and saturated (SI = 0).
Aragonite is absent and undersaturated (SI = −0.14), and pH is 8.28, the
PHREEQC value.

"""Write and execute the discopt-speciation example notebooks.

    uv run python examples/build_notebooks.py        # both
    uv run python examples/build_notebooks.py 02     # just one
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

EX = pathlib.Path(__file__).resolve().parent


def nb(cells):
    out = new_notebook()
    out.cells = [new_markdown_cell(c[3:].strip("\n")) if c.startswith("md:") else new_code_cell(c.strip("\n")) for c in cells]
    out.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return out


# ============================================================================ 01
GETTING_STARTED = [
"""md:
# Getting started with `discopt.speciation`

`discopt.speciation` computes chemical equilibrium in water: which dissolved species form, how much of each, and which solids precipitate. You declare species and reactions, each with an equilibrium constant $\\log_{10} K$. The equations are solved with the NLP solvers of [discopt](https://github.com/jkitchin/discopt). The package reads PHREEQC databases (Parkhurst and Appelo 2013, [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)), but in this notebook every constant is typed in by hand.

**What you will learn.**
- How to declare species, reactions and solids: operator form, string form, and YAML/dict specs.
- How to read a `Result`: phases present, saturation indices, molalities, residuals.
- How to use activity models, fixed pH, metastable polymorphs, and the three solution methods.

**Prerequisites.** Basic aqueous chemistry. No earlier notebook is needed.

**The equilibrium conditions.** The plugin finds the state that satisfies:

- **mass action** for every reaction: $\\sum_i \\nu_i \\log_{10} a_i = \\log_{10} K$. Here $\\nu_i$ is the stoichiometric coefficient (positive for products) and $a_i=\\gamma_i m_i$ is the *activity*: molality $m_i$ (mol/kg water) times an activity coefficient $\\gamma_i$.
- **conservation** of every quantity the reactions cannot change, such as element totals. These are computed from the network, not typed in.
- **complementarity** for every candidate solid $k$: $n_k \\ge 0$, $\\mathrm{SI}_k \\le 0$, and $n_k\\,\\mathrm{SI}_k = 0$. The saturation index is $\\mathrm{SI}_k=\\log_{10}(Q_k/K_k)$, where $Q_k$ is the activity product. A solid is either present and saturated ($\\mathrm{SI}=0$), or absent and undersaturated ($\\mathrm{SI}<0$).

The third condition is the one that is easy to get wrong. Without it, the answer depends on how reactions are written and on which phases are listed as candidates. This is the problem with the SI²-minimization approach of Laliwala, Amusat and Torres (2025, [doi:10.69997/sct.132742](https://doi.org/10.69997/sct.132742)). [`02-validation.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/02-validation.ipynb) re-solves their cases.

**The notebooks.**

| notebook | topic |
|---|---|
| **01-getting-started** | this notebook |
| 02-validation | the Laliwala et al. cases and evidence of correctness |
| 03-databases-and-your-own-data | PHREEQC databases, adding and overriding data |
| 04-fitting-thermodynamic-data | fitting constants to measurements |
| 05-diagrams | speciation, Pourbaix and Ellingham diagrams |
| 06-rare-earths-and-thorium | a larger worked application |
""",
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml

from discopt.speciation import System, solve_spec
""",
"""md:
## Declaring a system: gypsum solubility

**Goal.** Build the smallest useful system: calcium, sulfate, their neutral ion pair, and gypsum (CaSO₄·2H₂O).

- Species are created on a `System`. Charges come from the name suffix (`"SO4-2"` → −2) unless given explicitly.
- In a reaction, `+` joins species, `2 * X` sets a stoichiometric coefficient, and `>>` separates reactants from products.
- `logK` is base 10, at 25 °C, for the reaction as written (products over reactants).
- Solids have unit activity. Every solid needs exactly one reaction that links it to aqueous species, i.e. its solubility product $K_{sp}$.
- The default activity model is ideal ($\\gamma_i = 1$).
""",
"""
s = System("gypsum")
Ca, SO4, CaSO4 = s.aqueous("Ca+2"), s.aqueous("SO4-2"), s.aqueous("CaSO4")
gypsum = s.solid("Gypsum")

s.reaction(Ca + SO4 >> CaSO4, logK=2.36)      # aqueous ion pair
s.reaction(gypsum >> Ca + SO4, logK=-4.61)    # solubility product
s
""",
"""md:
`equilibrate` takes input amounts in mol per kg of water (molal units). Only the conserved totals (here total Ca and total SO₄) matter. So adding 0.0324 mol Ca²⁺ and 0.025 mol SO₄²⁻ is the same as adding 0.025 mol CaSO₄ plus 0.0074 mol Ca²⁺.
""",
"""
res = s.equilibrate({Ca: 0.0324, SO4: 0.025})
print(res.summary())
""",
"""md:
**Result.** 0.0169 mol/kg gypsum precipitates, and its SI is exactly zero. What stays dissolved is mostly free Ca²⁺ (0.0099 mol/kg) and the CaSO₄ ion pair (0.0056 mol/kg). The last line shows the residuals of the three conditions, which are all at machine precision. `I` is the ionic strength, $I=\\tfrac12\\sum_i m_i z_i^2$.

**Charge.** `charge_imbalance` is 0.0148 = 2(0.0324 − 0.025), because the excess Ca²⁺ has no counter-ion. Electroneutrality is not imposed as a constraint: it follows from whatever you add. Add a counter-ion such as Cl⁻, or pass `charge_balance=`, for a neutral solution.

**Reading a `Result`.**
- `res["Ca+2"]` gives a molality, or a solid amount, by name.
- `res.present` lists the solids at equilibrium.
- `res.SI` gives $\\log_{10}(Q/K)$ for every candidate solid.
- `res.pH` is `None` here, because this system has no H⁺.

**Undersaturated case.** The next cell writes the same system with string reactions, which are equivalent to the operator form. It then adds less material, so gypsum should be absent with a negative SI.
""",
"""
s2 = System("gypsum (string form)")
for name in ["Ca+2", "SO4-2", "CaSO4"]:
    s2.aqueous(name)
s2.solid("Gypsum")
s2.reaction("Ca+2 + SO4-2 = CaSO4", logK=2.36)
s2.reaction("Gypsum = Ca+2 + SO4-2", logK=-4.61)

dilute = s2.equilibrate({"Ca+2": 1e-3, "SO4-2": 1e-3})
print("present:", dilute.present, " SI:", dilute.SI, " converged:", dilute.converged)
""",
"""md:
## Solubility curve and activity coefficients

**Goal.** See how solubility depends on the activity model. We sweep the amount of CaSO₄ added from 10⁻³ to 10⁻¹ mol/kg. Dissolved calcium should rise until the solution saturates, then stay flat as the extra material precipitates.

Real ions interact, so their activity is lower than their molality ($\\gamma_i<1$). The Davies equation (Davies 1938, [doi:10.1039/JR9380002093](https://doi.org/10.1039/JR9380002093)) estimates $\\gamma_i$ from the charge $z_i$ and the ionic strength $I$ alone:
$\\log_{10}\\gamma_i=-A z_i^2\\left(\\frac{\\sqrt I}{1+\\sqrt I}-0.3I\\right)$, with $A\\approx0.51$ at 25 °C. It is a reasonable model up to $I$ of a few tenths of a mol/kg. Other models are covered in [`docs/activity-models.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/activity-models.md).
""",
"""
added = np.logspace(-3, -1, 41)
curves = {}
for model in ["ideal", "davies"]:
    dissolved, solid = [], []
    for c in added:
        r = s.equilibrate({Ca: c, SO4: c}, activity=model)
        assert r.converged
        dissolved.append(r["Ca+2"] + r["CaSO4"])
        solid.append(r["Gypsum"])
    curves[model] = (np.array(dissolved), np.array(solid))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for model, (d, n) in curves.items():
    ax[0].loglog(added, d, label=model)
    ax[1].semilogx(added, n, label=model)
ax[0].loglog(added, added, "k:", lw=0.8, label="all dissolved")
ax[0].set(xlabel="CaSO$_4$ added (mol/kg)", ylabel="dissolved Ca (mol/kg)")
ax[1].set(xlabel="CaSO$_4$ added (mol/kg)", ylabel="gypsum (mol/kg)")
ax[0].legend()
ax[1].legend()
plt.tight_layout()
""",
"""md:
**Result.** Left panel: dissolved Ca follows the "all dissolved" line until saturation. It then plateaus at 0.0106 mol/kg with ideal activities and 0.0160 mol/kg with Davies, about 50 % higher. Right panel: gypsum is zero before saturation and grows linearly after it; the log x-axis makes that line look curved. At the plateau, $I$ is 0.020 (ideal) and 0.041 (Davies). Davies gives $\\gamma_{\\mathrm{Ca^{2+}}}=0.48$ there, so more ions must dissolve to reach $K_{sp}$. The sharp corner is where complementarity switches gypsum from absent to present.

**Temperature.** `equilibrate(..., T=...)` takes kelvin. `logK` values are for 25 °C and change with $T$ only if a reaction has `dH`, `dCp` or `analytic=` data (see [`docs/custom-data.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/custom-data.md)). The gypsum reactions have none, so at 50 °C only the Davies $A$ parameter changes: dissolved Ca moves from 0.01596 to 0.01649 mol/kg.
""",
"""
for T in [298.15, 323.15]:
    r = s.equilibrate({Ca: 0.1, SO4: 0.1}, T=T, activity="davies")
    print(f"T = {T} K   dissolved Ca = {r['Ca+2'] + r['CaSO4']:.5f} mol/kg")
""",
"""md:
## Fixed activities: iron(III) solubility versus pH

**Goal.** Compute Fe(III) solubility from pH 1 to 13, and see how complementarity picks between two solids of the same composition.

**Fixed activities.** A species can be held at a fixed $\\log_{10}$ activity, which models an open reservoir. `pH=7` is shorthand for `fixed={"H+": -7}`. The amount of H⁺ is then not conserved: acid or base is exchanged as needed.

**The system.** Fe(III) hydrolysis species and two iron oxyhydroxides, goethite and ferrihydrite. The constants are from the minteq.v4 database (Allison et al. 1991, [OSTI 5673069](https://www.osti.gov/biblio/5673069)). Water is left out of these reactions, which is the same as giving it activity 1.

**Polymorphs.** The two solids have the same stoichiometry, but goethite's $\\log_{10} K$ is 2.7 units lower (0.491 vs 3.191), so it is more stable. When both are candidates, only goethite may form, whatever order they are declared in. Ferrihydrite is *metastable*: it can persist in practice but is not the equilibrium phase. To compute its higher solubility, either leave goethite out of the system, as done below, or pass `equilibrate(..., suppress=["Goethite"])`.
""",
"""
def iron_system(solids):
    fe = System("iron(III)")
    for name in ["H+", "Fe+3", "FeOH+2", "Fe(OH)2+", "Fe(OH)3", "Fe(OH)4-", "Fe2(OH)2+4", "Fe3(OH)4+5"]:
        fe.aqueous(name)
    fe.reaction("Fe+3 = FeOH+2 + H+", logK=-2.187)
    fe.reaction("Fe+3 = Fe(OH)2+ + 2 H+", logK=-4.594)
    fe.reaction("Fe+3 = Fe(OH)3 + 3 H+", logK=-12.56)
    fe.reaction("Fe+3 = Fe(OH)4- + 4 H+", logK=-21.588)
    fe.reaction("2 Fe+3 = Fe2(OH)2+4 + 2 H+", logK=-2.854)
    fe.reaction("3 Fe+3 = Fe3(OH)4+5 + 4 H+", logK=-6.288)
    logK = {"Ferrihydrite": 3.191, "Goethite": 0.491}
    for name in solids:
        fe.solid(name)
        fe.reaction(f"{name} + 3 H+ = Fe+3", logK=logK[name])
    return fe

# Fe atoms per aqueous species, to sum up total dissolved Fe(III)
IRON = {"Fe+3": 1, "FeOH+2": 1, "Fe(OH)2+": 1, "Fe(OH)3": 1, "Fe(OH)4-": 1, "Fe2(OH)2+4": 2, "Fe3(OH)4+5": 3}
pH = np.linspace(1, 13, 61)
fig, ax = plt.subplots(figsize=(6, 4))
for solids, style in [(["Ferrihydrite", "Goethite"], "-"), (["Ferrihydrite"], "--")]:
    system = iron_system(solids)
    dissolved, phase = [], []
    for p in pH:
        r = system.equilibrate({"Fe+3": 1e-3}, pH=p)
        assert r.converged, p
        dissolved.append(sum(c * r[k] for k, c in IRON.items()))
        phase.append(r.present)
    ax.semilogy(pH, dissolved, style, label=" + ".join(solids) + " candidates")
    print(" + ".join(solids), "-> phases found:", sorted({x for ph in phase for x in ph}))
ax.set(xlabel="pH", ylabel="dissolved Fe(III) (mol/kg)", ylim=(1e-14, 3e-3))
ax.legend()
plt.tight_layout()
""",
"""md:
**Result.** The printout confirms that ferrihydrite never forms when goethite is a candidate.
- **Low pH.** All 10⁻³ mol/kg Fe stays dissolved at the lowest pH. Goethite appears from about pH 1.2; ferrihydrite alone does not saturate until about pH 2.2.
- **Minimum.** Both curves reach a minimum near pH 8.5: about 1.4×10⁻¹² mol/kg over goethite and 6.8×10⁻¹⁰ over ferrihydrite. The ratio, about 500, is exactly $10^{2.7}$, the difference in $\\log_{10} K$.
- **High pH.** Solubility rises again as Fe(OH)₄⁻ takes over.

At every pH where goethite is present, ferrihydrite has SI = −2.7.
""",
"""md:
## Three solution methods

**Goal.** Check one answer three independent ways. `method=` chooses how the phase assemblage, i.e. the set of solids present, is found. All three should give the same state, and after the solve every method checks the equilibrium conditions to tight tolerance.

| method | idea | notes |
|---|---|---|
| `"complementarity"` (default) | log-molality mass action plus a smoothed Fischer–Burmeister equation per solid (Fischer 1992, [doi:10.1080/02331939208843795](https://doi.org/10.1080/02331939208843795)), driven through an ε-homotopy (Kanzow 1996, [doi:10.1137/S0895479894273134](https://doi.org/10.1137/S0895479894273134)) | any activity model |
| `"gibbs"` | minimize $G/RT$ subject to conservation (Smith and Missen 1982) | convex for ideal activities, so a local solve is global; needs an activity model with an excess Gibbs energy (ideal, Davies, SIT, Pitzer; not Truesdell–Jones or B-dot) |
| `"enumerate"` | solve for every subset of present solids and keep the one that satisfies complementarity | proves uniqueness; cost grows as $2^{\\#\\text{solids}}$ (limit 12 phases) |

Leal et al. (2017, [doi:10.1515/pac-2016-1107](https://doi.org/10.1515/pac-2016-1107)) review mass-action and Gibbs-minimization methods. The cell below solves the first gypsum problem with each method.
""",
"""
import time
for method in ["complementarity", "gibbs", "enumerate"]:
    t0 = time.perf_counter()
    r = s.equilibrate({Ca: 0.0324, SO4: 0.025}, method=method)
    print(f"{method:16s} gypsum = {r['Gypsum']:.12e}   SI = {r.SI['Gypsum']:+.1e}   "
          f"converged = {r.converged}   ({time.perf_counter() - t0:.2f} s)")
""",
"""md:
**Result.** All three methods give the same gypsum amount to 12 significant figures, with SI zero to rounding (≤ 10⁻¹⁵). Run times are hundredths of a second and vary between runs. The `02-validation` notebook uses this agreement as one of its lines of evidence.
""",
"""md:
## Declarative specs (YAML / JSON / dict)

**Goal.** Write the same problem as data instead of code. The schema is strict, so a misspelled key raises an error instead of being silently ignored. `solve_spec` accepts YAML, JSON or a dict and returns a JSON-serializable dict, which suits scripts and LLM agents. Each species is listed with its `phase` (aqueous is the default) and each reaction as an `equation` string. The same schema is documented in [`docs/custom-data.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/custom-data.md).
""",
"""
spec = '''
name: gypsum
activity: davies
species:
  - {name: Ca+2}
  - {name: SO4-2}
  - {name: CaSO4}
  - {name: Gypsum, phase: solid}
reactions:
  - {equation: "Ca+2 + SO4-2 = CaSO4", logK: 2.36}
  - {equation: "Gypsum = Ca+2 + SO4-2", logK: -4.61}
problem:
  amounts: {Ca+2: 0.0324, SO4-2: 0.025}
'''
out = solve_spec(spec)
print({k: out[k] for k in ["converged", "present", "solids", "SI", "ionic_strength"]})
""",
"""md:
**Result.** This spec sets `activity: davies`, so it matches the Davies curve rather than the first (ideal) calculation. Less gypsum precipitates (0.0119 vs 0.0169 mol/kg) and the ionic strength is higher (0.045 vs 0.025).

`System.to_spec()` goes the other way: it exports a system built in code. The export writes out the defaults: `temperature` in K, `activity: ideal`, charges, and each reaction as a `stoich` dict (negative for reactants).
""",
"""
# any System can be exported to the same schema
print(yaml.safe_dump(s.to_spec(), sort_keys=False))
""",
"""md:
## Summary

- A `System` holds aqueous species, solids and reactions with `logK`. Build it with operators (`>>`), with equation strings, or from a YAML/dict spec.
- `equilibrate` enforces mass action, conservation and complementarity. Solids come out either present with SI = 0 or absent with SI < 0.
- Complementarity picks the stable polymorph automatically. Use `suppress=` or leave the stable phase out to get metastable equilibria.
- The activity model matters even at moderate ionic strength: Davies raised gypsum solubility by about 50 % here.
- The three methods agree, which is a built-in consistency check.

## Next steps

- [`02-validation.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/02-validation.ipynb): why complementarity is needed, with the Laliwala et al. (2025) cases checked against PHREEQC.
- [`03-databases-and-your-own-data.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/03-databases-and-your-own-data.ipynb): load [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) and other databases instead of typing constants.
- Docs: [`docs/concepts.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/concepts.md) (conditions, methods, units), [`docs/activity-models.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/activity-models.md), [`docs/custom-data.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/custom-data.md).
""",
"""md:
## References

- Allison, J.D., Brown, D.S., and Novo-Gradac, K.J. (1991). *MINTEQA2/PRODEFA2, a geochemical assessment model for environmental systems: version 3.0 user's manual.* EPA/600/3-91/021. <https://www.osti.gov/biblio/5673069>
- Davies, C.W. (1938). The extent of dissociation of salts in water. Part VIII. *Journal of the Chemical Society*, 2093. <https://doi.org/10.1039/JR9380002093>
- Fischer, A. (1992). A special Newton-type optimization method. *Optimization* 24, 269–284. <https://doi.org/10.1080/02331939208843795>
- Kanzow, C. (1996). Some noninterior continuation methods for linear complementarity problems. *SIAM Journal on Matrix Analysis and Applications* 17, 851–868. <https://doi.org/10.1137/S0895479894273134>
- Kitchin, J. *discopt*. <https://github.com/jkitchin/discopt>
- Laliwala, C.A., Amusat, O.O., and Torres, A.I. (2025). An optimization-based law of mass action precipitation/dissolution model. *Systems and Control Transactions* 4, 2140–2145. <https://doi.org/10.69997/sct.132742>
- Leal, A.M.M., Kulik, D.A., Smith, W.R., and Saar, M.O. (2017). An overview of computational methods for chemical equilibrium and kinetic calculations for geochemical and reactive transport modeling. *Pure and Applied Chemistry* 89, 597–643. <https://doi.org/10.1515/pac-2016-1107>
- Parkhurst, D.L., and Appelo, C.A.J. (2013). *Description of input and examples for PHREEQC version 3.* USGS Techniques and Methods 6-A43. <https://doi.org/10.3133/tm6A43>
- Smith, W.R., and Missen, R.W. (1982). *Chemical Reaction Equilibrium Analysis: Theory and Algorithms.* Wiley, New York.

The full bibliography is in [`docs/references.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/references.md).
""",
]

# ============================================================================ 02
VALIDATION = [
"""md:
# Validation: the Laliwala, Amusat & Torres (2025) examples, and how we know the answers are right

**What you will learn.** How to rebuild published equilibrium problems from database data, and how to check any `discopt.speciation` result yourself instead of trusting the solver. The checks are: independent residuals, a PHREEQC comparison, alternative algorithms, invariance tests and a Gibbs-energy test.

**Before you start.** Read [`01-getting-started.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/01-getting-started.ipynb) first. It introduces activities, the saturation index $\\mathrm{SI}_k=\\log_{10}(Q_k/K_k)$, the Davies equation and complementarity (a candidate solid is either present with $\\mathrm{SI}_k=0$, or absent with $\\mathrm{SI}_k\\le 0$). The equations are summarized in [`docs/concepts.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/concepts.md). Amounts are in mol per kg of water (mol/kgw), and $I=\\tfrac12\\sum_i z_i^2 m_i$ is the ionic strength. Running the notebook takes about a minute, mostly for the exhaustive enumeration of Case 3. Log lines starting with `INFO pounce...` come from the underlying NLP solver and can be ignored.

Laliwala, Amusat & Torres (2025, [doi:10.69997/sct.132742](https://doi.org/10.69997/sct.132742)) compute aqueous equilibrium with precipitation by minimizing $z=\\sum_k \\mathrm{SI}_k^2$ over reaction extents. They validate the method on three cases against PHREEQC (Parkhurst & Appelo 2013, [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)). That objective leaves out complementarity: an absent solid does not need $\\mathrm{SI}=0$, only $\\mathrm{SI}\\le 0$. So minimizing $z$ can return states that are not at equilibrium. Its answers can also change when a reaction is rescaled or an inert candidate phase is added. A separate analysis, which is not part of this repository, built small counterexamples for these failures. We re-solve them below (E4).

The notebook collects five independent kinds of evidence that the results are correct equilibria:

1. **E1: Direct verification.** The equilibrium conditions are re-checked from the declared reactions and the returned molalities, with none of the solver's internals. Element totals come from the database formulas; activity coefficients are recomputed from scratch.
2. **E2: An independent code.** Results are compared with PHREEQC 3 on the same minteq.v4 data for every case.
3. **E3: Independent algorithms.** Complementarity homotopy, convex Gibbs minimization, and exhaustive active-set enumeration are run and compared.
4. **E4: Thermodynamic invariances.** Adding a less stable polymorph (another crystal form of the same compound), adding an inert phase, or rescaling a reaction must not change the state. These are counterexamples where the SI² objective fails.
5. **E5: The second law.** Every feasible perturbation of the reaction extents away from the computed state raises $G$.

**Data.** [`data/minteq_cases.json`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/data/minteq_cases.json) holds, for each case, the minteq.v4 log K values and formulas and the PHREEQC results. [`data/notebook_reference.json`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/data/notebook_reference.json) holds independent Pyomo/IPOPT active-set solutions for Case 2 and the counterexamples. Their provenance is recorded in [`tests/data/README.md`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/data/README.md).
""",
"""
import json
import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt

from discopt.speciation import System

DATA = pathlib.Path("data")
CASES = json.loads((DATA / "minteq_cases.json").read_text())
REF = json.loads((DATA / "notebook_reference.json").read_text())
LN10 = np.log(10.0)
""",
"""md:
## Building the paper's systems from database data

**Goal.** Turn each case in [`data/minteq_cases.json`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/data/minteq_cases.json) into a `System`.

The constants come from [`minteq.v4.dat`](https://github.com/usgs-coupled/phreeqc3/blob/master/database/minteq.v4.dat) as distributed with PHREEQC, which derives from MINTEQA2 (Allison et al. 1991, [OSTI 5673069](https://www.osti.gov/biblio/5673069)). Each case lists:

- **primary species** (the building blocks, e.g. Ag⁺, Cl⁻);
- **secondary aqueous species**, each with a formation reaction from primaries and its $\\log_{10}K$ (e.g. Ag⁺ + Cl⁻ = AgCl);
- **candidate solids**, each with a dissolution reaction (solid → ions) and $\\log_{10}K$.

`System.reaction` takes a coefficient dict with products positive and reactants negative; `logK` refers to the reaction as written. pH is held fixed as in the paper (a "pH-stat"). The acid or base PHREEQC added to hold it is already included in the totals, which is why Na⁺ or NO₃⁻ are not round numbers. Two more helpers:

- `composition` returns each species' formula in terms of primaries. The independent checker below uses it.
- `solve_case` builds and solves a case and times it.
""",
"""
def minteq_system(case, name="minteq"):
    s = System(name)
    prim = case["primaries"]
    for p in prim:
        s.aqueous(p, case["species"][p]["z"])
    for sp, d in case["species"].items():
        if sp in prim or not d["nu"]:          # skip primaries and H2O
            continue
        s.aqueous(sp, d["z"])
        s.reaction({**{k: -v for k, v in d["nu"].items()}, sp: 1.0}, logK=d["logK"])
    for ph, d in case["phases"].items():
        s.solid(ph)
        s.reaction({ph: -1.0, **d["nu"]}, logK=d["logK"])
    return s


def composition(case):
    \"\"\"Formula of every species/phase in terms of primaries (the chemistry, not the solver's matrix).\"\"\"
    comp = {sp: d["nu"] for sp, d in case["species"].items()}
    comp.update({ph: d["nu"] for ph, d in case["phases"].items()})
    return comp


def solve_case(name, activity="ideal", method="complementarity"):
    case = CASES[name]
    s = minteq_system(case, name)
    t0 = time.perf_counter()
    res = s.equilibrate(case["totals"], pH=case["pH"], activity=activity, method=method)
    return s, res, time.perf_counter() - t0


s1 = minteq_system(CASES["Case 1"], "Case 1")
print(s1)
for r in s1.reactions:
    print(f"  {r.equation():40s} logK = {r.logK:+.3f}")
""",
"""md:
**Output.** Case 1 has 12 aqueous species, 2 candidate solids and 10 reactions. Each line is printed as reactants = products. Two lines need explaining:

- `0 = H+ + OH-` is water dissociation, with H₂O omitted (its activity is 1), and $\\log K = -14.0$.
- The Ag₂O line is written with H⁺ on the left: Ag₂O + 2 H⁺ = 2 Ag⁺. Its $\\log K$ belongs to that direction.
""",
"""md:
## E1: An independent checker

**Goal.** Solver residuals only show that the solver met its own equations. `verify` instead recomputes the equilibrium conditions from scratch. It uses only the declared reactions, the input amounts and the returned molalities and solid amounts. It checks:

- **conservation.** For database cases, every element total is computed from the species formulas. For hand-written systems it uses the left null space of the stoichiometric matrix, found by SVD. The package itself uses exact rational row reduction.
- **mass action.** $\\log_{10}Q - \\log_{10}K$ for every aqueous reaction. Activity coefficients are recomputed here from the molalities and charges, for both ideal and Davies.
- **complementarity.** Every $n_k \\ge 0$, every $\\mathrm{SI}_k \\le 0$, and $\\min(n_k/n_{\\rm scale}, -\\mathrm{SI}_k) = 0$.

Each number is a maximum over all species or reactions, so a value near zero (about 1e-8 or less) is a pass. `n_scale` is the largest input amount, which makes solid amounts dimensionless. The function also reports the paper's objective $z=\\sum_k (\\log_{10}Q_k-\\log_{10}K_k)^2$, taken over the solid reactions *as written*. This equals $\\sum_k\\mathrm{SI}_k^2$ only when each reaction dissolves one formula unit (see E4).

`show_checks` prints one line of these numbers. `compare_phreeqc` prints our solid amounts and SI next to PHREEQC's.
""",
"""
# Davies equation: log10(gamma) = -A z^2 (sqrt(I)/(1+sqrt(I)) - b I).
# Use the same Debye-Hueckel A(25 C) as the package (0.5100; 0.509 is an older textbook value).
from discopt.speciation.water import debye_huckel_A
A_DAVIES, B_DAVIES = debye_huckel_A(298.15), 0.3


def verify(system, res, amounts, pH=None, activity="ideal", composition=None):
    \"\"\"Recheck conservation, mass action and complementarity from the reaction data and returned amounts only.\"\"\"
    fixed = {"H+": -pH} if pH is not None else {}
    names = [sp.name for sp in system.species if sp.name not in fixed]
    amount = lambda n: res.molality[n] if n in res.molality else res.solids[n]
    state = np.array([amount(n) for n in names])
    init = np.array([float(amounts.get(n, 0.0)) for n in names])

    # -- conservation
    if composition is not None:
        comps = sorted({c for n in names for c in composition[n] if c not in fixed})
        B = np.array([[composition[n].get(c, 0.0) for n in names] for c in comps])
    else:
        N = np.array([[r.stoich.get(system[n], 0.0) for r in system.reactions] for n in names])
        U, sv, _ = np.linalg.svd(N)
        B = U[:, int((sv > 1e-9).sum()):].T
    balance = np.max(np.abs(B @ (state - init)) / np.maximum(np.abs(B) @ init, 1e-300))

    # -- activities recomputed from molalities
    z = {sp.name: sp.charge for sp in system.species}
    I = 0.5 * sum(z[n] ** 2 * m for n, m in res.molality.items())
    def log10_gamma(n):
        if activity == "ideal" or z[n] == 0:
            return 0.0
        sI = np.sqrt(I)
        return -A_DAVIES * z[n] ** 2 * (sI / (1 + sI) - B_DAVIES * I)
    def log10_a(n):
        if n in fixed:
            return fixed[n]
        if system[n].is_solid:
            return 0.0
        return np.log10(res.molality[n]) + log10_gamma(n)

    # -- mass action and saturation
    mass_action, SI, z_paper = [], {}, 0.0
    for r in system.reactions:
        dev = sum(c * log10_a(sp.name) for sp, c in r.stoich.items()) - r.logK
        solid = [sp for sp in r.stoich if sp.is_solid]
        if solid:
            SI[solid[0].name] = dev / (-r.stoich[solid[0]])
            z_paper += dev ** 2
        else:
            mass_action.append(abs(dev))
    n = {k: res.solids[k] for k in SI}
    scale = max(init.max(), max(n.values(), default=0.0))
    return {
        "I": I,
        "balance": balance,
        "mass_action": max(mass_action, default=0.0),
        "max_SI": max(SI.values(), default=-np.inf),
        "min_solid": min(n.values(), default=0.0),
        "complementarity": max((abs(min(n[k] / scale, -SI[k])) for k in SI), default=0.0),
        "z_paper": z_paper,
        "SI": SI,
    }


def show_checks(label, c):
    print(f"{label:28s} balance {c['balance']:.1e} | mass action {c['mass_action']:.1e} | "
          f"max SI {c['max_SI']:+.1e} | min solid {c['min_solid']:+.1e} | complementarity {c['complementarity']:.1e}")


def compare_phreeqc(name, res, activity):
    ref = CASES[name]["phreeqc_ideal" if activity == "ideal" else "phreeqc_activity"]
    print(f"{'phase':<18}{'discopt.speciation':>16}{'PHREEQC':>14}{'SI (ours)':>12}{'SI (PHREEQC)':>14}")
    for ph, nref in ref["n"].items():
        print(f"{ph:<18}{res.solids[ph]:>16.6e}{nref:>14.6e}{res.SI[ph]:>12.4f}{ref['SI'][ph]:>14.4f}")
""",
"""md:
## Case 1: Silver chloride at pH 5

**Setup.** Ag⁺, Cl⁻ and NO₃⁻ are each 10⁻⁴ mol/kgw. The candidate solids are the Ag/Cl/N solids in the database: cerargyrite (AgCl) and Ag₂O. **Expected.** Cerargyrite precipitates, and Ag₂O stays far undersaturated at pH 5.

The cell solves the case twice, with ideal activities and with Davies activities (Davies 1938, [doi:10.1039/JR9380002093](https://doi.org/10.1039/JR9380002093)). Each run is compared with PHREEQC and then passed to the independent checker.
""",
"""
checks = {}
for activity in ["ideal", "davies"]:
    s, res, dt = solve_case("Case 1", activity)
    print(f"--- {activity} ({dt:.2f} s), converged = {res.converged}, I = {res.ionic_strength:.3e}")
    compare_phreeqc("Case 1", res, activity)
    case = CASES["Case 1"]
    c = verify(s, res, case["totals"], case["pH"], activity, composition(case))
    checks[("Case 1", activity)] = (res, c)
    show_checks("independent check", c)
""",
"""md:
**Result.** 8.63 × 10⁻⁵ mol/kgw of cerargyrite precipitates, about 86% of the silver. Ag₂O has SI = −12.3. Solid amounts and SI match PHREEQC to the printed digits (ideal) or to 1 part in 10⁷ (Davies). The ionic strength is tiny ($I\\approx 7\\times10^{-5}$), so the activity model barely matters here. The checker line shows conservation and mass action near machine precision, no negative solids and no supersaturation.
""",
"""md:
## Case 2: Gypsum in a multicomponent brine

**Setup.** The paper's Table 1 brine (Na, K, Ca, Mg, Cl, SO₄) at pH 7, with gypsum (CaSO₄·2H₂O) as the only candidate solid. **Why it is interesting.** The ionic strength is about 0.03–0.05 mol/kgw, so activity corrections matter. Ideal activities overestimate how much gypsum precipitates.

PHREEQC's activity model uses the WATEQ Debye–Hückel form (Truesdell & Jones 1974, [USGS](https://pubs.usgs.gov/publication/70232379)) for ions with size parameters and Davies for the rest. A difference well under 1% from our pure Davies model is therefore expected.

We also compare with a second, independent reference: an active-set solution computed with Pyomo/IPOPT, stored in [`data/notebook_reference.json`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/data/notebook_reference.json). The analysis that produced it is not part of this repository.
""",
"""
for activity in ["ideal", "davies"]:
    s, res, dt = solve_case("Case 2", activity)
    print(f"--- {activity} ({dt:.2f} s), converged = {res.converged}, I = {res.ionic_strength:.4f}")
    compare_phreeqc("Case 2", res, activity)
    case = CASES["Case 2"]
    c = verify(s, res, case["totals"], case["pH"], activity, composition(case))
    checks[("Case 2", activity)] = (res, c)
    show_checks("independent check", c)

ref = REF["Case2_davies"]
res = checks[("Case 2", "davies")][0]
print(f"\\nDavies vs Pyomo reference: gypsum {res['Gypsum']:.6e} vs {ref['gypsum']:.6e}, "
      f"I {res.ionic_strength:.6f} vs {ref['I']:.6f}")
""",
"""
labels = ["ideal", "Davies"]
ours = [checks[("Case 2", a)][0]["Gypsum"] for a in ["ideal", "davies"]]
phq = [CASES["Case 2"]["phreeqc_ideal"]["n"]["Gypsum"], CASES["Case 2"]["phreeqc_activity"]["n"]["Gypsum"]]
x = np.arange(2)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(x - 0.2, np.array(ours) * 1e3, 0.4, label="discopt.speciation")
ax.bar(x + 0.2, np.array(phq) * 1e3, 0.4, label="PHREEQC")
ax.axhline(REF["Case2_davies"]["gypsum"] * 1e3, color="k", ls=":", lw=1, label="Pyomo reference (Davies)")
ax.set_xticks(x, labels)
ax.set_ylabel("gypsum precipitated (mmol/kg)")
ax.legend(fontsize=8)
plt.tight_layout()
""",
"""md:
**Result.** Gypsum precipitated is 16.85 mmol/kg with ideal activities and 11.79 mmol/kg with Davies, so the ideal model overpredicts by about 43%. In the bar chart the two codes are indistinguishable. The PHREEQC differences are 0.005% (ideal) and 0.05% (activity-corrected). The Pyomo reference is 11.81 mmol/kg ($I$ = 0.04570 versus our 0.04578). The 0.2% gap comes from the Davies constant alone: the reference used the textbook $A=0.509$, and re-solving here with `activity={"model": "davies", "A": 0.509}` reproduces it (11.811 mmol/kg, $I$ = 0.045700).
""",
"""md:
## Case 3: Trace iron with seven candidate precipitates

**Setup.** Case 2's brine plus 10⁻⁶ mol/kgw Fe(III), with seven candidate phases: six iron phases and gypsum. It is solved at pH 7 and again at pH 7.5. **Expected.** Gypsum still precipitates. The iron goes into the most stable Fe phase, hematite (Fe₂O₃), and every other Fe phase stays undersaturated.

This case separates a correct solver from the SI² objective most sharply. That objective can push several Fe phases towards saturation at the same time, although only one of them can coexist with the fixed iron total.

The bar charts show the ideal-activity SI of each phase. The black ticks are PHREEQC's values. The two phases that are present have SI = 0, so their bars have zero length.
""",
"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, name in zip(axes, ["Case 3", "Case 3 (pH 7.5)"]):
    for activity in ["ideal", "davies"]:
        s, res, dt = solve_case(name, activity)
        print(f"--- {name}, {activity} ({dt:.2f} s), converged = {res.converged}, present = {res.present}")
        compare_phreeqc(name, res, activity)
        case = CASES[name]
        c = verify(s, res, case["totals"], case["pH"], activity, composition(case))
        checks[(name, activity)] = (res, c)
        show_checks("independent check", c)
    res = checks[(name, "ideal")][0]
    phases = list(res.SI)
    ax.barh(phases, [res.SI[p] for p in phases], color=["C2" if p in res.present else "C0" for p in phases], label="discopt.speciation (ideal)")
    ax.plot([CASES[name]["phreeqc_ideal"]["SI"][p] for p in phases], phases, "k|", ms=14, mew=2, label="PHREEQC (ideal)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(right=1.0)
    ax.set(title=f"{name}: saturation indices", xlabel="SI = log10(Q/K)")
axes[0].legend(fontsize=8, loc="lower left")
plt.tight_layout()
""",
"""md:
**Result.** At both pH values the assemblage is hematite + gypsum, in both codes. Hematite is 5.0 × 10⁻⁷ mol/kgw, i.e. all 10⁻⁶ mol of Fe as Fe₂O₃. Gypsum is unchanged from Case 2. The closest competitors are Fe(OH)₂.₇Cl₀.₃ (SI ≈ −0.8) and goethite (SI = −1.2).

Raising pH to 7.5 shifts only the phases whose dissolution reactions release a different number of H⁺ than hematite does: magnesioferrite rises by 1.0 and K-jarosite falls by 1.5. The goethite and ferrihydrite SI stay the same, because saturated hematite fixes the Fe³⁺ and OH⁻ activity product for them. Every SI agrees with PHREEQC to within 5 × 10⁻⁴.
""",
"""md:
## E1 + E2 summary

The table collects every run above. Its columns are:

- whether the phase assemblage matches PHREEQC (✓);
- the largest relative difference in solid amount and the largest SI difference from PHREEQC;
- the three independent-checker residuals.
""",
"""
print(f"{'case':<18}{'activity':<9}{'assemblage (ours = PHREEQC)':<30}{'max rel dev n':>14}{'max |dSI|':>11}"
      f"{'balance':>10}{'mass act.':>10}{'compl.':>10}")
for (name, activity), (res, c) in checks.items():
    ref = CASES[name]["phreeqc_ideal" if activity == "ideal" else "phreeqc_activity"]
    same = set(res.present) == {k for k, v in ref["n"].items() if v > 0}
    dev = max(abs(res.solids[k] - v) / v for k, v in ref["n"].items() if v > 0)
    dsi = max(abs(res.SI[k] - v) for k, v in ref["SI"].items())
    print(f"{name:<18}{activity:<9}{(', '.join(res.present) + ('  ✓' if same else '  ✗')):<30}{dev:>14.1e}{dsi:>11.1e}"
          f"{c['balance']:>10.1e}{c['mass_action']:>10.1e}{c['complementarity']:>10.1e}")
""",
"""md:
**Result.**
- **Assemblages.** Identical to PHREEQC in all eight runs.
- **Ideal runs.** Solid amounts agree to 5 × 10⁻⁵ relative and SI to 2 × 10⁻⁴. This is the size of PHREEQC's small water-mass and water-activity effects, which are not modeled here.
- **Activity-corrected runs.** Amounts agree to 5 × 10⁻⁴ (0.05%) and SI to 4.4 × 10⁻⁴. This is consistent with Davies versus PHREEQC's mixed Debye–Hückel/Davies model.
- **Independent checker.** Every run passes: the largest balance residual is 3 × 10⁻¹² and the largest mass-action and complementarity residuals are below 10⁻¹⁴, with no supersaturated solid and no negative amount.
""",
"""md:
## E3: Three algorithms, one answer

**Goal.** Show that the answer does not depend on one algorithm's quirks. `equilibrate(method=...)` offers three methods that find the phase assemblage in different ways:

- `"complementarity"` (default): a smoothed Fischer–Burmeister reformulation of the complementarity conditions, solved by homotopy (Fischer 1992, [doi:10.1080/02331939208843795](https://doi.org/10.1080/02331939208843795); Kanzow 1996, [doi:10.1137/S0895479894273134](https://doi.org/10.1137/S0895479894273134));
- `"gibbs"`: convex minimization of the Gibbs energy, which never looks at saturation indices (Smith & Missen 1982);
- `"enumerate"`: solves every one of the $2^{n_\\text{solids}}$ assemblages (128 for Case 3, so it takes 30 s or so) and keeps those that satisfy the equilibrium conditions. If more than one passes the numerical tolerance, it keeps the one with the lowest $G$.

The cell compares each method's molalities and solid amounts with the complementarity result.
""",
"""
for name in ["Case 1", "Case 2", "Case 3"]:
    results = {}
    for method in ["complementarity", "gibbs", "enumerate"]:
        s, res, dt = solve_case(name, "ideal", method)
        results[method] = res
        print(f"{name:8s} {method:16s} converged={res.converged}  present={res.present}  ({dt:5.2f} s)")
    base = results["complementarity"]
    for method in ["gibbs", "enumerate"]:
        r = results[method]
        dm = max(abs(r.molality[k] - v) / v for k, v in base.molality.items() if v > 1e-30)
        dn = max((abs(r.solids[k] - v) / v for k, v in base.solids.items() if v > 0), default=0.0)
        print(f"{'':8s}   {method} vs complementarity: max rel diff molality {dm:.1e}, solids {dn:.1e}")
    notes = [n for n in results["enumerate"].info.notes if "assemblages" in n]
    print(f"{'':8s}   enumeration: {notes[0] if notes else 'exactly one assemblage satisfies the equilibrium conditions'}\\n")
""",
"""md:
**Result.** All three methods return the same assemblage in every case. Molalities agree to 2 × 10⁻¹⁰ relative or better, and solid amounts to 3 × 10⁻¹¹ or better. For Cases 1 and 2, enumeration finds exactly one valid assemblage. For Case 3, two assemblages pass the enumeration tolerance, and the lower-Gibbs-energy one is hematite + gypsum, the same answer as the other two methods.
""",
"""md:
## E4: Invariances (counterexamples to the SI² objective)

**Goal.** Test properties any true equilibrium must have, using tiny systems built to break the SI² objective. M, L, N and X are generic aqueous species, ML is a complex, and NL(s), MX(s) and MX′(s) are solids. A true equilibrium must be unchanged by:

- **CE1:** listing a less stable polymorph (ferrihydrite next to goethite);
- **CE2 → CE3:** adding an inert, undersaturated candidate phase MX′(s);
- **CE2 → CE4:** writing the MX(s) reaction doubled, with doubled log K.

CE2 also tests coupling through a complex: NL(s) precipitation changes the free-ligand concentration, and so the MX(s) saturation. The reference amounts are the independent Pyomo/IPOPT active-set solutions in [`data/notebook_reference.json`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/data/notebook_reference.json).

We also print the paper's objective $z$ evaluated at the true equilibrium. If $z$ were a valid criterion it would be zero there, and the same for all equivalent descriptions of one system.
""",
"""
def ce2_system(extra_solid=False, double_MX=False):
    s = System("CE2")
    M, L, ML, N, X = (s.aqueous(n) for n in ("M", "L", "ML", "N", "X"))
    NL, MX = s.solid("NL(s)"), s.solid("MX(s)")
    s.reaction(M + L >> ML, logK=3.0)
    s.reaction(NL >> N + L, logK=-4.0)
    if double_MX:
        s.reaction(2 * MX >> 2 * M + 2 * X, logK=-12.0)
    else:
        s.reaction(MX >> M + X, logK=-6.0)
    if extra_solid:
        s.reaction(s.solid("MX'(s)") >> M + X, logK=-3.0)
    return s

def ce1_system():
    s = System("CE1")
    Fe, NO3, H, OH = s.aqueous("Fe+3"), s.aqueous("NO3-"), s.aqueous("H+"), s.aqueous("OH-")
    s.reaction({H: 1, OH: 1}, logK=-14.0)
    s.reaction(s.solid("Ferrihydrite") >> Fe + 3 * OH, logK=4.891 - 42.0)
    s.reaction(s.solid("Goethite") >> Fe + 3 * OH, logK=-43.0)
    return s

CE2_IN = {"M": 1e-3, "L": 2e-2, "N": 2e-2, "X": 1e-3}
CE1_IN = {"Fe+3": 1e-3, "NO3-": 3e-3, "H+": 1e-7, "OH-": 1e-7}
systems = {"CE1": (ce1_system(), CE1_IN), "CE2": (ce2_system(), CE2_IN),
           "CE3": (ce2_system(extra_solid=True), CE2_IN), "CE4": (ce2_system(double_MX=True), CE2_IN)}

ce = {}
print(f"{'':5}{'solid':<14}{'discopt.speciation':>16}{'reference':>14}{'SI':>10}   {'z_paper at equilibrium':>22}")
for key, (s, amounts) in systems.items():
    res = s.equilibrate(amounts)
    c = verify(s, res, amounts)
    ce[key] = res
    for i, (k, v) in enumerate(REF[key]["n"].items()):
        label = key if i == 0 else ""
        z_col = f"{c['z_paper']:22.4f}" if i == 0 else ""
        print(f"{label:5}{k:<14}{res.solids[k]:>16.10e}{v:>14.6e}{res.SI[k]:>10.4f}   {z_col}")
    show_checks(f"  {key} independent check", c)

print("\\nmax relative change in aqueous state, CE3 vs CE2:", max(abs(ce["CE3"].molality[k] / v - 1) for k, v in ce["CE2"].molality.items()))
print("max relative change in aqueous state, CE4 vs CE2:", max(abs(ce["CE4"].molality[k] / v - 1) for k, v in ce["CE2"].molality.items()))
""",
"""md:
**Result.**
- **CE1.** Goethite precipitates and ferrihydrite stays undersaturated (SI = −5.89).
- **CE2, CE3 and CE4.** They give identical states: NL(s) = 9.537 × 10⁻³ mol/kgw, MX(s) absent with SI = −1.02, and a relative change in the aqueous state of exactly 0. All amounts match the reference. Its CE4 value of 10⁻³³ for MX(s) is zero to solver precision.
- **The objective.** $z$ at this single equilibrium is 1.05 for CE2, 17.24 for CE3 (the inert phase adds its SI²) and 4.19 for CE4 (doubling the reaction doubles its deviation). A method that minimizes $z$ would therefore move away from the equilibrium, by a different amount for each description.
""",
"""md:
**Scan across a phase boundary.** Next we vary the X input from 10⁻⁴ to 5 × 10⁻² mol/kgw. Lines are `discopt.speciation` on an 80-point grid, and circles are the independent active-set reference.

**Result.** NL(s) is present throughout. MX(s) is zero until X ≈ 10⁻² mol/kgw, then grows. Above that point NL(s) also rises slightly, because MX(s) formation pulls M out of the ML complex and frees L. Every reference point lies on the curves, and no MX(s) appears below the boundary.
""",
"""
X0 = np.logspace(-4, np.log10(5e-2), 80)
n_NL, n_MX = [], []
s = ce2_system()
for x in X0:
    r = s.equilibrate(dict(CE2_IN, X=x))
    assert r.converged
    n_NL.append(r["NL(s)"])
    n_MX.append(r["MX(s)"])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogx(X0, np.array(n_NL) * 1e3, label="NL(s)")
ax.semilogx(X0, np.array(n_MX) * 1e3, label="MX(s)")
ax.semilogx([p["X0"] for p in REF["CE2_scan"]], [p["n"]["NL(s)"] * 1e3 for p in REF["CE2_scan"]], "ko", mfc="none", label="reference")
ax.semilogx([p["X0"] for p in REF["CE2_scan"]], [p["n"]["MX(s)"] * 1e3 for p in REF["CE2_scan"]], "ko", mfc="none")
ax.set(xlabel="X input (mol/kg)", ylabel="solid amount (mmol/kg)", title="CE2 scan")
ax.legend()
plt.tight_layout()
""",
"""md:
## E5: The computed state minimizes the Gibbs energy

**Goal.** Check the second law directly. At constant $T$, $P$ and composition, equilibrium is the global minimum of $G$ over all states reachable by the reactions (Smith & Missen 1982). For ideal activities,

$$\\frac{G}{RT} = \\sum_{i\\in\\text{aq}} m_i\\left(\\mu^\\circ_i + \\ln m_i - 1\\right) + \\sum_{k\\in\\text{solids}} n_k \\mu^\\circ_k,\\qquad \\sum_i \\nu_{ij}\\,\\mu^\\circ_i = -\\ln(10)\\log_{10} K_j.$$

We compute $\\mu^\\circ$ here directly from the log K values. Any solution of the second equation gives the same $G$ differences between reachable states. We then perturb the computed state along 5000 random feasible reaction-extent directions $d = N\\xi$, where absent solids may only form. If the state were not the minimum, some directions would lower $G$.

Trace species (down to ~10⁻¹⁵ mol/kg) limit the step size, so a naive $G(x+\\delta)-G(x)$ is lost to round-off. We therefore evaluate the exact difference in cancellation-free form:
$$\\Delta G/RT = \\nabla G\\cdot\\delta + \\sum_{i\\in\\text{aq}}\\left[(m_i+\\delta_i)\\ln\\!\\left(1+\\delta_i/m_i\\right)-\\delta_i\\right].$$
The second term is nonnegative for any $\\delta$. The first term is the real test: $\\nabla G\\cdot d$ is a combination of reaction affinities, $\\ln 10\\,(\\log Q_j-\\log K_j)$. It vanishes only if mass action holds, and it is nonnegative only if every absent solid is undersaturated. Both are computed from $\\mu^\\circ$ and the returned molalities, not from the solver's residuals.

$G$ is strictly convex in the aqueous amounts. A state where no feasible direction has a negative directional derivative is therefore the unique global minimum.
""",
"""
def gibbs_perturbations(system, res, pH=None, samples=5000, seed=1):
    \"\"\"Directional derivative and exact ΔG/RT for random feasible reaction-extent steps (ideal activities).\"\"\"
    fixed = {"H+": -pH} if pH is not None else {}
    names = [sp.name for sp in system.species if sp.name not in fixed]
    solid = np.array([system[n].is_solid for n in names])
    N = np.array([[r.stoich.get(system[n], 0.0) for r in system.reactions] for n in names])
    logK = np.array([r.logK - sum(c * fixed[sp.name] for sp, c in r.stoich.items() if sp.name in fixed)
                     for r in system.reactions])
    mu0, *_ = np.linalg.lstsq(N.T, -LN10 * logK, rcond=None)
    x = np.array([res.molality[n] if n in res.molality else res.solids[n] for n in names])
    grad = np.where(solid, mu0, mu0 + np.log(np.where(solid, 1.0, x)))   # dG/dx at the computed state

    rng, first, total = np.random.default_rng(seed), [], []
    absent = solid & (x <= 0)
    aq = ~solid
    while len(total) < samples:
        d = N @ rng.normal(size=N.shape[1])
        if np.any(d[absent] < 0):
            if np.all(d[absent] <= 0):
                d = -d
            else:
                continue
        movable = (~absent) & (d != 0)
        t = 0.1 / np.max(np.abs(d[movable]) / x[movable])   # change no amount by more than 10%
        delta = t * d
        # exact, cancellation-free: G(x+δ) - G(x) = ∇G·δ + Σ_aq [(m+δ) log1p(δ/m) - δ]
        lin = grad @ delta
        curv = np.sum((x[aq] + delta[aq]) * np.log1p(delta[aq] / x[aq]) - delta[aq])
        first.append(lin / t)
        total.append(lin + curv)
    return np.array(first), np.array(total)

fig, ax = plt.subplots(figsize=(6, 3.6))
for label, (system, res, pH) in {
    "CE2": (systems["CE2"][0], ce["CE2"], None),
    "Case 2 (ideal)": (minteq_system(CASES["Case 2"]), checks[("Case 2", "ideal")][0], 7.0),
    "Case 1 (ideal)": (minteq_system(CASES["Case 1"]), checks[("Case 1", "ideal")][0], 5.0),
}.items():
    slope, dG = gibbs_perturbations(system, res, pH)
    print(f"{label:16s} {len(dG)} feasible directions: min directional derivative = {slope.min():+.1e}, "
          f"min ΔG/RT = {dG.min():.2e}, fraction with ΔG > 0: {np.mean(dG > 0):.4f}")
    ax.hist(np.log10(dG[dG > 0]), bins=60, histtype="step", label=label)
ax.set(xlabel="log10(ΔG/RT) for random feasible perturbations", ylabel="count")
ax.legend(fontsize=8)
plt.tight_layout()
""",
"""md:
**Result.** For all 5000 directions in each system, $\\Delta G > 0$ (fraction = 1.0000).

- **Directional derivatives.** The smallest is +1.8 × 10⁻⁴ (CE2), −1.2 × 10⁻¹³ (Case 2; zero to round-off) and +6.5 × 10⁻³ (Case 1). Positive minima mean every feasible direction moves toward an undersaturated solid or away from mass action at first order.
- **Histogram.** The spread of $\\Delta G$ between systems has nothing to do with correctness. Each step is capped so that no amount changes by more than 10%, and trace species (down to about 10⁻¹⁵ mol/kgw in Case 1) force very small steps.
""",
"""md:
## Summary

For every example in the paper and every counterexample:

- **E1:** the returned state satisfies conservation, mass action and solid complementarity to machine precision (every checker residual is 3 × 10⁻¹² or smaller). This was checked from the reaction data and formulas alone, with activity coefficients recomputed from scratch.
- **E2:** the phase assemblages match PHREEQC exactly. Amounts agree to 5 × 10⁻⁵ (ideal) and 5 × 10⁻⁴ (activity-corrected), and SI to 5 × 10⁻⁴. These gaps are the size of the known model differences, such as water activity and Debye–Hückel versus Davies.
- **E3:** complementarity homotopy, Gibbs minimization and exhaustive enumeration give the same state. Where enumeration finds more than one assemblage within tolerance (Case 3), the lowest-$G$ one is that same state.
- **E4:** adding polymorphs or inert phases, or rescaling a reaction, leaves the state unchanged. The SI² objective's value at that same state changes with these choices, so minimizing it cannot give a unique equilibrium.
- **E5:** none of the 5000 random feasible reaction-extent perturbations per system lowers the Gibbs energy. The directional derivatives are nonnegative to within round-off, about 10⁻¹³.

**Using this on your own problems.** `verify` and `gibbs_perturbations` work on any `System`, so you can reuse them. Every result also carries `res.residuals` and `res.converged`; see [`docs/validation.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/validation.md).

## Next steps

- [`03-databases-and-your-own-data.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/03-databases-and-your-own-data.ipynb): load full PHREEQC databases instead of hand-built case files.
- [`04-fitting-thermodynamic-data.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/04-fitting-thermodynamic-data.ipynb): estimate $\\log K$ values from data.
- [`docs/validation.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/validation.md): the full PHREEQC comparison table, covering Truesdell–Jones, Pitzer, LLNL, SIT, exchange, surfaces and gases.
- [`docs/concepts.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/concepts.md): the equations, methods and conventions.
""",
"""md:
## References

- Laliwala, C.A., Amusat, O.O., and Torres, A.I. (2025). An optimization-based law of mass action precipitation/dissolution model. *Systems and Control Transactions* 4, 2140–2145. [doi:10.69997/sct.132742](https://doi.org/10.69997/sct.132742)
- Parkhurst, D.L., and Appelo, C.A.J. (2013). Description of input and examples for PHREEQC version 3. USGS Techniques and Methods 6-A43. [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)
- Allison, J.D., Brown, D.S., and Novo-Gradac, K.J. (1991). MINTEQA2/PRODEFA2 version 3.0 user's manual. EPA/600/3-91/021. [OSTI 5673069](https://www.osti.gov/biblio/5673069)
- Davies, C.W. (1938). The extent of dissociation of salts in water. Part VIII. *J. Chem. Soc.*, 2093. [doi:10.1039/JR9380002093](https://doi.org/10.1039/JR9380002093)
- Truesdell, A.H., and Jones, B.F. (1974). WATEQ, a computer program for calculating chemical equilibria of natural waters. *J. Res. U.S. Geol. Survey* 2, 233–248. [USGS](https://pubs.usgs.gov/publication/70232379)
- Fischer, A. (1992). A special Newton-type optimization method. *Optimization* 24, 269–284. [doi:10.1080/02331939208843795](https://doi.org/10.1080/02331939208843795)
- Kanzow, C. (1996). Some noninterior continuation methods for linear complementarity problems. *SIAM J. Matrix Anal. Appl.* 17, 851–868. [doi:10.1137/S0895479894273134](https://doi.org/10.1137/S0895479894273134)
- Smith, W.R., and Missen, R.W. (1982). *Chemical Reaction Equilibrium Analysis: Theory and Algorithms.* Wiley.

The full bibliography is in [`docs/references.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/references.md).
""",
]

# ============================================================================ 03
DATABASES = [
r"""md:
# PHREEQC databases and your own data

Every equilibrium calculation needs thermodynamic data: the species that exist, the equilibrium constant $\log K$ of each reaction, and the parameters of the activity model. This notebook shows how to:

1. load one of the five bundled PHREEQC databases and build a system from it;
2. change a constant or an activity parameter from a database, or add species and reactions to it;
3. declare your own species, reactions and activity model in Python;
4. write a small database of your own in PHREEQC format;
5. describe all of this in a YAML specification;
6. choose an activity model by solving one problem with two databases.

**Prerequisites.** Read notebook 01 first; it introduces `System`, `equilibrate` and the result object. Notebook 04 covers one more source of data: estimating constants from measurements.

**Units and conventions.** Amounts are in mol. Concentrations are molalities, mol per kg of water (written m). $\log K$ is base 10, for the reaction as written, at 25 °C unless a temperature expression is given.

Reference documentation: [`docs/databases.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/databases.md) and [`docs/custom-data.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/custom-data.md).
""",
r"""
import numpy as np
import matplotlib.pyplot as plt
import yaml
from discopt.speciation import (PhreeqcDatabase, available_databases, System, SIT, from_yaml, solve_spec, plot)

for name in available_databases():
    d = PhreeqcDatabase.load(name)
    print(f"{name:16s} {len(d.species):5d} aqueous species {len(d.phases):5d} phases  "
          f"default activity model: {type(d.default_activity()).__name__}")
""",
r"""md:
## 1. A bundled database

A PHREEQC database is a text file of species, reactions and phases with their constants (Parkhurst and Appelo 2013, [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)). The five bundled files, and the activity model each one selects, are listed above. Below, $I = \tfrac12\sum_i m_i z_i^2$ is the ionic strength.

| file | activity model | typical use |
|------|----------------|-------------|
| [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) | Truesdell-Jones | fresh and brackish water, I ≲ 1 m |
| [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat) | Pitzer | brines and evaporites |
| [`wateq4f_PWN.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat) | Truesdell-Jones | natural waters, trace metals |
| [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) | B-dot | many elements including rare earths, 0–300 °C |
| [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) | SIT | ThermoChimie: actinides, radioactive waste |

Sources, provenance and terms of use are in [`docs/databases.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/databases.md) and the References at the end.

Parsing gives plain Python objects. Each species and phase keeps its reaction, `log_k`, `delta_h`, `-analytic` coefficients and `-gamma` parameters.
""",
r"""
db = PhreeqcDatabase.load("phreeqc.dat")
print(len(db.masters), "master species,", len(db.species), "aqueous species,", len(db.phases), "phases")
for name in ["Calcite", "Aragonite", "CO2(g)"]:
    e = db.phases[name]
    print(f"{name:10s} {e.nu}\n{'':10s} log_k={e.logK}  delta_h={e.dH:.4g} J/mol  analytic={e.analytic}")
db.species["CaHCO3+"]
""",
r"""md:
**Reading the output.** [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) has 50 master species, 231 aqueous species and 71 phases.

- `nu` holds stoichiometric coefficients: negative for reactants, positive for products. Calcite dissolves as Calcite = Ca²⁺ + CO₃²⁻.
- `log_k` is $\log_{10} K$ at 25 °C for that reaction; for calcite $K_{sp} = 10^{-8.48}$.
- `delta_h` is the reaction enthalpy, converted to J/mol.
- `analytic` holds coefficients of a $\log K(T)$ expression. When present, it takes precedence over `log_k` and `delta_h`, as in PHREEQC.
- Aqueous species such as CaHCO₃⁺ are formation reactions from master species; `a0` (ion size, Å) and `b` are their Truesdell-Jones parameters.
""",
r"""md:
**Building a system.** `db.system(components, phases=...)` builds a `System` containing every species that the requested elements, or valence states such as `"C(4)"` (carbonate carbon), allow. The `phases` are *candidates*: each is present only if the solution becomes saturated with it.

The example is calcite and aragonite, two polymorphs of CaCO₃, in water open to atmospheric CO₂:
- `{"Calcite": 1.0}` adds 1 mol of calcite to 1 kg of water;
- `fixed={"CO2(g)": -3.4}` fixes $\log_{10} P_{\mathrm{CO_2}} = -3.4$ (atm), close to the atmospheric value.

Only calcite should remain, because aragonite is the metastable polymorph.
""",
r"""
s = db.system(["Ca", "C(4)"], phases=["Calcite", "Aragonite", "CO2(g)"])
print(len(s.species), "species:", ", ".join(sp.name for sp in s.species))
r = s.equilibrate({"Calcite": 1.0}, fixed={"CO2(g)": -3.4})
print(r.summary())
""",
r"""md:
**Result.** The solution reaches pH 8.21 and 0.9995 mol of calcite remains. The `SI` column is the saturation index, $\mathrm{SI} = \log_{10}(\mathrm{IAP}/K)$, where IAP is the ion activity product: SI = 0 at saturation, SI < 0 when undersaturated. Calcite has SI = 0 and aragonite SI = −0.144, so aragonite is correctly absent.

This is *complementarity*: for each solid, either its amount is zero or its SI is zero. The `residuals` line shows that mass balance, mass action and complementarity hold to round-off.

**Metastable equilibrium.** To compute the equilibrium with aragonite, for example to interpret a precipitate that has not yet converted to calcite, remove the stable phase with `suppress=["Calcite"]`. The helper `dissolved_ca` sums the molalities of all calcium species.
""",
r"""
def dissolved_ca(res):
    return sum(v for k, v in res.molality.items() if k.startswith("Ca"))

meta = s.equilibrate({"Aragonite": 1.0}, fixed={"CO2(g)": -3.4}, suppress=["Calcite"])
print(f"stable:     present={r.present}  Ca={dissolved_ca(r):.4e} m  SI(Aragonite)={r.SI['Aragonite']:+.3f}")
print(f"metastable: present={meta.present}  Ca={dissolved_ca(meta):.4e} m")
""",
r"""md:
**Result.** With calcite suppressed, aragonite is saturated and dissolved calcium rises from 5.33 × 10⁻⁴ to 6.01 × 10⁻⁴ m, as expected for the more soluble polymorph.
""",
r"""md:
## 2. Changing and extending database values

Every reaction in the system is a `Reaction` named after its database species or phase. Its constants are ordinary attributes that you can change.

**Precedence.** An `analytic` expression takes precedence over `logK`/`dH`. To use your own 25 °C value, set `analytic = None` first; otherwise your `logK` is ignored.

The cell below tries two values of the calcite constant, then restores the database values so later cells are unaffected.
""",
r"""
calcite = next(rx for rx in s.reactions if rx.name == "Calcite")
print(calcite.equation(), "| logK(25 C) from analytic:", round(calcite.logK_at(298.15), 4))
saved = (calcite.logK, calcite.dH, calcite.analytic)

calcite.analytic, calcite.logK = None, -8.40        # e.g. your own measured value
r2 = s.equilibrate({"Calcite": 1.0}, fixed={"CO2(g)": -3.4})
print(f"logK -8.40: present={r2.present}, dissolved Ca {dissolved_ca(r):.4e} -> {dissolved_ca(r2):.4e} m")

calcite.logK = -8.30                                 # now less stable than aragonite (logK -8.336)
r3 = s.equilibrate({"Calcite": 1.0}, fixed={"CO2(g)": -3.4})
print(f"logK -8.30: present={r3.present}, dissolved Ca {dissolved_ca(r3):.4e} m (set by aragonite)")

calcite.logK, calcite.dH, calcite.analytic = saved   # restore
""",
r"""md:
**Result.**
- With $\log K = -8.40$, calcite is more soluble than with the database value (−8.48) but still controls the solution; dissolved calcium rises from 5.33 × 10⁻⁴ to 5.70 × 10⁻⁴ m.
- With $\log K = -8.30$, calcite is more soluble than aragonite ($\log K = -8.336$). The calculation precipitates aragonite instead, and dissolved calcium is 6.01 × 10⁻⁴ m, the metastable aragonite value found above.

Changing one constant can change which phase is stable, and complementarity handles that without any change to the problem set-up.
""",
r"""md:
**Adding a species.** [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) has no CaCl⁺ complex. The cell below adds one, with an illustrative constant, to gypsum (CaSO₄·2H₂O) in 0.5 m NaCl. A new species needs two calls: `aqueous` declares it with its activity parameters, and `reaction` links it to existing species.

**Activity parameters.** Per-species activity parameters live in `species.params`, a dict you can edit. The second part of the cell changes the Truesdell-Jones ion size `a0` of Ca²⁺, then restores it.
""",
r"""
g = db.system(["Ca", "S(6)", "Na", "Cl"], phases=["Gypsum"])
brine = {"Gypsum": 1.0, "Na": 0.5, "Cl": 0.5}                 # gypsum in 0.5 m NaCl
before = g.equilibrate(brine)
print("CaCl+ in database system:", "CaCl+" in g)

g.aqueous("CaCl+", a0=5.0, b=0.0)
g.reaction("Ca+2 + Cl- = CaCl+", logK=0.4, name="CaCl+")     # illustrative value
after = g.equilibrate(brine)
print(f"add CaCl+: dissolved Ca {dissolved_ca(before):.4e} -> {dissolved_ca(after):.4e} m; "
      f"CaCl+ = {after.molality['CaCl+']:.3e} m")

print("Ca+2 params:", g["Ca+2"].params)
g["Ca+2"].params["a0"] = 6.0                                  # illustrative: larger ion size
bigger = g.equilibrate(brine)
g["Ca+2"].params["a0"] = 5.0                                  # restore
print(f"a0 5 -> 6: gamma(Ca+2) {after.gamma['Ca+2']:.4f} -> {bigger.gamma['Ca+2']:.4f}, "
      f"dissolved Ca {dissolved_ca(after):.4e} -> {dissolved_ca(bigger):.4e} m")
""",
r"""md:
**Result.**
- Adding CaCl⁺ raises dissolved calcium from 0.0332 to 0.0372 m: the complex takes up free Ca²⁺, so more gypsum dissolves before saturation.
- A larger `a0` raises γ(Ca²⁺) from 0.254 to 0.298, so gypsum saturates sooner and dissolved calcium falls to 0.0352 m.
""",
r"""md:
## 3. Your own system, with your own activity model

**Declaring species.**
- A species' charge is read from its name.
- Truesdell-Jones and B-dot take per-species `a0` and `b`.
- SIT takes ion-pair interaction coefficients $\varepsilon_{ij}$ (kg/mol).
- Pitzer takes a `PitzerParameters` table.

**Declaring reactions.** Each reaction takes `logK` at 25 °C and, optionally, `dH` (J/mol), `dCp`, `dV` or `analytic` for temperature and pressure dependence.

The cell below builds a system from scratch: NaCl solutions with halite (NaCl(s)) as the candidate solid, using the specific ion interaction theory, SIT (Guggenheim and Turgeon 1955, [doi:10.1039/TF9555100747](https://doi.org/10.1039/TF9555100747), in the NEA form of Grenthe et al. 1992). SIT adds a term $\varepsilon_{ij} m_j$ for each oppositely charged ion to the Debye–Hückel term. Increasing amounts of halite are added to 1 kg of water.
""",
r"""
nacl = System("NaCl with SIT", activity=SIT(eps={("Na+", "Cl-"): 0.03}))
nacl.aqueous("Na+")
nacl.aqueous("Cl-")
nacl.solid("Halite")
nacl.reaction("Halite = Na+ + Cl-", logK=1.570, dH=3841.0, name="Halite")
for m in [0.1, 1.0, 3.0, 8.0]:
    res = nacl.equilibrate({"Halite": m})
    print(f"NaCl added {m:4.1f} m: gamma(Na+)={res.gamma['Na+']:.3f}  dissolved={res.molality['Na+']:.3f} m  present={res.present}")
""",
r"""md:
**Result.** Up to 3 m all the halite dissolves (`present=[]`, no solid). γ(Na⁺) falls from 0.783 at 0.1 m to 0.670 at 1 m, then rises to 0.699 at 3 m as the ε term grows with molality. At 8 m the solution saturates: halite is present with 7.02 m dissolved. That point is beyond the usual SIT range (about 3–4 m), so with a single ε value it is an illustration, not a prediction of halite solubility.
""",
r"""md:
## 4. A database file of your own

Any PHREEQC-format text works, from a full database to a handful of species. `delta_h` is in kJ/mol unless it is followed by `kcal`.
""",
r"""
MINI = '''
SOLUTION_MASTER_SPECIES
H       H+      -1.0    H       1.008
H(1)    H+      -1.0    0
E       e-      0       0.0     0
O       H2O     0       O       16.0
O(-2)   H2O     0       0
Ca      Ca+2    0       Ca      40.08
S       SO4-2   0       SO4     32.064
S(6)    SO4-2   0       SO4
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
'''
mine = PhreeqcDatabase(MINI)          # or PhreeqcDatabase.from_file("mine.dat")
mini = mine.system(["Ca", "S(6)"], phases=["Gypsum"])
full = db.system(["Ca", "S(6)"], phases=["Gypsum"])
for label, sysm in [("mini database", mini), ("phreeqc.dat", full)]:
    res = sysm.equilibrate({"Gypsum": 1.0})
    print(f"{label:14s} gypsum solubility {dissolved_ca(res):.5f} m, pH {res.pH:.3f}, species: {sorted(res.molality)}")
""",
r"""md:
**Result.** The five-species file gives a gypsum solubility of 0.01511 m, against 0.01508 m from [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat). Its pH is exactly 7.000 because it has no HSO₄⁻ or CaOH⁺, so dissolving gypsum neither consumes nor releases H⁺; [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) gives pH 7.064. A small database can reproduce the main quantity, but only the species it contains can appear in the result.
""",
r"""md:
## 5. Everything as a YAML specification

A specification can start from a database, override constants, and add species and reactions. The schema is strict, so a misspelled key is an error. `solve_spec` returns a JSON-serializable dict, and `System.to_spec()` writes any system back out as explicit data.
""",
r"""
spec = yaml.safe_load('''
name: gypsum in brine
database: {file: phreeqc.dat, components: [Ca, "S(6)", Na, Cl], phases: [Gypsum]}
overrides:
  Gypsum: {logK: -4.50}
species:
  - {name: CaCl+, a0: 5.0}
reactions:
  - {equation: "Ca+2 + Cl- = CaCl+", logK: 0.4, name: CaCl+}
problem:
  amounts: {Gypsum: 1.0, Na: 0.5, Cl: 0.5}
''')
out = solve_spec(spec)
print(out["converged"], out["present"], {k: f"{v:.3e}" for k, v in out["molality"].items() if k.startswith("Ca")})

explicit = from_yaml(spec).to_spec()
print("explicit spec sections:", list(explicit), "|", len(explicit["species"]), "species,", len(explicit["reactions"]), "reactions")
""",
r"""md:
**Result.** The specification repeats the CaCl⁺ example of Section 2 with the gypsum constant raised from −4.58 to −4.50. Gypsum is present and total dissolved calcium (Ca²⁺ + CaSO₄ + CaCl⁺) is 0.0413 m, up from 0.0372 m with the database constant. `to_spec()` writes the whole system as explicit data (15 species, 9 reactions) with no reference to [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat), which keeps a calculation reproducible if the database changes.
""",
r"""md:
## 6. Choosing an activity model: gypsum in NaCl brines

The activity $a_i = \gamma_i m_i$ is the effective concentration that enters mass action. The activity model computes the activity coefficients $\gamma_i$, and matters most in concentrated solutions. The same question can be answered with two databases:

- [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) with Truesdell-Jones (Truesdell and Jones 1974, [pubs.usgs.gov/publication/70232379](https://pubs.usgs.gov/publication/70232379)), an ion-association model intended for I ≲ 1 m;
- [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat) with Pitzer's specific-interaction model (Pitzer 1973, [doi:10.1021/j100621a026](https://doi.org/10.1021/j100621a026); database of Plummer et al. 1988, [doi:10.3133/wri884153](https://doi.org/10.3133/wri884153)), intended for brines.

The two agree in dilute solution and diverge as the NaCl concentration grows. Which curve to trust follows from the range each model was parameterized for, not from which one is "more accurate" in general.
""",
r"""
pz = PhreeqcDatabase.load("pitzer.dat")
systems = {"phreeqc.dat (Truesdell-Jones)": db.system(["Ca", "S(6)", "Na", "Cl"], phases=["Gypsum"]),
           "pitzer.dat (Pitzer)": pz.system(["Ca", "S(6)", "Na", "Cl"], phases=["Gypsum"])}
nacl_m = np.array([0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
fig, ax = plt.subplots(figsize=(5, 3.5))
for label, sysm in systems.items():
    sw = plot.sweep(sysm, lambda m: {"amounts": {"Gypsum": 1.0, "Na": m, "Cl": m}}, nacl_m)
    ca = sw.get(dissolved_ca)
    print(label, "converged:", sw.converged.all(), "| dissolved Ca (m):", np.round(ca, 4))
    ax.plot(nacl_m, ca, "o-", label=label)
ax.set(xlabel="NaCl (mol/kg)", ylabel="dissolved Ca (mol/kg)", title="Gypsum solubility, 25 °C")
ax.legend()
plt.tight_layout()
""",
r"""md:
**Result.** Both models give 0.016 m in nearly pure water, and solubility rises steeply as NaCl is added. At 1 m NaCl Pitzer gives 0.0454 m and Truesdell-Jones 0.0413 m; at 3 m both give about 0.057 m. Above 3 m they diverge: Pitzer has a maximum at 3 m and falls to 0.0520 m at 5 m, while Truesdell-Jones keeps rising to 0.0634 m, outside the range it was parameterized for.
""",
r"""md:
**Changing Pitzer parameters.** The Pitzer parameters are a `PitzerParameters` table on the system's activity model. You can change or add entries with `add(kind, species, coefficients)`. The coefficients are PHREEQC's six temperature terms, or a single number.
""",
r"""
gp = systems["pitzer.dat (Pitzer)"]
table = gp.activity.params                     # PitzerParameters of this system
# pair keys are unordered: find the Na+/Cl- entry of B0 (beta0)
key = next(k for k in table.data["B0"] if set(k) == {"Na+", "Cl-"})
original = table.data["B0"][key]               # six temperature coefficients
brine3 = {"Gypsum": 1.0, "Na": 3.0, "Cl": 3.0}  # gypsum in 3 m NaCl
base = gp.equilibrate(brine3)
table.add("B0", list(key), [original[0] + 0.01, *original[1:]])   # raise the 25 °C term by 0.01
changed = gp.equilibrate(brine3)
table.data["B0"][key] = original               # restore
print(f"B0(NaCl) {original[0]:.4f} -> {original[0] + 0.01:.4f}: gamma(Ca+2) {base.gamma['Ca+2']:.4f} -> {changed.gamma['Ca+2']:.4f}, "
      f"dissolved Ca {dissolved_ca(base):.4e} -> {dissolved_ca(changed):.4e}")
""",
r"""md:
**Result.** Raising $\beta^{(0)}$ for Na⁺–Cl⁻ from 0.0753 to 0.0853 raises γ(Ca²⁺) in 3 m NaCl from 0.540 to 0.608, but dissolved calcium changes by only 0.35 % (0.05767 to 0.05788 m). A single activity coefficient can be sensitive to a parameter that the solubility hardly depends on; notebook 04 shows how to find which parameters data can determine.
""",
r"""md:
## Summary

- `PhreeqcDatabase.load(name)` parses a bundled database; `db.system(components, phases=...)` builds a system with the activity model that the file defines.
- Database constants are ordinary attributes. Clear `analytic` before setting your own `logK`. Add species with `aqueous` plus `reaction`; edit activity parameters through `species.params` or the Pitzer table.
- The same data can come from Python, a PHREEQC-format string or file, or a strict YAML specification; `to_spec()` exports any system as explicit data.
- Choose the database, and with it the activity model, to match the ionic strength of your problem.

**Next steps.** Notebook 04 fits constants and activity parameters to measurements; notebook 05 draws diagrams; notebook 06 uses [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) and [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) for rare earths and thorium. Reference pages: [`docs/databases.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/databases.md), [`docs/custom-data.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/custom-data.md), [`docs/activity-models.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/activity-models.md).
""",
r"""md:
## References

- Parkhurst, D.L., and Appelo, C.A.J. (2013). Description of input and examples for PHREEQC version 3. USGS Techniques and Methods 6-A43. [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)
- Plummer, L.N., Parkhurst, D.L., Fleming, G.W., and Dunkle, S.A. (1988). A computer program incorporating Pitzer's equations for calculation of geochemical reactions in brines. USGS WRI Report 88-4153. [doi:10.3133/wri884153](https://doi.org/10.3133/wri884153)
- Appelo, C.A.J. (2015). Principles, caveats and improvements in databases for calculating hydrogeochemical reactions in saline waters from 0 to 200 °C and 1 to 1000 atm. *Applied Geochemistry* 55, 62–71. [doi:10.1016/j.apgeochem.2014.11.007](https://doi.org/10.1016/j.apgeochem.2014.11.007)
- Ball, J.W., and Nordstrom, D.K. (1991). User's manual for WATEQ4F. USGS Open-File Report 91-183. [doi:10.3133/ofr91183](https://doi.org/10.3133/ofr91183)
- Delany, J.M., and Lundeen, S.R. (1991). The LLNL thermochemical data base. LLNL UCID-21658. <https://www.osti.gov/biblio/138193>
- Giffaut, E., et al. (2014). Andra thermodynamic database for performance assessment: ThermoChimie. *Applied Geochemistry* 49, 225–236. [doi:10.1016/j.apgeochem.2014.05.007](https://doi.org/10.1016/j.apgeochem.2014.05.007)
- Truesdell, A.H., and Jones, B.F. (1974). WATEQ, a computer program for calculating chemical equilibria of natural waters. *J. Res. U.S. Geol. Survey* 2(2), 233–248. <https://pubs.usgs.gov/publication/70232379>
- Pitzer, K.S. (1973). Thermodynamics of electrolytes. I. Theoretical basis and general equations. *J. Phys. Chem.* 77, 268–277. [doi:10.1021/j100621a026](https://doi.org/10.1021/j100621a026)
- Guggenheim, E.A., and Turgeon, J.C. (1955). Specific interaction of ions. *Trans. Faraday Soc.* 51, 747. [doi:10.1039/TF9555100747](https://doi.org/10.1039/TF9555100747)
- Grenthe, I., et al. (1992). *Chemical Thermodynamics of Uranium.* NEA-OECD, North-Holland. <https://www.oecd-nea.org/jcms/pl_13704>

Full bibliography: [`docs/references.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/references.md).
""",
]

# ============================================================================ 04
FITTING = [
r"""md:
# Fitting thermodynamic data with confidence intervals

**What you will learn.** How to estimate equilibrium constants ($\log K$), a reaction enthalpy ($\Delta H$) and an activity-model parameter from measurements with `discopt.speciation.fit`, and how to judge how well the data determine them:

1. gypsum solubility in Na₂SO₄ solutions: two $\log K$ values, standard errors, correlations, predictions, profile likelihood, and what happens when you fit too many parameters;
2. gypsum solubility versus temperature: $\log K$ and $\Delta H$ together;
3. calcite in CO₂-bearing water: refitting constants in a PHREEQC database.

**Prerequisites.** Notebook 01 (building a `System`, `equilibrate`, saturation index). Section 3 uses `PhreeqcDatabase` from notebook 03.

**How the fit works.** Every data point is modelled by a complete equilibrium calculation, so a solid can be present at some conditions and absent at others with no special treatment. The fit adjusts the parameters to minimize $\chi^2=\sum_k \big((y_k^{\text{model}}-y_k)/\sigma_k\big)^2$, where $\sigma_k$ is the standard deviation of measurement $k$.

**Statistics used here.**
- *Standard error*: the estimated standard deviation of a fitted parameter.
- *95 % confidence interval*: a range that would contain the true value in 95 % of repeated experiments.
- *Correlation* $r$: near ±1 means the data fix a combination of two parameters but not each one separately.

**Data.** So that the answers can be checked, the "measurements" are generated from a known model with added noise. With real data, replace the generating cells with your own numbers and standard deviations.
""",
r"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from discopt.speciation import System, PhreeqcDatabase, fit
""",
r"""md:
## 1. Gypsum solubility in Na₂SO₄ solutions

**Goal.** Fit the gypsum solubility product $K_{sp}$ and the CaSO₄⁰ ion-pair constant $K_{pair}$ to dissolved-calcium measurements (mol per kg of water, mol/kg).

**Model.** Gypsum dissolves to Ca²⁺ and SO₄²⁻, which also form the neutral CaSO₄⁰ ion pair. Activities (effective concentrations) are corrected with the Davies equation ([Davies 1938](https://doi.org/10.1039/JR9380002093)), which depends on the ionic strength $I=\tfrac12\sum m_i z_i^2$. Added sulfate lowers the solubility (the common-ion effect), while the ion pair and activity corrections raise it.

**What each part of the data determines.** While gypsum is present, the ion-pair molality is fixed at $K_{sp}K_{pair}$ (the pair is neutral, so its activity coefficient is 1), whatever the sulfate level. Free Ca²⁺ falls as sulfate is added. Pure water alone gives only the sum of the two contributions; the sulfate series separates them.

**The code below.** `gypsum_system` builds the model with given constants. `fit.total` makes a function that sums the molalities of Ca²⁺ and CaSO₄⁰. `conditions` gives 1 mol gypsum (an excess of solid) plus $c$ mol Na₂SO₄. The "measured" values are the true model plus 2 % random noise.
""",
r"""
def gypsum_system(logK=-4.58, pair=2.30, dH=-460.0):
    s = System("gypsum", activity="davies")
    Ca, SO4, CaSO4 = s.aqueous("Ca+2"), s.aqueous("SO4-2"), s.aqueous("CaSO4")
    s.aqueous("Na+")
    s.reaction(Ca + SO4 >> CaSO4, logK=pair, dH=5500.0, name="pair")
    s.reaction(s.solid("Gypsum") >> Ca + SO4, logK=logK, dH=dH, name="gypsum")
    return s

dissolved_ca = fit.total("Ca+2", "CaSO4")
def conditions(c, T=298.15):
    return {"amounts": {"Gypsum": 1.0, "Na+": 2 * c, "SO4-2": c}, "T": T}

rng = np.random.default_rng(1)
truth = gypsum_system()
na2so4 = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
measured = np.array([dissolved_ca(truth.equilibrate(**conditions(c))) for c in na2so4])
measured *= 1 + 0.02 * rng.normal(size=measured.size)             # 2 % measurement noise
list(zip(na2so4, measured.round(6)))
""",
r"""md:
**Setting up the fit.**
- `fit.Observation(conditions, quantity, value, sigma)` is one measurement. `sigma` is its standard deviation in the units of the quantity, here 2 % of the value.
- `fit.logK(model, "gypsum", lower=..., upper=...)` declares a parameter to fit, with bounds. The fit starts from the values in `model`, which are deliberately wrong here.
- `absolute_sigma=True` means the `sigma` values are real standard deviations. With the default (`False`) they are only relative weights, and the errors are rescaled by the scatter of the fit.
""",
r"""
model = gypsum_system(logK=-4.0, pair=1.5)                        # deliberately poor starting values
obs = [fit.Observation(conditions(c), dissolved_ca, y, sigma=0.02 * y, label=f"Na2SO4={c}")
       for c, y in zip(na2so4, measured)]
params = [fit.logK(model, "gypsum", lower=-6, upper=-3), fit.logK(model, "pair", lower=0, upper=4)]
res = fit.fit_equilibrium(model, obs, params, absolute_sigma=True)
print(res.summary())
print("truth: logK gypsum = -4.58, logK pair = 2.30")
""",
r"""md:
**Reading the summary.**
- **Recovery.** Both constants are within one standard error of the truth: $\log K_{sp} = -4.573 \pm 0.015$ (true −4.58) and $\log K_{pair} = 2.289 \pm 0.026$ (true 2.30), starting from −4.0 and 1.5.
- **Goodness of fit.** $\chi^2/\text{dof} = 0.74$ (dof = data points − parameters = 5). A value near 1 is what you expect when the model is right and the stated σ are real, which supports `absolute_sigma=True`. Much larger than 1 would mean the model or the σ are wrong.
- **Correlation.** $r = -0.97$: a higher $K_{sp}$ with a lower $K_{pair}$ fits nearly as well. The data still determine both, but each interval is wider than if the other constant were known.

**Predictions.** `res.predict(conditions, quantity)` returns a value and its standard error, propagated from the full parameter covariance, at any conditions, including ones that were not measured. The plot shows the fit with a 95 % band (±1.96 standard errors).
""",
r"""
grid = np.linspace(0, 0.25, 26)
pred = np.array([res.predict(conditions(c), dissolved_ca) for c in grid])
free_ca = np.array([res.predict(conditions(c), "molality:Ca+2") for c in grid])
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.errorbar(na2so4, measured, yerr=0.02 * measured, fmt="ko", label="data (±1σ)")
ax.plot(grid, pred[:, 0], "C0", label="fit: dissolved Ca")
ax.fill_between(grid, pred[:, 0] - 1.96 * pred[:, 1], pred[:, 0] + 1.96 * pred[:, 1], color="C0", alpha=0.3, label="95% band")
ax.plot(grid, free_ca[:, 0], "C1--", label="fit: free Ca²⁺")
ax.set(xlabel="Na₂SO₄ added (mol/kg)", ylabel="mol/kg")
ax.legend()
plt.tight_layout()
""",
r"""md:
**Result.** The fit passes through all seven points. The 95 % band is narrow inside the data and widens slightly beyond 0.2 mol/kg, where there are no measurements. The gap between dissolved Ca and free Ca²⁺ is the CaSO₄⁰ ion pair. It stays at about 0.005 mol/kg at every sulfate level, so at 0.2 mol/kg Na₂SO₄ most dissolved calcium is paired.
""",
r"""md:
### Profile likelihood

**Why.** The intervals in the summary are *linearized*: they assume the model is linear in the parameters near the best fit. A profile-likelihood interval does not ([Venzon & Moolgavkar 1988](https://doi.org/10.2307/2347496)).

**Method.** `res.profile(name)` fixes one parameter at each grid value, re-optimizes the others, and records the minimum $\chi^2$. With absolute σ the 95 % interval is where $\chi^2 \le \chi^2_{\min} + 3.84$ (the 95 % point of $\chi^2$ with one degree of freedom), the dashed line in the plots.

**Interpretation.** When the model is nearly linear, the profile and linearized intervals agree; when they disagree, trust the profile. A profile that never crosses the threshold means the data cannot bound that parameter ([Raue et al. 2009](https://doi.org/10.1093/bioinformatics/btp358)).
""",
r"""
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, name in zip(axs, res.names):
    pr = res.profile(name, n=13)
    pr.plot(ax)
    lo, hi = res.confidence_intervals[name]
    print(f"{name:14s} linearized [{lo:.4f}, {hi:.4f}]   profile [{pr.interval[0]:.4f}, {pr.interval[1]:.4f}]")
plt.tight_layout()
""",
r"""md:
**Result.** Both profiles are close to parabolas, and the profile and linearized intervals agree to within 0.001, so the linearized errors can be trusted here. The `INFO pounce::linesearch` line is a diagnostic from the inner optimizer, not an error.
""",
r"""md:
### Asking for more than the data can tell

**Test.** Refit the same data with a third parameter, the Davies $b$ (the coefficient of the linear $bI$ term, 0.3 in truth). There is no ready-made helper for it, so it is a custom `fit.Parameter(name, set, get)`: `set` writes a value into the model and `get` reads it back. This pattern works for any number in the model.

**What to expect.** In these data $b$ and the ion-pair constant both raise solubility as ionic strength increases, so the data cannot tell them apart. The fit warns when the largest correlation exceeds |r| = 0.99, or when the sensitivity matrix is ill-conditioned (condition number above 10¹⁰).
""",
r"""
model3 = gypsum_system()
davies_b = fit.Parameter("Davies b", set=lambda v: setattr(model3.activity, "b", v),
                         get=lambda: model3.activity.b, lower=-0.5, upper=1.0)
params3 = [fit.logK(model3, "gypsum", lower=-6, upper=-3), fit.logK(model3, "pair", lower=0, upper=4), davies_b]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res3 = fit.fit_equilibrium(model3, obs, params3, absolute_sigma=True)
print(res3.summary())
for w in caught:
    print("\nwarning:", w.message)
print("\nstandard error of logK pair: 2 parameters", round(res.standard_errors["logK pair"], 3),
      "-> 3 parameters", round(res3.standard_errors["logK pair"], 3))
""",
r"""md:
**Result.**
- **The fit is still good** ($\chi^2/\text{dof} = 0.80$): a good fit does not mean well-determined parameters.
- **All three correlations exceed 0.99** in magnitude; the warning names only the largest (gypsum–pair, $r=-0.999$).
- **Intervals widen.** $b = 0.21 \pm 0.17$ (interval −0.13 to 0.55, which includes the true 0.3), and the standard error of $\log K_{pair}$ grows from 0.026 to 0.56, about 20 times.

**What would help.** Fix $b$ at a literature value, or add data that separate the effects, such as a different ionic medium (NaCl) that changes the ionic strength without changing sulfate. The `discopt-doe` plugin can choose such experiments systematically.
""",
r"""md:
## 2. Temperature dependence: fitting logK and ΔH together

**Goal.** Section 1 worked at 25 °C. Here you fit how the gypsum constant changes with temperature.

**Data.** Gypsum solubility in pure water from 5 to 80 °C, with 1 % noise. The truth uses $\Delta H = -4$ kJ/mol; the fit starts from $\Delta H = 0$. The ion-pair constants are held at their true values.

**Model.** The van 't Hoff form $\log K(T) = \log K_{298} - \frac{\Delta H}{R\ln 10}\left(\frac1T - \frac1{298.15}\right)$, with both $\log K_{298}$ and $\Delta H$ fitted.

**Units.** `System.reaction(..., dH=...)` takes J/mol, while `fit.enthalpy` works in kJ/mol so its steps are comparable in size to $\log K$.

**Identifiability.** A wide temperature range is what makes $\Delta H$ identifiable; over a few kelvin, $\log K$ and $\Delta H$ would be almost perfectly correlated.
""",
r"""
T_data = np.array([5, 15, 25, 35, 50, 65, 80]) + 273.15           # K
truthT = gypsum_system(dH=-4000.0)                                  # J/mol here; fit.enthalpy reports kJ/mol
yT = np.array([dissolved_ca(truthT.equilibrate(**conditions(0.0, T))) for T in T_data])
yT *= 1 + 0.01 * rng.normal(size=T_data.size)                       # 1 % measurement noise
modelT = gypsum_system(dH=0.0)
obsT = [fit.Observation(conditions(0.0, T), dissolved_ca, y, sigma=0.01 * y) for T, y in zip(T_data, yT)]
resT = fit.fit_equilibrium(modelT, obsT, [fit.logK(modelT, "gypsum", lower=-6, upper=-3),
                                          fit.enthalpy(modelT, "gypsum", lower=-50, upper=50)], absolute_sigma=True)
print(resT.summary())
print("truth: logK = -4.58, dH = -4.0 kJ/mol (the ion-pair constants are held at their true values)")

Tg = np.linspace(273.15, 363.15, 40)
band = np.array([resT.predict(conditions(0.0, T), dissolved_ca) for T in Tg])
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar(T_data - 273.15, yT, yerr=0.01 * yT, fmt="ko", label="data")
ax.plot(Tg - 273.15, band[:, 0], "C0", label="fit")
ax.fill_between(Tg - 273.15, band[:, 0] - 1.96 * band[:, 1], band[:, 0] + 1.96 * band[:, 1], alpha=0.3)
ax.set(xlabel="T (°C)", ylabel="gypsum solubility (mol/kg)"); ax.legend(); plt.tight_layout()
""",
r"""md:
**Result.**
- **Recovery.** $\log K = -4.5784 \pm 0.0024$ and $\Delta H = -4.12 \pm 0.15$ kJ/mol, both within one standard error of the truth (−4.58, −4.0).
- **Correlation** is only −0.48, because the reference temperature (25 °C) lies inside the data range.
- **Low χ².** $\chi^2/\text{dof} = 0.14$ is below the expected ≈ 1; with only 7 points this can happen by chance.
- **The curve** passes through a shallow minimum near 60 °C: the model combines the negative $\Delta H$ of gypsum with the positive $\Delta H$ of the ion pair and temperature-dependent activity corrections. The band widens outside 5–80 °C, where the curve is extrapolated.
""",
r"""md:
## 3. Fitting against a PHREEQC database: calcite in CO₂-bearing water

**Goal.** The same tools work on a full database system. Start from [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) ([Parkhurst & Appelo 2013](https://doi.org/10.3133/tm6A43)) and refit two of its constants to pH and dissolved-calcium measurements at eight CO₂ partial pressures:

- the calcite solubility product ($\log K \approx -8.48$ for CaCO₃ = CO₃²⁻ + Ca²⁺);
- the CaHCO₃⁺ stability constant ($\log K \approx 11.43$ for Ca²⁺ + CO₃²⁻ + H⁺ = CaHCO₃⁺, as written in the database).

**Conditions.** `"fixed": {"CO2(g)": lp}` holds $\log_{10} p_{\text{CO}_2}$ (atm) constant, like water open to a gas of that composition; excess calcite is present at every point. `fit.logK(...).get()` reads the current database value. Here those values generate the synthetic data, and `initial=` starts the fit elsewhere.

**Temperature dependence.** The database stores these constants as analytic functions of $T$. `fit.logK` shifts the constant term, so the fitted parameter is still $\log K$ at 25 °C and the database's temperature dependence is kept.
""",
r"""
db = PhreeqcDatabase.load("phreeqc.dat")
cal = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
ca_total = fit.total("Ca+2", "CaHCO3+", "CaCO3", "CaOH+")
# database log K values at 25 °C; they generate the synthetic data below
true_values = {"Calcite": fit.logK(cal, "Calcite").get(), "CaHCO3+": fit.logK(cal, "CaHCO3+").get()}

obs_cal = []
for lp in np.linspace(-4.5, -1.0, 8):
    cond = {"amounts": {"Calcite": 1.0}, "fixed": {"CO2(g)": lp}}
    r = cal.equilibrate(cond["amounts"], fixed=cond["fixed"])
    obs_cal.append(fit.Observation(cond, "pH", r.pH + rng.normal(0, 0.01), sigma=0.01))
    obs_cal.append(fit.Observation(cond, ca_total, ca_total(r) * (1 + 0.02 * rng.normal()), sigma=0.02 * ca_total(r)))

p_cal = [fit.logK(cal, "Calcite", initial=-8.0, lower=-10, upper=-6), fit.logK(cal, "CaHCO3+", initial=10.5, lower=8, upper=14)]
res_cal = fit.fit_equilibrium(cal, obs_cal, p_cal, absolute_sigma=True)
print(res_cal.summary())
print("database values used to generate the data:", {k: round(v, 4) for k, v in true_values.items()})
""",
r"""md:
**Result.**
- **Recovery.** $\log K_{\text{Calcite}} = -8.486 \pm 0.009$ (database −8.480) and $\log K_{\text{CaHCO}_3^+} = 11.43 \pm 0.23$ (database 11.435), both within one standard error.
- **Goodness of fit.** $\chi^2/\text{dof} = 1.03$ with 16 observations (8 pH, 8 calcium).
- **CaHCO₃⁺ is less certain.** Its standard error is about 25 times the calcite one because CaHCO₃⁺ is a minor species, so the data are only weakly sensitive to it. The correlation ($r = -0.72$) is moderate: the two constants are separable.
""",
r"""
lps = np.linspace(-5, 0, 21)
pH_band = np.array([res_cal.predict({"amounts": {"Calcite": 1.0}, "fixed": {"CO2(g)": lp}}, "pH") for lp in lps])
data_lp = [o.conditions["fixed"]["CO2(g)"] for o in obs_cal if o.quantity == "pH"]
data_pH = [o.value for o in obs_cal if o.quantity == "pH"]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
ax.errorbar(data_lp, data_pH, yerr=0.01, fmt="ko", label="data")
ax.plot(lps, pH_band[:, 0], "C0", label="fit")
ax.fill_between(lps, pH_band[:, 0] - 1.96 * pH_band[:, 1], pH_band[:, 0] + 1.96 * pH_band[:, 1], alpha=0.3, label="95% band")
ax.set(xlabel="log10 pCO2 (atm)", ylabel="pH at calcite saturation"); ax.legend()
# the band is thinner than the line on the left, so show its half-width directly
ax2.plot(lps, 1.96 * pH_band[:, 1], "C0")
ax2.axvspan(min(data_lp), max(data_lp), color="0.9", label="measured range")
ax2.set(xlabel="log10 pCO2 (atm)", ylabel="95% half-width of predicted pH", ylim=(0, None)); ax2.legend()
plt.tight_layout()
""",
r"""md:
**Result.** The predicted pH at calcite saturation falls linearly with $\log p_{\text{CO}_2}$ through all the data. The 95 % band is too narrow to see on the left, so the right panel plots its half-width: about 0.005 pH units, smallest inside the measured range near $\log p_{\text{CO}_2} = -1$ and rising to 0.006 when extrapolated to 1 atm.
""",
r"""md:
## Summary

| step | call |
|------|------|
| choose what to fit | `fit.logK`, `fit.enthalpy`, `fit.activity_parameter`, or `fit.Parameter(name, set, get)` |
| describe measurements | `fit.Observation(conditions, quantity, value, sigma)`, where the quantity is `"pH"`, `"SI:Gypsum"`, `fit.total(...)`, a callable, ... |
| fit | `res = fit.fit_equilibrium(system, observations, parameters, absolute_sigma=...)` |
| uncertainty | `res.summary()`, `res.confidence_intervals`, `res.correlation`, `res.profile(name)` |
| use the result | the system now holds the fitted values; `res.predict(conditions, quantity)` gives value ± standard error |

**Take-aways.**
- Warnings about ill-conditioning, strong correlation or non-converged points mean the data do not determine the parameters as posed: fix a parameter or add more informative data.
- A good $\chi^2$ does not guarantee well-determined parameters; check correlations and profiles.
- For concentrations spanning decades, fit `"log10_molality:..."` with a constant `sigma`.

## Next steps

- [`docs/fitting.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/fitting.md): all parameter helpers, observation quantities, and practical advice (cost, held-out validation).
- [`docs/activity-models.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/activity-models.md): the activity models whose parameters you can fit.
- Notebook 05 draws speciation and phase diagrams from a model; notebook 06 applies the package to rare earths and thorium.
""",
r"""md:
## References

- Davies, C.W. (1938). The extent of dissociation of salts in water. Part VIII. *J. Chem. Soc.*, 2093. [doi:10.1039/JR9380002093](https://doi.org/10.1039/JR9380002093)
- Parkhurst, D.L., and Appelo, C.A.J. (2013). Description of input and examples for PHREEQC version 3. USGS Techniques and Methods 6-A43. [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)
- Venzon, D.J., and Moolgavkar, S.H. (1988). A method for computing profile-likelihood-based confidence intervals. *Applied Statistics* 37, 87–94. [doi:10.2307/2347496](https://doi.org/10.2307/2347496)
- Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U., and Timmer, J. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25, 1923–1929. [doi:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358)

The full bibliography is in [`docs/references.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/references.md).
""",
]

# ============================================================================ 05
DIAGRAMS = [
r"""md:
# Diagrams: speciation, Pourbaix, phase transitions, Ellingham

`discopt.speciation.plot` builds diagrams from complete equilibrium calculations. Every point on every diagram is a full solve, so aqueous complexes, activity corrections and solid phases all come from one consistent model.

**You will learn to**
- sweep one variable (pH, temperature, CO₂ pressure) and plot speciation, saturation index and solubility along it;
- compute a Pourbaix (Eh–pH) diagram and read its fields and water stability lines;
- draw an Ellingham diagram from $\log K(T)$ functions.

**Prerequisites.** [`01-getting-started.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/01-getting-started.ipynb), which covers systems, `equilibrate`, saturation index (SI) and complementarity. [`03-databases-and-your-own-data.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/03-databases-and-your-own-data.ipynb) shows how `db.system` selects species from a database.

**Data.** All aqueous calculations use [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) (Parkhurst and Appelo 2013, [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)). Amounts are in mol per kg of water (mol/kgw).

**Runtime.** The Pourbaix cell does about 1,400 equilibrium solves and takes 2–3 minutes. The other cells take seconds.
""",
r"""
import time
import numpy as np
import matplotlib.pyplot as plt
from discopt.speciation import PhreeqcDatabase, plot, fit

db = PhreeqcDatabase.load("phreeqc.dat")
""",
r"""md:
## Iron speciation and ferrihydrite precipitation vs pH

**Goal.** Follow 1 mmol/kgw of iron (added with 3 mmol/kgw chloride) from pH 2 to 12. `Fe(OH)3(a)` is amorphous ferric hydroxide (ferrihydrite) and is allowed to precipitate.

**Terms.**
- **pe** $= -\log_{10} a_{e^-}$ sets how oxidizing the solution is. Higher pe favours Fe(III), lower pe favours Fe(II). Holding pe = 4 fixes the redox level, so the Fe(II)/Fe(III) split is computed, not assumed from the salt added.
- **SI** (saturation index) $= \log_{10}(\mathrm{IAP}/K_{sp})$. A solid is present only when SI = 0 and absent when SI < 0 (complementarity).

**Code.** `plot.sweep(system, "pH", values, ...)` runs one `equilibrate` per value. `Sweep.get(quantity)` returns an array along the sweep, for example `"SI:Fe(OH)3(a)"` or `"solids:Fe(OH)3(a)"`. `plot.plot_sweep` plots molalities on a log axis, with solids dashed.
""",
r"""
fe = db.system(["Fe", "Cl"], phases=["Fe(OH)3(a)"])
pHs = np.linspace(2, 12, 41)
sw = plot.sweep(fe, "pH", pHs, amounts={"Fe": 1e-3, "Cl": 3e-3}, pe=4)
print("all converged:", sw.converged.all())
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
plot.plot_sweep(sw, ["Fe+3", "FeOH+2", "Fe(OH)2+", "Fe(OH)3", "Fe(OH)4-", "Fe+2"], ax=axs[0], floor=1e-14)
axs[0].set_title("aqueous Fe (pe = 4)")
# right panel: amount of precipitate (black, left axis) and its saturation index (red, right axis)
axs[1].plot(pHs, sw.get("solids:Fe(OH)3(a)") * 1e3, "k")
axs[1].set(xlabel="pH", ylabel="Fe(OH)3(a) (mmol/kgw)", title="precipitate")
ax2 = axs[1].twinx()
ax2.plot(pHs, sw.get("SI:Fe(OH)3(a)"), "C3--")
ax2.set_ylabel("SI", color="C3")
plt.tight_layout()
""",
r"""md:
**Result.**
- **Low pH.** Below pH ≈ 5.5 all the iron is dissolved, and at pe = 4 almost all of it is Fe²⁺ (1.0 × 10⁻³ mol/kgw). Fe³⁺ is only about 10⁻¹² mol/kgw. SI of Fe(OH)₃(a) rises from −11 at pH 2 to −0.5 at pH 5.5.
- **Onset of precipitation.** SI reaches 0 between pH 5.5 and 5.75, and ferrihydrite appears at the same point. By pH 6.25 about 98 % of the iron is solid. The solid amount is zero wherever SI < 0, which is complementarity at work.
- **Above the onset.** Dissolved Fe(OH)₃ is fixed by the solid at about 2 × 10⁻⁸ mol/kgw. Fe²⁺ falls steeply because ferrihydrite now limits Fe(III) and pe ties Fe(II) to it.
- **High pH.** Fe(OH)₄⁻ grows about tenfold per pH unit (2 × 10⁻⁵ mol/kgw at pH 12), so a little solid redissolves (0.98 mmol/kgw at pH 12).
""",
r"""md:
## Pourbaix (Eh–pH) diagram for iron

**What it shows.** A Pourbaix diagram (Pourbaix 1966) maps which species or solid dominates as a function of acidity (pH, x-axis) and oxidizing power (y-axis). The y-axis is the redox potential Eh in volts, related to pe by $E_h = (RT\ln 10/F)\,\mathrm{pe} \approx 0.0592\,\mathrm{pe}$ V at 25 °C. Oxidized species (Fe³⁺, ferrihydrite) sit at the top and reduced species (Fe²⁺) at the bottom.

**Water stability lines (dashed).** Water is stable only between two lines:
- Above $E_h = 1.23 - 0.0592\,\mathrm{pH}$ it is oxidized to O₂ (1 atm).
- Below $E_h = -0.0592\,\mathrm{pH}$ it is reduced to H₂ (1 atm).

Fields outside these lines cannot persist in open contact with water.

**Method.** `plot.pourbaix` solves the system at every (pH, pe) grid point. It labels each point with the candidate that has the largest amount: molality for aqueous species, mol/kgw for solids. Here there are 31 × 31 points at 10⁻³ mol/kgw Fe and a coarser 21 × 21 grid at 10⁻⁶ mol/kgw to save time.

**Dependence on total concentration.** Textbook diagrams are drawn at one assumed activity. These boundaries move with the total iron, which the second panel shows.
""",
r"""
# candidate species/solids to compare; keep only names that exist in this system
cands = [n for n in ["Fe+2", "Fe+3", "FeOH+", "FeOH+2", "Fe(OH)2+", "Fe(OH)3", "Fe(OH)4-", "Fe(OH)3(a)"] if n in fe]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, total, n in [(axs[0], 1e-3, 31), (axs[1], 1e-6, 21)]:   # (axis, total Fe in mol/kgw, grid points per axis)
    t = time.time()
    pH_grid = np.linspace(1, 13, n)
    pe_grid = np.linspace(-12, 20, n)   # Eh from about -0.71 to +1.18 V
    pr, _ = plot.pourbaix(fe, cands, pH_grid, pe_grid, amounts={"Fe": total, "Cl": 3 * total}, ax=ax)
    ax.set_title(f"Fe total {total:g} mol/kg")
    # pr.grid holds an index into pr.labels for each point (-1 = not converged)
    fields = sorted({pr.labels[i] for i in np.unique(pr.grid) if i >= 0})
    print(f"{total:g}: {pr.grid.size} points in {time.time() - t:.0f} s, fields: {fields}")
plt.tight_layout()
""",
r"""md:
**Result.**
- **Reading the diagram.** Between the water lines, Fe²⁺ fills the low-pH, low-Eh region and ferrihydrite, Fe(OH)₃(a), the oxidized neutral-to-basic region. Fe³⁺ and FeOH²⁺ only appear at pH below about 3 and Eh above about 0.75 V. The sloped Fe²⁺/Fe(OH)₃(a) boundary is the reaction Fe(OH)₃ + 3H⁺ + e⁻ = Fe²⁺ + 3H₂O, which involves both electrons and protons.
- **At 10⁻³ mol/kgw** there are six fields. Ferrihydrite spans pH ≈ 3 to 13 at the top of the diagram, and Fe(OH)₄⁻ is confined to a corner at pH 13, low Eh.
- **At 10⁻⁶ mol/kgw** the ferrihydrite field shrinks to pH ≈ 5.5–10. Fe(OH)₂⁺ gets its own field (pH ≈ 3.7–5.5, high Eh), and Fe(OH)₄⁻ takes over above pH ≈ 10. With less iron, the solid is stable over a narrower range.
- **Caveats.**
  - The step-like boundaries come from the coarse grid. Use more points for smooth lines, at a cost of one solve per point.
  - The FeOH⁺ field sits at very low Eh, around the H₂ line, at the edge of water stability.
  - Metallic iron is not a candidate here, so no Fe(s) field appears.
""",
r"""md:
## Gypsum–anhydrite transition with temperature

The previous sections varied pH and pe. A sweep can also run over temperature, and then the stable solid itself can change.

**Setup.** Gypsum (CaSO₄·2H₂O) and anhydrite (CaSO₄) are both candidate solids. We add 0.5 mol of gypsum per kg of water, far more than dissolves, so one solid is always present. Temperature is swept from 0 to 100 °C in 4 °C steps. Each phase's $\log K(T)$ comes from [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat). Nothing tells the solver where the transition is.
""",
r"""
cs = db.system(["Ca", "S(6)"], phases=["Gypsum", "Anhydrite"])
Ts = np.linspace(0, 100, 26) + 273.15
swT = plot.sweep(cs, "T", Ts, amounts={"Gypsum": 0.5})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
tC = Ts - 273.15
axs[0].plot(tC, swT.get("solids:Gypsum"), label="Gypsum")
axs[0].plot(tC, swT.get("solids:Anhydrite"), label="Anhydrite")
axs[0].set(xlabel="T (°C)", ylabel="mol/kgw", title="stable solid")
axs[0].legend()
# dissolved Ca = sum of the molalities of the main Ca-bearing aqueous species
ca_dissolved = fit.total("Ca+2", "CaSO4", "CaHSO4+")
axs[1].plot(tC, swT.get(ca_dissolved), "k")
axs[1].set(xlabel="T (°C)", ylabel="dissolved Ca (mol/kg)", title="solubility of the stable phase")
plt.tight_layout()
switch = tC[np.argmax(swT.get("solids:Anhydrite") > 0)]
print(f"anhydrite first stable at about {switch:.0f} °C in this grid (phreeqc.dat data)")
""",
r"""md:
**Result.**
- **Stable solid.** Gypsum is the only solid up to 52 °C and anhydrite from 56 °C, so the transition lies between these grid points with [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) data. Below the transition anhydrite is undersaturated (SI = −0.30 at 25 °C), and above it gypsum is.
- **Solubility.** The right panel follows whichever solid is stable. Dissolved Ca rises to about 0.0154 mol/kgw near 40 °C (gypsum), then falls to about 0.006 mol/kgw at 100 °C (anhydrite, whose solubility decreases with temperature). The kink near 55 °C marks the phase change.
""",
r"""md:
## Open-system calcite: pH and solubility vs CO₂ partial pressure

**Goal.** Find the pH and dissolved calcium of water in equilibrium with calcite (CaCO₃) and a CO₂ gas phase whose pressure is held fixed ("open system"). This is the textbook picture of water in contact with limestone.

**Code.** The sweep variable `"fixed:CO2(g)"` holds $\log_{10} p_{\mathrm{CO_2}}$ (atm) at each value, here from −5 to 0 (atmospheric air is about −3.4). One mole of calcite per kg of water is an excess, so calcite stays present.
""",
r"""
cal = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
lps = np.linspace(-5, 0, 21)
swc = plot.sweep(cal, "fixed:CO2(g)", lps, amounts={"Calcite": 1.0})
ca_dissolved = fit.total("Ca+2", "CaHCO3+", "CaCO3", "CaOH+")
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(lps, swc.get("pH"), "C0")
ax.set_xlabel("log10 pCO2 (atm)")
ax.set_ylabel("pH", color="C0")
ax2 = ax.twinx()   # second y-axis on the right
ax2.semilogy(lps, swc.get(ca_dissolved), "C1")
ax2.set_ylabel("dissolved Ca (mol/kg)", color="C1")
plt.tight_layout()
""",
r"""md:
**Result.** More CO₂ makes the water more acidic and dissolves more calcite (CaCO₃ + CO₂ + H₂O = Ca²⁺ + 2HCO₃⁻).

| log pCO₂ (atm) | pH | dissolved Ca (mol/kgw) |
|---|---|---|
| −5 | 9.24 | 1.8 × 10⁻⁴ |
| −3.5 (about air) | 8.28 | 4.9 × 10⁻⁴ |
| −2 (soil gas) | 7.30 | 1.6 × 10⁻³ |
| 0 | 6.00 | 9.3 × 10⁻³ |

pH falls almost linearly with log pCO₂, and solubility rises about 50-fold over the range.
""",
r"""md:
## Ellingham diagram

The last diagram leaves aqueous chemistry. It uses the same $\log K(T)$ machinery for high-temperature oxidation reactions.

**What is plotted.** An Ellingham diagram (Ellingham 1944) plots the standard Gibbs energy of oxide formation, $\Delta G^\circ(T) = -RT\ln 10\,\log K(T)$, against temperature. Every reaction is written per mole of O₂ so the lines can be compared. `plot.ellingham` accepts `Reaction` objects with temperature data, or any function `T -> log10 K`.

**Reading the diagram.**
- **Height.** Lower lines mean more stable oxides. A metal (or C, or H₂) can reduce the oxide of any element whose line lies *above* its own at that temperature. Where two lines cross, the direction of reduction reverses.
- **Slope.** With $\Delta G^\circ \approx \Delta H^\circ - T\Delta S^\circ$, the slope is $-\Delta S^\circ$. Metal oxidation consumes O₂ gas, so $\Delta S^\circ < 0$ and those lines rise. C + O₂ = CO₂ keeps the moles of gas constant, so its line is flat. 2C + O₂ = 2CO doubles the moles of gas, so its line falls.

**Data used here.** The curves are rounded linear fits $\Delta G^\circ \approx A + BT$ (A in kJ/mol O₂, B in kJ/(mol O₂·K)) of standard tabulated data. They are for illustration only, not for design calculations, and ignore melting.
""",
r"""
R = 8.314462618e-3   # gas constant, kJ/(mol K)

# turn a fit dG = A + B*T (kJ/mol O2) into a function T -> log10 K
def from_linear(A, B):
    return lambda T: -(A + B * T) / (R * T * np.log(10))

oxides = {
    "2 Ni + O2 = 2 NiO": from_linear(-471.0, 0.172),
    "2 Fe + O2 = 2 FeO": from_linear(-528.0, 0.130),
    "2 H2 + O2 = 2 H2O(g)": from_linear(-492.0, 0.109),
    "C + O2 = CO2": from_linear(-394.0, -0.001),
    "2 C + O2 = 2 CO": from_linear(-223.0, -0.175),
    "4/3 Al + O2 = 2/3 Al2O3": from_linear(-1120.0, 0.214),
}
fig, ax = plt.subplots(figsize=(6, 4.5))
# every reaction is already written per 1 mol O2; passing `per` labels the axis in kJ/mol O2
T, dG = plot.ellingham(oxides, np.linspace(300, 2000, 60), per={name: 1 for name in oxides}, ax=ax)
ax.legend(fontsize=8)
plt.tight_layout()
""",
r"""md:
**Result.** The axis runs from 27 to 1727 °C.
- **Aluminium.** Al₂O₃ lies far below every other line, so aluminium reduces all the other oxides shown at every temperature.
- **Carbon and iron.** The falling 2C + O₂ = 2CO line crosses FeO near 730 °C. Above that temperature carbon can reduce FeO to iron, which is the basis of the blast furnace. It crosses NiO already near 440 °C.
- **Hydrogen and iron.** The H₂/H₂O line lies above FeO until about 1440 °C, where the two lines cross. With these fits and all gases at 1 atm, H₂ does not reduce FeO at lower temperatures.
""",
r"""md:
## Summary

| diagram | call | what it shows |
|---|---|---|
| sweep | `sw = plot.sweep(system, variable, values, **conditions)`, then `plot.plot_sweep(sw, species)` or `sw.get(quantity)` | any result along pH, pe, T, P, an amount or `"fixed:<gas>"` |
| predominance / Pourbaix | `pr, ax = plot.pourbaix(system, candidates, pH, pe, amounts=...)` | dominant species or solid at each (pH, Eh), with water lines; `plot.predominance` for any other pair of axes |
| Ellingham | `T, dG = plot.ellingham(reactions, T, per=...)` | $\Delta G^\circ(T)$ per mol O₂; lower line = more stable oxide |

**Things to remember.**
- Every point is a full equilibrium solve, so boundaries depend on total concentrations and activity corrections.
- Grid size sets both cost and boundary smoothness.

**Next steps.**
- [`06-rare-earths-and-thorium.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/06-rare-earths-and-thorium.ipynb) applies sweeps to Th and rare-earth solubility with [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) and [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat).
- [`docs/diagrams.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/diagrams.md) gives the full API, including other sweep variables and per-atom weighting of candidates.
- [`docs/databases.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/databases.md) covers choosing a database.
""",
r"""md:
## References

- Parkhurst, D.L., and Appelo, C.A.J. (2013). *Description of input and examples for PHREEQC version 3.* USGS Techniques and Methods 6-A43. [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43). Source of [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) and its conventions.
- Pourbaix, M. (1966). *Atlas of Electrochemical Equilibria in Aqueous Solutions.* Pergamon Press, Oxford.
- Ellingham, H.J.T. (1944). Reducibility of oxides and sulphides in metallurgical processes. *Journal of the Society of Chemical Industry* 63(5), 125–133.

The full bibliography, with the sources for each database and activity model, is in [`docs/references.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/references.md).
""",
]

# ============================================================================ 06
RARE_EARTHS = [
r"""md:
# Rare earths and thorium

This is the most advanced notebook. It uses the two bundled databases that cover rare-earth elements (REE: the lanthanides and Y) and thorium.

**Read first.** Notebooks 01 (systems, `equilibrate`, complementarity) and 03 (loading PHREEQC databases). Background: [`docs/databases.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/databases.md) and [`docs/activity-models.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/activity-models.md).

**Terms.**
- *Molality* (m, mol/kgw): moles per kg of water. *Ionic strength* $I = \tfrac12\sum_i m_i z_i^2$.
- *Activity coefficient* γ: activity = γ × molality; far from 1 for highly charged ions such as Th⁴⁺.
- *Hydrolysis*: a metal ion binding OH⁻ as pH rises (Th⁴⁺ → ThOH³⁺ → … → Th(OH)₄(aq)).
- *Saturation index* SI = log(IAP/K). By complementarity a solid is either present (SI = 0) or absent (SI ≤ 0).
- $\log K$ is for dissolution, with the solid on the left.

| database | activity model | strengths | limits |
|----------|----------------|-----------|--------|
| [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) (LLNL thermo.com V8 R6; Delany & Lundeen 1991) | LLNL B-dot (Helgeson 1969, [doi:10.2475/ajs.267.7.729](https://doi.org/10.2475/ajs.267.7.729)), A/B/ḃ tabulated to 300 °C | every lanthanide, Y, Th; hydroxides, carbonates, fluorides, hydrated phosphates; high temperature | I ≲ 1 m; many values are estimates; "relatively untested" per its header |
| [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) (ThermoChimie v12a; Giffaut et al. 2014, [doi:10.1016/j.apgeochem.2014.05.007](https://doi.org/10.1016/j.apgeochem.2014.05.007)) | SIT with 606 ε pairs | NEA-reviewed Th hydrolysis/carbonate data; ThO₂ at five crystallinities; brines | 25 °C for most data; of the lanthanides only Sm, Eu and Ho |

It covers:
1. ThO₂ solubility vs pH, in both databases and for several ThO₂ crystallinities;
2. rare-earth phosphates (La, Ce, Nd);
3. thorium in NaCl brines with SIT;
4. temperature dependence;
5. ore minerals that are **not** in either database (monazite, xenotime, bastnäsite, thorite), and how to add them.

Section 0 shows the evidence that the activity models and speciation reproduce PHREEQC for these databases.
""",
r"""
import json, math, pathlib, re, time
import numpy as np
import matplotlib.pyplot as plt
from discopt.speciation import PhreeqcDatabase

ll = PhreeqcDatabase.load("llnl.dat")
si = PhreeqcDatabase.load("sit.dat")
for name, db in [("llnl.dat", ll), ("sit.dat", si)]:
    print(f"{name}: {len(db.species)} aqueous species, {len(db.phases)} phases, default model {db.default_activity()}")

def element_total(res, el):
    # Total dissolved molality of an element (mol/kgw): for each aqueous species,
    # atoms of `el` in its name (e.g. 4 in "Th4(OH)8+8") times its molality.
    # Solids are not in res.molality, so with a solid present this is its solubility.
    pat = re.compile(rf"{el}(?![a-z])(\d*)")
    return sum(sum(int(n or 1) for n in pat.findall(k)) * v for k, v in res.molality.items() if k != "H2O")
""",
r"""md:
## 0. Evidence: agreement with PHREEQC

**Goal.** Check that this package reproduces PHREEQC 3 (Parkhurst & Appelo 2013, [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)) on these two databases. You can skip this section on a first read.

[`tests/data/phreeqc_llnl_sit.json`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/data/phreeqc_llnl_sit.json) holds PHREEQC results computed on the same files: Th and CO₂ in 0.1–3 m NaCl at pH 3, 25–150 °C, with Cl adjusted for charge balance. For each case the cell recomputes the speciation and prints the worst difference in log activity (species above 1e-20 m), the relative difference in ionic strength and the difference in log water activity. The file ships with the source repository; the cell skips itself if it is absent.
""",
r"""
ref_path = pathlib.Path("..") / "tests" / "data" / "phreeqc_llnl_sit.json"
if not ref_path.exists():
    print(f"{ref_path} not found (it ships with the source repository); skipping the PHREEQC comparison.")
else:
    ref = json.loads(ref_path.read_text())
    dbs = {"llnl.dat": ll, "sit.dat": si}
    print(f"{'database':9s} {'T/°C':>5s} {'NaCl':>5s} {'species':>7s} {'max|Δ log a|':>13s} {'ΔI/I':>9s} {'Δ log a_w':>10s}")
    for c in ref["cases"]:
        s = dbs[c["database"]].system(["Na", "Cl", "Th", "C(4)"], T=273.15 + c["T_C"])
        r = s.equilibrate({"Na": c["Na"], "Cl": c["Na"], "Th": 1e-6, "C(4)": 1e-3}, pH=3, charge_balance="Cl-")
        cmp = [k for k in c["la"] if c["molality"][k] > 1e-20]   # species compared
        worst = max(abs(r.log10_activity[k] - c["la"][k]) for k in cmp)
        print(f"{c['database']:9s} {c['T_C']:5.0f} {c['Na']:5.1f} {len(cmp):7d} {worst:13.1e} "
              f"{r.ionic_strength / c['mu'] - 1:9.1e} {r.log10_activity['H2O'] - c['la_H2O']:10.1e}  converged={r.converged}")
""",
r"""md:
**Result.** Log activities agree within about 2e-4, and ionic strength within 2e-5 relative. In [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) the largest differences are for polynuclear Th species such as Th₂(OH)₃⁵⁺ and Th₄(OH)₈⁸⁺, whose $\log\gamma$ scales with $z^2$ (25–64); in [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) they are for Th chloride and hydroxide complexes. The same checks run as `tests/test_phreeqc.py::test_llnl_sit_match_phreeqc`.
""",
r"""md:
## 1. ThO₂ solubility vs pH

**Data.** Both databases have crystalline ThO₂: `Thorianite` in [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) ($\log K = 1.86$) and `ThO2(cr)` in [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) ($\log K = 1.77$), for ThO₂ + 4H⁺ = Th⁴⁺ + 2H₂O. The two independent compilations agree to 0.1 log units.

**Crystallinity.** A solid can form as well-ordered crystals or as a disordered, hydrated, fine-grained material; the less ordered forms are more soluble. [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) has such forms with much larger $\log K$: `ThO2(mcr)` (microcrystalline) 3.0, `ThO2(aged)` 8.5, `ThO2(fresh)` 9.3 and `ThO2(coll)` (colloidal) 11.1. Measured thorium solubility near room temperature is controlled by the hydrous, poorly crystalline solid, not by ThO₂(cr).

**Metastable equilibrium.** A *metastable* phase is not the most stable one but persists because conversion is slow. With all forms as candidates, complementarity keeps only ThO₂(cr). To compute the solubility of a metastable form, pass the more stable forms in `suppress=`.

**What the code does.** `th_curve` equilibrates 1e-2 mol/kgw of solid in 0.1 m NaCl at each pH and records dissolved Th and the phases present, for three cases: [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) Thorianite, [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) with all four forms, and [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) ThO₂(fresh) with the others suppressed.
""",
r"""
pHs = np.arange(3.0, 10.01, 0.5)
curves = {}

def th_curve(db, phases, solid, suppress=()):
    s = db.system(["Na", "Cl", "Th"], phases=phases)
    out, t = [], time.time()
    for pH in pHs:
        r = s.equilibrate({"Na": 0.1, "Cl": 0.1, solid: 1e-2}, pH=pH, charge_balance="Cl-", suppress=list(suppress))
        assert r.converged
        out.append((element_total(r, "Th"), r.present))
    return out, time.time() - t

allforms = ["ThO2(cr)", "ThO2(mcr)", "ThO2(aged)", "ThO2(fresh)"]
curves["llnl.dat: Thorianite"], t1 = th_curve(ll, ["Thorianite"], "Thorianite")
curves["sit.dat: all forms (stable = ThO2(cr))"], t2 = th_curve(si, allforms, "ThO2(fresh)")
curves["sit.dat: ThO2(fresh), others suppressed"], t3 = th_curve(si, allforms, "ThO2(fresh)", ["ThO2(cr)", "ThO2(mcr)", "ThO2(aged)"])
print(f"solve times: {t1:.1f} s, {t2:.1f} s, {t3:.1f} s for {len(pHs)} points each")
print("phases present with all sit.dat forms as candidates:", sorted({p for _, pres in curves["sit.dat: all forms (stable = ThO2(cr))"] for p in pres}))

fig, ax = plt.subplots(figsize=(6, 4))
for (label, vals), style in zip(curves.items(), ["C0-", "C1--", "C3-"]):
    tot = np.array([v for v, _ in vals])
    ax.plot(pHs, np.log10(tot), style, label=label)
ax.set(xlabel="pH", ylabel="log10 dissolved Th (mol/kg)", title="ThO2 solubility in 0.1 m NaCl, 25 °C", ylim=(-17, -1))
ax.legend(fontsize=8); plt.tight_layout()
""",
r"""md:
**Reading the plot.**
- **Acid side.** Solubility falls steeply, by 3–4 log units per pH unit. The slope is −4 where Th⁴⁺ dominates (ThO₂ + 4H⁺ = Th⁴⁺ + 2H₂O) and shallower where ThOH³⁺ and Th(OH)₂²⁺ contribute.
- **Above pH ~5 (llnl.dat) or ~7 (sit.dat).** The curve flattens where neutral Th(OH)₄(aq) dominates. Its concentration is independent of pH.
- **Between the two ThO₂ forms.** The gap is the difference in $\log K$ (≈ 7.5 log units). The crystallinity of the solid matters far more than the activity model or the database.
- **Between the databases.** Crystalline ThO₂ agrees, but the plateaus differ by about 1.5 log units. The two files have different constants for Th(OH)₄(aq).
- **ThO₂(fresh) at pH 3–3.5.** The red curve is flat at 1e-2 m because all 1e-2 mol of the solid dissolved. The solid is absent there, and the value is the input amount, not a solubility.
""",
r"""md:
## 2. Rare-earth phosphates (llnl.dat)

**Data.** [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) has a hydrated phosphate `LnPO4:10H2O` for each lanthanide, written as LnPO₄·10H₂O + H⁺ = Ln³⁺ + HPO₄²⁻ + 10H₂O. The amorphous hydroxide `Ln(OH)3(am)` is added as a competing phase.

**Method.** A 1 mmol/kg phosphate solid is equilibrated in 0.01 m NaCl across pH. Complementarity decides which solid, if any, is present.
""",
r"""
ln_pHs = np.arange(2.0, 9.01, 0.5)
fig, ax = plt.subplots(figsize=(6, 4))
for el in ["La", "Ce", "Nd"]:
    phos, hyd = f"{el}PO4:10H2O", f"{el}(OH)3(am)"
    print(f"{phos}: log K = {ll.phases[phos].logK:.4f}")
    s = ll.system([el, "P(5)", "Na", "Cl"], phases=[phos, hyd])
    tot, present = [], set()
    for pH in ln_pHs:
        r = s.equilibrate({phos: 1e-3, "Na": 0.01, "Cl": 0.01}, pH=pH, charge_balance="Cl-")
        assert r.converged
        tot.append(element_total(r, el)); present |= set(r.present)
    ax.plot(ln_pHs, np.log10(tot), label=f"{el}  (present: {', '.join(sorted(present))})")
ax.set(xlabel="pH", ylabel="log10 dissolved Ln (mol/kg)", title="LnPO4:10H2O solubility, llnl.dat, 25 °C")
ax.legend(fontsize=8); plt.tight_layout()
""",
r"""md:
**Reading the plot.**
- **Phases present.** Only the phosphate, at every pH; `Ln(OH)3(am)` never forms up to pH 9.
- **Slope.** Dissolved Ln falls by about 1 log unit per pH unit from pH 2 to 7 (about 1e-4 to 1e-9 m), because the reaction consumes one H⁺.
- **Above pH 8.** The curves flatten; Ce and Nd level off while La keeps falling.

**Two warnings.**

**The constants look estimated.** The three `log K` values differ by exactly 0.1000. That pattern is typical of estimated, not measured, data. Differences between neighbouring lanthanides in these curves reflect aqueous complexation more than independent solubility measurements.

**The phases are not monazite.** These hydrated phosphates are like *rhabdophane*, the hydrated LnPO₄·nH₂O that precipitates from water at low temperature. They are not the anhydrous monazite of ore deposits, which is expected to be less soluble (section 5 shows how to add it).
""",
r"""md:
## 3. Thorium in NaCl brines (sit.dat, SIT)

**Goal.** See how the activity model changes the Th⁴⁺ activity coefficient and ThO₂ solubility up to 3 m NaCl.

**Activity coefficients.** In the specific ion interaction theory, SIT (Guggenheim & Turgeon 1955, [doi:10.1039/TF9555100747](https://doi.org/10.1039/TF9555100747); NEA form in Grenthe et al. 1992), the activity coefficient of Th⁴⁺ ($z^2 = 16$) is
$\log_{10}\gamma = -16\,A\sqrt I/(1+1.5\sqrt I) + \sum_j \varepsilon(\mathrm{Th^{4+}}, j)\,m_j$.
The first term is Debye–Hückel; the sum runs over oppositely charged ions $j$, each with an *interaction coefficient* ε (kg/mol) fitted to experiments for that ion pair. [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) gives $\varepsilon(\mathrm{Th^{4+}},\mathrm{Cl^-}) = 0.25$ kg/mol. The Debye–Hückel term lowers log γ as I grows, but the ε term raises it linearly, so γ passes through a minimum.

**LLNL B-dot comparison.** B-dot (Helgeson 1969) adds one ḃ·I term that is the same for every ion; it has no ion-specific interaction term. It is shown here beyond its intended range (I ≲ 1 m).

**What the code does.** At each NaCl molality it computes γ(Th⁴⁺) with trace Th (1e-9 m) at pH 4 in both databases, then the solubility of ThO₂(fresh) at pH 5 and 6 with SIT.
""",
r"""
nacl = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
print("SIT epsilon with Th+4:", {k: v for k, v in si.sit.items() if "Th+4" in k})
sit_s = si.system(["Na", "Cl", "Th"], phases=["ThO2(fresh)"])
ll_s = ll.system(["Na", "Cl", "Th"])
g_sit, g_ll, sol = [], [], {5.0: [], 6.0: []}
t = time.time()
for m in nacl:
    r1 = sit_s.equilibrate({"Na": m, "Cl": m, "Th": 1e-9}, pH=4, charge_balance="Cl-")
    r2 = ll_s.equilibrate({"Na": m, "Cl": m, "Th": 1e-9}, pH=4, charge_balance="Cl-")
    g_sit.append(math.log10(r1.gamma["Th+4"]))
    g_ll.append(math.log10(r2.gamma["Th+4"]))
    for pH in sol:
        r = sit_s.equilibrate({"Na": m, "Cl": m, "ThO2(fresh)": 1e-2}, pH=pH, charge_balance="Cl-")
        assert r.converged and r.present == ["ThO2(fresh)"]
        sol[pH].append(math.log10(element_total(r, "Th")))
print(f"{4 * len(nacl)} equilibria in {time.time() - t:.1f} s")   # per NaCl value: 2 for gamma, 2 for solubility

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(nacl, g_sit, "o-", label="SIT (sit.dat)")
axs[0].plot(nacl, g_ll, "s--", label="LLNL B-dot (llnl.dat)")
axs[0].set(xlabel="NaCl (mol/kg)", ylabel="log10 γ(Th4+)", title="Th4+ activity coefficient"); axs[0].legend()
for pH, v in sol.items():
    axs[1].plot(nacl, v, "o-", label=f"pH {pH:g}")
axs[1].set(xlabel="NaCl (mol/kg)", ylabel="log10 dissolved Th (mol/kg)", title="ThO2(fresh) solubility, SIT"); axs[1].legend()
plt.tight_layout()
""",
r"""md:
**Reading the plots.**
- **Where the models differ.** They differ already at 0.05 m, by about 0.35 log units, because B-dot uses the ion-size parameter å·B√I in the denominator while SIT uses a fixed 1.5√I. By 1 m they differ by more than a log unit. SIT γ reaches a shallow minimum near 2 m, where the ε term, fitted to Th data in chloride media, starts to win.
- **Why pH 5 depends on salt.** At pH 5, solubility is controlled by charged hydrolysis species. It rises by about 0.7 log units up to 1–2 m NaCl as their activity coefficients fall, then levels off.
- **Why pH 6 barely does.** At pH 6 the neutral Th(OH)₄(aq) dominates, and the curve hardly depends on NaCl.

**Choosing a model.** Above about 0.1 m NaCl the two models differ by up to 1.4 log units in γ(Th⁴⁺), and B-dot has no ion-specific data. For concentrated brines, such as those of radioactive waste repositories, use SIT with reviewed data ([`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat)); keep [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) for dilute, high-temperature systems.
""",
r"""md:
## 4. Temperature (llnl.dat)

**How temperature enters.** [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) gives each reaction an analytic $\log K(T)$, and the B-dot parameters are tabulated from 0.01 to 300 °C. Calculations outside that table raise an error, as in PHREEQC.

**Caution.** Each species also has its own valid range (the `Range:` comments in the file), which PHREEQC and this package do not enforce.

**What is computed.** Thorianite solubility at pH 3 and LaPO₄·10H₂O solubility at pH 4, in 0.1 m NaCl from 25 to 300 °C. Temperature is passed to `equilibrate(..., T=...)` in kelvin. The cell ends with a deliberate call at 350 °C to show the out-of-range error.
""",
r"""
Tc = np.array([25, 50, 75, 100, 150, 200, 250, 300])
th_s = ll.system(["Na", "Cl", "Th"], phases=["Thorianite"])
la_s = ll.system(["La", "P(5)", "Na", "Cl"], phases=["LaPO4:10H2O"])
thT, laT = [], []
for T in Tc:
    r = th_s.equilibrate({"Na": 0.1, "Cl": 0.1, "Thorianite": 1e-2}, pH=3, charge_balance="Cl-", T=273.15 + T)
    q = la_s.equilibrate({"Na": 0.1, "Cl": 0.1, "LaPO4:10H2O": 1e-2}, pH=4, charge_balance="Cl-", T=273.15 + T)
    assert r.converged and q.converged
    thT.append(math.log10(element_total(r, "Th")))
    laT.append(math.log10(element_total(q, "La")))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Tc, thT, "o-", label="Th, Thorianite, pH 3")
ax.plot(Tc, laT, "s-", label="La, LaPO4:10H2O, pH 4")
ax.set(xlabel="T (°C)", ylabel="log10 dissolved (mol/kg)", title="solubility vs temperature, llnl.dat"); ax.legend()
plt.tight_layout()
try:
    th_s.equilibrate({"Na": 0.1, "Cl": 0.1, "Thorianite": 1e-2}, pH=3, charge_balance="Cl-", T=273.15 + 350)
except ValueError as e:
    print("350 °C:", e)
""",
r"""md:
**Result.** The two solids go in opposite directions at fixed pH. *Retrograde* solubility decreases with temperature; *prograde* solubility increases.
- **Thorianite is retrograde.** Dissolved Th falls by about 5 log units from 25 to 200 °C, then levels off.
- **LaPO₄·10H₂O is prograde.** Dissolved La rises by about 2.6 log units, mostly above 100 °C.

Both trends come straight from the `-analytic` $\log K(T)$ expressions in [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat).

**Caveat on fixed pH.** A fixed pH hides the change of $K_w$ with temperature. In [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat), $\log K_w$ goes from −14.0 at 25 °C to −11.2 at 250 °C, so neutral pH drops to about 5.6. A fixed pH 3 is therefore *less* acidic relative to neutral at high temperature, and the trends partly reflect that.
""",
r"""md:
## 5. Ore minerals not in the databases

Neither database contains the main REE and thorium ore minerals:

| mineral | formula | status |
|---------|---------|--------|
| monazite | (Ce,La,Nd,Th)PO₄ | not in either database; [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) has only the hydrated LnPO₄·10H₂O |
| xenotime | YPO₄ | not in either database ([`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) has Y³⁺ aqueous species) |
| bastnäsite | (Ce,La)CO₃F | not in either database |
| thorite / huttonite | ThSiO₄ | not in either database; no Th–silicate complexes |

**How to add one.** Declare the solid and its dissolution reaction, written with species already in the system, on a system built from the database. Complementarity then decides whether the new mineral replaces an existing phase.

**Monazite example.** The constant below, $\log K = -26.0$ for CePO₄ = Ce³⁺ + PO₄³⁻, is an **illustrative round number** of the magnitude reported for monazite. For real work, replace it with a critically evaluated value and its temperature dependence (`dH`, or `analytic`).

**What the code does.** It computes a baseline with only CePO₄·10H₂O, declares the new solid with `ce.solid(...)` and its reaction with `ce.reaction(...)`, then repeats the pH sweep with `rx.logK` set to −25, −26 and −27 to show the sensitivity to the uncertain constant.
""",
r"""
ce_pHs = np.arange(2.0, 9.01, 0.5)
ce = ll.system(["Ce(+3)", "P(5)", "Na", "Cl"], phases=["CePO4:10H2O"])
base = []
for pH in ce_pHs:
    r = ce.equilibrate({"CePO4:10H2O": 1e-3, "Na": 0.01, "Cl": 0.01}, pH=pH, charge_balance="Cl-")
    base.append(element_total(r, "Ce"))

mz = ce.solid("CePO4(monazite)")
rx = ce.reaction(mz >> ce["Ce+3"] + ce["PO4-3"], logK=-26.0, name="CePO4(monazite)")   # illustrative value
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ce_pHs, np.log10(base), "k--", label="CePO4:10H2O only")
for lk in [-25.0, -26.0, -27.0]:
    rx.logK = lk
    tot, present = [], set()
    for pH in ce_pHs:
        r = ce.equilibrate({"CePO4:10H2O": 1e-3, "Na": 0.01, "Cl": 0.01}, pH=pH, charge_balance="Cl-")
        assert r.converged
        tot.append(element_total(r, "Ce")); present |= set(r.present)
    ax.plot(ce_pHs, np.log10(tot), label=f"+ monazite, log K {lk:g}  (present: {', '.join(sorted(present))})")
ax.set(xlabel="pH", ylabel="log10 dissolved Ce (mol/kg)", title="adding a mineral that is not in llnl.dat")
ax.legend(fontsize=7); plt.tight_layout()
rx.logK = -26.0
r = ce.equilibrate({"CePO4:10H2O": 1e-3, "Na": 0.01, "Cl": 0.01}, pH=5.0, charge_balance="Cl-")
print("log K -26, pH 5: present =", r.present, " SI(CePO4:10H2O) =", round(r.SI["CePO4:10H2O"], 2))
""",
r"""md:
**What happens.**
- The hydrated phosphate was given as the input solid. Monazite is more stable at all three log K values, so the Ce and P end up as monazite and the hydrated phase dissolves completely (SI = −1.38 at log K −26, pH 5).
- Dissolved Ce drops to the value set by monazite.
- **Sensitivity.** Each unit of $\log K$ moves dissolved Ce by about 0.5 log units. Ce and phosphate both come from the solid, so dissolved Ce scales roughly as $\sqrt{K}$.
- **At log K −25,** monazite is only slightly more stable than the hydrated phase (their $\log K$ values on the same basis are about −25 and −24.6), so the curves nearly coincide.

**If no reliable constant exists,** fit it to your own solubility data with confidence intervals, as in notebook 04. Use `fit.logK(system, "CePO4(monazite)")` as the parameter and dissolved Ce as the observation.

**Other ore minerals.** The same pattern adds the rest:
- xenotime: `YPO4 = Y+3 + PO4-3`;
- bastnäsite: `CeCO3F = Ce+3 + CO3-2 + F-`;
- thorite: `ThSiO4 + 4 H+ = Th+4 + SiO2 + 2 H2O`, using the `SiO2` aqueous species of [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat).
""",
r"""md:
## Summary

- [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) covers every lanthanide, Y and Th to 300 °C with LLNL B-dot; many constants are estimates, and it is meant for I ≲ 1 m.
- [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) (ThermoChimie) has reviewed Th data, several ThO₂ crystallinities and SIT for brines, mostly at 25 °C.
- Both reproduce PHREEQC to about 2e-4 in log activity.
- The crystallinity of the solid (≈ 7.5 log units for ThO₂) matters more than the database or activity model; use `suppress=` for metastable solubilities.
- Above about 0.1 m NaCl, SIT and B-dot differ strongly for Th⁴⁺; use SIT for brines.
- Ore minerals missing from both databases can be added with `system.solid` and `system.reaction`, and their log K fitted when no reliable value exists.

**Next steps.**
- Fit a constant such as the monazite log K: notebook 04 and [`docs/fitting.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/fitting.md).
- Diagrams for these systems: notebook 05 and [`docs/diagrams.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/diagrams.md).
- Model parameters and validity ranges: [`docs/activity-models.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/activity-models.md); overriding database values: [`docs/custom-data.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/custom-data.md).
""",
r"""md:
## References

- Parkhurst, D.L., and Appelo, C.A.J. (2013). *Description of input and examples for PHREEQC version 3.* USGS Techniques and Methods 6-A43. [doi:10.3133/tm6A43](https://doi.org/10.3133/tm6A43)
- Delany, J.M., and Lundeen, S.R. (1991). *The LLNL thermochemical data base: revised data and file format for the EQ3/6 package.* LLNL UCID-21658. [OSTI 138193](https://www.osti.gov/biblio/138193) — [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat)
- Giffaut, E., et al. (2014). Andra thermodynamic database for performance assessment: ThermoChimie. *Applied Geochemistry* 49, 225–236. [doi:10.1016/j.apgeochem.2014.05.007](https://doi.org/10.1016/j.apgeochem.2014.05.007) — [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat), v12a at [thermochimie-tdb.com](https://www.thermochimie-tdb.com)
- Helgeson, H.C. (1969). Thermodynamics of hydrothermal systems at elevated temperatures and pressures. *American Journal of Science* 267, 729–804. [doi:10.2475/ajs.267.7.729](https://doi.org/10.2475/ajs.267.7.729) — B-dot
- Drummond, S.E. (1981). *Boiling and mixing of hydrothermal fluids: chemical effects on mineral precipitation.* PhD thesis, Pennsylvania State University — CO₂(aq) activity coefficient in [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat)
- Scatchard, G. (1936). Concentrated solutions of strong electrolytes. *Chemical Reviews* 19, 309–327. [doi:10.1021/cr60064a008](https://doi.org/10.1021/cr60064a008) — SIT
- Guggenheim, E.A., and Turgeon, J.C. (1955). Specific interaction of ions. *Transactions of the Faraday Society* 51, 747. [doi:10.1039/TF9555100747](https://doi.org/10.1039/TF9555100747) — SIT
- Ciavatta, L. (1980). The specific interaction theory in evaluating ionic equilibria. *Annali di Chimica (Rome)* 70, 551–567 — SIT
- Grenthe, I., et al. (1992). *Chemical Thermodynamics of Uranium.* NEA-OECD, North-Holland. [NEA](https://www.oecd-nea.org/jcms/pl_13704) — the SIT form used by the NEA TDB and ThermoChimie

The full list is in [`docs/references.md`](https://github.com/jkitchin/discopt-speciation/blob/main/docs/references.md).
""",
]


if __name__ == "__main__":
    which = sys.argv[1:] or ["01", "02", "03", "04", "05", "06"]
    books = {"01": ("01-getting-started.ipynb", GETTING_STARTED), "02": ("02-validation.ipynb", VALIDATION),
             "03": ("03-databases-and-your-own-data.ipynb", DATABASES),
             "04": ("04-fitting-thermodynamic-data.ipynb", FITTING),
             "05": ("05-diagrams.ipynb", DIAGRAMS),
             "06": ("06-rare-earths-and-thorium.ipynb", RARE_EARTHS)}
    for key in which:
        fname, cells = books[key]
        book = nb(cells)
        client = NotebookClient(book, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(EX)}})
        try:
            client.execute()
        finally:
            nbformat.write(book, EX / fname)
        print("wrote", fname)

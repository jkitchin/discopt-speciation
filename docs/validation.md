# Validation

**Reference data.** References were computed with [PHREEQC 3](references.md#parkhurst2013), via [phreeqc-python](references.md#vitens)
1.6.2, on the same database files. They are stored in [`tests/data/*.json`](https://github.com/jkitchin/discopt-speciation/tree/main/tests/data) and
checked by [`tests/test_phreeqc.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_phreeqc.py), [`tests/test_minteq.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_minteq.py) and
[`tests/test_counterexamples.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_counterexamples.py).

| case | database / model | quantity | agreement |
|------|------------------|----------|-----------|
| NaCl 0.1 m + Ca, SO₄, 25 and 60 °C | [phreeqc.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat), Truesdell-Jones | log γ, I, a_w | ≤ 6e-6 |
| NaCl 1 m and 4 m + Ca, SO₄, 25 and 60 °C | [pitzer.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat), Pitzer | log a (incl. H₂O, OH⁻, HSO₄⁻) | ≤ 5e-5 |
| NaCl 0.1 and 1 m + Th, CO₂, pH 3, 25/60/150 °C | [llnl.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat), LLNL B-dot | log a of every species, I, a_w | ≤ 1.3e-4, 2e-5 rel, 3e-7 |
| NaCl 0.1, 1 and 3 m + Th, CO₂, pH 3, 25/60 °C | [sit.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat), SIT | log a of every species, I, a_w | ≤ 1.8e-4, 2e-5 rel, 1e-6 |
| Na/Ca/Cl with `Cl charge` | phreeqc.dat | adjusted Cl total | 5e-12 rel |
| Na/Ca on exchanger X | phreeqc.dat | NaX, CaX₂ | 5e-12 rel |
| Zn on Hfo, diffuse double layer | phreeqc.dat | ψ, surface species | ≤ 1.4e-4 rel |
| Fe(II)/Fe(III) at pe 10 | phreeqc.dat | log a | ≤ 3e-6 |
| Calcite + CO₂(g) at 10⁻³·⁵ atm | phreeqc.dat | pH, I, water mass | 1e-7, 1e-6, 1e-9 rel |
| Na–CO₂ solution with a 1 L gas volume | phreeqc.dat, Peng–Robinson | pH, n(CO₂(g)) | 1.2e-6 rel, 1.4e-5 rel |
| Pure CO₂ and CH₄, CO₂–CH₄–N₂–H₂O mixtures, 1–200 atm, 25–100 °C (39 states) | phreeqc.dat, Peng–Robinson | φ, molar volume, P | φ ≤ 4e-8 (fixed P), ≤ 3.5e-6 (fixed V); V_m, P ≤ 3e-5 rel |
| [MINTEQ](references.md#allison1991) benchmark systems | ideal/Davies | full speciation | see [`test_minteq.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_minteq.py) |
| [Torres](references.md#laliwala2025) mass-action examples | ideal | solids, SI, invariance | see notebook 02 |

**Remaining differences.**
- **Gas cases:** the remaining 3e-5 relative difference in molar volume and
  pressure is the gas constant: PHREEQC uses 0.0820597 L atm/(mol K), the package
  $R/101.325 = 0.0820574$. The Peng–Robinson states are checked in
  [`tests/test_peng_robinson.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_peng_robinson.py),
  against PHREEQC's own `PR_PHI`, `PR_P` and `GAS_VM`.
- **Surface case:** the difference in ψ is at the level of physical constants
  and PHREEQC's convergence tolerance.
- **llnl.dat and sit.dat cases:** the largest differences are for highly
  charged Th polymers, such as Th₄(OH)₈⁸⁺, whose $\log\gamma$ scales with $z^2$.
  They follow from the 2e-5 relative difference in I, which is within
  PHREEQC's convergence tolerance. PHREEQC's input also sets pe 4, which fixes
  O₂(aq). Here O(0) is balanced instead, so the O₂-dependent chlorine species
  (below 1e-24 m) are not compared.

**Solver-independent checks.** Every result carries `residuals`: the
mass-action, balance and complementarity residuals, recomputed from the
species amounts. `converged` is true only when all are below 1e-8.

The validation notebook also shows:

- that the complementarity, Gibbs and enumeration methods give the same state;
- that no feasible perturbation lowers the Gibbs energy.

## PHREEQC example problems

[`tests/test_phreeqc_examples.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_phreeqc_examples.py) checks the package against the PHREEQC 3 example problems ([Parkhurst and Appelo 2013](references.md#parkhurst2013)) and a set of adversarial stress cases.
- **Reference data:** PHREEQC 3.8.7 (IPhreeqc 3.8.7-17149, through [phreeqpython](references.md#vitens) 1.6.2), run on the same database files.
- **Stored with each case** ([`tests/data/phreeqc_examples.json`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/data/phreeqc_examples.json), written by [`tests/data/generate_phreeqc_examples.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/data/generate_phreeqc_examples.py)): the PHREEQC input, every selected-output row, and a step-by-step translation.
- **How a step is posed:** the package receives PHREEQC's species moles, plus the same reactants. Element, charge and valence totals are therefore identical.
- **Scale:** every step is one test, 227 steps in all.

**Tolerances.** The limits below are absolute unless marked *rel*.

| quantity | limit | why |
|---|---|---|
| pH, pe, log a(H₂O) | 1e-4 | PHREEQC convergence is ~1e-5 |
| I, water mass | 1e-4 rel | |
| log activities (species above 1e-30 m), SI | 5e-4 | trace redox species carry several times the pe error (N₂(aq) in ex7: 2e-4) |
| phase and gas moles | 1e-6 + 1e-4·n mol | a phase amount is the difference between the element total and what stays dissolved; a 5e-5 pH error moves 5e-4 mol of gas beside 4 mmol of carbon by 2.5e-8 mol |
| exchange, surface, solid-solution moles | 5e-4 rel (below 1e-12 mol ignored) | surface species are exponentially sensitive to ψ (ex8: 1.3e-4) |
| ψ | 1e-4 rel | |

**Fast and slow runs.** Steps slower than about 1 s are marked `slow`: the Pitzer temperature extremes, ex1 redox, ex5, the evaporite, and some ex6 steps. Steps that expose a limitation are `xfail(strict=True)`, with the diagnosis as the reason.

### Examples

| example | compared | max deviation | status |
|---|---|---|---|
| ex1 seawater, initial speciation (non-redox) | log a, I, a_w | log a 1.5e-5, I 6e-6 rel | pass |
| ex1 seawater re-equilibrated, one pe | pH, pe, log a | pe 8.3, log a 180 (not converged) | **xfail**: log-molality floor |
| ex2 gypsum/anhydrite, 25–75 °C (51 T) | SI, phase moles, I, water | phase 1.4e-7 mol, log a 4e-6 | pass |
| ex3 mixing, calcite/dolomite (A–E) | pH, log a, SI, phases | log a 4.8e-5, pH 1.6e-5 | pass |
| ex4 evaporation, 52.73 mol H₂O | water, pH, pe, log a | water 7.6e-4 rel, log a 8.9e-4 | **xfail**: water gfw |
| ex4, same water *mass* removed | as above | log a 3.2e-6 | pass |
| ex5 pyrite oxidation (6 steps) | pH, pe, phases, SI | log a 1.5e-4 (CH₄), SI 1.4e-4 | pass (slow) |
| ex6 K-feldspar path, 14 of 20 steps | pH, log a, SI, phases | log a 3.3e-7, SI 1.2e-7 | pass |
| ex6 A1, A5, B 0.04/0.08/1/4 µmol | as above | pH 2, SI 28 (not converged) | **xfail**: non-convergence |
| ex7 solution 1 + fixed-P gas at 1–3 mmol (no gas forms) | pH, pe, log a, SI | log a 2e-4 (N₂), pe 4.2e-5 | pass |
| ex7 fixed-P ≥ 4 mmol and fixed-V gas, Peng–Robinson (21 steps) | gas moles, pH, log a | gas 3.3e-6 mol (6.9e-4 rel), log a 1.6e-4 | pass |
| ex7 with ideal gases in PHREEQC (25 steps) | as above | gas 2.5e-7 mol (7e-4 rel), log a 2e-4 | pass |
| ex8 Zn on Hfo, DDL (26 steps) | surface species, ψ, Zn²⁺ | surface 1.3e-4 rel, ψ 4.7e-5 rel | pass |
| ex10 aragonite–strontianite SS, ideal (8 steps) | SS moles, SI, log a | log a 1.5e-5 | pass |
| ex10 solution 1 (aragonite + CO₂(g), φ = 0.9945) | pH, log a | pH 1.3e-6, log a 2.5e-6 | pass |
| ex10 with ideal gases in PHREEQC (9 steps) | as above | log a 1.5e-5 | pass |
| ex11 exchange, one mixing cell | exchange species, log a | NaX 3e-12 rel | pass |
| ex14 brine (I 6.9) + carbonates, + As surface | pH, surface, ψ, phases | log a 1.9e-5, ψ 1.7e-5 rel | pass |
| ex19 Langmuir Cd isotherm (6 steps) | sorbed Cd, Cd²⁺ | 3.7e-6 rel | pass |

### Stress cases

| case | compared | max deviation | status |
|---|---|---|---|
| Black Sea water evaporated ([pitzer.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat)), 36 mol H₂O removed | phases, SI, a_w, water | water 7e-5 rel, SI 2e-4 | pass (slow) |
| same, 48–54.876 mol removed | as above | water 3.7e-3 rel, SI 0.023 | **xfail**: water gfw (slow) |
| same, 55.026 mol removed | as above | not converged after 600 s (pH off 4.3) | **xfail**: non-convergence (slow) |
| same, equal water mass, 36 and 48 mol | as above | SI 2.1e-4 (polyhalite), log a 4e-5 | pass (slow) |
| same, equal water mass, 54–55.026 mol (I 8–12) | as above | log a_w 3.8e-3, log a 9.1e-3, SI 0.026 | **xfail**: non-electroneutral Pitzer a_w (slow; 230–315 s per step) |
| halite saturation, Truesdell–Jones / Pitzer | halite moles, a_w, log a | 1.4e-5 / 8.5e-5 mol, log a 1.1e-5 | pass |
| 2.5 m MgSO₄, 1.5 m Na₂SO₄ (pitzer.dat, no Cl⁻) | pH (charge balance), log a | pH 0.45, log a 0.89 | **xfail**: MacInnes scaling |
| 2.5 m MgSO₄ with 1 mM NaCl | as above | pH 7.2e-6, log a 5.3e-5 | pass |
| Zn/Pb/Cd at 1e-30 m, fixed pH | log a of all trace species | NaN, pH None, `converged=True` | **xfail**: silent NaN |
| same, re-equilibrated | as above | not converged | **xfail**: traces lost below the floor |
| pe 13.75 (O₂ limit) and −6.95 (H₂ limit), Fe/Mn/S/N/C | log a of every valence, SI | log a 9.4 / 94 (not converged) | **xfail**: log-molality floor |
| 14 phases incl. polymorphs | phases present, moles, SI | log a 1.2e-4, SI 2.5e-5 | pass |
| calcite at SI 0 and ±1e-9 mol | presence, moles, SI | 1.5e-10 mol, SI 1.3e-6 | pass |
| three phases dissolving completely | moles → 0, SI < 0 | log a 3.8e-6 | pass |
| 0.5 m NaCl + calcite + gypsum at 0.01 °C (phreeqc, pitzer, llnl, sit) | phases, pH, log a | log a 1.5e-5, gypsum 2.4e-6 mol (pitzer), pH 1.8e-6 | pass (pitzer: slow, 19 s) |
| same at 200 °C ([phreeqc.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat), pitzer.dat) | as above | pH 3e-3 / 4.8e-3, log a 0.015, gypsum 9.7e-4 mol | **xfail**: no pressure term in log K |
| same at 300 °C ([llnl.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat)), 100 °C ([sit.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat)) | as above | log a 7.3e-8 / 1.1e-7, pH 3e-9 | pass (sit: slow, 9 s) |
| Hfo with 100× ex8's sites at I 1e-4 | pH, surface, ψ | not converged | **xfail**: start-point failure |
| Na exchanger vs Ca, Mg, Sr, Ba | exchange species | exchange 9e-13 rel, log a 5e-6 | pass |
| charge balance on F⁻ | adjusted total, log a | log a 2.3e-6 | pass |
| fixed-volume CO₂ absorbed to ~1e-10 of its moles | gas moles, pH | gas 8.8e-6 rel, SI 3.3e-6 | pass |

### Limitations found

Each item below is marked `xfail(strict=True)` in the tests. The repro inputs are in the xfail reasons.

1. **Log-molality floor.** `LOG_LB = -200` in [`solve.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/solve.py).
   - **What happens:** in oxic water a vanishing valence state (Fe(HS)₃⁻ at pe 8, Mn(NO₃)₂ at pe −7) is pinned at the bound, mass balance fails, and the solve reports `converged=False`.
   - **Minimal repro:** [phreeqc.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat), Na/Fe/S/O, Na⁺ 0.02, SO₄²⁻ 0.01, O₂ 2e-4, Fe(OH)₃ 1e-8 at pH 7, pe 8. It converges at pe 0 and pe 4.
   - **Affected:** ex1 re-equilibrated, where pe is 4.07 against 12.41, and both pe-limit cases. Removing U, Mn or N does not help; removing Fe alone makes ex1 converge and agree to 1e-5.
2. **Water gfw.** PHREEQC takes the gram-formula weight of water from the database, 18.016 in phreeqc.dat and [pitzer.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat). The package uses 18.01528.
   - **Effect:** removing *N* mol of water leaves a different mass. For ex4 the water mass differs by 7.6e-4 rel. The equal-mass steps agree.
   - **Status:** a convention difference. It is documented here, not a bug.
3. **No pressure dependence of log K.** For 0.5 m NaCl with Ca, C(4) and S(6) at 25 °C and 200 atm, the package gives pH 4.3144 and PHREEQC 4.2633. The two agree at 1 atm.
   - **Above 100 °C:** PHREEQC works at the saturation pressure (15.3 atm at 200 °C), which gives pH deviations of 3–5e-3.
4. **ex6 non-convergence.** With the four K–Al–Si phases, small K-feldspar additions fail to converge. In a scan, 1e-8, 2e-8, 4e-8, 8e-8, 1e-6 and 4e-6 mol fail, and 1.6e-7 and 1e-5 mol converge.
   - **At 2.658e-8 mol:** pH 6.68 against PHREEQC's 7.0037, where Gibbsite should be at SI 0. The complementarity residual is about 1.
   - **Nearby points converge:** 1.6e-7 and 1e-5 mol both converge.
5. **Silent NaN for 1e-30 m traces.** At fixed pH the result has NaN molalities and `pH=None`, yet reports `converged=True` with zero residuals. Re-equilibrating the same solution returns `converged=False`, with a balance residual of 1.
6. **Surface start point.** The system is phreeqc.dat Na/Cl/Zn with Hfo (area 600 m²/g, mass 9 g) and amounts {Na⁺ 1e-4, Cl⁻ 1e-4, Zn²⁺ 1e-6, Hfo_w 0.02, Hfo_s 5e-4}.
   - **Without H⁺:** it converges and matches PHREEQC (pH 8.0678 against 8.0679).
   - **With 1e-7 mol H⁺ added:** it ends at `converged=False`, with Hfo_sOH at 5e-198.
7. **MacInnes scaling without Cl⁻.** The package rescales individual-ion activity coefficients to KCl only when Cl⁻ is present. PHREEQC always rescales.
   - **Effect in 2.5 m MgSO₄:** log γ(Mg²⁺) is off by −0.895 and log γ(SO₄²⁻) by +0.895, so the charge-balanced pH is off by 0.45.
   - **What still agrees:** neutral combinations and a_w. With 1 mM NaCl added, everything agrees within tolerance.
8. **Pitzer water activity of a non-electroneutral solution.**
   - **Where it shows:** the Black Sea analysis in the evaporite case carries a 2.36 meq charge imbalance, which evaporation concentrates to 0.13–0.27 eq/kgw. From there log a_w differs by 6e-4 to 3.8e-3, and minerals near saturation shift with it (SI Epsomite up to 0.026).
   - **Isolated:** the same brine at fixed pH is off by 1.1e-3 as given, but agrees to 1.6e-5 once Na makes it electroneutral. Every Na–Cl–X subsystem at those concentrations agrees to 2e-6.
   - **Likely cause:** the package takes a_w from the derivative of the excess Gibbs energy, while PHREEQC sums its osmotic coefficient. The two coincide only when Σ z m = 0.
9. **Performance and convergence of the concentrated Pitzer brine.**
   - **Runtime:** the evaporite steps at 54.876 and 55.026 mol removed take 180–380 s each. The Pitzer temperature extremes take 15–19 s.
   - **Non-convergence:** removing a literal 55.026 mol of water ends at `converged=False` after 600 s. The equal-mass step converges.
   - **Slow-set total:** 25 min, almost all in these four steps.

pe is not compared for [llnl.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) and [sit.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) at 0.01 °C, or for sit.dat at 100 °C. These solutions contain no redox couple, and PHREEQC's pe there comes from H(0) ≈ 1e-14 m, a species that is not in the package system.

### Out of scope

| example | reason |
|---|---|
| ex9 | kinetic oxidation of Fe(II) (`KINETICS`, `RATES`) |
| ex6C | kinetic K-feldspar dissolution |
| ex11 (column), ex12, ex13, ex15, ex21 | 1-D advective / diffusive / dual-porosity transport, heat transport, multicomponent diffusion; only ex11's single mixing cell is compared |
| ex16, ex17, ex18 | inverse modeling (`INVERSE_MODELING`); ex17's evaporation of Black Sea water is used forward as a Pitzer stress case |
| ex20 | isotope distribution ([`iso.dat`](https://github.com/usgs-coupled/phreeqc3/blob/master/database/iso.dat), isotope ratios) |
| ex22 | CO₂ and H₂S solubility at high pressure: needs pressure-dependent log K (item 3) |
| ex10, non-ideal part | Guggenheim a₀/a₁ solid solution is not supported; the ideal solid solution is compared |
| ex19, linear and Freundlich isotherms | 1e100 sites with log K −100.7/−102.6 and a non-integer (0.722) sorption stoichiometry are not representable |
| ex1, redox couples in the initial solution | PHREEQC distributes Fe, Mn, U and N with separate couples; the package has one pe per solution (the re-equilibrated state is compared) |
| ex4b | `MIX` with factor 20 only rescales ex4a's water |
| [wateq4f_PWN.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat) | not used: IPhreeqc 3.8.7 rejects the bundled file ("Expecting log k"), so there is no reference |

Some examples are adapted rather than run literally; each case's `note` in the JSON records how:
- **ex6A:** the "alternative reactant" is replaced by PHREEQC's dissolved amount.
- **ex8:** `Fix_H+` is replaced by a fixed pH plus the Na PHREEQC added.
- **ex14:** the surface is reacted in a batch.
- **ex7 fixed volume:** the initial gas moles are taken from PHREEQC's carbon balance.

## Reproducing the references

`tests/data` records the PHREEQC input for each case, under `input`. With
[phreeqc-python](references.md#vitens) installed:

```python
from phreeqpython import PhreeqPython
pp = PhreeqPython(database="phreeqc.dat")
pp.ip.run_string(case["input"] + "\nSELECTED_OUTPUT\n -activities Ca+2 ...")
```

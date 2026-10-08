"""Generate the figures in docs/diagrams.md.

    uv run python docs/figures/make_figures.py

The code for each figure is repeated verbatim on the Diagrams page.
"""
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from discopt.speciation import PhreeqcDatabase, fit, plot  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
db = PhreeqcDatabase.load("phreeqc.dat")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- sweep
s = db.system(["Fe", "Cl"], phases=["Fe(OH)3(a)"])
sw = plot.sweep(s, "pH", np.linspace(2, 12, 41), amounts={"Fe": 1e-3, "Cl": 3e-3}, pe=4)

fig, ax = plt.subplots(figsize=(6.5, 4))
plot.plot_sweep(sw, ["Fe+3", "FeOH+2", "Fe(OH)2+", "Fe(OH)3", "Fe(OH)4-", "Fe+2"], ax=ax, floor=1e-14)
ax.set_title("1 mmol/kg Fe, 3 mmol/kg Cl, pe = 4")
save(fig, "fe_sweep.png")
print("  all converged:", sw.converged.all())

# ---------------------------------------------------------------- Pourbaix
t = time.time()
cands = [n for n in ["Fe+2", "Fe+3", "FeOH+", "FeOH+2", "Fe(OH)2+", "Fe(OH)3", "Fe(OH)4-", "Fe(OH)3(a)"] if n in s]
fig, ax = plt.subplots(figsize=(6, 4.8))
pr, _ = plot.pourbaix(s, cands, pH=np.linspace(1, 13, 41), pe=np.linspace(-12, 20, 41),
                      amounts={"Fe": 1e-3, "Cl": 3e-3}, ax=ax)
ax.set_title("Fe, 1 mmol/kg total")
save(fig, "fe_pourbaix.png")
print(f"  {pr.grid.size} points in {time.time() - t:.0f} s, failed: {(pr.grid < 0).sum()}")

# ---------------------------------------------------------------- solubility curve
cs = db.system(["Ca", "S(6)"], phases=["Gypsum", "Anhydrite"])
T = np.linspace(0, 100, 26) + 273.15
swT = plot.sweep(cs, "T", T, amounts={"Gypsum": 0.5})
dissolved_ca = fit.total("Ca+2", "CaSO4", "CaHSO4+")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.plot(T - 273.15, swT.get("solids:Gypsum"), label="Gypsum")
a1.plot(T - 273.15, swT.get("solids:Anhydrite"), label="Anhydrite")
a1.set(xlabel="T (°C)", ylabel="solid (mol/kg water)", title="stable solid")
a1.legend()
a2.plot(T - 273.15, swT.get(dissolved_ca) * 1e3, "k")
a2.set(xlabel="T (°C)", ylabel="dissolved Ca (mmol/kg)", title="solubility")
save(fig, "gypsum_anhydrite.png")
print("  anhydrite first stable at", (T - 273.15)[np.argmax(swT.get("solids:Anhydrite") > 0)], "°C")

# ---------------------------------------------------------------- Ellingham
R = 8.314462618e-3  # kJ/(mol K)


def from_linear(A, B):
    """log10 K(T) for dG = A + B*T in kJ per mol O2."""
    return lambda T: -(A + B * T) / (R * T * np.log(10))


oxides = {
    "2 Ni + O2 = 2 NiO": from_linear(-471.0, 0.172),
    "2 Fe + O2 = 2 FeO": from_linear(-528.0, 0.130),
    "2 H2 + O2 = 2 H2O(g)": from_linear(-492.0, 0.109),
    "C + O2 = CO2": from_linear(-394.0, -0.001),
    "2 C + O2 = 2 CO": from_linear(-223.0, -0.175),
    "4/3 Al + O2 = 2/3 Al2O3": from_linear(-1120.0, 0.214),
}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
plot.ellingham(oxides, np.linspace(300, 2000, 60), per={k: 1 for k in oxides}, ax=ax)
save(fig, "ellingham.png")

# ---------------------------------------------------------------- CO2 solubility at pressure
# (repeated on the "Gases and pressure" page)
import json  # noqa: E402

dissolved_c = fit.total("CO2", "HCO3-", "CO3-2", "(CO2)2", weights=[1, 1, 1, 2])
pressures = np.array([5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 500.0])
curves = {}
for tc in (25.0, 50.0, 100.0):
    for eos in ("auto", "ideal"):  # Peng-Robinson CO2 and H2O(g), or ideal gases
        c = []
        for p in pressures:
            s = db.system(["C(4)"], gas_phase={"members": ["CO2(g)", "H2O(g)"], "pressure": float(p)}, gas_eos=eos)
            r = s.equilibrate({"CO2(g)": 50.0}, T=273.15 + tc)  # excess CO2; P is the gas phase's
            c.append(dissolved_c(r) if r.converged else np.nan)
        curves[tc, eos] = np.array(c)

# PHREEQC's example 22 (fixed-volume gas), from the package's validation data
ref = json.loads((OUT.parents[1] / "tests" / "data" / "phreeqc_examples.json").read_text())
ex22 = next(c for c in ref["cases"] if c["id"] == "ex22")["rows"]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for tc, color in zip((25.0, 50.0, 100.0), ("C0", "C1", "C3")):
    ax.plot(pressures, curves[tc, "auto"], color=color, label=f"{tc:g} °C")
    ax.plot(pressures, curves[tc, "ideal"], color=color, ls=":", lw=1)
    pts = [(r["P"], r["tot"]["C(4)"]) for r in ex22 if r["state"] == "react" and abs(r["tc"] - tc) < 0.1
           and r["P"] <= 500 and r["tot"].get("C(4)")]
    ax.plot(*zip(*pts), "o", color=color, mfc="none", ms=4)
ax.plot([], [], "k-", label="Peng–Robinson + molar volumes")
ax.plot([], [], "k:", lw=1, label="ideal gases")
ax.plot([], [], "ko", mfc="none", ms=4, label="PHREEQC (example 22)")
ax.set(xlabel="pressure (atm)", ylabel="dissolved CO₂ (mol/kg water)", ylim=(0, 3.2),
       title="CO₂ solubility in water")
ax.legend(fontsize=8, ncol=2)
save(fig, "co2_pressure.png")

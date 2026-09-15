"""Speciate a water analysis and report saturation indices (like a PHREEQC SOLUTION block).

Replace ``analysis`` and ``pH`` with the measured values. Totals are mol/kg water.
The listed minerals are suppressed, so none precipitates: the result is the
sample as measured, and ``SI > 0`` means supersaturated.
"""

from discopt.speciation import PhreeqcDatabase

db = PhreeqcDatabase.load("phreeqc.dat")
minerals = ["Calcite", "Aragonite", "Dolomite", "Gypsum", "Halite"]
s = db.system(["Ca", "Mg", "Na", "Cl", "S(6)", "C(4)"], phases=minerals)

analysis = {"Ca": 2.0e-3, "Mg": 0.8e-3, "Na": 3.0e-3, "Cl": 3.0e-3, "S(6)": 1.0e-3, "C(4)": 4.5e-3}
r = s.equilibrate(analysis, pH=7.6, charge_balance="Cl-", suppress=minerals)

assert r.converged
print(f"pH {r.pH:.2f}   ionic strength {r.ionic_strength:.4g} mol/kg")
print(f"Cl adjusted by {r.charge_balance_adjustment:+.3e} mol/kg for electroneutrality")
print("saturation indices:")
for m in minerals:
    print(f"  {m:10s} {r.SI[m]:+.3f}")
print("main species (mol/kg):")
for name, m in sorted(r.molality.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {name:10s} {m:.3e}")

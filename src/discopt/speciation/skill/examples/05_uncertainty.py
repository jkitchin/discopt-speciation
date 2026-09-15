"""How uncertain is calcite solubility in air-equilibrated water, given uncertain log K data?

1. Exact local sensitivities (implicit function theorem, no re-solves).
2. First-order propagation of independent log K standard deviations.
3. The range over a ±2σ box of log K values, from its corners.
"""

import numpy as np

from discopt.speciation import PhreeqcDatabase, fit
from discopt.speciation import optimize as O

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])   # a gas must be a phase to fix its pressure
amounts, conditions = {"Calcite": 1.0}, {"fixed": {"CO2(g)": -3.5}}
dissolved_ca = fit.total("Ca+2", "CaHCO3+", "CaCO3", "CaOH+")

sigma = {"Calcite": 0.02, "CaHCO3+": 0.05, "CaCO3": 0.05}   # assumed 1σ of each log K
r = s.equilibrate(amounts, **conditions)
S = r.sensitivity([fit.logK(s, k) for k in sigma], ["pH", dissolved_ca])
assert S.ok, S.issues

print(f"pH {r.pH:.3f}, dissolved Ca {dissolved_ca(r) * 1e3:.4f} mmol/kg")
print("d(dissolved Ca, mmol/kg) / d(log K):")
for j, k in enumerate(sigma):
    print(f"  {k:8s} {S.jacobian[1, j] * 1e3:+.4f}")
sd = np.sqrt(sum((S.jacobian[1, j] * sigma[k]) ** 2 for j, k in enumerate(sigma)))
print(f"linear 1σ: ± {sd * 1e3:.4f} mmol/kg")

box = {k: 2 * u for k, u in sigma.items()}                 # ±2σ around each nominal log K
b = O.corner_bounds(s, amounts, O.dissolved("Ca"), box, **conditions)
print(f"±2σ corners: [{b.lower * 1e3:.4f}, {b.upper * 1e3:.4f}] mmol/kg  (an inner estimate, not a bound)")

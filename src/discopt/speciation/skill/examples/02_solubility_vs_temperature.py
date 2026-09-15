"""Solubility of calcium sulfate in 0.1 mol/kg NaCl from 25 to 95 °C.

Gypsum and anhydrite are both candidates; complementarity picks the stable one
at each temperature, so the solubility curve follows the phase transition.
"""

import numpy as np

from discopt.speciation import PhreeqcDatabase, fit, plot

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Ca", "S(6)", "Na", "Cl"], phases=["Gypsum", "Anhydrite"])

T_C = np.arange(25, 96, 10)
sw = plot.sweep(s, "T", T_C + 273.15, amounts={"Gypsum": 0.5, "Na": 0.1, "Cl": 0.1})
dissolved_ca = fit.total("Ca+2", "CaSO4", "CaHSO4+")

ca = sw.get(dissolved_ca)
gyp, anh = sw.get("solids:Gypsum"), sw.get("solids:Anhydrite")
assert sw.converged.all()
print(" T (°C)  dissolved Ca (mmol/kg)  stable solid")
for t, c, g, a in zip(T_C, ca, gyp, anh):
    print(f"{t:6.0f}  {1e3 * c:22.2f}  {'Gypsum' if g > a else 'Anhydrite'}")

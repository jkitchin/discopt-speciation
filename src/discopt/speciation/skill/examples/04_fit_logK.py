"""Fit log K of gypsum and of the CaSO4 ion pair to solubility data in Na2SO4 solutions.

``data`` is (added Na2SO4 in mol/kg, measured dissolved Ca in mol/kg). Here it is
synthetic, made from logK = -4.58 and 2.30 with 2% noise, so the fit can be
checked; replace it with real measurements.
"""

import numpy as np

from discopt.speciation import System, fit

s = System("gypsum", activity="davies")
Ca, SO4, CaSO4 = s.aqueous("Ca+2"), s.aqueous("SO4-2"), s.aqueous("CaSO4")
s.aqueous("Na+")                       # spectator ion: every input must be a declared species
s.reaction(Ca + SO4 >> CaSO4, logK=2.30, name="CaSO4")
s.reaction(s.solid("Gypsum") >> Ca + SO4, logK=-4.58, name="Gypsum")
dissolved_ca = fit.total("Ca+2", "CaSO4")

# synthetic data from the "true" constants above
rng = np.random.default_rng(1)
data = []
for c in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]:
    r = s.equilibrate({"Gypsum": 1.0, "Na+": 2 * c, "SO4-2": c})
    data.append((c, dissolved_ca(r) * (1 + 0.02 * rng.standard_normal())))

# start the fit away from the truth
params = [fit.logK(s, "Gypsum", initial=-4.3), fit.logK(s, "CaSO4", initial=2.0)]
obs = [fit.Observation({"amounts": {"Gypsum": 1.0, "Na+": 2 * c, "SO4-2": c}}, dissolved_ca, y, sigma=0.02 * y)
       for c, y in data]
res = fit.fit_equilibrium(s, obs, params, absolute_sigma=True)
print(res.summary())

value, stderr = res.predict({"amounts": {"Gypsum": 1.0, "Na+": 0.3, "SO4-2": 0.15}}, dissolved_ca)
print(f"predicted dissolved Ca in 0.15 mol/kg Na2SO4: {value * 1e3:.3f} ± {stderr * 1e3:.3f} mmol/kg")

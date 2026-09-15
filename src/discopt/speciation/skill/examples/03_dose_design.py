"""Smallest NaOH dose that brings dissolved Al below 1 µmol/kg by precipitating gibbsite.

An equilibrium-constrained optimization: the equilibrium conditions are
constraints, the dose is the decision variable. The answer is re-checked with
an ordinary equilibrate at the optimal dose (``verification``).
"""

from discopt.speciation import PhreeqcDatabase
from discopt.speciation import optimize as O

db = PhreeqcDatabase.load("phreeqc.dat")
s = db.system(["Al", "Na", "Cl"], phases=["Gibbsite"], activity="ideal")

r = O.design(
    s,
    {"Al": 1e-3, "Cl": 3e-3},                       # 1 mmol/kg AlCl3
    {"NaOH": (0, 6e-3)},                             # decision and its range (mol/kg)
    objective="NaOH",                                # minimize the dose
    constraints=[(O.dissolved("Al"), "<=", 1e-6)],
    reagents={"NaOH": {"Na": 1, "OH-": 1}},
)

print("status:", r.status, "  certified:", r.certified)
print(f"NaOH dose: {r.decisions['NaOH'] * 1e3:.4f} mmol/kg")
for c in r.constraints:
    print("constraint:", c)
print(f"pH at the optimum: {r.result.pH:.3f}")
print("verified by equilibrate:", r.verification["agrees"])

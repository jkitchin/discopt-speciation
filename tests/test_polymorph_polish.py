"""Active-set repair when the assemblage guess is singular (two phases of one formula)."""

from discopt.speciation import PhreeqcDatabase


def test_polish_drops_singular_polymorph():
    ll = PhreeqcDatabase.load("llnl.dat")
    ce = ll.system(["Ce(+3)", "P(5)", "Na", "Cl"], phases=["CePO4:10H2O"])
    mz = ce.solid("CePO4(monazite)")
    ce.reaction(mz >> ce["Ce+3"] + ce["PO4-3"], logK=-26.0, name="CePO4(monazite)")
    r = ce.equilibrate({"CePO4:10H2O": 1e-3, "Na": 0.01, "Cl": 0.01}, pH=8.0, charge_balance="Cl-")
    assert r.converged
    assert r.present == ["CePO4(monazite)"]
    assert abs(r.solids["CePO4(monazite)"] - 1e-3) < 1e-6

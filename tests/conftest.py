import json
import pathlib

import pytest

from discopt.speciation import System

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def notebook_ref():
    return json.loads((DATA / "notebook_reference.json").read_text())


@pytest.fixture(scope="session")
def minteq():
    return json.loads((DATA / "minteq_cases.json").read_text())


def ce2_system(extra_solid=False, double_MX=False):
    """Counterexample 2 of the review: coupling through a complex."""
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


CE2_AMOUNTS = {"M": 1e-3, "L": 2e-2, "N": 2e-2, "X": 1e-3}


def ce1_system():
    """Counterexample 1: degenerate polymorphs (ferrihydrite vs goethite)."""
    s = System("CE1")
    Fe, NO3, H, OH = s.aqueous("Fe+3"), s.aqueous("NO3-"), s.aqueous("H+"), s.aqueous("OH-")
    s.reaction({H: 1, OH: 1}, logK=-14.0)  # H2O = H+ + OH-, water activity 1
    s.reaction(s.solid("Ferrihydrite") >> Fe + 3 * OH, logK=4.891 - 42.0)
    s.reaction(s.solid("Goethite") >> Fe + 3 * OH, logK=-43.0)
    return s


CE1_AMOUNTS = {"Fe+3": 1e-3, "NO3-": 3e-3, "H+": 1e-7, "OH-": 1e-7}


def minteq_system(case: dict, name: str = "minteq") -> System:
    """Build a System from a PHREEQC minteq.v4 selection (tests/data/minteq_cases.json)."""
    s = System(name)
    prim = case["primaries"]
    for p in prim:
        s.aqueous(p, case["species"][p]["z"])
    for sp, d in case["species"].items():
        if sp in prim or not d["nu"]:
            continue
        s.aqueous(sp, d["z"])
        s.reaction({**{k: -v for k, v in d["nu"].items()}, sp: 1.0}, logK=d["logK"])
    for ph, d in case["phases"].items():
        s.solid(ph)
        s.reaction({ph: -1.0, **d["nu"]}, logK=d["logK"])
    return s

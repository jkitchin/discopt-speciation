import numpy as np
import pytest

from discopt.speciation import System, parse_charge, parse_equation
from discopt.speciation.network import build_network

from conftest import CE2_AMOUNTS, ce2_system


@pytest.mark.parametrize(
    "name,z", [("Ca+2", 2), ("SO4-2", -2), ("Na+", 1), ("Cl-", -1), ("Ca++", 2), ("CaSO4", 0), ("Fe(OH)2+", 1)]
)
def test_parse_charge(name, z):
    assert parse_charge(name) == z


def test_parse_equation_keeps_ion_names():
    left, right = parse_equation("Ca+2 + 2 H+ <=> CaH2+4")
    assert left == [(1.0, "Ca+2"), (2.0, "H+")]
    assert right == [(1.0, "CaH2+4")]
    assert parse_equation("Gypsum = Ca+2 + SO4-2")[1] == [(1.0, "Ca+2"), (1.0, "SO4-2")]


def test_parse_equation_errors():
    with pytest.raises(ValueError):
        parse_equation("A + B")


def test_dsl_string_and_dict_forms_agree():
    s = System()
    a, b, c = s.aqueous("A"), s.aqueous("B"), s.aqueous("C")
    r1 = s.reaction(a + 2 * b >> c, logK=1.0)
    r2 = s.reaction("A + 2 B = C", logK=1.0)
    r3 = s.reaction({"A": -1, "B": -2, "C": 1}, logK=1.0)
    names = lambda r: {k.name: v for k, v in r.stoich.items()}
    assert names(r1) == names(r2) == names(r3) == {"A": -1.0, "B": -2.0, "C": 1.0}


def test_undeclared_species():
    s = System()
    s.aqueous("A")
    with pytest.raises(KeyError):
        s.reaction("A = B", logK=0.0)


def test_foreign_species_rejected():
    s1, s2 = System("one"), System("two")
    a = s1.aqueous("A")
    b = s2.aqueous("B")
    with pytest.raises(ValueError):
        s2.reaction(a >> b, logK=0.0)


def test_solid_needs_exactly_one_reaction():
    s = System()
    s.aqueous("A")
    s.solid("S")
    with pytest.raises(ValueError, match="exactly one"):
        s.validate()


def test_dependent_reactions_rejected():
    s = System()
    s.aqueous("A"), s.aqueous("B"), s.aqueous("C")
    s.reaction("A = B", logK=1.0)
    s.reaction("B = C", logK=1.0)
    s.reaction("A = C", logK=2.0)
    with pytest.raises(ValueError, match="linearly dependent"):
        s.equilibrate({"A": 1.0})


def test_network_conservation_and_mu0():
    s = ce2_system()
    net = build_network(s.species, s.reactions, {s[k]: v for k, v in CE2_AMOUNTS.items()}, {}, 298.15)
    cols = net.aq + net.solids
    N = np.zeros((len(cols), len(s.reactions)))
    for j, r in enumerate(s.reactions):
        for sp, c in r.stoich.items():
            N[cols.index(sp), j] = c
    assert np.allclose(net.A @ N, 0.0)
    assert net.A.shape[0] == len(cols) - len(s.reactions)
    logK = np.array([r.logK for r in s.reactions])
    assert np.allclose(N.T @ net.mu0, -np.log(10) * logK)


def test_absent_components_are_pruned():
    s = System()
    s.aqueous("Ca+2"), s.aqueous("SO4-2"), s.aqueous("CaSO4"), s.aqueous("Mg+2"), s.aqueous("MgSO4")
    s.reaction("Ca+2 + SO4-2 = CaSO4", logK=2.36)
    s.reaction("Mg+2 + SO4-2 = MgSO4", logK=2.26)
    res = s.equilibrate({"Ca+2": 1e-3, "SO4-2": 1e-3})
    assert res.converged
    assert set(res.pruned) == {"Mg+2", "MgSO4"}
    assert res["MgSO4"] == 0.0
    assert res["Ca+2"] + res["CaSO4"] == pytest.approx(1e-3, rel=1e-10)

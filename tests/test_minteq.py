"""Validation cases from the review (minteq.v4 thermodynamics) against PHREEQC
and the notebook's Pyomo active-set reference."""

import pytest

from discopt.speciation import Davies

from conftest import minteq_system

CASES = ["Case 1", "Case 2", "Case 3", "Case 3 (pH 7.5)"]


def run(case, **kw):
    s = minteq_system(case)
    return s.equilibrate(case["totals"], pH=case["pH"], **kw)


@pytest.mark.parametrize("name", CASES)
def test_ideal_matches_phreeqc(name, minteq):
    case = minteq[name]
    res = run(case)
    assert res.converged, res.summary()
    ref = case["phreeqc_ideal"]
    for ph, n in ref["n"].items():
        assert res.solids[ph] == pytest.approx(n, rel=2e-3, abs=1e-10), ph
        assert res.SI[ph] == pytest.approx(ref["SI"][ph], abs=5e-3), ph
    assert res.ionic_strength == pytest.approx(ref["mu"], rel=5e-3)


@pytest.mark.parametrize("name", CASES)
def test_davies_assemblage_matches_phreeqc(name, minteq):
    case = minteq[name]
    res = run(case, activity="davies")
    assert res.converged, res.summary()
    ref = case["phreeqc_activity"]
    assert set(res.present) == {k for k, v in ref["n"].items() if v > 0}
    for ph, n in ref["n"].items():
        assert res.solids[ph] == pytest.approx(n, rel=1e-2, abs=1e-10), ph


def test_case2_davies_matches_notebook(minteq, notebook_ref):
    ref = notebook_ref["Case2_davies"]
    res = run(minteq["Case 2"], activity=Davies(A=0.509))
    assert res.converged
    assert res.ionic_strength == pytest.approx(ref["I"], rel=1e-3)
    assert res.solids["Gypsum"] == pytest.approx(ref["gypsum"], rel=1e-3)


@pytest.mark.parametrize("name", ["Case 1", "Case 3"])
def test_methods_agree(name, minteq):
    case = minteq[name]
    a = run(case, method="complementarity")
    b = run(case, method="gibbs")
    assert a.converged and b.converged
    assert a.present == b.present
    for k, v in a.solids.items():
        assert b.solids[k] == pytest.approx(v, rel=1e-6, abs=1e-14)
    for k, v in a.molality.items():
        assert b.molality[k] == pytest.approx(v, rel=1e-6, abs=1e-25)


@pytest.mark.slow
def test_case3_enumerate(minteq):
    case = minteq["Case 3"]
    a = run(case)
    c = run(case, method="enumerate")
    assert c.converged and c.present == a.present

"""Counterexamples CE1-CE4 from the mass-action review: the SI^2 objective fails
on all of them; a correct equilibrium solver must match the active-set reference
and be invariant to polymorph degeneracy, inert phases and reaction scaling."""

import pytest

from conftest import CE1_AMOUNTS, CE2_AMOUNTS, ce1_system, ce2_system

METHODS = ["complementarity", "gibbs", "enumerate"]


def check(res, ref, rel=1e-6, atol=1e-12):
    assert res.converged, res.summary()
    for k, v in ref["n"].items():
        assert res.solids[k] == pytest.approx(v, rel=rel, abs=atol), k
    for k, v in ref["C"].items():
        assert res.molality[k] == pytest.approx(v, rel=rel, abs=atol), k
    for k, v in res.SI.items():
        assert v <= 1e-8
        assert min(res.solids[k], -v) <= 1e-8


@pytest.mark.parametrize("method", METHODS)
def test_ce1_polymorphs(method, notebook_ref):
    res = ce1_system().equilibrate(CE1_AMOUNTS, method=method)
    check(res, notebook_ref["CE1"])
    assert res.present == ["Goethite"]
    # goethite saturated => SI_ferrihydrite = logK_goethite - logK_ferrihydrite = -43 - (4.891 - 42)
    assert res.SI["Ferrihydrite"] == pytest.approx(-5.891, abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_ce2_coupling(method, notebook_ref):
    res = ce2_system().equilibrate(CE2_AMOUNTS, method=method)
    check(res, notebook_ref["CE2"])
    assert res.present == ["NL(s)"]
    assert res.SI["MX(s)"] == pytest.approx(notebook_ref["CE2"]["SI"]["MX(s)"], abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_ce3_inert_phase_invariance(method, notebook_ref):
    base = ce2_system().equilibrate(CE2_AMOUNTS, method=method)
    res = ce2_system(extra_solid=True).equilibrate(CE2_AMOUNTS, method=method)
    check(res, notebook_ref["CE3"])
    for k, v in base.molality.items():
        assert res.molality[k] == pytest.approx(v, rel=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_ce4_scaling_invariance(method, notebook_ref):
    base = ce2_system().equilibrate(CE2_AMOUNTS, method=method)
    res = ce2_system(double_MX=True).equilibrate(CE2_AMOUNTS, method=method)
    check(res, notebook_ref["CE4"])
    for k, v in base.molality.items():
        assert res.molality[k] == pytest.approx(v, rel=1e-8)
    assert res.gibbs_energy == pytest.approx(base.gibbs_energy, rel=1e-10)


@pytest.mark.parametrize("method", ["complementarity", "gibbs"])
def test_ce2_scan_crosses_phase_boundary(method, notebook_ref):
    """Sweep X0 through the MX(s) saturation boundary; both solids appear/disappear correctly."""
    for row in notebook_ref["CE2_scan"]:
        res = ce2_system().equilibrate(dict(CE2_AMOUNTS, X=row["X0"]), method=method)
        assert res.converged, (row["X0"], res.summary())
        for k, v in row["n"].items():
            assert res.solids[k] == pytest.approx(v, rel=1e-6, abs=1e-11), (row["X0"], k)


def test_gibbs_energy_minimal_among_assemblages():
    """Equilibrium G is below G of every other square (fixed-assemblage) state.

    G/RT carries a gauge (mu0 is only fixed up to conservation laws), so states
    are compared within one network.
    """
    import itertools

    import numpy as np

    from discopt.speciation import Ideal
    from discopt.speciation import solve as sv
    from discopt.speciation.network import build_network

    s = ce2_system()
    amt = {s[k]: v for k, v in CE2_AMOUNTS.items()}
    net = build_network(s.species, s.reactions, amt, {}, 298.15)
    act = Ideal()
    info = sv.SolveInfo("test")
    opts = dict(sv.NLP_OPTIONS)
    x_eq, active, ok = sv.solve_complementarity(net, act, opts, info)
    assert ok
    g_eq = sv.gibbs_energy(net, x_eq, act)
    x_spec = sv.solve_speciation(net, act, opts, info)
    for S in itertools.chain.from_iterable(itertools.combinations(range(net.ns), k) for k in range(net.ns + 1)):
        if set(S) == active:
            continue
        x = sv.solve_square(net, act, set(S), [x_spec], opts, info)
        if x is None or np.any(x[net.na :] < 0):  # negative solid amounts are not physical states
            continue
        assert sv.gibbs_energy(net, x, act) > g_eq

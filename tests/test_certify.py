"""Global stability certificates and phase splitting for non-ideal solid solutions.

Adversarial cases: states that satisfy every local equilibrium condition but are
metastable or unstable (inside a miscibility gap), gaps at the edge of existence,
near-zero members, ternary gaps, and PHREEQC example 10's Guggenheim parameters.
"""

import math

import numpy as np
import pytest

from discopt.speciation import System
from discopt.speciation.phases import SolidSolution
from discopt.speciation.stability import Iv, _IVO, stability_certificate, tpd_minimum
from discopt.speciation.network import PhaseInfo


def binary(W=None, gugg=None, logK=(-2.0, -2.0)):
    s = System("toy", activity="ideal")
    s.solvent("H2O")
    for n, z in [("A+", 1), ("B+", 1), ("L-", -1)]:
        s.aqueous(n, z)
    s.solid("AL"); s.solid("BL")
    s.reaction("AL = A+ + L-", logK=logK[0], name="AL")
    s.reaction("BL = B+ + L-", logK=logK[1], name="BL")
    if gugg is not None:
        s.solid_solution("SS", ["AL", "BL"], guggenheim=gugg)
    elif W is None:
        s.solid_solution("SS", ["AL", "BL"])
    else:
        s.solid_solution("SS", ["AL", "BL"], margules={("AL", "BL"): W})
    return s


def regular_binodal(W):
    """Symmetric regular solution: ln(x/(1-x)) = W(2x-1)."""
    from scipy.optimize import brentq

    x = brentq(lambda x: math.log(x / (1 - x)) - W * (2 * x - 1), 1e-12, 0.5 - 1e-9)
    return x, 1 - x


def fractions_of(res):
    return sorted(d["mole_fraction"]["AL"] for d in res.solid_solutions.values() if d["present"])


# ------------------------------------------------------------------ excess models
@pytest.mark.parametrize("model", [SolidSolution("S", ["a", "b"], guggenheim=[3.43, -1.82, 0.4]),
                                   SolidSolution("S", ["a", "b", "c"],
                                                 margules={("a", "b"): 2.5, ("b", "c"): -1.0, ("a", "c"): 3.0})])
def test_ln_lambda_is_derivative_of_excess(model):
    """ln lambda_j = d(n G_ex)/dn_j, by central differences in amounts."""
    from discopt.speciation.activity import NUM

    rng = np.random.default_rng(0)
    for _ in range(5):
        nj = rng.uniform(0.1, 1.0, len(model.members))
        lnl = model.ln_lambda(NUM, list(nj / nj.sum()))
        for j in range(len(nj)):
            h = 1e-6
            up, dn = nj.copy(), nj.copy()
            up[j] += h; dn[j] -= h
            g = lambda v: v.sum() * model.excess(NUM, list(v / v.sum()))
            assert lnl[j] == pytest.approx((g(up) - g(dn)) / (2 * h), abs=1e-7)


def test_excess_bound_is_lipschitz():
    from discopt.speciation.activity import NUM

    m = SolidSolution("S", ["a", "b"], guggenheim=[3.43, -1.82, 0.4])
    xs = np.linspace(0, 1, 2001)
    g = np.array([m.excess(NUM, [1 - x, x]) for x in xs])
    assert np.abs(np.diff(g) / np.diff(xs)).max() <= m.excess_bound()


def test_interval_ops_enclose():
    a, b = Iv(-0.3, 0.7), Iv(0.2, 0.5)
    for u in np.linspace(a.lo, a.hi, 11):
        for v in np.linspace(b.lo, b.hi, 11):
            for r, iv in [(u * v, a * b), (u - v, a - b), (u + v, a + b)]:
                assert iv.lo <= r <= iv.hi
    assert _IVO.sum([a, b]).lo <= 0.3 - 0.3 + 0.2


# ------------------------------------------------------------------ TPD bounds without a solve
def info(model):
    return PhaseInfo(name=model.name, kind="ss", members=list(range(len(model.members))), model=model,
                     mpos=list(range(len(model.members))))


@pytest.mark.parametrize("W", [2.5, 3.0, 4.0])
def test_tpd_detects_interior_instability(W):
    """At SI_j = ln(x lambda)/ln10 of a composition inside the spinodal, the TPD is negative elsewhere."""
    m = SolidSolution("S", ["a", "b"], margules={("a", "b"): W})
    x0 = 0.5
    SI = [(math.log(x0) + W * (1 - x0) ** 2) / math.log(10)] * 2
    out = tpd_minimum(info(m), SI)
    assert out["upper"] < -1e-3 and out["lower"] <= out["upper"]
    assert min(out["x"]) == pytest.approx(regular_binodal(W)[0], abs=5e-3)


def test_tpd_certifies_binodal_tangent():
    """At the binodal the plane is a double tangent: min D = 0, proven to within tol."""
    W = 3.0
    xb = regular_binodal(W)[0]
    SI = [(math.log(xb) + W * (1 - xb) ** 2) / math.log(10), (math.log(1 - xb) + W * xb**2) / math.log(10)]
    out = tpd_minimum(info(SolidSolution("S", ["a", "b"], margules={("a", "b"): W})), SI)
    assert out["lower"] >= -1e-6 and out["complete"]


def test_tpd_near_zero_member():
    """A member with SI = -30 drives the minimizer to x ~ 1e-30; the eps-truncation margin must still hold."""
    m = SolidSolution("S", ["a", "b"], margules={("a", "b"): 1.0})
    SI = [0.0, -30.0]
    out = tpd_minimum(info(m), SI)
    exact = math.log(1 + 10**-30 * math.exp(-1.0))  # ~0: pure a has D = 0 and nearly all mass stays there
    assert out["lower"] <= -exact + 1e-12 and out["lower"] >= -1e-6


def test_tpd_ternary_gap():
    """Ternary regular solution with one strongly non-ideal pair: an off-edge minimum is found and bounded."""
    m = SolidSolution("S", ["a", "b", "c"], margules={("a", "b"): 3.5, ("a", "c"): 0.5, ("b", "c"): 0.5})
    x0 = np.array([0.45, 0.45, 0.10])
    lnl = m.ln_lambda(__import__("discopt.speciation.activity", fromlist=["NUM"]).NUM, list(x0))
    SI = [(math.log(x) + l) / math.log(10) for x, l in zip(x0, lnl)]
    out = tpd_minimum(info(m), SI)
    assert out["upper"] < -1e-3
    assert out["lower"] <= out["upper"] and out["complete"]


def test_lower_bound_is_rigorous_on_random_ternaries():
    """The proven lower bound never exceeds the minimum of D on a dense grid, for random (also negative) W."""
    from discopt.speciation.activity import NUM

    rng = np.random.default_rng(7)
    g = np.linspace(0, 1, 301)
    P = np.clip(np.array([(a, b, 1 - a - b) for a in g for b in g if a + b <= 1]), 1e-300, 1.0)
    for trial in range(12):
        W = {("a", "b"): rng.uniform(-2, 5), ("a", "c"): rng.uniform(-2, 5), ("b", "c"): rng.uniform(-2, 5)}
        m = SolidSolution("S", ["a", "b", "c"], margules=W)
        SI = rng.uniform(-1.0, 0.5, 3)
        out = tpd_minimum(info(m), list(SI))
        D = (P * (np.log(P) - math.log(10) * SI)).sum(1) + np.array([m.excess(NUM, list(p)) for p in P])
        assert out["lower"] <= D.min() + 1e-12
        assert out["complete"]
        assert out["upper"] <= D.min() + 1e-3  # and the incumbent is near the true minimum


# ------------------------------------------------------------------ equilibrium with phase splitting
@pytest.mark.parametrize("W", [3.0, 2.2])
def test_symmetric_split_reaches_binodal(W):
    """Equal totals start the solver at x = 0.5, a stationary point inside the spinodal."""
    res = binary(W).equilibrate({"A+": 1.0, "B+": 1.0, "L-": 2.0})
    assert res.converged and res.certificate
    lo, hi = regular_binodal(W)
    assert fractions_of(res) == pytest.approx([lo, hi], abs=1e-6)
    assert res.certificate.phases[0].lower_bound >= -1e-6


def test_metastable_start_between_binodal_and_spinodal():
    """Bulk x_A = 0.85 with W = 3 lies between spinodal (0.789) and binodal (0.929): locally stable, globally not."""
    res = binary(3.0).equilibrate({"A+": 1.7, "B+": 0.3, "L-": 2.0})
    assert res.converged and res.certificate
    lo, hi = regular_binodal(3.0)
    assert fractions_of(res) == pytest.approx([lo, hi], abs=1e-6)


def test_certify_false_leaves_metastable_state():
    res = binary(3.0).equilibrate({"A+": 1.7, "B+": 0.3, "L-": 2.0}, certify=False)
    assert res.certificate is None
    assert len(fractions_of(res)) == 1
    cert = stability_certificate(res)
    assert cert.status == "unstable" and not cert


@pytest.mark.parametrize("W", [1.9, 2.0])
def test_no_gap_at_or_below_critical(W):
    """W <= 2 has no gap (W = 2 is the critical point, where D is flat to fourth order)."""
    res = binary(W).equilibrate({"A+": 1.0, "B+": 1.0, "L-": 2.0})
    assert res.converged and res.certificate and len(fractions_of(res)) == 1


def test_barely_supercritical():
    """W = 2.01: a gap of width ~0.17 around x = 0.5 with a TPD depth of ~1e-6 at the scale of tol."""
    res = binary(2.01).equilibrate({"A+": 1.0, "B+": 1.0, "L-": 2.0})
    assert res.converged
    assert res.certificate.status in ("certified", "inconclusive")
    if len(fractions_of(res)) == 2:
        assert fractions_of(res) == pytest.approx(list(regular_binodal(2.01)), abs=1e-4)


def test_outside_gap_single_phase():
    res = binary(3.0).equilibrate({"A+": 1.98, "B+": 0.02, "L-": 2.0})
    assert res.converged and res.certificate and len(fractions_of(res)) == 1


def test_ideal_default_not_certified_but_available():
    s = binary(None)
    res = s.equilibrate({"A+": 1.0, "B+": 1.0, "L-": 2.0})
    assert res.certificate is None
    cert = s.equilibrate({"A+": 1.0, "B+": 1.0, "L-": 2.0}, certify=True).certificate
    assert cert and all(q.method == "exact" for q in cert.phases)


def test_absent_nonideal_phase_supersaturated_off_stationary_point():
    """Undersaturated at its stationary composition yet supersaturated elsewhere: an absent phase must form.

    With W = 3.5 and equal totals, both members have SI ~ 0.03 > 0, so the pure end
    compositions have D < 0. The stationary point x = 0.5 has s = log10(0.5) + W/(4 ln 10) - SI
    ~ 0.05 > 0, so a solution with the phase absent and undersaturated there satisfies every
    local condition.
    """
    s = binary(3.5)
    res = s.equilibrate({"A+": 0.0735, "B+": 0.0735, "L-": 0.147})
    assert res.converged and res.certificate
    assert any(d["present"] for d in res.solid_solutions.values())


# ------------------------------------------------------------------ PHREEQC example 10
EX10_BINODAL = (0.0048032, 0.857863)  # mole fraction of Strontianite (comp2), PHREEQC ex10 output


def test_ex10_miscibility_gap():
    """PHREEQC ex10: Aragonite (comp1)-Strontianite (comp2), Gugg_nondim a0 = 3.43, a1 = -1.82.

    For a bulk inside the gap the two coexisting compositions are the binodal PHREEQC prints.
    """
    s = binary(gugg=[3.43, -1.82], logK=(-2.0, -2.0))
    # x_B = 0.4 bulk, most mass in the solid
    res = s.equilibrate({"A+": 1.2, "B+": 0.8, "L-": 2.0})
    assert res.converged and res.certificate
    xb = sorted(d["mole_fraction"]["BL"] for d in res.solid_solutions.values() if d["present"])
    assert xb == pytest.approx(list(EX10_BINODAL), rel=2e-4)


@pytest.mark.parametrize("xb", [0.1, 0.5, 0.8])
def test_ex10_gap_independent_of_bulk(xb):
    s = binary(gugg=[3.43, -1.82])
    res = s.equilibrate({"A+": 2 * (1 - xb), "B+": 2 * xb, "L-": 2.0})
    xs = sorted(d["mole_fraction"]["BL"] for d in res.solid_solutions.values() if d["present"])
    assert res.certificate and xs == pytest.approx(list(EX10_BINODAL), rel=2e-4)


@pytest.mark.parametrize("xb", [0.001, 0.9, 0.99])
def test_ex10_outside_gap(xb):
    res = binary(gugg=[3.43, -1.82]).equilibrate({"A+": 2 * (1 - xb), "B+": 2 * xb, "L-": 2.0})
    assert res.converged and res.certificate and len(fractions_of(res)) == 1


def test_ternary_split():
    s = System("t3", activity="ideal")
    s.solvent("H2O")
    for n, z in [("A+", 1), ("B+", 1), ("C+", 1), ("L-", -1)]:
        s.aqueous(n, z)
    for m in "ABC":
        s.solid(f"{m}L")
        s.reaction(f"{m}L = {m}+ + L-", logK=-2.0, name=f"{m}L")
    s.solid_solution("SS", ["AL", "BL", "CL"], margules=[["AL", "BL", 3.5], ["AL", "CL", 0.5], ["BL", "CL", 0.5]])
    res = s.equilibrate({"A+": 0.9, "B+": 0.9, "C+": 0.2, "L-": 2.0})
    assert res.converged and res.certificate
    comps = [d["mole_fraction"] for d in res.solid_solutions.values() if d["present"]]
    assert len(comps) == 2
    # symmetric in A and B: mirror images with equal C
    assert comps[0]["AL"] == pytest.approx(comps[1]["BL"], abs=1e-6)
    assert comps[0]["CL"] == pytest.approx(comps[1]["CL"], abs=1e-6)
    assert abs(comps[0]["AL"] - comps[0]["BL"]) > 0.5


@pytest.mark.parametrize("kw", [dict(activity="davies"), dict(method="gibbs"), dict(method="enumerate")])
def test_split_other_models_and_methods(kw):
    res = binary(3.0).equilibrate({"A+": 1.7, "B+": 0.3, "L-": 2.0}, **kw)
    assert res.converged and res.certificate
    assert fractions_of(res) == pytest.approx(list(regular_binodal(3.0)), abs=1e-5)
    if "activity" in kw:
        assert "assumed for 'davies'" in res.certificate.assumptions[0]


def test_sensitivity_after_split():
    """IFT sensitivities of a split state: in the gap, the solution activities do not depend on the totals."""
    res = binary(3.0).equilibrate({"A+": 1.7, "B+": 0.3, "L-": 2.0})
    sr = res.sensitivity(["total:A+"], ["log10_activity:A+"])
    assert sr.ok, sr.issues
    # two coexisting solids of fixed composition + common L-: A+ activity changes only through L- (A+ and L- both
    # added): check against a re-solve instead of an analytic value
    h = 1e-4
    up = binary(3.0).equilibrate({"A+": 1.7 + h, "B+": 0.3, "L-": 2.0}).log10_activity["A+"]
    dn = binary(3.0).equilibrate({"A+": 1.7 - h, "B+": 0.3, "L-": 2.0}).log10_activity["A+"]
    assert sr.jacobian[0, 0] == pytest.approx((up - dn) / (2 * h), rel=1e-4, abs=1e-8)


def test_spec_round_trip_keeps_nonideal_parameters():
    s = binary(gugg=[3.43, -1.82])
    s2 = System.from_spec(s.to_spec())
    r1 = s.equilibrate({"A+": 1.2, "B+": 0.8, "L-": 2.0})
    r2 = s2.equilibrate({"A+": 1.2, "B+": 0.8, "L-": 2.0})
    assert fractions_of(r1) == pytest.approx(fractions_of(r2), abs=1e-9)

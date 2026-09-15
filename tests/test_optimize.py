"""Adversarial tests for discopt.speciation.optimize (design and worst-case bounds).

Every optimum is compared with an independent dense scan of ``System.equilibrate``;
every certificate (when one is issued) is checked for soundness against the scan.
"""

import math

import numpy as np
import pytest

from discopt.speciation import PhreeqcDatabase, System
from discopt.speciation import optimize as O

NAOH = {"NaOH": {"Na": 1, "OH-": 1}}


# --------------------------------------------------------------------------- systems
def toy(ka=-5.0, ksp=-10.0, beta=5.0):
    """M+ / HL ligand at fixed pH: dissolved M is non-monotone in log Ka when ML(s) is present."""
    s = System("toy")
    for n in ("H+", "M+", "L-", "HL", "ML2-"):
        s.aqueous(n)
    s.solid("ML(s)")
    s.reaction("HL = H+ + L-", logK=ka, name="Ka")
    s.reaction("ML(s) = M+ + L-", logK=ksp, name="Ksp")
    s.reaction("M+ + 2 L- = ML2-", logK=beta, name="beta2")
    return s


TOY_AMOUNTS = {"M+": 1e-3, "HL": 1e-1}
M_TOTAL = O.dissolved({"M+": 1, "ML2-": 1})


def scan(system, amounts, quantity, key, grid, **kw):
    out = []
    for g in grid:
        val, res = O.evaluate(system, amounts, quantity, {key: float(g)}, **kw)
        assert res.converged
        out.append(val)
    return np.array(out)


@pytest.fixture(scope="module")
def gibbsite():
    db = PhreeqcDatabase.load("phreeqc.dat")
    return db.system(["Al", "Na", "Cl"], phases=["Gibbsite"], activity="ideal")


AL = {"Al": 1e-3, "Cl": 3e-3}


# --------------------------------------------------------------------------- building blocks
def test_formula_parsing():
    assert O._parse_formula("Al(OH)4-") == {"Al": 1, "O": 4, "H": 4}
    assert O._parse_formula("Th6(OH)15+9") == {"Th": 6, "O": 15, "H": 15}
    assert O._parse_formula("NdPO4:10H2O") == {"Nd": 1, "P": 1, "O": 14, "H": 20}
    assert O._parse_formula("ThO2(fresh)") == {"Th": 1, "O": 2}
    assert O._parse_formula("Gibbsite") is None


def test_composition_of_mineral_from_reaction(gibbsite):
    assert O._count(gibbsite, "Gibbsite", "Al") == pytest.approx(1.0)
    assert O._count(gibbsite, "Al(OH)2+", "Al") == 1.0


def test_parameter_validation():
    s = toy()
    with pytest.raises(KeyError):
        O.worst_case_bounds(s, TOY_AMOUNTS, M_TOTAL, {"nonexistent": 0.5}, method="local", pH=5)
    with pytest.raises(ValueError):
        O.worst_case_bounds(s, TOY_AMOUNTS, M_TOTAL, {"Ka": (-3, -7)}, method="local", pH=5)
    with pytest.raises(ValueError):  # pH both fixed and a decision
        O.design(s, TOY_AMOUNTS, {"pH": (4, 6)}, objective=M_TOTAL, pH=5)


def test_quantity_on_result_matches_molalities(gibbsite):
    r = gibbsite.equilibrate({**AL, "Na": 3e-3, "OH-": 3e-3})
    names = ["Al+3", "AlOH+2", "Al(OH)2+", "Al(OH)3", "Al(OH)4-"]
    assert O.dissolved("Al").value(r) == pytest.approx(sum(r[n] for n in names), rel=1e-12)
    fp = O.precipitated_fraction("Al").value(r)
    assert fp == pytest.approx(r.solids["Gibbsite"] / (r.solids["Gibbsite"] + r.water_mass * sum(r[n] for n in names)))


# --------------------------------------------------------------------------- bounds
def test_interior_extreme_beats_corners():
    """min of dissolved M over log Ka in [-7, -3] is interior (dense scan); corners miss it."""
    s = toy()
    q = O.log10_dissolved({"M+": 1, "ML2-": 1})
    b = O.worst_case_bounds(s, TOY_AMOUNTS, q, {"Ka": (-7, -3)}, method="local", pH=5)
    coarse = np.linspace(-7, -3, 17)
    fine = np.clip(b.argmin["Ka"] + np.linspace(-0.1, 0.1, 9), -7, -3)
    grid = np.concatenate([coarse, fine])
    vals = scan(s, TOY_AMOUNTS, q, "Ka", grid, pH=5)
    c = O.corner_bounds(s, TOY_AMOUNTS, q, {"Ka": (-7, -3)}, include_center=False, pH=5)
    assert -7 + 0.1 < b.argmin["Ka"] < -3 - 0.1
    assert b.lower <= vals.min() + 1e-6  # at least as good as the scan
    assert b.lower >= vals.min() - 1e-3  # and attained (verified with equilibrate), not an artifact
    assert b.minimize.verification["agrees"]
    assert c.lower > b.lower + 0.2  # corner enumeration misses the interior minimum
    assert b.upper == pytest.approx(vals.max(), abs=1e-6)
    assert not b.certified and b.lower_bound is None  # a local method never claims a bound


@pytest.mark.slow
def test_range_crossing_phase_appearance():
    """log Ksp in [-12, -4]: ML(s) dissolves completely above about -6; the range spans both assemblages."""
    s = toy()
    grid = np.linspace(-12, -4, 17)
    vals = scan(s, TOY_AMOUNTS, M_TOTAL, "Ksp", grid, pH=5)
    present = [bool(O.evaluate(s, TOY_AMOUNTS, M_TOTAL, {"Ksp": g}, pH=5)[1].present) for g in (-12, -4)]
    assert present == [True, False]
    b = O.worst_case_bounds(s, TOY_AMOUNTS, M_TOTAL, {"Ksp": (-12, -4)}, method="local", pH=5)
    assert b.lower == pytest.approx(vals.min(), rel=1e-6)
    assert b.upper == pytest.approx(vals.max(), rel=1e-6)  # the plateau (all M dissolved)
    assert b.upper == pytest.approx(1e-3, rel=1e-6)
    assert b.maximize.verification["agrees"] and b.minimize.verification["agrees"]


def ten_complex_system():
    s = System("ten")
    s.aqueous("M+2")
    s.aqueous("L-")
    for i in range(10):
        s.aqueous(f"ML{i}+")
    s.solid("ML2(s)")
    s.reaction("ML2(s) = M+2 + 2 L-", logK=-12.0, name="Ksp")
    for i in range(10):
        s.reaction(f"M+2 + L- = ML{i}+", logK=2.0 + 0.1 * i, name=f"K{i}")
    return s


@pytest.mark.slow
def test_monte_carlo_underestimates_ten_parameter_range():
    """Dissolved M with 10 uncertain complexes (+-1): the maximum is the all-high corner.

    1000 uniform samples essentially never come near it; the optimizer does (and all 1024 corners confirm).
    """
    s = ten_complex_system()
    am = {"M+2": 1e-3, "L-": 2e-3}
    unc = {f"K{i}": 1.0 for i in range(10)}
    q = O.log10_dissolved({"M+2": 1, **{f"ML{i}+": 1 for i in range(10)}})
    mc = O.monte_carlo_bounds(s, am, q, unc, n_samples=1000, seed=1)
    b = O.worst_case_bounds(s, am, q, unc, method="local", n_starts=2)
    corners = O.corner_bounds(s, am, q, unc, max_dim=10)
    assert b.maximize.verification["agrees"]
    assert b.upper == pytest.approx(corners.upper, abs=1e-6)
    assert b.lower == pytest.approx(corners.lower, abs=1e-6)
    # The maximum is the all-high corner, which 1000 uniform samples miss by > 0.1 in log10. The low side is
    # compressed (dissolved M is floored by Ksp when complexation is weak), so sampling gets closer there; it must
    # still be an inner estimate, never beyond the optimizer's (corner-confirmed) range.
    assert mc.upper < b.upper - 0.1
    assert b.lower <= mc.lower and mc.upper <= b.upper


# --------------------------------------------------------------------------- design
def test_minimum_dose_at_constraint(gibbsite):
    """Minimum NaOH such that dissolved Al <= 1e-6: the constraint is active; left edge of the feasible window."""
    r = O.design(gibbsite, AL, {"NaOH": (0.0, 6e-3)}, objective="NaOH",
                 constraints=[(O.dissolved("Al"), "<=", 1e-6)], reagents=NAOH)
    assert r.status == "local_optimum" and not r.certified
    assert r.verification["agrees"] and r.constraints[0]["satisfied"]
    d = r.decisions["NaOH"]
    q = O.dissolved("Al")

    def diss(x):
        return q.value(gibbsite.equilibrate({**AL, "Na": x, "OH-": x}))

    assert diss(d) == pytest.approx(1e-6, rel=1e-4)
    assert diss(d * (1 - 1e-3)) > 1e-6  # slightly less base is infeasible
    grid = np.linspace(0, 6e-3, 25)
    feas = [x for x in grid if diss(x) <= 1e-6]
    assert feas and d <= min(feas) + 1e-12


# Extra HCl so that the solution without base is undersaturated in Gibbsite (SI = -4.2 at pH 2.3). Without it
# (AL alone) Gibbsite is already saturated at zero dose (pH 3.72), so "min NaOH with SI >= 0" sits at the bound.
ACID = {"Al": 1e-3, "Cl": 8e-3, "H+": 5e-3}


@pytest.mark.slow
def test_optimum_exactly_at_phase_boundary(gibbsite):
    """Minimum NaOH such that Gibbsite is saturated: the optimum is where the phase appears (N = 0, SI = 0).

    Scan: SI = -2.13 at 4 mM NaOH, Gibbsite present at 5 mM, so the boundary is strictly interior.
    """
    r = O.design(gibbsite, ACID, {"NaOH": (0.0, 6e-3)}, objective="NaOH",
                 constraints=[(O.saturation_index("Gibbsite"), ">=", 0.0)], reagents=NAOH)
    assert r.status == "local_optimum" and r.verification["converged"]
    d = r.decisions["NaOH"]
    assert 4e-3 < d < 5e-3

    # bisection on the recomputed equilibria: SI < 0 below the boundary, the phase is present above it
    def saturated(x):
        return bool(gibbsite.equilibrate({**ACID, "Na": x, "OH-": x}).present)

    lo, hi = 4e-3, 5e-3
    assert not saturated(lo) and saturated(hi)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if saturated(mid) else (mid, hi)
    assert d == pytest.approx(hi, rel=1e-5)
    assert r.result.solids.get("Gibbsite", 0.0) <= 1e-9  # the phase is exactly at its appearance point


def test_infeasible_specification_is_reported(gibbsite):
    """NaOH <= 1 mM cannot bring dissolved Al below 1e-6 (the whole range stays acidic)."""
    r = O.design(gibbsite, AL, {"NaOH": (0.0, 1e-3)}, objective="NaOH",
                 constraints=[(O.dissolved("Al"), "<=", 1e-6)], reagents=NAOH)
    assert r.status == "no_feasible_point_found"
    assert not r.certified and r.decisions == {} and r.objective is None
    grid = np.linspace(0, 1e-3, 6)
    assert min(O.dissolved("Al").value(gibbsite.equilibrate({**AL, "Na": x, "OH-": x})) for x in grid) > 1e-5


@pytest.mark.slow
def test_multiple_local_optima_single_start_is_worse(gibbsite):
    """max dissolved Al over NaOH in [0, 6 mM] has local maxima at both ends (interior minimum between).

    A local solve started at the right end stays there (6.5e-4); multistart finds the global maximum at d = 0.
    """
    kw = dict(objective=O.dissolved("Al"), sense="max", reagents=NAOH)
    one = O.design(gibbsite, AL, {"NaOH": (0.0, 6e-3)}, starts=[{"NaOH": 6e-3}], **kw)
    many = O.design(gibbsite, AL, {"NaOH": (0.0, 6e-3)}, **kw)
    assert one.decisions["NaOH"] == pytest.approx(6e-3, rel=1e-6)
    assert many.decisions["NaOH"] == pytest.approx(0.0, abs=1e-9)
    assert many.objective > one.objective * 1.2
    grid = np.linspace(0, 6e-3, 13)
    best = max(O.dissolved("Al").value(gibbsite.equilibrate({**AL, "Na": x, "OH-": x})) for x in grid)
    assert many.objective == pytest.approx(best, rel=1e-6)


# --------------------------------------------------------------------------- global / certificates
@pytest.mark.slow
def test_global_certificate_soundness_toy():
    """Global method on the 1-parameter toy: whatever it claims must be consistent with a dense scan."""
    s = toy()
    q = O.log10_dissolved({"M+": 1, "ML2-": 1})
    grid = np.linspace(-7, -3, 81)
    vals = scan(s, TOY_AMOUNTS, q, "Ka", grid, pH=5)
    for log_span in (None, 12.0):
        b = O.worst_case_bounds(s, TOY_AMOUNTS, q, {"Ka": (-7, -3)}, method="global", time_limit=30,
                                log_span=log_span, pH=5)
        assert b.lower <= vals.min() + 1e-6 and b.upper >= vals.max() - 1e-6
        for side, true in ((b.minimize, vals.min()), (b.maximize, vals.max())):
            if side.certified:
                assert log_span is None  # assumed bounds never yield a certificate
                assert (side.bound <= true + 1e-6) if side is b.minimize else (side.bound >= true - 1e-6)
            if side.conditional_bound is not None and log_span is None:
                cb = side.conditional_bound
                assert (cb <= true + 1e-6) if side is b.minimize else (cb >= true - 1e-6)
        if log_span is not None:
            assert not b.certified and any("assumed" in a for a in b.assumptions)


@pytest.mark.slow
def test_global_design_infeasible_or_honest(gibbsite):
    r = O.design(gibbsite, AL, {"NaOH": (0.0, 1e-3)}, objective="NaOH",
                 constraints=[(O.dissolved("Al"), "<=", 1e-6)], reagents=NAOH, method="global", time_limit=30)
    assert r.objective is None
    assert r.status in ("infeasible", "no_feasible_point_found")
    assert r.certified == (r.status == "infeasible")


def test_variable_bounds_are_valid_at_solutions(gibbsite):
    """LP-derived bounds must contain the equilibrium at every dose in the box."""
    params = [O._make_param(gibbsite, "NaOH", (0.0, 6e-3), "amount", NAOH)]
    prob = O._Problem(gibbsite, AL, params, {})
    lb, ub, rigorous, _ = prob.variable_bounds()
    assert rigorous
    wide = prob.default_bounds()
    for x in np.linspace(0, 6e-3, 5):
        res = prob.equilibrate_at(np.array([x]))
        pt = prob.x_from_result(res, wide)
        assert np.all(pt <= ub + 1e-9) and np.all(pt >= lb - 1e-9)
    # Tightness actually implied by the LP: conservation alone allows all water to become H+ (min W is the
    # default floor, not ~1 kg), so log10 m <= log10(n_max / W_min). Al species: n_max = Al total.
    L, net = prob.L, prob.net
    wmin = lb[L.W]
    dlb, dub = wide
    for i, k in enumerate(net.yidx):
        if net.kind[k] == "aq" and "Al" in net.species[k].name:
            assert ub[i] == pytest.approx(math.log10(1e-3 / wmin), abs=1e-5)
            assert ub[i] < dub[i]


def test_log_span_is_flagged(gibbsite):
    params = [O._make_param(gibbsite, "NaOH", (0.0, 6e-3), "amount", NAOH)]
    prob = O._Problem(gibbsite, AL, params, {})
    _, _, rigorous, notes = prob.variable_bounds(log_span=10.0)
    assert not rigorous and notes


def test_log_quantity_is_finite():
    s = toy()
    val, res = O.evaluate(s, TOY_AMOUNTS, O.log10_dissolved({"M+": 1, "ML2-": 1}), {"Ka": -5.0}, pH=5)
    assert math.isfinite(val) and res.converged

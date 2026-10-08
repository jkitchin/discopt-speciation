"""Implicit-function-theorem sensitivities vs finite differences of re-solves."""

import math

import numpy as np
import pytest

from discopt.speciation import PhreeqcDatabase, System, fit


@pytest.fixture(scope="module")
def db():
    return PhreeqcDatabase.load("phreeqc")


def _solve_at(system, amounts, kw, p, v):
    """Re-solve with parameter ``p`` (Parameter or spec string) at value ``v``."""
    if hasattr(p, "set"):
        v0 = p.get()
        p.set(v)
        try:
            return system.equilibrate(amounts, **kw)
        finally:
            p.set(v0)
    kind, _, name = p.partition(":")
    if kind == "total":
        return system.equilibrate(dict(amounts, **{name: v}), **kw)
    if kind == "fixed":
        return system.equilibrate(amounts, **dict(kw, fixed=dict(kw["fixed"], **{name: v})))
    return system.equilibrate(amounts, **dict(kw, **{p: v}))


def _value(amounts, kw, p):
    if hasattr(p, "get"):
        return p.get()
    kind, _, name = p.partition(":")
    return {"total": lambda: amounts[name], "fixed": lambda: kw["fixed"][name]}.get(kind, lambda: kw[p])()


def fd(system, amounts, kw, p, quantities, rel=1e-3):
    """Richardson-extrapolated central differences of re-solves (O(h^4); steps well above solver noise)."""
    v0 = _value(amounts, kw, p)
    h = rel * (abs(v0) if isinstance(p, str) and p.startswith("total") else max(abs(v0), 1.0))

    def D(hh):
        up = _solve_at(system, amounts, kw, p, v0 + hh)
        dn = _solve_at(system, amounts, kw, p, v0 - hh)
        assert up.converged and dn.converged
        return np.array([(fit.observe(up, q) - fit.observe(dn, q)) / (2 * hh) for q in quantities])

    return (4 * D(h / 2) - D(h)) / 3


def check(system, amounts, kw, params, quantities, rel=1e-6, abs_=1e-12, fd_rel=1e-3):
    r = system.equilibrate(amounts, **kw)
    assert r.converged
    S = r.sensitivity(params, quantities)
    assert S.ok, S.issues
    for j, p in enumerate(params):
        step = fd_rel.get(S.parameters[j], 1e-3) if isinstance(fd_rel, dict) else fd_rel
        ref = fd(system, amounts, kw, p, quantities, rel=step)
        for i, q in enumerate(quantities):
            assert S.jacobian[i, j] == pytest.approx(ref[i], rel=rel, abs=abs_), (S.quantities[i], S.parameters[j])
    return r, S


# --------------------------------------------------------------------------- agreement with FD
def test_calcite_co2_logK_and_fixed_pressure(db):
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
    qs = ["pH", "log10_molality:Ca+2", "solids:Calcite", "gamma:Ca+2", "SI:Calcite", fit.total("Ca+2", "CaHCO3+")]
    r, S = check(s, {"Calcite": 10.0}, {"fixed": {"CO2(g)": -3.5}}, [fit.logK(s, "Calcite"), "fixed:CO2(g)"], qs)
    assert S.linear == [True, True]  # residuals affine in logK and fixed log activities
    assert S.get("SI:Calcite", 0) == pytest.approx(0.0, abs=1e-14)  # saturated phase stays saturated
    assert r.sensitivity([fit.logK(s, "Calcite")], ["pH"]).jacobian.shape == (1, 1)


def test_affine_parameters_are_step_independent(db):
    from discopt.speciation.sensitivity import sensitivities

    # an ideal gas: log10 P enters affinely (a Peng-Robinson gas adds log10 phi(P), which is not affine)
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"], gas_eos="ideal")
    r = s.equilibrate({"Calcite": 10.0}, fixed={"CO2(g)": -3.5})
    params = [fit.logK(s, "Calcite"), "fixed:CO2(g)"]
    a = sensitivities(r, params, ["pH", "solids:Calcite"], rel_step=1e-5).jacobian
    b = sensitivities(r, params, ["pH", "solids:Calcite"], rel_step=0.1).jacobian
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_enthalpy_away_from_25C():
    s = System("g")
    Ca, SO4 = s.aqueous("Ca+2"), s.aqueous("SO4-2")
    s.reaction(s.solid("Gypsum") >> Ca + SO4, logK=-4.58, dH=-1000.0, name="gypsum")
    check(s, {"Gypsum": 1.0}, {"T": 333.15}, [fit.enthalpy(s, "gypsum"), fit.logK(s, "gypsum")],
          ["log10_molality:Ca+2", "solids:Gypsum"])


def test_analytic_logK_far_from_25C(db):
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
    check(s, {"Calcite": 1.0}, {"fixed": {"CO2(g)": -2.0}, "T": 363.15},
          [fit.logK(s, "Calcite"), fit.logK(s, "CO3-2 + H+ = HCO3-")], ["pH", "log10_molality:Ca+2"])


def test_totals_pH_truesdell_jones_a0_charge_balance(db):
    s = db.system(["Na", "Ca", "Cl"])
    a0 = fit.activity_parameter(s, "Ca+2", "a0", initial=5.0)
    _, S = check(s, {"Na": 0.01, "Ca": 0.005, "Cl": 0.02}, {"pH": 7, "water_mass": 1.0, "charge_balance": "Cl-"},
                 ["total:Ca", "pH", a0], ["log10_activity:Ca+2", "ionic_strength", "gamma:Ca+2", "molality:Cl-"])
    assert S.get("molality:Cl-", "total:Ca") == pytest.approx(2.0, rel=1e-5)  # charge balance adds 2 Cl- per Ca


def test_exchange(db):
    s = db.system(["Na", "Ca", "Cl"], exchange=["X"])
    check(s, {"Na": 0.01, "Ca": 0.005, "Cl": 0.02, "X": 0.01}, {"pH": 7}, ["total:X", "total:Ca"],
          ["exchange:CaX2", "exchange:NaX", "log10_molality:Ca+2"])


def test_ddl_surface(db):
    s = db.system(["Na", "Cl", "Zn"], surfaces={"Hfo": {"area": 600, "mass": 0.09}})
    # d I / d Hfo_w is ~2e-5 at I = 0.1: a step of 1e-3 * 2e-4 moves I by ~4e-12, so re-solve differences
    # are rounding-limited (~1e-5 relative); Richardson FD converges to the IFT value at steps of 1e-2..3e-2.
    # pH needs the default step: at 0.05 pH units the reference itself has truncation error.
    _, S = check(s, {"Na": 0.1, "Cl": 0.1, "Zn": 1e-6, "Hfo_w": 2e-4, "Hfo_s": 5e-6}, {"pH": 5},
                 ["total:Hfo_w", "total:Zn", "pH"], ["surface:Hfo_sOZn+", "log10_molality:Zn+2", "ionic_strength"],
                 fd_rel={"total:Hfo_w": 1e-2})
    assert S.linear[0] is False  # site totals enter the residuals logarithmically


def test_fixed_volume_gas_and_free_water(db):
    s = db.system(["Na", "Cl", "C(4)"], gas_phase={"members": ["CO2(g)"], "volume": 1.0})
    check(s, {"Na": 0.01, "Cl": 0.01, "C(4)": 0.1}, {"pH": 5}, ["total:C(4)", fit.logK(s, "CO2(g) = CO2")],
          ["gases:CO2(g)", "log10_partial_pressure:CO2(g)", "water_mass"])
    h = db.system(["Na", "Cl"], phases=["Halite"])
    check(h, {"Na": 8.0, "Cl": 8.0}, {"pH": 7}, ["total:Na", fit.logK(h, "Halite")],
          ["water_mass", "molality:Na+", "solids:Halite"])


@pytest.mark.slow
def test_pitzer_brine():
    pz = PhreeqcDatabase.load("pitzer.dat")
    s = pz.system(["Na", "Cl", "Ca", "Mg", "S(6)"], phases=["Gypsum", "Halite"])
    P = s.activity.params
    key = ("Cl-", "Na+")

    def set_b0(v):
        P.data["B0"][key] = (float(v),) + P.data["B0"][key][1:]

    b0 = fit.Parameter("B0 Na-Cl", set_b0, lambda: P.data["B0"][key][0], scale=0.1)
    r, S = check(s, {"Na": 5.0, "Cl": 5.0, "Ca": 0.05, "Mg": 0.5, "S(6)": 0.3}, {"pH": 7},
                 [b0, "total:Mg"], ["log10_activity:Ca+2", "molality:SO4-2", "log10_activity:H2O", "solids:Gypsum"])
    assert r.ionic_strength > 6 and r.present == ["Gypsum"]
    assert S.linear == [False, True]  # Pitzer coefficients enter nonlinearly (Richardson difference)


# --------------------------------------------------------------------------- adversarial
def _saturated_gypsum():
    s = System("b")
    Ca, SO4 = s.aqueous("Ca+2"), s.aqueous("SO4-2")
    s.reaction(s.solid("Gypsum") >> Ca + SO4, logK=-4.0, name="gypsum")
    return s, {"Ca+2": 1e-2, "SO4-2": 1e-2}  # ideal: exactly at saturation, no solid


def test_phase_at_saturation_boundary_is_flagged_with_one_sided_values():
    s, a = _saturated_gypsum()
    pK = fit.logK(s, "gypsum")
    qs = ["log10_molality:SO4-2", "solids:Gypsum"]
    S = s.equilibrate(a).sensitivity(["total:Ca+2", pK], qs)
    assert not S.ok and S.boundary == ["Gypsum"] and "boundary" in S.issues[0]
    assert np.all(np.isnan(S.jacobian))

    def f(amounts):
        r = s.equilibrate(amounts)
        return np.array([fit.observe(r, q) for q in qs])

    h, f0 = 1e-7, f(a)
    right = (f(dict(a, **{"Ca+2": 1e-2 + h})) - f0) / h  # adding Ca precipitates gypsum
    left = (f0 - f(dict(a, **{"Ca+2": 1e-2 - h}))) / h
    pK.set(-4.0 + h)
    right_k = (f(a) - f0) / h  # raising K keeps everything dissolved
    pK.set(-4.0 - h)
    left_k = (f0 - f(a)) / h
    pK.set(-4.0)
    one_sided_err = dict(rel=1e-5, abs=1e-9)  # O(h) error of one-sided differences
    assert S.right[:, 0] == pytest.approx(right, **one_sided_err)
    assert S.left[:, 0] == pytest.approx(left, **one_sided_err)
    assert S.right[:, 1] == pytest.approx(right_k, **one_sided_err)
    assert S.left[:, 1] == pytest.approx(left_k, **one_sided_err)
    assert S.right[1, 0] == pytest.approx(0.5, rel=1e-9) and S.left[1, 0] == 0.0


def test_fit_falls_back_to_fd_at_boundary():
    s, a = _saturated_gypsum()
    kw = dict(method="complementarity", activity=None, fd_step=1e-4, tol=1e-10, max_nfev=10, strict=False,
              failures=[], jacobian="ift", fallbacks=[], solves=0)
    obs = [fit.Observation({"amounts": a}, "log10_molality:SO4-2", 0.0, label="boundary"),
           fit.Observation({"amounts": {"Gypsum": 1.0}}, "log10_molality:SO4-2", 0.0, label="saturated")]
    J, _ = fit._model_jacobian(s, [fit.logK(s, "gypsum")], [-4.0], obs, kw)
    assert [lab for lab, _ in kw["fallbacks"]] == ["boundary"]
    assert np.all(np.isfinite(J)) and J[1, 0] == pytest.approx(0.5, rel=1e-12)


def test_trace_species_relative_derivatives():
    s = System("t")
    Fe, H, OH, FeOH = s.aqueous("Fe+3"), s.aqueous("H+"), s.aqueous("OH-"), s.aqueous("FeOH+2")
    s.aqueous("Cl-")
    s.reaction({H: 1, OH: 1}, logK=-14.0)
    s.reaction(Fe + OH >> FeOH, logK=11.8, name="FeOH")
    total = 1e-25
    a = {"Fe+3": total, "Cl-": 3 * total}
    r = s.equilibrate(a, pH=3)
    S = r.sensitivity(["total:Fe+3", fit.logK(s, "FeOH")], ["log10_molality:FeOH+2", "log10_molality:Fe+3"])
    assert S.ok
    frac = 10 ** 0.8 / (1 + 10 ** 0.8)  # K a_OH / (1 + K a_OH), ideal
    # exact: log10 m = log10 total + const, and d log10 m(Fe+3) / d logK = -frac
    assert total * S.jacobian[0, 0] == pytest.approx(1 / math.log(10), rel=1e-9)
    assert S.jacobian[1, 1] == pytest.approx(-frac, rel=1e-9)
    assert S.jacobian[0, 1] == pytest.approx(1 - frac, rel=1e-9)
    # the fit's default FD step (h = fd_step * max(|theta|, scale=1)) cannot even be taken for this total
    with pytest.raises(ValueError):
        s.equilibrate(dict(a, **{"Fe+3": total - 1e-4}), pH=3)


def test_nearly_identical_polymorphs_are_flagged_singular():
    s = System("CE1")
    Fe, H, OH = s.aqueous("Fe+3"), s.aqueous("H+"), s.aqueous("OH-")
    s.aqueous("NO3-")
    s.reaction({H: 1, OH: 1}, logK=-14.0)
    s.reaction(s.solid("Ferrihydrite") >> Fe + 3 * OH, logK=-43.0 + 1e-9, name="fh")
    s.reaction(s.solid("Goethite") >> Fe + 3 * OH, logK=-43.0, name="goe")
    r = s.equilibrate({"Fe+3": 1e-3, "NO3-": 3e-3}, pH=4)
    S = r.sensitivity([fit.logK(s, "goe"), "total:Fe+3"], ["log10_molality:Fe+3", "solids:Goethite"])
    assert not S.ok
    assert any("singular" in m or "ill-conditioned" in m or "boundary" in m for m in S.issues)
    assert not np.all(np.isfinite(S.jacobian))


def test_surface_at_very_low_ionic_strength(db):
    s = db.system(["Na", "Cl", "Zn"], surfaces={"Hfo": {"area": 600, "mass": 0.09}})
    _, S = check(s, {"Na": 1e-7, "Cl": 1e-7, "Zn": 1e-9, "Hfo_w": 2e-4, "Hfo_s": 5e-6}, {"pH": 7},
                 ["total:Zn"], ["surface:Hfo_sOZn+", "log10_molality:Zn+2"])
    assert S.condition < 1e12


def test_independent_parameter_gives_exact_zero(db):
    s = db.system(["Na", "Ca", "Cl", "C(4)"], phases=["Calcite"])
    r = s.equilibrate({"Na": 0.01, "Cl": 0.01}, pH=7)  # no Ca or C: calcite cannot form
    S = r.sensitivity([fit.logK(s, "Calcite")], ["molality:Na+", "pH", lambda res: res.ionic_strength])
    assert S.ok and np.all(S.jacobian == 0.0)


def test_callable_quantity_matches_builtin(db):
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
    r = s.equilibrate({"Calcite": 10.0}, fixed={"CO2(g)": -3.5})
    S = r.sensitivity([fit.logK(s, "Calcite")], ["molality:Ca+2", lambda res: res.molality["Ca+2"]])
    assert S.jacobian[1, 0] == pytest.approx(S.jacobian[0, 0], rel=1e-9)


def test_bad_specs(db):
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
    r = s.equilibrate({"Calcite": 10.0}, fixed={"CO2(g)": -3.5})
    with pytest.raises(KeyError):
        r.sensitivity(["pH"], ["pH"])  # pH was not fixed
    with pytest.raises(ValueError):
        r.sensitivity(["bogus"], ["pH"])


# --------------------------------------------------------------------------- fitting
def test_fit_ift_matches_fd_with_fewer_solves():
    from test_fit_plot import gypsum_system, synthetic

    out = {}
    for jm in ("fd", "ift"):
        s = gypsum_system(logK=-4.3, pair=2.0)
        params = [fit.logK(s, "gypsum", lower=-6, upper=-3), fit.logK(s, "pair", lower=0, upper=4)]
        with pytest.warns(UserWarning):
            out[jm] = fit.fit_equilibrium(s, synthetic(), params, absolute_sigma=True, jacobian=jm)
    fd_, ift = out["fd"], out["ift"]
    assert ift.jacobian_method == "ift" and not ift.jacobian_fallbacks
    for k in fd_.parameters:
        assert ift.parameters[k] == pytest.approx(fd_.parameters[k], abs=1e-6)
        assert ift.standard_errors[k] == pytest.approx(fd_.standard_errors[k], rel=1e-4)
    assert ift.n_solves < fd_.n_solves / 2
    with pytest.raises(ValueError):
        fit.fit_equilibrium(gypsum_system(), synthetic(), [fit.logK(gypsum_system(), "gypsum")], jacobian="bad")

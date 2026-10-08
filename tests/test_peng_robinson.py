"""Peng-Robinson gases: the equation of state against PHREEQC, and its use in equilibrium calculations.

Reference data: ``tests/data/phreeqc_peng_robinson.json`` (written by
``tests/data/generate_phreeqc_pr.py``, PHREEQC 3.8.7 through phreeqpython). The
PHREEQC example problems that depend on Peng-Robinson (ex7 gas phases, ex10
solution 1) are in ``test_phreeqc_examples.py``.
"""

import json
import math
import pathlib

import numpy as np
import pytest

from discopt.speciation import PhreeqcDatabase, System
from discopt.speciation.activity import NUM
from discopt.speciation.database import DATA_DIR
from discopt.speciation.eos import PengRobinson, pure_ln_phi

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "phreeqc_peng_robinson.json").read_text())
LN10 = math.log(10.0)
# PHREEQC's gas constant is 0.0820597 L atm/(mol K); the package's is R/101.325 = 0.0820574. At fixed P that moves
# V_m by 3e-5 relative and phi by < 1e-7; at fixed V it moves P by 3e-5 relative and phi by < 4e-6.
PHI_TOL = {"pressure": 1e-7, "fixed_si": 1e-7, "volume": 5e-6}


@pytest.fixture(scope="module")
def db_system():
    return PhreeqcDatabase.load("phreeqc.dat").system(["C", "N"], phases=["CO2(g)", "CH4(g)", "N2(g)", "H2O(g)"])


@pytest.mark.parametrize("case", REF["cases"], ids=[c["id"] for c in REF["cases"]])
def test_fugacity_coefficients_match_phreeqc(case, db_system):
    T = case["tc"] + 273.15
    if case["kind"] == "fixed_si":
        phi = math.exp(pure_ln_phi(db_system[case["gas"]], T, 10.0 ** case["log_p"]))
        assert abs(phi - case["phi"]) < PHI_TOL["fixed_si"]
        # PHREEQC's SI of the gas is its log10 fugacity
        assert abs(case["log_p"] + math.log10(phi) - case["si"]) < 1e-7
        return
    gases = {g: v for g, v in case["gases"].items() if v["moles"] > 0}
    names = list(gases)
    n = [gases[g]["moles"] for g in names]
    x = [v / sum(n) for v in n]
    model = PengRobinson.for_members([db_system[g] for g in names])
    if case["kind"] == "pressure":
        P = case["P"]
        phi = np.exp(model.ln_phi(x, T, P))
        assert abs(model.molar_volume(x, T, P) / case["Vm"] - 1.0) < 1e-4
    else:
        V = case["value"]
        P = model.pressure(NUM, n, V, T)
        assert abs(P / case["P"] - 1.0) < 1e-4
        phi = np.exp(model.ln_fugacity(NUM, n, V, T)) / (np.array(x) * P)
    for g, ph in zip(names, phi):
        assert abs(ph - gases[g]["phi"]) < PHI_TOL[case["kind"]], g
        assert abs(P * gases[g]["moles"] / sum(n) / gases[g]["p"] - 1.0) < 1e-4, g


def test_fugacities_are_thermodynamically_consistent(db_system):
    """d ln f_i / d n_j is symmetric (f from a Gibbs energy) and sum_j n_j d ln f_i / d n_j = 0 (Gibbs-Duhem)."""
    model = PengRobinson.for_members([db_system[g] for g in ("CO2(g)", "CH4(g)", "N2(g)", "H2O(g)")])
    T, P = 330.0, 60.0

    def lnf(n):
        x = np.asarray(n) / sum(n)
        return np.array(model.ln_fugacity(NUM, list(x), model.molar_volume(x, T, P), T))

    n0, h = np.array([0.5, 0.3, 0.15, 0.05]), 1e-6
    J = np.column_stack([(lnf(n0 + h * e) - lnf(n0 - h * e)) / (2 * h) for e in np.eye(4)])
    assert np.max(np.abs(J - J.T)) < 1e-7
    assert np.max(np.abs(J @ n0)) < 1e-7


def _co2_water(eos=None, gas_eos="auto", **gas):
    s = System("co2", gas_eos=gas_eos)
    s.solvent("H2O")
    for name in ("H+", "OH-", "CO2", "HCO3-"):
        s.aqueous(name)
    co2g = s.gas("CO2(g)", **gas)
    s.reaction("H2O = H+ + OH-", logK=-14.0)
    s.reaction("CO2 = H+ + HCO3-", logK=-6.35)
    s.reaction("CO2(g) = CO2", logK=-1.468)
    return s, co2g


def test_user_critical_constants_and_ideal_override():
    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    s.gas_phase(["CO2(g)"], pressure=50.0)
    r = s.equilibrate({"CO2(g)": 10.0})
    assert r.converged and r.gas_phases["gas"]["eos"] == "peng-robinson"
    phi = r.fugacity_coefficient["CO2(g)"]
    assert abs(phi - math.exp(pure_ln_phi(s["CO2(g)"], s.T, 50.0))) < 1e-9
    # mass action uses the fugacity: log a(CO2) = logK + log10(phi P)
    assert abs(r.log10_activity["CO2"] - (-1.468 + math.log10(phi * 50.0))) < 1e-8
    assert abs(r.SI["CO2(g)"] - math.log10(phi * 50.0)) < 1e-8
    assert abs(r.partial_pressure["CO2(g)"] - 50.0) < 1e-9

    ideal = System.from_spec(dict(s.to_spec(), gas_eos="ideal"))
    ri = ideal.equilibrate({"CO2(g)": 10.0})
    assert ri.gas_phases["gas"]["eos"] == "ideal" and ri.fugacity_coefficient["CO2(g)"] == 1.0
    assert abs(ri.log10_activity["CO2"] - (-1.468 + math.log10(50.0))) < 1e-8
    # about 26 % less dissolved CO2 at 50 atm
    assert ri.molality["CO2"] / r.molality["CO2"] == pytest.approx(1.0 / phi, rel=1e-6)


def test_fixed_volume_gas():
    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    s.gas_phase(["CO2(g)"], volume=1.0)
    r = s.equilibrate({"CO2(g)": 3.0})
    assert r.converged
    d = r.gas_phases["gas"]
    n = r.gases["CO2(g)"]
    model = PengRobinson.for_members([s["CO2(g)"]])
    assert d["pressure"] == pytest.approx(model.pressure(NUM, [n], 1.0, s.T), rel=1e-12)
    assert r.fugacity_coefficient["CO2(g)"] == pytest.approx(10.0 ** r.SI["CO2(g)"] / d["pressure"], rel=1e-10)
    assert r.fugacity_coefficient["CO2(g)"] < 0.85  # ~37 atm of CO2
    assert r.SI["CO2(g)"] == pytest.approx(model.ln_fugacity(NUM, [n], 1.0, s.T)[0] / LN10, abs=1e-8)
    # fewer moles than the ideal gas would need for the same fugacity
    assert n > 10.0 ** r.SI["CO2(g)"] * 1.0 / (0.0820574 * s.T)


def test_fixed_partial_pressure_is_a_fugacity():
    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    r = s.equilibrate({}, fixed={"CO2(g)": math.log10(20.0)})
    phi = math.exp(pure_ln_phi(s["CO2(g)"], s.T, 20.0))
    assert r.partial_pressure["CO2(g)"] == pytest.approx(20.0)
    assert r.fugacity_coefficient["CO2(g)"] == pytest.approx(phi)
    assert r.SI["CO2(g)"] == pytest.approx(math.log10(20.0 * phi))
    assert r.log10_activity["CO2"] == pytest.approx(-1.468 + math.log10(20.0 * phi), abs=1e-8)
    plain, _ = _co2_water()  # no critical constants: ideal
    rp = plain.equilibrate({}, fixed={"CO2(g)": math.log10(20.0)})
    assert rp.fugacity_coefficient["CO2(g)"] == 1.0
    assert rp.log10_activity["CO2"] == pytest.approx(-1.468 + math.log10(20.0), abs=1e-8)


def test_binary_parameters_and_absent_phase():
    db = PhreeqcDatabase.load("phreeqc.dat")
    base = dict(gas_phase={"members": ["CO2(g)", "CH4(g)"], "pressure": 100.0})
    amounts = {"CO2(g)": 2.0, "CH4(g)": 2.0}
    r0 = db.system(["C"], **base).equilibrate(amounts)
    base["gas_phase"]["kij"] = [["CO2(g)", "CH4(g)", 0.1]]
    r1 = db.system(["C"], **base).equilibrate(amounts)
    assert r0.converged and r1.converged
    # a positive k_ij weakens the attraction between unlike molecules: higher fugacity coefficients
    assert r1.fugacity_coefficient["CO2(g)"] > r0.fugacity_coefficient["CO2(g)"] + 1e-3
    x = [r1.gas_phases["gas"]["mole_fraction"][g] for g in ("CO2(g)", "CH4(g)")]
    model = PengRobinson.for_members([db.system(["C"], **base)[g] for g in ("CO2(g)", "CH4(g)")],
                                     kij={("CO2(g)", "CH4(g)"): 0.1})
    assert np.allclose(np.exp(model.ln_phi(x, 298.15, 100.0)),
                       [r1.fugacity_coefficient[g] for g in ("CO2(g)", "CH4(g)")], rtol=1e-8)
    # too little gas for 100 atm: the phase is absent, with phi of the composition that would form
    r2 = db.system(["C"], **base).equilibrate({"CO2(g)": 0.01, "CH4(g)": 1e-4})
    assert r2.converged and r2.present == [] and r2.partial_pressure["CO2(g)"] == 0.0
    assert 0.3 < r2.fugacity_coefficient["CO2(g)"] < 1.0


def test_mixed_constants_raise_and_auto_stays_ideal_without_constants():
    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    s.aqueous("Ar")
    s.gas("Ar(g)")
    s.reaction("Ar(g) = Ar", logK=-2.85)
    s.gas_phase(["CO2(g)", "Ar(g)"], pressure=1.0)
    with pytest.raises(ValueError, match="no critical constants"):
        s.equilibrate({"CO2(g)": 0.1, "Ar(g)": 0.1})
    s.critical_constants("Ar(g)", Tc=150.8, Pc=48.0, omega=0.0)
    r = s.equilibrate({"CO2(g)": 0.1, "Ar(g)": 0.1})
    assert r.converged and r.gas_phases["gas"]["eos"] == "peng-robinson"
    with pytest.raises(ValueError, match="critical constants need"):
        s.critical_constants("Ar(g)", Tc=150.8, Pc=None)


def test_spec_round_trip_keeps_constants_eos_and_kij():
    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    s.gas_phase(["CO2(g)"], pressure=30.0, eos="peng-robinson")
    spec = s.to_spec()
    gas = next(sp for sp in spec["species"] if sp["name"] == "CO2(g)")
    assert gas["params"] == {"Tc": 304.2, "Pc": 72.86, "omega": 0.225}
    assert spec["gas_phases"][0]["eos"] == "peng-robinson"
    s2 = System.from_spec(spec)
    a, b = s.equilibrate({"CO2(g)": 5.0}), s2.equilibrate({"CO2(g)": 5.0})
    assert a.fugacity_coefficient == pytest.approx(b.fugacity_coefficient)
    text = (pathlib.Path(DATA_DIR) / "phreeqc.dat").read_bytes().decode("latin-1")
    db = PhreeqcDatabase(text + "\nGAS_BINARY_PARAMETERS\nCO2(g) CH4(g) 0.12\n")
    assert db.gas_binary_parameters == {frozenset(("CO2(g)", "CH4(g)")): 0.12}
    sd = db.system(["C"], gas_phase={"members": ["CO2(g)", "CH4(g)"], "pressure": 10.0})
    kij = sd.to_spec()["gas_phases"][0]["kij"]
    assert [sorted(t[:2]) + [t[2]] for t in kij] == [["CH4(g)", "CO2(g)", 0.12]]


def test_database_reads_dashless_options():
    db = PhreeqcDatabase.load("pitzer.dat")
    assert db.phases["Oxg(g)"].options["t_c"] == 154.6  # written "T_c  154.6 ; -P_c ..." in pitzer.dat
    assert db.phases["Mtg(g)"].options["p_c"] == 45.4


def test_gibbs_method_and_optimize_refuse_peng_robinson():
    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    s.gas_phase(["CO2(g)"], pressure=10.0)
    with pytest.raises(NotImplementedError, match="Peng-Robinson"):
        s.equilibrate({"CO2(g)": 1.0}, method="gibbs")
    from discopt.speciation import optimize as O

    with pytest.raises(NotImplementedError, match="Peng-Robinson"):
        O.design(s, {"CO2(g)": 1.0}, {"H+": (0.0, 1e-3)}, objective="H+")


def test_certificate_lists_the_vapor_assumption():
    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    s.gas_phase(["CO2(g)"], pressure=10.0)
    r = s.equilibrate({"CO2(g)": 1.0}, certify=True)
    assert r.certificate.status == "certified"
    assert any("Peng-Robinson" in a for a in r.certificate.assumptions)
    assert r.certificate.phases[0].method == "stationary point (assumed)"


def test_sensitivity_through_peng_robinson_gas():
    from discopt.speciation import fit

    s, _ = _co2_water(Tc=304.2, Pc=72.86, omega=0.225)
    s.gas_phase(["CO2(g)"], pressure=40.0)
    amounts = {"CO2(g)": 5.0}
    pK = fit.logK(s, "CO2(g) = CO2")
    S = s.equilibrate(amounts).sensitivity([pK], ["log10_activity:CO2", "gases:CO2(g)"])
    assert S.ok, S.issues
    h, k0 = 1e-5, pK.get()
    out = []
    for v in (k0 + h, k0 - h):
        pK.set(v)
        out.append(s.equilibrate(amounts).gases["CO2(g)"])
    pK.set(k0)
    assert S.get("gases:CO2(g)", 0) == pytest.approx((out[0] - out[1]) / (2 * h), rel=1e-5)
    # the gas stays at 40 atm with the same fugacity, so log a(CO2) moves one for one with logK
    assert S.get("log10_activity:CO2", 0) == pytest.approx(1.0, abs=1e-8)


def test_sensitivity_to_a_fixed_peng_robinson_pressure():
    """d/d log10 P includes d log10 phi / d log10 P (not affine), and agrees with finite differences."""
    db = PhreeqcDatabase.load("phreeqc.dat")
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
    amounts, lp = {"Calcite": 10.0}, 1.3
    S = s.equilibrate(amounts, fixed={"CO2(g)": lp}).sensitivity(["fixed:CO2(g)"], ["pH", "log10_activity:CO2"])
    assert S.ok
    h = 1e-4
    up, dn = (s.equilibrate(amounts, fixed={"CO2(g)": lp + c * h}) for c in (1, -1))
    assert S.get("pH", 0) == pytest.approx((up.pH - dn.pH) / (2 * h), rel=1e-6)
    dphi = (math.log10(up.fugacity_coefficient["CO2(g)"]) - math.log10(dn.fugacity_coefficient["CO2(g)"])) / (2 * h)
    assert S.get("log10_activity:CO2", 0) == pytest.approx(1.0 + dphi, rel=1e-6)
    assert dphi < -0.02  # ~20 atm of CO2

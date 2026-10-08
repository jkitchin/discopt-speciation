"""Pressure: log K from molar volumes, water properties at P, and which pressure a calculation uses.

The comparison with PHREEQC at 100-500 atm (phreeqc.dat, pitzer.dat, sit.dat, llnl.dat), at the saturation
pressure above 100 C, and for example 22 (CO2 to ~1500 atm) is in ``test_phreeqc_examples.py``.
"""

import math

import pytest

from discopt.speciation import PhreeqcDatabase, System
from discopt.speciation import water as W
from discopt.speciation.volume import pressure_factor, reaction_volume


@pytest.fixture(scope="module")
def db():
    return PhreeqcDatabase.load("phreeqc.dat")


def test_water_properties():
    w1, w500 = W.water_state(298.15, 1.0), W.water_state(298.15, 500.0)
    assert w1.rho == pytest.approx(0.99704, abs=2e-5)          # IAPWS-95: 0.997047 g/cm3
    assert w1.Av == pytest.approx(1.87, abs=0.03)              # Debye-Hueckel volume slope, cm3 kg^0.5 mol^-1.5
    assert w1.kappa == pytest.approx(4.52e-5, rel=0.01)        # 1/atm
    assert w500.rho > w1.rho * 1.02
    assert W.saturation_pressure(373.15) == pytest.approx(1.0, abs=2e-3)
    # below the vapor pressure the properties are those at the vapor pressure
    assert W.water_state(473.15, 1.0).P == pytest.approx(W.saturation_pressure(473.15))
    assert W.density(298.15) == pytest.approx(w1.rho)


def test_no_correction_at_or_below_one_atm():
    assert pressure_factor(298.15, 1.0) == 0.0 and pressure_factor(298.15, 0.5) == 0.0
    assert pressure_factor(298.15, 101.0) == pytest.approx(100 * 101325e-9 / (math.log(10) * 0.0083147 * 298.15))


def test_calcite_dissolution_volume(db):
    s = db.system(["Ca", "C(4)"], phases=["Calcite"])
    rxn = next(r for r in s.reactions if r.name == "Calcite")
    dv0, terms = reaction_volume(rxn.stoich, W.water_state(298.15, 1.0))
    assert -62.0 < dv0 < -54.0  # literature: about -58 cm3/mol at infinite dilution
    assert terms  # ionic-strength terms of Ca+2 and CO3-2


def test_log_k_follows_the_volume_change(db):
    s = db.system(["Ca", "C(4)"], phases=["Calcite"])
    r1 = s.equilibrate({"Calcite": 1.0}, P=1.0)
    r500 = s.equilibrate({"Calcite": 1.0}, P=500.0)
    assert r1.converged and r500.converged and r500.pressure == 500.0
    # calcite is more soluble under pressure (dV < 0)
    assert r500.molality["Ca+2"] > 1.3 * r1.molality["Ca+2"]


def test_user_molar_volumes():
    def system(vm):
        s = System("ab")
        s.aqueous("A+", Vm=vm)
        s.aqueous("B-")
        s.solid("AB", Vm=30.0)
        s.reaction("AB = A+ + B-", logK=-5.0, name="AB")
        return s

    s = system([10.0])  # a1 only: 10 * 4.184 cm3/mol after PHREEQC's scaling (cal/bar -> cm3)
    r = s.equilibrate({"AB": 1.0}, P=201.0)
    ws = W.water_state(298.15, 201.0)
    dV = 10.0 * 4.184004 + 0.5 * ws.Av * math.sqrt(r.ionic_strength) - 30.0
    lk = -5.0 - dV * 200 * 101325e-9 / (math.log(10) * 0.0083147 * 298.15)
    assert r.log10_activity["A+"] + r.log10_activity["B-"] == pytest.approx(lk, abs=1e-9)
    spec = s.to_spec()
    s2 = System.from_spec(spec)
    assert s2.equilibrate({"AB": 1.0}, P=201.0).molality["A+"] == pytest.approx(r.molality["A+"], rel=1e-10)


def test_pressure_sources(db):
    s = db.system(["C(4)"], gas_phase={"members": ["CO2(g)"], "pressure": 50.0})
    r = s.equilibrate({"CO2": 5.0})
    assert r.pressure == 50.0 and r._state["pressure_source"] == "gas phase"
    assert s.equilibrate({"CO2": 5.0}, P=10.0).pressure == 10.0

    sv = db.system(["C(4)"], gas_phase={"members": ["CO2(g)", "H2O(g)"], "volume": 1.0})
    rv = sv.equilibrate({"CO2": 10.0}, T=323.15)
    assert rv.converged and rv.pressure == pytest.approx(rv.gas_phases["gas"]["pressure"], rel=1e-10)
    assert rv.pressure > 50.0

    sf = db.system(["C(4)"], phases=["CO2(g)"])
    assert sf.equilibrate({}, fixed={"CO2(g)": 1.5}).pressure == pytest.approx(10.0**1.5)

    hot = db.system(["Na", "Cl"]).equilibrate({"Na": 1.0, "Cl": 1.0}, T=473.15)
    aw = 10.0 ** hot.log10_activity["H2O"]
    assert hot.pressure == pytest.approx(W.saturation_pressure(473.15) * aw, rel=1e-9)
    assert "saturation" in hot._state["pressure_source"]


def test_pitzer_b_changes_above_one_atm():
    pdb = PhreeqcDatabase.load("pitzer.dat")
    s = pdb.system(["Na", "Cl"])
    ctx1 = s.activity.prepare([s["Na+"], s["Cl-"]], 298.15, 1.01325)
    ctx2 = s.activity.prepare([s["Na+"], s["Cl-"]], 298.15, 200 * 1.01325)
    assert ctx1.data["bz"] is None
    b1, b2 = ctx2.data["bz"][1], ctx2.data["bz"][2]
    assert b1 == pytest.approx(1.2 - (7e-5 + 1.93e-9 * 48.15**2) * 200)
    assert b2 == pytest.approx(1.2 - 9.65e-10 * 35.15**2.773 * 200**0.623)


def test_gibbs_and_global_refuse_ionic_strength_dependent_log_k(db):
    s = db.system(["Ca", "C(4)"], phases=["Calcite"], activity="davies")  # an excess Gibbs energy, so gibbs works
    with pytest.raises(NotImplementedError, match="ionic strength"):
        s.equilibrate({"Calcite": 1.0}, P=200.0, method="gibbs")
    assert s.equilibrate({"Calcite": 1.0}, P=1.0, method="gibbs").converged


def test_sensitivity_through_ionic_strength_dependent_log_k(db):
    from discopt.speciation import fit

    s = db.system(["Ca", "C(4)", "Na", "Cl"], phases=["Calcite"])
    amounts, kw = {"Calcite": 1.0, "Na": 0.5, "Cl": 0.5}, {"P": 300.0, "fixed": {"H+": -7.5}}
    p = fit.logK(s, "Calcite")
    S = s.equilibrate(amounts, **kw).sensitivity([p, "total:Na"], ["log10_molality:Ca+2", "ionic_strength"])
    assert S.ok, S.issues
    for j, (get, set_, h) in enumerate([(p.get, p.set, 1e-5), (None, None, 1e-6)]):
        if get is None:
            up = s.equilibrate({**amounts, "Na": 0.5 + h}, **kw)
            dn = s.equilibrate({**amounts, "Na": 0.5 - h}, **kw)
        else:
            k0 = get()
            set_(k0 + h)
            up = s.equilibrate(amounts, **kw)
            set_(k0 - h)
            dn = s.equilibrate(amounts, **kw)
            set_(k0)
        fd = (math.log10(up.molality["Ca+2"]) - math.log10(dn.molality["Ca+2"])) / (2 * h)
        assert S.jacobian[0, j] == pytest.approx(fd, rel=1e-5, abs=1e-9)


def test_sensitivity_with_the_saturation_pressure(db):
    """Above 100 C the pressure is p_sat a_w, which moves with the solution: the derivatives include it."""
    s = db.system(["Ca", "C(4)", "Na", "Cl"], phases=["Calcite"])
    amounts, kw = {"Calcite": 1.0, "Na": 2.0, "Cl": 2.0}, {"T": 473.15, "fixed": {"H+": -6.0}}
    r = s.equilibrate(amounts, **kw)
    assert "saturation" in r._state["pressure_source"] and r.pressure > 10.0
    S = r.sensitivity(["total:Na"], ["log10_molality:Ca+2"])
    assert S.ok, S.issues
    h = 1e-5
    up = s.equilibrate({**amounts, "Na": 2.0 + h}, **kw)
    dn = s.equilibrate({**amounts, "Na": 2.0 - h}, **kw)
    assert up.pressure != dn.pressure
    fd = (math.log10(up.molality["Ca+2"]) - math.log10(dn.molality["Ca+2"])) / (2 * h)
    assert S.jacobian[0, 0] == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("label", ["50C-10mol", "100C-20mol", "150C-25mol"])
def test_example_22_finds_the_gas_pressure_itself(label):
    """PHREEQC example 22 without being told the pressure: it follows the fixed-volume gas, as in PHREEQC."""
    import json
    import pathlib

    data = json.loads((pathlib.Path(__file__).parent / "data" / "phreeqc_examples.json").read_text())
    case = next(c for c in data["cases"] if c["id"] == "ex22")
    step = next(s for s in case["steps"] if s["label"] == label)
    ref = case["rows"][step["row"]]
    s = PhreeqcDatabase.load("phreeqc.dat").system(**step["system"])
    amounts = {k: v / step["W0"] for k, v in step["add"].items()}
    for i, f, groups in step["base"]:
        for name, n in case["rows"][i]["n"].items():
            if name in s and s[name].phase not in ("solvent", "electron"):
                amounts[name] = amounts.get(name, 0.0) + f * n / step["W0"]
    r = s.equilibrate(amounts, T=273.15 + ref["tc"])
    assert r.converged and r._state["pressure_source"] == "gas volume"
    assert r.pressure == pytest.approx(ref["P"], rel=2e-6)
    assert r.pH == pytest.approx(ref["pH"], abs=2e-5)

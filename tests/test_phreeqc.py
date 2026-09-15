"""Bundled PHREEQC databases: parsing, and agreement with PHREEQC 3 itself.

Reference values in ``tests/data/phreeqc_*.json`` were computed with PHREEQC 3
(phreeqpython 1.6.2) from the same database files.
"""

import json
import math
import pathlib

import pytest

from discopt.speciation import PhreeqcDatabase, available_databases
from discopt.speciation.database import parse_phreeqc_equation

DATA = pathlib.Path(__file__).parent / "data"
FEATURES = {c["label"]: c for c in json.loads((DATA / "phreeqc_features.json").read_text())}
ACTIVITY = json.loads((DATA / "phreeqc_activity.json").read_text())


@pytest.fixture(scope="module")
def db():
    return PhreeqcDatabase.load("phreeqc")


@pytest.fixture(scope="module")
def pz():
    return PhreeqcDatabase.load("pitzer.dat")


def close(a, b, rel=0.0, abs_=0.0):
    return a == pytest.approx(b, rel=rel, abs=abs_)


# --------------------------------------------------------------------------- parsing
def test_bundled():
    assert {"phreeqc.dat", "pitzer.dat", "wateq4f_PWN.dat", "llnl.dat", "sit.dat"} <= set(available_databases())
    with pytest.raises(FileNotFoundError):
        PhreeqcDatabase.load("nope")


def test_equation_parser():
    assert parse_phreeqc_equation("Ca+2 + 2X- = CaX2") == ([(1.0, "Ca+2"), (2.0, "X-")], [(1.0, "CaX2")])
    lhs, rhs = parse_phreeqc_equation("CO3-2 + 2 H+ = CO2 + H2O")
    assert lhs == [(1.0, "CO3-2"), (2.0, "H+")] and rhs == [(1.0, "CO2"), (1.0, "H2O")]


def test_database_contents(db, pz):
    assert db.master_key("C(4)") == "C(+4)"
    cal = db.phases["Calcite"]
    assert close(cal.logK, -8.48, abs_=1e-9) and cal.nu["Ca+2"] == 1 and cal.nu["Calcite"] == -1
    assert db.phases["CO2(g)"].gas
    assert db.species["CaHCO3+"].a0 is not None
    assert "X" in db.exchange_masters and "Hfo_w" in db.surface_masters
    assert db.pitzer is None and pz.pitzer is not None
    s = db.system(["Ca", "C(4)"], phases=["Calcite"])
    assert "CaCO3" in s and "Calcite" in s
    with pytest.raises(ValueError):
        db.system(["Ca"], phases=["Gypsum"])  # needs S(6)


# --------------------------------------------------------------------------- activity models
def test_truesdell_jones_matches_phreeqc(db):
    ref = ACTIVITY["truesdell_jones"]
    s = db.system(list(ref["amounts"]))
    for case in ref["cases"]:
        r = s.equilibrate(ref["amounts"], pH=7, water_mass=1.0, T=case["T"])
        assert r.converged
        for name, lg in case["lg"].items():
            assert close(math.log10(r.gamma[name]), lg, abs_=2e-5), name
        if "mu" in case:
            assert close(r.ionic_strength, case["mu"], rel=1e-4)
            assert close(10 ** r.log10_activity["H2O"], case["a_w"], abs_=2e-6)


def test_pitzer_matches_phreeqc(pz):
    ref = ACTIVITY["pitzer"]
    s = pz.system(list(ref["amounts_per_m"]) + list(ref["fixed_amounts"]))
    for case in ref["cases"]:
        amounts = {k: v * case["m"] for k, v in ref["amounts_per_m"].items()} | ref["fixed_amounts"]
        r = s.equilibrate(amounts, pH=7, water_mass=1.0, T=case["T"])
        assert r.converged
        for name, la in case["la"].items():
            assert close(r.log10_activity[name], la, abs_=1e-4), (case["T"], case["m"], name)


# --------------------------------------------------------------------------- features
def test_charge_balance(db):
    rf = FEATURES["charge_balance"]
    s = db.system(["Na", "Ca", "Cl"])
    r = s.equilibrate({"Na": 0.01, "Ca": 0.005, "Cl": 0.02}, pH=7, water_mass=1.0, charge_balance="Cl-")
    assert r.converged
    assert close(0.02 + r.charge_balance_adjustment, rf["totCl"], rel=1e-8)
    for n in ("Na+", "Ca+2", "Cl-"):
        assert close(r.log10_activity[n], rf["la_" + n], abs_=1e-5)


def test_exchange(db):
    rf = FEATURES["exchange"]
    s = db.system(["Na", "Ca", "Cl"], exchange=["X"])
    amt = {"Na": 0.01 + rf["m_NaX"], "Ca": 0.005 + rf["m_CaX2"], "Cl": rf["totCl"], "X": 0.01}
    r = s.equilibrate(amt, pH=7, water_mass=1.0)
    assert r.converged
    assert close(r.exchange["NaX"], rf["m_NaX"], rel=1e-8)
    assert close(r.exchange["CaX2"], rf["m_CaX2"], rel=1e-8)


def test_surface_ddl(db):
    rf = FEATURES["surface_ddl"]
    s = db.system(["Na", "Cl", "Zn"], surfaces={"Hfo": {"area": 600, "mass": 0.09}})
    amt = {"Na": 0.1, "Cl": rf["totCl"], "Zn": 1e-6 + rf["m_Hfo_sOZn+"] + rf["m_Hfo_wOZn+"],
           "Hfo_w": 2e-4, "Hfo_s": 5e-6}
    r = s.equilibrate(amt, pH=5, water_mass=1.0)
    assert r.converged
    assert close(r.potential["Hfo"], rf["psi"], rel=1e-4)
    for n in ("Hfo_wOH2+", "Hfo_wO-", "Hfo_sOZn+"):
        assert close(r.surface[n], rf["m_" + n], rel=5e-4), n


def test_calcite_open_to_co2(db):
    rf = FEATURES["calcite_co2"]
    s = db.system(["Ca", "C(4)"], phases=["Calcite", "CO2(g)"])
    r = s.equilibrate({"Calcite": 10.0}, fixed={"CO2(g)": -3.5})
    assert r.converged and "Calcite" in r.present
    assert close(r.pH, rf["pH"], abs_=1e-5)
    assert close(r.ionic_strength, rf["mu"], rel=1e-5)
    assert close(r.water_mass, rf["water"], rel=1e-7)


def test_calcite_closed_in_pure_water(db):
    # only a solid is given: the proton balance has zero total and zero input
    s = db.system(["Ca", "C(4)"], phases=["Calcite"])
    r = s.equilibrate({"Calcite": 1e-2}, water_mass=1.0)
    assert r.converged and r.present == ["Calcite"]
    assert 9.5 < r.pH < 10.5


def test_redox(db):
    rf = FEATURES["redox_fe"]
    s = db.system(["Fe", "Cl"])
    r = s.equilibrate({"Fe": 1e-3, "Cl": 3e-3}, pH=3, pe=10, water_mass=1.0)
    assert r.converged
    for n in ("Fe+2", "Fe+3", "FeCl+2", "FeOH+2"):
        assert close(r.log10_activity[n], rf["la_" + n], abs_=1e-5), n


def test_fixed_volume_gas(db):
    rf = FEATURES["gas_volume"]
    r0 = db.system(["Na", "C(4)"]).equilibrate({"Na": 0.01, "C(4)": 0.01}, pH=7, water_mass=1.0)
    amt = {k: v * r0.water_mass for k, v in r0.molality.items() if k != "H2O"}
    s = db.system(["Na", "C(4)"], gas_phase={"members": ["CO2(g)"], "volume": 1.0})
    r = s.equilibrate(amt)
    assert r.converged
    # PHREEQC uses a slightly different gas treatment; agreement is ~1e-4 relative
    assert close(r.pH, rf["pH"], abs_=1e-3)
    assert close(r.gases["CO2(g)"], rf["n_co2g"], rel=1e-3)


# --------------------------------------------------------------------------- llnl.dat and sit.dat
LLNL_SIT = json.loads((DATA / "phreeqc_llnl_sit.json").read_text())


@pytest.fixture(scope="module")
def thdbs():
    return {name: PhreeqcDatabase.load(name) for name in ("llnl.dat", "sit.dat")}


def test_llnl_sit_parsing(thdbs):
    from discopt.speciation import LLNL, SIT

    ll, si = thdbs["llnl.dat"], thdbs["sit.dat"]
    assert isinstance(ll.default_activity(), LLNL) and isinstance(si.default_activity(), SIT)
    assert len(ll.llnl["temperatures"]) == 8 and len(ll.llnl["co2_coefs"]) == 5
    assert len(si.sit) == 606
    # reactions written with a leading minus sign ("- 4 H+ ... = O2") are read
    assert "O2" in si.species
    # -add_logk Log_K_O2 0.5 adds half the named expression's log_k
    assert close(ll.species["S2O6-2"].logK, -1.16865, abs_=1e-9)
    assert ll.species["CO2"].options.get("gamma") == "co2" and ll.species["HClO"].options.get("gamma") == "llnl"
    act = ll.default_activity()
    A, B, bdot = act.interpolate(298.15)
    assert close(A, 0.5114, abs_=1e-4) and close(bdot, 0.0410, abs_=1e-4)
    with pytest.raises(ValueError):
        act.interpolate(700.0)


@pytest.mark.parametrize("case", LLNL_SIT["cases"], ids=lambda c: f"{c['database']}-{c['T_C']:g}C-{c['Na']:g}m")
def test_llnl_sit_match_phreeqc(thdbs, case):
    s = thdbs[case["database"]].system(["Na", "Cl", "Th", "C(4)"], T=273.15 + case["T_C"])
    na = case["Na"]
    r = s.equilibrate({"Na": na, "Cl": na, "Th": 1e-6, "C(4)": 1e-3}, pH=3, charge_balance="Cl-")
    assert r.converged
    assert close(r.ionic_strength, case["mu"], rel=1e-4)
    assert close(r.log10_activity["H2O"], case["la_H2O"], abs_=5e-6)
    assert close(0.0 if na == 0 else na + r.charge_balance_adjustment, case["totals"]["Cl"], rel=1e-4)
    for name, la in case["la"].items():
        # PHREEQC's input has pe 4, which sets O2(aq); here O(0) is balanced instead, so the
        # O2-dependent chlorine species (all < 1e-24 m) differ. Compare everything above 1e-20 m.
        if case["molality"][name] > 1e-20:
            assert close(r.log10_activity[name], la, abs_=5e-4), name

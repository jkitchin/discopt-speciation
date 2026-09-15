import pydantic
import pytest

from discopt.speciation import System, from_yaml, load_spec, solve_spec

from conftest import CE2_AMOUNTS, ce2_system

GYPSUM = """
name: gypsum
activity: davies
species:
  - {name: Ca+2}
  - {name: SO4-2}
  - {name: CaSO4}
  - {name: Gypsum, phase: solid}
reactions:
  - {equation: "Ca+2 + SO4-2 = CaSO4", logK: 2.36}
  - {equation: "Gypsum = Ca+2 + SO4-2", logK: -4.61}
problem:
  amounts: {Ca+2: 0.0324, SO4-2: 0.025}
"""


def test_yaml_roundtrip_and_solve():
    s = from_yaml(GYPSUM)
    assert s["SO4-2"].charge == -2
    out = solve_spec(GYPSUM)
    assert out["converged"]
    assert out["present"] == ["Gypsum"]
    ideal = solve_spec(GYPSUM, activity="ideal", method="gibbs")
    assert ideal["solids"]["Gypsum"] > out["solids"]["Gypsum"]  # activity corrections dissolve gypsum


def test_to_spec_roundtrip():
    s = ce2_system()
    s2 = System.from_spec(s.to_spec())
    a = s.equilibrate(CE2_AMOUNTS)
    b = s2.equilibrate(CE2_AMOUNTS)
    assert b.solids == pytest.approx(a.solids)


def test_strict_schema():
    with pytest.raises(pydantic.ValidationError):
        load_spec({"species": [{"name": "A", "phse": "solid"}], "reactions": []})
    with pytest.raises(pydantic.ValidationError):
        load_spec({"species": [{"name": "A"}], "reactions": [{"logK": 1.0}]})
    with pytest.raises(pydantic.ValidationError):
        load_spec({"species": [{"name": "A"}, {"name": "A"}], "reactions": []})


MINI_DB = """
SOLUTION_MASTER_SPECIES
H       H+      -1.0    H       1.008
H(1)    H+      -1.0    0
E       e-      0       0.0     0
O       H2O     0       O       16.0
O(-2)   H2O     0       0
Ca      Ca+2    0       Ca      40.08
S       SO4-2   0       SO4     32.064
S(6)    SO4-2   0       SO4
SOLUTION_SPECIES
H+ = H+
    log_k 0
e- = e-
    log_k 0
H2O = H2O
    log_k 0
Ca+2 = Ca+2
    log_k 0
    -gamma 5.0 0.165
SO4-2 = SO4-2
    log_k 0
    -gamma 5.0 -0.04
H2O = OH- + H+
    log_k -14.0
    delta_h 13.362 kcal
    -gamma 3.5 0
Ca+2 + SO4-2 = CaSO4
    log_k 2.25
    delta_h 1.325 kcal
PHASES
Gypsum
    CaSO4:2H2O = Ca+2 + SO4-2 + 2 H2O
    log_k -4.58
    delta_h -0.109 kcal
END
"""


def test_custom_phreeqc_text_database():
    from discopt.speciation import PhreeqcDatabase

    db = PhreeqcDatabase(MINI_DB)
    s = db.system(["Ca", "S(6)"], phases=["Gypsum"])
    r = s.equilibrate({"Gypsum": 1.0})
    assert r.converged and r.present == ["Gypsum"]
    assert 0.01 < r.molality["Ca+2"] < 0.02
    assert s["Ca+2"].params["a0"] == 5.0


def test_database_spec_with_overrides_and_additions():
    spec = {
        "name": "gyp",
        "database": {"file": "phreeqc.dat", "components": ["Ca", "S(6)"], "phases": ["Gypsum"]},
        "overrides": {"Gypsum": {"logK": -4.0}},
        "species": [{"name": "Foo", "phase": "solid"}],
        "reactions": [{"equation": "Foo = Ca+2 + SO4-2", "logK": -2.0, "name": "Foo"}],  # more soluble: stays absent
        "problem": {"amounts": {"Gypsum": 1.0}, "charge_balance": None},
    }
    out = solve_spec(spec)
    base = solve_spec({**spec, "overrides": {}})
    assert out["converged"] and base["converged"] and out["present"] == ["Gypsum"]
    assert out["molality"]["Ca+2"] > base["molality"]["Ca+2"]  # larger Ksp dissolves more
    s = from_yaml(spec)
    s2 = System.from_spec(s.to_spec())  # database systems round-trip as explicit data
    a, b = s.equilibrate({"Gypsum": 1.0}), s2.equilibrate({"Gypsum": 1.0})
    assert b.molality["Ca+2"] == pytest.approx(a.molality["Ca+2"], rel=1e-8)
    with pytest.raises(ValueError):
        from_yaml({**spec, "overrides": {"nope": {"logK": 1.0}}})


def test_spec_phases_and_parameters():
    spec = {
        "activity": {"model": "sit", "eps": [["Na+", "Cl-", 0.03]]},
        "species": [{"name": "Na+"}, {"name": "Cl-"}, {"name": "NaCl"}, {"name": "CO2"}, {"name": "CO2(g)", "phase": "gas"}],
        "reactions": [{"equation": "Na+ + Cl- = NaCl", "logK": -0.5, "dH": 2000.0},
                      {"equation": "CO2(g) = CO2", "analytic": [-1.47]}],
        "gas_phases": [{"members": ["CO2(g)"], "pressure": 1.0}],
        "problem": {"amounts": {"Na+": 0.5, "Cl-": 0.5, "CO2(g)": 1.0}, "temperature": 310.0},
    }
    out = solve_spec(spec)
    assert out["converged"]
    assert out["partial_pressure"]["CO2(g)"] == pytest.approx(1.0, rel=1e-6)
    s = from_yaml(spec)
    assert System.from_spec(s.to_spec()).reactions[1].analytic == (-1.47,)
    with pytest.raises(pydantic.ValidationError):
        load_spec({"species": [{"name": "Gyp", "phase": "solid", "site": "X"}]})
    with pytest.raises(pydantic.ValidationError):
        load_spec({"species": [{"name": "A"}], "reactions": [{"equation": "A = A"}]})


def test_gibbs_needs_excess_gibbs_energy():
    with pytest.raises(pydantic.ValidationError):
        solve_spec(GYPSUM, method="newton")
    # Truesdell-Jones (species-specific ion sizes) has no G^ex
    with pytest.raises(NotImplementedError):
        solve_spec(GYPSUM, method="gibbs", activity="truesdell-jones")
    # Davies has one, and Gibbs minimization agrees with complementarity
    a = solve_spec(GYPSUM, method="gibbs")
    b = solve_spec(GYPSUM)
    assert a["converged"] and a["present"] == b["present"]
    assert a["solids"]["Gypsum"] == pytest.approx(b["solids"]["Gypsum"], rel=1e-8)

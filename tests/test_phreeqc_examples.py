"""PHREEQC's example problems and adversarial stress cases, compared with PHREEQC 3.

Reference data: ``tests/data/phreeqc_examples.json``, written by
``tests/data/generate_phreeqc_examples.py`` (PHREEQC 3.8.7 through phreeqpython,
same database files). Each case holds the exact PHREEQC input, every selected-
output row, and ``steps`` that pose the same problem to this package: amounts
are PHREEQC's species moles (so element, charge and redox totals are identical)
plus any reactants, per kg of the water those moles came with (``W0``).

Every step is one test. Steps where the comparison exposes a package limitation
or bug are ``xfail(strict=True)`` with the diagnosis in the reason (see
``docs/validation.md``); a strict xfail that starts passing means the limitation
was fixed and the mark should go.
"""

import functools
import json
import math
import pathlib

import pytest

from discopt.speciation import PhreeqcDatabase
from discopt.speciation.database import DATA_DIR

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "phreeqc_examples.json").read_text())
CASES = {c["id"]: c for c in REF["cases"]}

# ---------------------------------------------------------------------------------------------------------------
# Tolerances (absolute unless *_rel). Both programs solve the same mass-action/mass-balance equations with the same
# logK and activity-model parameters, so agreement is limited only by convergence tolerances and constants.
TOL = {
    "pH": 1e-4,
    "pe": 1e-4,
    "la_H2O": 1e-4,
    "mu_rel": 1e-4,
    "water_rel": 1e-4,
    # log activities and saturation indices: 5e-4. Trace redox species (e.g. N2 at 1e-8 m in ex7, U(4) in ex1) carry
    # ~8 x (pe error) in their log activity, and PHREEQC's own pe convergence is ~1e-5..5e-5; the worst converged
    # comparison seen is 2e-4 (ex7_ideal_gas N2(aq)).
    "la": 5e-4,
    "SI": 5e-4,
    "psi_rel": 1e-4,
    # exchange/surface/solid-solution moles, relative. Surface amounts are exponentially sensitive to the potential
    # (Hfo_wOZn+ in ex8 moves 2.3 x the psi error), worst converged case 1.3e-4.
    "ex_rel": 5e-4,
    "surf_rel": 5e-4,
    "ss_rel": 5e-4,
}
# Pure-phase and gas-phase moles: |dn| <= 1e-6 + 1e-4 n mol. A phase amount is the small difference between the
# element total and what stays dissolved, so the allowed pH/pe error (1e-4) moves it by ~ln(10) * 1e-4 * (dissolved
# total): e.g. ex7_ideal_gas at 4 mmol, 5e-4 mol of gas beside ~4 mmol of carbon, pH off 4.6e-5 -> CO2(g) off
# 2.5e-8 mol (7e-4 relative) with the gas mole fractions agreeing to 1e-5.
PHASE_ABS, PHASE_REL = 1e-6, 1e-4
AMOUNT_FLOOR = 1e-12  # mol; PHREEQC reports ~1e-26 mol for gas components when no gas phase forms
LA_FLOOR = 1e-30  # mol/kgw; species below this are numerical noise in both programs

# pe is not defined in these reference solutions: no redox couple is present (O(0) = 0) and PHREEQC's pe comes from
# its H(0) ~ 1e-14 m, which the package system does not contain.
NO_PE = {"stress_T_llnl_0.01C", "stress_T_sit_0.01C", "stress_T_sit_100C"}

# ---------------------------------------------------------------------------------------------------------------
# Known package limitations / bugs, diagnosed in docs/validation.md. Keys: (case id, step label or "*").
LOG_FLOOR = ("log-molality lower bound LOG_LB=-200 (solve.py) pins a redox species of a vanishing valence state "
             "(e.g. Fe(HS)3- in oxic water, Mn(NO3)2 at pe -7), mass balance is violated and the solve reports "
             "converged=False; repro: phreeqc.dat Na/Fe/S/O, Na+ 0.02, SO4-2 0.01, O2 2e-4, Fe(OH)3 1e-8, pH 7, pe 8")
WATER_GFW = ("PHREEQC takes the water gfw from the database (phreeqc.dat/pitzer.dat: H 1.008, O 16.00 -> 18.016 g/mol) "
             "while the package uses 18.01528, so removing N mol of water leaves 7.6e-4 relative more water; the "
             "-mass-matched step with the same water mass agrees")
PENG_ROBINSON = ("phreeqc.dat gives CO2(g)/CH4(g)/N2(g)/H2O(g) critical constants, so PHREEQC uses Peng-Robinson "
                 "fugacity coefficients for a gas phase; the package gas phase is ideal (the *_ideal_gas variant, same "
                 "gases without -T_c/-P_c/-Omega, agrees)")
LOGK_PRESSURE = ("the package has no pressure dependence of logK (PHREEQC's Vm / -analytical pressure terms); above "
                 "100 C PHREEQC runs at P = Psat (15.3 atm at 200 C). Repro: Na/Cl 0.5, Ca 2e-3, C(4) 4e-3, S(6) 2e-3 "
                 "m at 25 C, 200 atm: pH 4.3144 vs PHREEQC 4.2633 (identical at 1 atm)")
EX6_NONCONV = ("equilibrate with 4 aluminosilicate phases fails to converge (converged=False, complementarity residual "
               "~1) for small K-feldspar additions; repro: phreeqc.dat K/Al/Si, phases Gibbsite, Kaolinite, K-mica, "
               "K-feldspar, pure water + 2.658e-8 mol KAlSi3O8 -> pH 6.68 vs PHREEQC 7.0037 (Gibbsite at SI 0); "
               "converges at 1.6e-7 and 1e-5 mol")
TRACE_NAN = ("with 1e-30 m Zn/Pb/Cd at fixed pH the package returns NaN molalities and pH=None yet converged=True with "
             "zero residuals (silent failure); PHREEQC gives Zn+2 7.96e-31 m")
TRACE_DROP = ("re-equilibrating the 1e-30 m trace solution: converged=False with mass-balance residual 1.0 (Zn+2 "
              "3.6e-95 m, the trace totals are lost below the log-molality floor)")
SURFACE_START = ("dense surface (Hfo_w 0.02, Hfo_s 5e-4 mol) at I = 1e-4: equilibrate fails (converged=False, Hfo_sOH "
                 "5e-198) as soon as H+ is among the input amounts; repro: phreeqc.dat Na/Cl/Zn, Hfo area 600 mass 9, "
                 "{Na+:1e-4, Cl-:1e-4, Zn+2:1e-6, Hfo_w:0.02, Hfo_s:5e-4} converges (pH 8.0678, PHREEQC 8.0679); "
                 "adding H+: 1e-7 does not")
MACINNES = ("the package applies PHREEQC's MacInnes scaling of individual-ion activity coefficients only when Cl- "
            "is present; PHREEQC applies it always, so in Cl--free Pitzer solutions log gamma differs by z*D "
            "(2.5 m MgSO4: Mg+2 -0.895, SO4-2 +0.895, pH off 0.45); electroneutral combinations and a_w agree")
PITZER_BRINE = ("Pitzer water activity of a NON-electroneutral solution differs from PHREEQC: the Black Sea analysis "
                "carries a 2.36 meq charge imbalance, concentrated to 0.13-0.27 eq/kgw at 54-55 mol water removed "
                "(I 8-12), giving log a_w off 6e-4..3.8e-3, la Ca+2 up to 9e-3, SI Epsomite up to 0.026. The same "
                "brine at fixed pH agrees to 1.6e-5 once Na is adjusted to electroneutrality, and off by 1.1e-3 "
                "without (likely cause: the package takes a_w from d(Gex)/dn_w, PHREEQC from its osmotic-coefficient "
                "sum; the two coincide only when sum z m = 0)")

XFAIL = {
    ("ex1", "reequilibrated-redox"): LOG_FLOOR,
    ("stress_pe_O2_limit", "*"): LOG_FLOOR,
    ("stress_pe_H2_limit", "*"): LOG_FLOOR,
    ("ex4", "4a"): WATER_GFW,
    ("stress_T_phreeqc_200C", "*"): LOGK_PRESSURE,
    ("stress_T_pitzer_200C", "*"): LOGK_PRESSURE,
    ("ex10", "solution1"): PENG_ROBINSON,
    ("stress_trace_1e-30", "fixed-pH"): TRACE_NAN,
    ("stress_trace_1e-30", "reequilibrated"): TRACE_DROP,
    ("stress_surface_dense_lowI", "*"): SURFACE_START,
    ("stress_pitzer_no_chloride_MgSO4", "*"): MACINNES,
    ("stress_pitzer_no_chloride_Na2SO4", "*"): MACINNES,
}
for _lab in ("A1", "A5", "B-0.04umol", "B-0.08umol", "B-1umol", "B-4umol"):
    XFAIL[("ex6", _lab)] = EX6_NONCONV
for _mm in (4, 8, 16, 32, 64, 125, 250, 500, 1000):
    XFAIL[("ex7", f"fixedP-{_mm}mmol")] = PENG_ROBINSON
for _mm in (1, 2, 3, 4, 8, 16, 32, 64, 125, 250, 500, 1000):
    XFAIL[("ex7", f"fixedV-{_mm}mmol")] = PENG_ROBINSON
# at 36 mol removed the gfw difference is 0.026 g in 0.35 kg (7e-5 relative), inside the 1e-4 tolerance
for _x in ("48", "54", "54.5", "54.876"):
    XFAIL[("stress_pitzer_evaporite", f"H2O-{_x}")] = WATER_GFW
XFAIL[("stress_pitzer_evaporite", "H2O-55.026")] = (
    "equilibrate does not converge (converged=False after ~600 s, pH off 4.3) for Black Sea water with 55.026 mol "
    "H2O removed (0.0086 kg left, I ~ 11.6, halite + polyhalite + glauberite + anhydrite); the same problem with "
    "the equal-mass water amount converges in ~330 s")
for _x in ("54", "54.5", "54.876", "55.026"):
    XFAIL[("stress_pitzer_evaporite", f"H2O-{_x}-mass-matched")] = PITZER_BRINE

# Steps slower than ~1 s (Pitzer temperature extremes 15-17 s, ex1 redox 16 s, the evaporite 4 s per step and 230 s
# at 54.876, ...) run only with the slow set.
SLOW_CASES = {"stress_pitzer_evaporite", "stress_T_pitzer_0.01C", "stress_T_pitzer_200C", "stress_T_sit_100C",
              "stress_pe_O2_limit", "stress_pe_H2_limit", "ex5"}
SLOW_STEPS = {("ex1", "reequilibrated-redox"), ("stress_trace_1e-30", "reequilibrated"),
              ("ex6", "A1"), ("ex6", "B-0.04umol"), ("ex6", "B-0.08umol"), ("ex6", "B-0.16umol")}


@functools.lru_cache(maxsize=None)
def database(name, additions=""):
    if not additions:
        return PhreeqcDatabase.load(name)
    text = (pathlib.Path(DATA_DIR) / name).read_bytes().decode("latin-1")
    return PhreeqcDatabase(text + "\n" + additions)


def run_step(case, step):
    """Pose one step to the package; return (result, reference row, W0)."""
    db = database(case["database"], case["db_additions"])
    rows = case["rows"]
    ref = rows[step["row"]]
    sysargs = dict(step["system"])
    s = db.system(sysargs.pop("components"), **sysargs)
    W0 = step["W0"]
    if step["totals"] is not None:
        amounts = dict(step["totals"])
    else:
        amounts = {}
        for i, f, groups in step["base"]:
            r = rows[i]
            for group in groups:
                for name, n in r[group].items():
                    if name in s and s[name].phase not in ("solvent", "electron"):
                        amounts[name] = amounts.get(name, 0.0) + f * n
        for k, v in step["add"].items():
            amounts[k] = amounts.get(k, 0.0) + v
        amounts = {k: v / W0 for k, v in amounts.items()}
    call = dict(step["call"])
    r = s.equilibrate(amounts, T=273.15 + ref["tc"], P=ref.get("P") or 1.0, **call)
    return r, ref, W0


def _worst(pairs):
    worst = (0.0, None)
    for name, dev in pairs:
        if not dev <= worst[0]:  # NaN counts as worst
            worst = (dev, name)
    return worst


def _finite(x):
    return x is not None and math.isfinite(x)


def deviations(case, step, r, ref, W0):
    """Deviation of every compared quantity (package vs PHREEQC); scalars or (worst value, species)."""
    d = {}
    W = r.water_mass * W0
    d["pH"] = abs(r.pH - ref["pH"]) if _finite(r.pH) else math.inf
    if case["id"] not in NO_PE and ref.get("pe") is not None and r.pe is not None:
        d["pe"] = abs(r.pe - ref["pe"])
    d["mu_rel"] = abs(r.ionic_strength / ref["mu"] - 1.0)
    d["water_rel"] = abs(W / ref["water"] - 1.0)
    if "H2O" in r.log10_activity:
        d["la_H2O"] = abs(r.log10_activity["H2O"] - ref["la_H2O"])
    d["la"] = _worst((name, abs(r.log10_activity[name] - la)) for name, la in ref["la"].items()
                     if name in r.log10_activity and ref["n"][name] / ref["water"] > LA_FLOOR)
    skip = set(step.get("skip_phases", ()))
    d["phase_mol"] = _worst((name, abs(r.solids[name] * W0 - n) / (PHASE_ABS + PHASE_REL * abs(n)))
                            for name, n in ref["equi"].items() if name in r.solids and name not in skip)
    d["SI"] = _worst((name, abs(r.SI[name] - si)) for name, si in ref["si"].items() if name in r.SI and si > -99)
    d["gas_mol"] = _worst((name, abs(r.gases[name] * W0 - n) / (PHASE_ABS + PHASE_REL * abs(n)))
                          for name, n in ref["gas"].items() if name in r.gases)
    for group, res in (("ex", r.exchange), ("surf", r.surface), ("ss", r.solids)):
        pairs = []
        for name, n in ref[group].items():
            if name not in res:
                continue
            ours = res[name] * W0
            if max(abs(ours), abs(n)) < AMOUNT_FLOOR:
                continue
            pairs.append((name, abs(ours - n) / max(abs(n), AMOUNT_FLOOR)))
        d[group + "_rel"] = _worst(pairs)
    for name, psi in ref["psi"].items():
        if name in r.potential:
            d["psi_rel"] = abs(r.potential[name] / psi - 1.0) if psi else abs(r.potential[name])
    return d


def _params():
    out = []
    for cid, case in CASES.items():
        for step in case["steps"]:
            marks = []
            reason = XFAIL.get((cid, step["label"])) or XFAIL.get((cid, "*"))
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            if cid in SLOW_CASES or (cid, step["label"]) in SLOW_STEPS:
                marks.append(pytest.mark.slow)
            out.append(pytest.param(cid, step["label"], marks=marks, id=f"{cid}:{step['label']}"))
    return out


def test_xfail_keys_exist():
    labels = {(cid, s["label"]) for cid, c in CASES.items() for s in c["steps"]} | {(cid, "*") for cid in CASES}
    assert set(XFAIL) <= labels
    assert SLOW_STEPS <= labels and SLOW_CASES <= set(CASES)


@pytest.mark.parametrize("cid, label", _params())
def test_phreeqc_example(cid, label):
    case = CASES[cid]
    step = next(s for s in case["steps"] if s["label"] == label)
    r, ref, W0 = run_step(case, step)
    assert r.converged, f"not converged: residuals {r.residuals}"
    d = deviations(case, step, r, ref, W0)
    bad = {}
    for key, v in d.items():
        dev, name = v if isinstance(v, tuple) else (v, None)
        limit = 1.0 if key in ("phase_mol", "gas_mol") else TOL[key]  # already scaled by their tolerance
        if not dev <= limit:
            bad[key] = (dev, name)
    assert not bad, f"deviations beyond tolerance: {bad}"

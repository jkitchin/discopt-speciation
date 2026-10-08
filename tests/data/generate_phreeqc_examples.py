"""Generate ``phreeqc_examples.json``: PHREEQC 3 results for the manual's examples and stress cases.

Run with an interpreter that has ``phreeqpython`` (it bundles IPhreeqc), for example::

    uv venv /tmp/phreeqc-venv && uv pip install -p /tmp/phreeqc-venv phreeqpython
    /tmp/phreeqc-venv/bin/python tests/data/generate_phreeqc_examples.py

phreeqpython is *not* a dependency of this package. The database files are the
ones bundled in ``src/discopt/speciation/data``.

Every case stores the exact PHREEQC input that was run, the database, the
PHREEQC version, all selected-output rows (a generic dump of every aqueous,
exchange, surface, gas, equilibrium-phase and solid-solution amount), and a
``steps`` list that says how to pose the same problem to ``discopt.speciation``:

* ``row``: index of the PHREEQC row holding the equilibrium to compare against.
* ``base``: ``[[row, factor], ...]``; the package amounts are
  ``sum(factor * moles)`` of the species in those rows (aqueous, exchange,
  surface and gas species), which reproduces PHREEQC's element, charge and
  redox totals exactly.
* ``add``: extra amounts (mol) keyed by species / phase / site name, e.g. a
  reactant written as a combination of species.
* ``W0``: the kg of water those amounts refer to. The package works per kg of
  initial water, so the test divides every amount by ``W0`` and multiplies
  extensive results back by ``W0``.
* ``system``: arguments of ``PhreeqcDatabase.system``; ``call``: extra keyword
  arguments of ``System.equilibrate`` (``fixed``, ``pH``, ``pe``, ``water_mass``,
  ``charge_balance``); ``T_C`` and ``P`` are taken from the PHREEQC row.
* ``totals``: when present, the package amounts are these element totals
  instead of species moles (used with a fixed pH, as for an initial solution).

Examples are from Parkhurst & Appelo (2013), USGS TM 6-A43, as distributed in
https://github.com/usgs-coupled/phreeqc3/tree/master/examples. Inputs are
copied from those files with USER_GRAPH / file output removed; any other change
is described in the case's ``note``.
"""

import functools
import importlib.metadata
import json
import math
import pathlib

from phreeqpython import PhreeqPython

HERE = pathlib.Path(__file__).parent
DBDIR = HERE.parents[1] / "src" / "discopt" / "speciation" / "data"
OUT = HERE / "phreeqc_examples.json"

DUMP = r"""
SELECTED_OUTPUT 1
 -reset false
 -simulation true
 -state true
 -solution true
 -step true
 -high_precision true
USER_PUNCH 1
 10 PUNCH "pH", -LA("H+"), "pe", -LA("e-"), "mu", MU, "water", TOT("water"), "la_H2O", LA("H2O"), "tc", TC, "P", PRESSURE, "cb", CHARGE_BALANCE
 20 t = SYS("aq", count, name$, type$, moles)
 30 FOR i = 1 TO count
 40 PUNCH "n:" + name$(i), moles(i), "la:" + name$(i), LA(name$(i))
 50 NEXT i
 60 t = SYS("elements", count, name$, type$, moles)
 70 FOR i = 1 TO count
 80 PUNCH "tot:" + type$(i) + ":" + name$(i), moles(i)
 90 NEXT i
 100 t = SYS("ex", count, name$, type$, moles)
 110 FOR i = 1 TO count
 120 PUNCH "ex:" + name$(i), moles(i)
 130 NEXT i
 140 t = SYS("surf", count, name$, type$, moles)
 150 FOR i = 1 TO count
 160 PUNCH "surf:" + name$(i), moles(i)
 170 NEXT i
 180 t = SYS("gas", count, name$, type$, moles)
 190 FOR i = 1 TO count
 200 PUNCH "gas:" + name$(i), moles(i)
 210 NEXT i
 220 t = SYS("equi", count, name$, type$, moles)
 230 FOR i = 1 TO count
 240 PUNCH "equi:" + name$(i), moles(i)
 250 NEXT i
 260 t = SYS("ss", count, name$, type$, moles)
 270 FOR i = 1 TO count
 280 PUNCH "ss:" + name$(i), moles(i)
 290 NEXT i
 300 PUNCH "gas_p", GAS_P
"""

# KAlSi3O8 written as species: K+ + Al+3 + 3 H4SiO4 - 4 H2O - 4 H+ = K+ + Al+3 + 3 H4SiO4 + 4 OH- - 8 H2O
# (the package accepts a negative amount only for water)
KFSP = {"K+": 1.0, "Al+3": 1.0, "H4SiO4": 3.0, "OH-": 4.0, "H2O": -8.0}


def dump_block(si=(), gases=(), edl=()):
    lines, n = [], 310
    for p in si:
        lines.append(f' {n} PUNCH "si:{p}", SI("{p}")')
        n += 10
    for g in gases:
        lines.append(f' {n} PUNCH "pp:{g}", SR("{g}")')
        n += 10
    for s in edl:
        lines.append(f' {n} PUNCH "psi:{s}", EDL("psi", "{s}")')
        n += 10
    return DUMP + "\n".join(lines) + "\n"


def parse_row(header, row):
    sim, state, soln, step = row[:4]
    it = row[4:]
    d = {it[i]: it[i + 1] for i in range(0, len(it) - 1, 2) if isinstance(it[i], str) and it[i]}
    out = {"sim": int(sim), "state": state, "soln": int(soln), "step": int(step),
           "n": {}, "la": {}, "tot": {}, "ex": {}, "surf": {}, "gas": {}, "equi": {}, "ss": {}, "si": {}, "pp": {},
           "psi": {}}
    for k, v in d.items():
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            v = None
        if ":" not in k:
            out[k] = v
            continue
        kind, name = k.split(":", 1)
        if kind == "tot":
            typ, el = name.split(":", 1)
            out["tot"][el if typ == "dis" else f"{typ}:{el}"] = v
        elif kind in ("n", "la"):
            out[kind][name] = v
        else:
            out[kind][name] = v
    # keep log activities only for species that are present
    out["la"] = {k: v for k, v in out["la"].items() if out["n"].get(k, 0.0) > 0.0}
    out["n"] = {k: v for k, v in out["n"].items() if v and v > 0.0}
    return out


def run_phreeqc(database, text):
    pp = PhreeqPython(database_directory=DBDIR, database="phreeqc.dat")
    ip = pp.ip
    # load from text with CR stripped (wateq4f_PWN.dat has CRLF line ends, which IPhreeqc's file loader rejects)
    dbtext = (DBDIR / database).read_bytes().decode("latin-1").replace("\r", "")
    ip.load_database_string(dbtext)
    if ip.phc_database_error_count:
        raise RuntimeError(f"cannot load {database}: {ip.get_error_string()}")
    status = ip.run_string(text)
    arr = ip.get_selected_output_array()
    if status or len(arr) < 2:
        raise RuntimeError(f"PHREEQC failed ({status}): {getattr(ip, 'get_error_string', lambda: '')()}")
    return [parse_row(arr[0], r) for r in arr[1:]]


def find(rows, sim, state, step=None, soln=None, nth=0):
    hits = [i for i, r in enumerate(rows)
            if r["sim"] == sim and r["state"] == state and (step is None or r["step"] == step)
            and (soln is None or r["soln"] == soln)]
    if len(hits) <= nth:
        raise LookupError(f"no row sim={sim} state={state} step={step} soln={soln}: "
                          f"{[(r['sim'], r['state'], r['soln'], r['step']) for r in rows]}")
    return hits[nth]


CASES = []


def case(cid, database, body, *, title, compared, steps, db_additions="", si=(), gases=(), edl=(), group="example",
         note="", scope="full"):
    CASES.append(dict(id=cid, database=database, body=body, title=title, compared=compared, steps=steps,
                      db_additions=db_additions, si=si, gases=gases, edl=edl, group=group, note=note, scope=scope))


def step(row, *, base=(), add=None, system, call=None, W0=None, totals=None, label="", skip_phases=()):
    """``row``/``base`` entries are ``(sim, state, step, soln)`` selectors, resolved after the run;
    ``add``/``W0`` may be callables of (rows, resolve). ``skip_phases``: phases whose PHREEQC amount has
    no package counterpart (e.g. the unreacted stock of an alternative reactant); their SI is still compared."""
    return dict(row=row, base=list(base), add=add or {}, system=system, call=call or {}, W0=W0, totals=totals,
                label=label, skip_phases=list(skip_phases))


def variant(cid, new_id, *, prepend="", title_suffix="", note=""):
    """Copy an already defined case with extra definitions prepended to its input and database."""
    c = dict(next(c for c in CASES if c["id"] == cid))
    c.update(id=new_id, body=prepend + c["body"], db_additions=prepend + c["db_additions"],
             title=c["title"] + title_suffix, note=note, group="variant")
    CASES.append(c)


# PHREEQC's water gfw is computed from the database's H and O gfw (1.008 + 16.00 in phreeqc.dat and pitzer.dat);
# the package uses 18.01528 g/mol. Removing the same MOLES of water therefore leaves different water masses.
WATER_GFW_RATIO = (2 * 1.008 + 16.00) / 18.01528

# phreeqc.dat's gas phases without the critical constants (-T_c, -P_c, -Omega): PHREEQC then treats
# GAS_PHASE and gases in EQUILIBRIUM_PHASES as ideal, like the package.
IDEAL_GASES = """
PHASES
CO2(g)
        CO2 = CO2
        -log_k  -1.468
        -delta_h -4.776 kcal
        -analytic   10.5624  -2.3547e-2  -3972.8  0  5.8746e5  1.9194e-5
H2O(g)
        H2O = H2O
        -log_k  1.506; delta_h -44.03 kJ
        -analytic   -16.5066 -2.0013E-3  2710.7  3.7646  0 2.24E-6
N2(g)
        N2 = N2
        -log_k           -3.1864
        -analytic -58.453 1.818e-3  3199  17.909 -27460
CH4(g)
        CH4 = CH4
        -log_k     -2.8502
        -analytic   10.44  -7.65e-3  -6669  0  1.014e6
"""
IDEAL_NOTE = ("Same as the example, with CO2(g), H2O(g), N2(g) and CH4(g) redefined without critical constants "
              "(in both the PHREEQC input and the package database), so PHREEQC also uses ideal gases.")

# ============================================================================ examples
SEAWATER_EX1 = """
SOLUTION 1  SEAWATER FROM NORDSTROM AND OTHERS (1979)
        units   ppm
        pH      8.22
        pe      8.451
        density 1.023
        temp    25.0
        redox   O(0)/O(-2)
        Ca              412.3
        Mg              1291.8
        Na              10768.0
        K               399.1
        Fe              0.002
        Mn              0.0002  pe
        Si              4.28
        Cl              19353.0
        Alkalinity      141.682 as HCO3
        S(6)            2712.0
        N(5)            0.29    gfw   62.0
        N(-3)           0.03    as    NH4
        U               3.3     ppb   N(5)/N(-3)
        O(0)            1.0     O2(g) -0.7
"""
U_DATA = """
SOLUTION_MASTER_SPECIES
        U       U+4     0.0     238.0290     238.0290
        U(4)    U+4     0.0     238.0290
        U(5)    UO2+    0.0     238.0290
        U(6)    UO2+2   0.0     238.0290
SOLUTION_SPECIES
        U+4 = U+4
                log_k          0.0
        U+4 + 4 H2O = U(OH)4 + 4 H+
                log_k          -8.538
                delta_h        24.760 kcal
        U+4 + 5 H2O = U(OH)5- + 5 H+
                log_k          -13.147
                delta_h        27.580 kcal
        U+4 + 2 H2O = UO2+ + 4 H+ + e-
                log_k          -6.432
                delta_h        31.130 kcal
        U+4 + 2 H2O = UO2+2 + 4 H+ + 2 e-
                log_k          -9.217
                delta_h        34.430 kcal
        UO2+2 + H2O = UO2OH+ + H+
                log_k          -5.782
                delta_h        11.015 kcal
        2UO2+2 + 2H2O = (UO2)2(OH)2+2 + 2H+
                log_k          -5.626
                delta_h        -36.04 kcal
        3UO2+2 + 5H2O = (UO2)3(OH)5+ + 5H+
                log_k          -15.641
                delta_h        -44.27 kcal
        UO2+2 + CO3-2 = UO2CO3
                log_k          10.064
                delta_h        0.84 kcal
        UO2+2 + 2CO3-2 = UO2(CO3)2-2
                log_k          16.977
                delta_h        3.48 kcal
        UO2+2 + 3CO3-2 = UO2(CO3)3-4
                log_k          21.397
                delta_h        -8.78 kcal
PHASES
        Uraninite
        UO2 + 4 H+ = U+4 + 2 H2O
        log_k          -3.490
        delta_h        -18.630 kcal
"""
MAJOR = ["Ca", "Mg", "Na", "K", "Si", "Cl", "C(4)", "S(6)"]
case("ex1", "phreeqc.dat", U_DATA + SEAWATER_EX1 + "END\nMIX 1\n 1 1.0\nEND\n", db_additions=U_DATA,
     title="Example 1: speciate seawater (with uranium)",
     si=["Calcite", "Aragonite", "Dolomite", "Gypsum", "Uraninite", "Goethite", "Pyrolusite", "Quartz"],
     compared="(a) initial speciation of the non-redox elements at pH 8.22; (b) the same seawater "
              "re-equilibrated (MIX 1 1.0) with every element and a single pe: pH, pe, I, a_w, log "
              "activities, valence totals, SI",
     note="PHREEQC distributes Fe, Mn and U in the initial solution with separate redox couples "
          "(O(0)/O(-2), pe, N(5)/N(-3)); the package has one pe per solution, so (a) compares only "
          "non-redox elements and (b) compares the re-equilibrated state, where PHREEQC also uses one pe.",
     scope="partial",
     steps=[step((1, "i_soln", None, 1), totals=MAJOR, system=dict(components=MAJOR),
                 call=dict(pH=8.22, water_mass=1.0), label="initial-major"),
            step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                 system=dict(components=["Ca", "Mg", "Na", "K", "Fe", "Mn", "Si", "Cl", "C", "S", "N", "U", "O"]),
                 label="reequilibrated-redox")])

case("ex2", "phreeqc.dat", """
SOLUTION 1 Pure water
        pH      7.0
        temp    25.0
EQUILIBRIUM_PHASES 1
        Gypsum          0.0     1.0
        Anhydrite       0.0     1.0
REACTION_TEMPERATURE 1
        25.0 75.0 in 51 steps
END
""", title="Example 2: gypsum/anhydrite solubility, 25-75 C", si=["Gypsum", "Anhydrite"],
     compared="SI, phase amounts, Ca molality, I, water mass, pH at each of 51 temperatures",
     steps=[step((1, "react", k, None), base=[((1, "i_soln", None, 1), 1.0)], add={"Gypsum": 1.0, "Anhydrite": 1.0},
                 system=dict(components=["Ca", "S(6)"], phases=["Gypsum", "Anhydrite"]), label=f"step{k}")
            for k in range(1, 52)])

SEAWATER_EX3 = """
SOLUTION 2  Seawater
        units   ppm
        pH      8.22
        pe      8.451
        density 1.023
        temp    25.0
        Ca              412.3
        Mg              1291.8
        Na              10768.0
        K               399.1
        Si              4.28
        Cl              19353.0
        Alkalinity      141.682 as HCO3
        S(6)            2712.0
"""
case("ex3", "phreeqc.dat", """
SOLUTION 1  Pure water
        pH      7.0
        temp    25.0
EQUILIBRIUM_PHASES
        CO2(g)          -2.0
        Calcite         0.0
SAVE solution 1
END
""" + SEAWATER_EX3 + """
END
MIX 1
        1      0.7
        2      0.3
SAVE solution   3
END
EQUILIBRIUM_PHASES 1
        Calcite         0.0
        Dolomite        0.0
USE solution 3
END
EQUILIBRIUM_PHASES 2
        Calcite         0.0
USE solution 3
END
""", title="Example 3: mixing groundwater and seawater", si=["Calcite", "Dolomite", "Aragonite", "Gypsum"],
     compared="A: calcite + CO2(g); B: seawater speciation; C: 70/30 mixture; D: + calcite and dolomite; "
              "E: + calcite. pH, I, log activities, SI, phase amounts",
     note="PHREEQC's CO2(g) at -2.0 is a 10 mol reservoir; the package holds the pressure fixed.",
     steps=[step((1, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"Calcite": 10.0},
                 system=dict(components=["Ca", "C(4)"], phases=["Calcite", "CO2(g)"]),
                 call=dict(fixed={"CO2(g)": -2.0}), label="A"),
            step((2, "i_soln", None, 2), totals=MAJOR, system=dict(components=MAJOR),
                 call=dict(pH=8.22, water_mass=1.0), label="B"),
            step((3, "react", 1, None), base=[((1, "react", 1, None), 0.7), ((2, "i_soln", None, 2), 0.3)],
                 system=dict(components=MAJOR), label="C"),
            step((4, "react", 1, None), base=[((3, "react", 1, None), 1.0)], add={"Calcite": 10.0, "Dolomite": 10.0},
                 system=dict(components=MAJOR, phases=["Calcite", "Dolomite"]), label="D"),
            step((5, "react", 1, None), base=[((3, "react", 1, None), 1.0)], add={"Calcite": 10.0},
                 system=dict(components=MAJOR, phases=["Calcite"]), label="E")])

case("ex4", "phreeqc.dat", """
SOLUTION 1  Precipitation from Central Oklahoma
        units           mg/L
        pH              4.5   # estimated
        temp            25.0
        Ca              .384
        Mg              .043
        Na              .141
        K               .036
        Cl              .236
        C(4)            .1      CO2(g)  -3.5
        S(6)            1.3
        N(-3)           .208
        N(5)            .237
REACTION 1
        H2O     -1.0
        52.73 moles
SAVE solution 2
END
""", title="Example 4a: rainwater evaporation", si=["Gypsum", "Calcite", "CO2(g)"],
     compared="evaporation of 52.73 mol H2O: water mass, pH, pe, I, log activities, N valence totals",
     note="Example 4b (MIX with factor 20) only rescales the water mass and is not compared separately. "
          "Step 4a-mass-matched removes 52.73 * 18.016/18.01528 mol, the same water MASS as PHREEQC, whose "
          "water gfw comes from the database (H 1.008 + O 16.00 = 18.016; the package uses 18.01528).",
     scope="partial",
     steps=[step((1, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"H2O": -52.73 * h2o},
                 system=dict(components=["Ca", "Mg", "Na", "K", "Cl", "C", "S", "N", "O"]), label=label)
            for h2o, label in ((1.0, "4a"), (WATER_GFW_RATIO, "4a-mass-matched"))])

EX5_X = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05]
case("ex5", "phreeqc.dat", """
SOLUTION 1  PURE WATER
        pH      7.0
        temp    25.0
EQUILIBRIUM_PHASES 1
        Pyrite          0.0
        Goethite        0.0
        Calcite         0.0
        CO2(g)         -3.5
        Gypsum          0.0     0.0
REACTION 1
        O2      1.0
        NaCl    0.5
        0.0     0.001   0.005   0.01   0.03   0.05
END
""", title="Example 5: pyrite oxidation", si=["Gypsum", "Pyrite", "Goethite", "Calcite"],
     compared="pH, pe, I, phase transfers, SI(gypsum), totals at 6 amounts of O2 + NaCl",
     steps=[step((1, "react", k + 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                 add={"Pyrite": 10.0, "Goethite": 10.0, "Calcite": 10.0, "Gypsum": 0.0,
                      "O2": x, "Na+": 0.5 * x, "Cl-": 0.5 * x},
                 system=dict(components=["Fe", "S", "Ca", "C", "Na", "Cl", "O"],
                             phases=["Pyrite", "Goethite", "Calcite", "Gypsum", "CO2(g)"]),
                 call=dict(fixed={"CO2(g)": -3.5}), label=f"O2={x:g}")
            for k, x in enumerate(EX5_X)])

EX6_PHASES = """
PHASES
        Gibbsite
                Al(OH)3 + 3 H+ = Al+3 + 3 H2O
                log_k          8.049
                delta_h        -22.792 kcal
        Kaolinite
                Al2Si2O5(OH)4 + 6 H+ = H2O + 2 H4SiO4 + 2 Al+3
                log_k          5.708
                delta_h        -35.306 kcal
        K-mica
                KAl3Si3O10(OH)2 + 10 H+ = 3 Al+3 + 3 H4SiO4 + K+
                log_k          12.970
                delta_h        -59.377 kcal
        K-feldspar
                KAlSi3O8 + 4 H2O + 4 H+ = Al+3 + 3 H4SiO4 + K+
                log_k          0.875
                delta_h        -12.467 kcal
"""
EX6_A = [  # (label, {phase: (SI, amount or "alt")})
    ("A1", [("Gibbsite", "alt"), ("Kaolinite", 0.0), ("K-mica", 0.0), ("K-feldspar", 0.0)]),
    ("A2", [("Gibbsite", 0.0), ("Kaolinite", "alt"), ("K-mica", 0.0), ("K-feldspar", 0.0)]),
    ("A3", [("Gibbsite", 0.0), ("Kaolinite", 0.0), ("K-mica", "alt"), ("K-feldspar", 0.0)]),
    ("A4", [("Gibbsite", 0.0), ("Kaolinite", 0.0), ("K-mica", 0.0), ("K-feldspar", "alt")]),
    ("A5", [("Gibbsite", "alt"), ("Kaolinite", 1.0)]),
    ("A6", [("Kaolinite", "alt"), ("K-mica", 1.0)]),
]
EX6_B = [0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 200.0]
body6 = EX6_PHASES + "SOLUTION 1  PURE WATER\n            pH    7.0 charge\n          temp   25.0\nEND\n"
steps6 = []
for i, (lab, ph) in enumerate(EX6_A):
    body6 += "USE solution 1\nEQUILIBRIUM_PHASES 1\n"
    for p, a in ph:
        body6 += f"        {p:<15} 0.0     " + ("KAlSi3O8        10.0" if a == "alt" else f"{a}") + "\n"
    body6 += "END\n"
    alt = next(p for p, a in ph if a == "alt")

    def add6(rows, resolve, ph=ph, alt=alt, sim=i + 2):
        r = rows[resolve((sim, "react", 1, None))]
        d = 10.0 - r["equi"].get(alt, 0.0)
        out = {p: (0.0 if a == "alt" else a) for p, a in ph}
        for sp, nu in KFSP.items():
            out[sp] = out.get(sp, 0.0) + nu * d
        return out

    steps6.append(step((i + 2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add=add6,
                       system=dict(components=["K", "Al", "Si"], phases=[p for p, _ in ph]), label=lab, skip_phases=[alt]))
body6 += "USE solution 1\nEQUILIBRIUM_PHASES 1\n        Kaolinite       0.0     0.0\n        Gibbsite        0.0     0.0\n" \
         "        K-mica          0.0     0.0\n        K-feldspar      0.0     0.0\nREACTION 1\n        K-feldspar      1.0\n" \
         "        0.04 0.08 0.16 0.32 0.64 1.0 2.0 4.0\n        8.0 16.0 32.0 64.0 100 200 umol\nEND\n"
for k, x in enumerate(EX6_B):
    steps6.append(step((8, "react", k + 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                       add={"Gibbsite": 0.0, "Kaolinite": 0.0, "K-mica": 0.0, "K-feldspar": 0.0}
                       | {sp: nu * x * 1e-6 for sp, nu in KFSP.items()},
                       system=dict(components=["K", "Al", "Si"],
                                   phases=["Gibbsite", "Kaolinite", "K-mica", "K-feldspar"]),
                       label=f"B-{x:g}umol"))
case("ex6", "phreeqc.dat", body6, db_additions=EX6_PHASES, si=["Gibbsite", "Kaolinite", "K-mica", "K-feldspar"],
     title="Example 6: K-feldspar reaction path (6A and 6B)",
     compared="6A1-A6: at PHREEQC's amount of K-feldspar dissolved, the target phase has SI = 0 and "
              "pH / log activities agree; 6B: 14 reaction steps with four candidate phases",
     note="6A uses PHREEQC's 'alternative reactant' (dissolve KAlSi3O8 until a phase reaches SI 0), which the "
          "package has no input for; the dissolved amount PHREEQC found is given to the package instead, so 6A "
          "checks that the package puts that point on the phase boundary. 6C (kinetics) is out of scope.",
     scope="partial", steps=steps6)

EX7_DATA = """
SOLUTION_MASTER_SPECIES
N(-3)    NH4+           0.0     N
SOLUTION_SPECIES
NH4+ = NH3 + H+
        log_k           -9.252
        delta_h 12.48   kcal
        -analytic    0.6322    -0.001225     -2835.76
NO3- + 10 H+ + 8 e- = NH4+ + 3 H2O
        log_k           119.077
        delta_h -187.055        kcal
        -gamma    2.5000    0.0000
PHASES
NH3(g)
        NH3 = NH3
        log_k           1.770
        delta_h -8.170  kcal
"""
EX7_X = [1., 2., 3., 4., 8., 16., 32., 64., 125., 250., 500., 1000.]


def add_ex7_fixed_volume(rows, resolve, k, x):
    """Reactant plus the gas that ``-equilibrate 1`` puts in the 23.19 L box before the reaction.

    PHREEQC's selected output has no row for that initial gas. Its carbon (all CO2(g); CH4(g) is
    ~1e-20 mol) follows exactly from PHREEQC's own carbon balance on the first step. Its H2O(g) is
    the ideal-gas amount at sol 1's H2O(g) saturation ratio. This is 0.0298 mol, and the ideal-gas
    error there (~3e-5 relative, judged from the CO2) shifts the water mass by only 1e-8 kg.
    """
    add = {sp: nu * x * 1e-3 for sp, nu in OM.items()}
    s1 = rows[resolve((1, "react", 1, None))]
    r1 = rows[resolve((3, "react", 1, None))]
    x1 = EX7_X[0] * 1e-3
    gas_c = sum(r1["gas"].get(g, 0.0) for g in ("CO2(g)", "CH4(g)"))
    add["CO2(g)"] = gas_c + r1["tot"]["C"] * r1["water"] - s1["tot"]["C"] * s1["water"] - x1
    add["H2O(g)"] = 10 ** s1["si"]["H2O(g)"] * 23.19 / (0.0820573 * (273.15 + s1["tc"]))
    return add
GAS7 = ["CO2(g)", "CH4(g)", "N2(g)", "H2O(g)"]
OM = {"CH4": 0.5, "CO2": 0.5, "NH3": 0.07}  # CH2O(NH3)0.07 as species (C, H, O, N balanced)
case("ex7", "phreeqc.dat", EX7_DATA + """
SOLUTION 1
EQUILIBRIUM_PHASES 1
        Calcite
        CO2(g)  -1.5
SAVE solution 1
END
USE solution 1
GAS_PHASE 1 Fixed-pressure gas phase
        -fixed_pressure
        -pressure       1.1
        CO2(g)          0.0
        CH4(g)          0.0
        N2(g)           0.0
        H2O(g)          0.0
REACTION 1
        CH2O(NH3)0.07     1.0
        1. 2. 3. 4. 8. 16. 32 64. 125. 250. 500. 1000. mmol
END
USE solution 1
USE reaction 1
GAS_PHASE 1 Fixed volume gas phase
        -fixed_volume
        -volume         23.19
        CO2(g)          0.0
        CH4(g)          0.0
        N2(g)           0.0
        H2O(g)          0.0
        -equilibrate 1
END
""", db_additions=EX7_DATA, si=["Calcite", "CO2(g)", "CH4(g)", "N2(g)", "NH3(g)", "H2O(g)"], gases=GAS7,
     title="Example 7: organic decomposition with fixed-pressure and fixed-volume gas phases",
     compared="solution 1 (calcite + CO2); fixed-P (1.1 atm) and fixed-V (23.19 L) gas: gas moles, "
              "partial pressures, pH, pe, I at 12 amounts of CH2O(NH3)0.07",
     note="phreeqc.dat gives CO2(g), CH4(g), N2(g) and H2O(g) critical constants, so PHREEQC 3 treats these gas "
          "phases with the Peng-Robinson equation of state; the package uses ideal gases.",
     steps=[step((1, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"Calcite": 10.0},
                 system=dict(components=["Ca", "C(4)"], phases=["Calcite", "CO2(g)"]),
                 call=dict(fixed={"CO2(g)": -1.5}), label="solution1")]
     + [step((2, "react", k + 1, None), base=[((1, "react", 1, None), 1.0)],
             add={sp: nu * x * 1e-3 for sp, nu in OM.items()},
             system=dict(components=["Ca", "C", "N", "O"], gas_phase={"members": GAS7, "pressure": 1.1}),
             label=f"fixedP-{x:g}mmol") for k, x in enumerate(EX7_X)]
     + [step((3, "react", k + 1, None), base=[((1, "react", 1, None), 1.0)],
             add=functools.partial(add_ex7_fixed_volume, k=k, x=x),
             system=dict(components=["Ca", "C", "N", "O"], gas_phase={"members": GAS7, "volume": 23.19}),
             label=f"fixedV-{x:g}mmol") for k, x in enumerate(EX7_X)])
variant("ex7", "ex7_ideal_gas", prepend=IDEAL_GASES, title_suffix=" (ideal-gas PHREEQC run)", note=IDEAL_NOTE)

EX8_DATA = """
SURFACE_SPECIES
     Hfo_sOH  + H+ = Hfo_sOH2+
     log_k  7.18
     Hfo_sOH = Hfo_sO- + H+
     log_k  -8.82
     Hfo_sOH + Zn+2 = Hfo_sOZn+ + H+
     log_k  0.66
     Hfo_wOH  + H+ = Hfo_wOH2+
     log_k  7.18
     Hfo_wOH = Hfo_wO- + H+
     log_k  -8.82
     Hfo_wOH + Zn+2 = Hfo_wOZn+ + H+
     log_k  -2.32
PHASES
     Fix_H+
     H+ = H+
     log_k  0.0
"""
EX8_PH = [5.0 + 0.25 * i for i in range(13)]
body8 = EX8_DATA + """
SURFACE 1
     Hfo_sOH        5e-6    600.    0.09
     Hfo_wOH        2e-4
END
SOLUTION 1
     -units  mmol/kgw
     pH      8.0
     Zn      0.0001
     Na      100.    charge
     N(5)    100.
END
SOLUTION 2
     -units  mmol/kgw
     pH      8.0
     Zn      0.1
     Na      100.    charge
     N(5)    100.
END
"""
steps8 = []
sim = 3
for sol in (1, 2):
    for ph in EX8_PH:
        sim += 1
        body8 += f"USE solution {sol}; USE surface 1\nEQUILIBRIUM_PHASES 1\n   Fix_H+ {-ph:g} NaOH 10.0\nEND\n"

        def add8(rows, resolve, sim=sim):
            r = rows[resolve((sim, "react", 1, None))]
            return {"Hfo_w": 2e-4, "Hfo_s": 5e-6, "Na+": 10.0 - r["equi"]["Fix_H+"]}

        steps8.append(step((sim, "react", 1, None), base=[((sol + 1, "i_soln", None, sol), 1.0)], add=add8,
                           system=dict(components=["Zn", "Na", "N(5)"], surfaces={"Hfo": {"area": 600, "mass": 0.09}}),
                           call=dict(pH=ph), label=f"Zn{'1e-7' if sol == 1 else '1e-4'}-pH{ph:g}"))
case("ex8", "phreeqc.dat", body8, db_additions=EX8_DATA, edl=["Hfo"],
     title="Example 8: Zn sorption on hydrous ferric oxide (DDL)",
     compared="surface species, Zn+2 molality and surface potential at pH 5-8 for total Zn 1e-7 and 1e-4 m",
     note="PHREEQC sets the pH with the phase Fix_H+ and NaOH; the package fixes pH directly and is given the "
          "same amount of Na that PHREEQC added.",
     steps=steps8)

EX10_DATA = """
PHASES
        Strontianite
                SrCO3 = CO3-2 + Sr+2
                log_k           -9.271
        Aragonite
                CaCO3 = CO3-2 + Ca+2
                log_k           -8.336
"""
EX10_X = [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.1, 10.0]
case("ex10", "phreeqc.dat", EX10_DATA + """
SOLID_SOLUTIONS 1
        Ca(x)Sr(1-x)CO3
                -comp1   Aragonite       0
                -comp2   Strontianite    0
END
SOLUTION 1
        -units mmol/kgw
        pH 5.93 charge
        Ca      3.932
        C       7.864
EQUILIBRIUM_PHASES 1
        CO2(g) -0.01265 10
        Aragonite
SAVE solution 1
END
USE solution 1
USE solid_solution 1
REACTION 1
        SrCO3   1.0
        1e-5 1e-4 5e-4 1e-3 2e-3 5e-3 0.1 10.0
END
""", db_additions=EX10_DATA, si=["Aragonite", "Strontianite", "Calcite"],
     title="Example 10 (ideal variant): aragonite-strontianite solid solution",
     compared="solution 1 (aragonite + CO2); solid-solution moles and mole fractions, pH, Sr and Ca "
              "molalities at 8 amounts of SrCO3",
     note="The example's solid solution is non-ideal (Guggenheim a0 = 3.43, a1 = -1.82), which the package does "
          "not support. The Guggenheim line is removed (ideal solid solution) and a short list of SrCO3 amounts "
          "replaces the 500-step path.",
     scope="partial",
     steps=[step((2, "react", 1, None), base=[((2, "i_soln", None, 1), 1.0)], add={"Aragonite": 10.0},
                 system=dict(components=["Ca", "C(4)"], phases=["Aragonite", "CO2(g)"]),
                 call=dict(fixed={"CO2(g)": -0.01265}), label="solution1")]
     + [step((3, "react", k + 1, None), base=[((2, "react", 1, None), 1.0)], add={"Sr+2": x, "CO3-2": x},
             system=dict(components=["Ca", "Sr", "C(4)"],
                         solid_solutions={"Ca(x)Sr(1-x)CO3": ["Aragonite", "Strontianite"]}),
             label=f"SrCO3={x:g}") for k, x in enumerate(EX10_X)])
variant("ex10", "ex10_ideal_gas", prepend=IDEAL_GASES, title_suffix=" (ideal-gas PHREEQC run)",
        note=IDEAL_NOTE + " The Guggenheim line is removed as in ex10.")

EX11_BODY = """
SOLUTION 0  CaCl2
        units            mmol/kgw
        temp             25.0
        pH               7.0     charge
        pe               12.5    O2(g)   -0.68
        Ca               0.6
        Cl               1.2
SOLUTION 1  Initial solution for column
        units            mmol/kgw
        temp             25.0
        pH               7.0     charge
        pe               12.5    O2(g)   -0.68
        Na               1.0
        K                0.2
        N(5)             1.2
END
EXCHANGE 1
        -equilibrate 1
        X                0.0011
END
USE solution 0
USE exchange 1
END
"""
case("ex11", "phreeqc.dat", EX11_BODY,
     title="Example 11 (equilibrium part): Na/K exchanger flushed by CaCl2",
     compared="one batch of the column's first cell: exchanger in equilibrium with solution 1, contacted with "
              "the CaCl2 inflow; exchange species, pH, log activities",
     note="Transport (ADVECTION/TRANSPORT) is out of scope; only a single mixing cell is computed.",
     scope="partial",
     steps=[step((3, "react", 1, None), base=[((1, "i_soln", None, 0), 1.0), ((2, "i_exch", None, None), 1.0, ["ex"])],
                 system=dict(components=["Ca", "Cl", "Na", "K"], exchange=["X"]), label="batch")])

EX14_DATA = """
SURFACE_MASTER_SPECIES
        Surf    SurfOH
SURFACE_SPECIES
        SurfOH = SurfOH
                log_k   0.0
        SurfOH  + H+ = SurfOH2+
                log_k   7.29
        SurfOH = SurfO- + H+
                log_k   -8.93
        SurfOH + AsO4-3 + 3H+ = SurfH2AsO4 + H2O
                log_k   29.31
        SurfOH + AsO4-3 + 2H+ = SurfHAsO4- + H2O
                log_k   23.51
        SurfOH + AsO4-3 = SurfOHAsO4-3
                log_k   10.58
SOLUTION_MASTER_SPECIES
        As       H3AsO4        -1.0     74.9216    74.9216
SOLUTION_SPECIES
        H3AsO4 = H3AsO4
                log_k           0.0
        H3AsO4 = AsO4-3 + 3H+
                log_k   -20.7
        H+ + AsO4-3 = HAsO4-2
                log_k   11.50
        2H+ + AsO4-3 = H2AsO4-
                log_k           18.46
"""
EX14_COMP = ["Ca", "Mg", "Na", "Cl", "C(4)", "S(6)", "As"]
case("ex14", "phreeqc.dat", EX14_DATA + """
SOLUTION 1 Brine
        pH      5.713
        pe      4.0     O2(g)   -0.7
        temp    25.
        units   mol/kgw
        Ca      .4655
        Mg      .1609
        Na      5.402
        Cl      6.642           charge
        C       .00396
        S       .004725
        As      .025 umol/kgw
END
USE solution 1
EQUILIBRIUM_PHASES 1
        Dolomite        0.0     1.6
        Calcite         0.0     0.1
SAVE solution 1
END
SURFACE 1
        SurfOH           0.07    600.    30.
END
USE solution 1
USE surface 1
END
""", db_additions=EX14_DATA, si=["Calcite", "Dolomite", "Gypsum", "Halite"], edl=["Surf"],
     title="Example 14 (equilibrium part): brine with dolomite, calcite and an arsenate-sorbing surface",
     compared="brine (I ~ 6.9, far beyond Truesdell-Jones' range) with dolomite and calcite; the brine "
              "contacted with the DDL surface: pH, surface species, potential",
     note="Transport is out of scope. The example equilibrates the surface with a fixed solution composition "
          "(SURFACE -equilibrate); here the bare surface is reacted with the brine in a batch instead, which the "
          "package can pose.",
     scope="partial",
     steps=[step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"Dolomite": 1.6, "Calcite": 0.1},
                 system=dict(components=EX14_COMP, phases=["Dolomite", "Calcite"]), label="brine+carbonates"),
            step((4, "react", 1, None), base=[((2, "react", 1, None), 1.0)], add={"Surf": 0.07},
                 system=dict(components=EX14_COMP, surfaces={"Surf": {"area": 600, "mass": 30}}),
                 label="brine+surface")])

EX19_DATA = """
SURFACE_MASTER_SPECIES
      Langmuir Langmuir
SURFACE_SPECIES
  Langmuir = Langmuir
  Langmuir + Cd+2 = LangmuirCd+2
      -log_k 6.56           # log10(112.4 / 30.9e-6)
"""
EX19_X = [1e-8, 5e-8, 1e-7, 2e-7, 4e-7, 7e-7]
case("ex19", "phreeqc.dat", EX19_DATA + """
SURFACE 1
      Langmuir 8.45e-8 1 1  # 9.5 / 112.4e6
      -no_edl
SOLUTION 1
      pH  6
      Ca  1
      Cl  2
END
USE solution 1
USE surface 1
REACTION 1
      CdCl2 1
      1e-8 5e-8 1e-7 2e-7 4e-7 7e-7
END
""", db_additions=EX19_DATA, edl=[],
     title="Example 19 (Langmuir part): Cd sorption isotherm without electrostatics",
     compared="sorbed Cd and Cd+2 molality at six Cd additions",
     note="The linear and Freundlich isotherms (1e100 sites, log_k -100.7 / -102.61, non-integer 0.722 "
          "stoichiometry) are not representable. For the Langmuir isotherm, -mole_balance LangmuirCdCl2 is dropped "
          "(it only changes how PHREEQC books the sorbed Cl) and a list of additions replaces '0.7e-6 in 20'.",
     scope="partial",
     steps=[step((2, "react", k + 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                 add={"Langmuir": 8.45e-8, "Cd+2": x, "Cl-": 2 * x},
                 system=dict(components=["Ca", "Cl", "Cd"], surfaces={"Langmuir": {"edl": None}}),
                 label=f"Cd={x:g}") for k, x in enumerate(EX19_X)])

# ============================================================================ stress cases
BLACK_SEA = """
SOLUTION 1  Black Sea water
        units   mg/L
        density 1.014
        pH      8.0     # estimated
        Ca      233
        Mg      679
        Na      5820
        K       193
        S(6)    1460
        Cl      10340
        Br      35
        C       1       CO2(g) -3.5
"""
EVAP_PHASES = ["Calcite", "Gypsum", "Anhydrite", "Glauberite", "Polyhalite", "Epsomite", "Kieserite", "Hexahydrite",
               "Halite", "Bischofite", "Carnallite"]
EVAP_X = [36.0, 48.0, 54.0, 54.5, 54.876, 55.026]
case("stress_pitzer_evaporite", "pitzer.dat", BLACK_SEA + """
END
USE solution 1
EQUILIBRIUM_PHASES 1
 CO2(g) -3.5 10; Calcite 0 0
 Gypsum 0 0;     Anhydrite 0 0;  Glauberite 0 0;  Polyhalite 0 0
 Epsomite 0 0;   Kieserite 0 0;  Hexahydrite 0 0
 Halite 0 0;     Bischofite 0 0; Carnallite 0 0
REACTION 1
 H2O -1; 36 48 54 54.5 54.876 55.026
END
""", group="stress", si=EVAP_PHASES,
     title="Pitzer: Black Sea water evaporated past halite saturation (ex17b, forward part)",
     compared="11 candidate evaporite minerals + CO2(g); water mass, I, a_w, phases present and amounts, SI",
     note="The -mass-matched steps remove x * 18.016/18.01528 mol of water, the same water MASS as PHREEQC "
          "(PHREEQC's water gfw comes from the database, H 1.008 + O 16.00; the package uses 18.01528).",
     steps=[step((2, "react", k + 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                 add={p: 0.0 for p in EVAP_PHASES} | {"H2O": -x * h2o},
                 system=dict(components=["Ca", "Mg", "Na", "K", "S(6)", "Cl", "Br", "C(4)"],
                             phases=EVAP_PHASES + ["CO2(g)"]),
                 call=dict(fixed={"CO2(g)": -3.5}), label=f"H2O-{x:g}{suffix}")
            for h2o, suffix in ((1.0, ""), (WATER_GFW_RATIO, "-mass-matched")) for k, x in enumerate(EVAP_X)])

for dbname, tag in (("phreeqc.dat", "truesdell_jones"), ("pitzer.dat", "pitzer")):
    case(f"stress_halite_{tag}", dbname, """
SOLUTION 1
EQUILIBRIUM_PHASES 1
    Halite 0 10
END
""", group="stress", si=["Halite"],
         title=f"Halite saturation (~6 m NaCl) with {dbname}",
         compared="halite dissolved, I, a_w, log activities",
         note="Truesdell-Jones is fitted to I < ~1-2 m; at 6 m both programs evaluate the same formula far outside "
              "its range" if tag == "truesdell_jones" else "",
         steps=[step((1, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"Halite": 10.0},
                     system=dict(components=["Na", "Cl"], phases=["Halite"]), label="halite")])

for tag, sol, comps in (("MgSO4", "Mg 2.5\n    S(6) 2.5", ["Mg", "S(6)"]),
                        ("Na2SO4", "Na 3.0\n    S(6) 1.5", ["Na", "S(6)"]),
                        ("MgSO4_trace_Cl", "Mg 2.5\n    S(6) 2.5\n    Na 1e-3\n    Cl 1e-3", ["Mg", "S(6)", "Na", "Cl"])):
    case(f"stress_pitzer_no_chloride_{tag}", "pitzer.dat", f"""
SOLUTION 1
    units mol/kgw
    pH 7 charge
    {sol}
END
""", group="stress",
         title=f"Pitzer, concentrated sulfate solution ({tag}): MacInnes scaling without / with Cl-",
         compared="pH (charge balance), I, a_w, log activities of the major ions",
         note="PHREEQC's MacInnes convention scales individual-ion activity coefficients to KCl at the solution's "
              "ionic strength whether or not Cl- is present; the trace-Cl variant switches it on in the package too.",
         steps=[step((1, "i_soln", None, 1), base=[((1, "i_soln", None, 1), 1.0)],
                     system=dict(components=comps), label="charge-balance")])

case("stress_trace_1e-30", "phreeqc.dat", """
SOLUTION 1
    units mmol/kgw
    pH 7.0
    Na 12
    Cl 10
    C(4) 2
    Pb 1e-27
    Zn 1e-27
    Cd 1e-27
END
MIX 1
    1 1.0
END
""", group="stress", title="Trace metals at 1e-30 mol/kgw",
     compared="log activities of every Pb, Zn and Cd species (down to ~1e-36 m), fixed pH and re-equilibrated",
     steps=[step((1, "i_soln", None, 1), totals=["Na", "Cl", "C(4)", "Pb", "Zn", "Cd"],
                 system=dict(components=["Na", "Cl", "C(4)", "Pb", "Zn", "Cd"]), call=dict(pH=7.0, water_mass=1.0),
                 label="fixed-pH"),
            step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                 system=dict(components=["Na", "Cl", "C(4)", "Pb", "Zn", "Cd"]), label="reequilibrated")])

REDOX_COMP = ["Fe", "Mn", "S", "N", "C", "Na", "Cl"]
for tag, pe in (("O2_limit", 13.75), ("H2_limit", -6.95)):
    case(f"stress_pe_{tag}", "phreeqc.dat", f"""
SOLUTION 1
    units mmol/kgw
    pH 7.0
    pe {pe}
    Fe 0.1
    Mn 0.1
    S 1
    N 1
    C 2
    Na 3
    Cl 1
END
""", group="stress", si=["Goethite", "Pyrolusite", "Siderite", "Pyrite", "Mackinawite", "Rhodochrosite", "Sulfur"],
         title=f"pe {pe} at pH 7 (water stability {tag.replace('_', ' ')}) with Fe, Mn, S, N, C",
         compared="log activities of all valence states and SI at fixed pH and pe",
         steps=[step((1, "i_soln", None, 1), totals=REDOX_COMP, system=dict(components=REDOX_COMP),
                     call=dict(pH=7.0, pe=pe, water_mass=1.0), label=tag)])

MANY = ["Calcite", "Aragonite", "Dolomite", "Gypsum", "Anhydrite", "Quartz", "Chalcedony", "SiO2(a)", "Talc",
        "Sepiolite", "Chrysotile", "Fluorite", "Celestite", "Strontianite"]
case("stress_many_phases", "phreeqc.dat", """
SOLUTION 1
END
USE solution 1
EQUILIBRIUM_PHASES 1
    CO2(g) -2.0 10
""" + "".join(f"    {p} 0 1\n" for p in MANY) + "END\n", group="stress", si=MANY,
     title="14 candidate phases incl. polymorphs (calcite/aragonite, quartz/chalcedony/SiO2(a), gypsum/anhydrite)",
     compared="phases present, amounts, pH, SI",
     steps=[step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={p: 1.0 for p in MANY},
                 system=dict(components=["Ca", "Mg", "C(4)", "S(6)", "Si", "F", "Sr"], phases=MANY + ["CO2(g)"]),
                 call=dict(fixed={"CO2(g)": -2.0}), label="14 phases")])

case("stress_si_near_zero", "phreeqc.dat", """
SOLUTION 1
END
USE solution 1
EQUILIBRIUM_PHASES 1
    Calcite 0 10
SAVE solution 2
END
USE solution 2
EQUILIBRIUM_PHASES 2
    Calcite 0 0
END
USE solution 2
EQUILIBRIUM_PHASES 2
    Calcite 0 0
REACTION 1
    CaCO3 1
    1e-9
END
USE solution 2
EQUILIBRIUM_PHASES 2
    Calcite 0 0
REACTION 1
    CaCO3 1
    -1e-9
END
""", group="stress", si=["Calcite", "Aragonite"],
     title="Calcite exactly at saturation, and perturbed by +/-1e-9 mol CaCO3 (SI ~ +/-1e-5)",
     compared="whether calcite is present, its amount (to 1e-12 mol) and SI",
     steps=[step((3, "react", 1, None), base=[((2, "react", 1, None), 1.0)], add={"Calcite": 0.0},
                 system=dict(components=["Ca", "C(4)"], phases=["Calcite"]), label="exact"),
            step((4, "react", 1, None), base=[((2, "react", 1, None), 1.0)],
                 add={"Calcite": 0.0, "Ca+2": 1e-9, "CO3-2": 1e-9},
                 system=dict(components=["Ca", "C(4)"], phases=["Calcite"]), label="plus1e-9"),
            step((5, "react", 1, None), base=[((2, "react", 1, None), 1.0)],
                 add={"Calcite": 0.0, "Ca+2": -1e-9, "CO3-2": -1e-9},
                 system=dict(components=["Ca", "C(4)"], phases=["Calcite"]), label="minus1e-9")])

case("stress_dissolve_all", "phreeqc.dat", """
SOLUTION 1
END
USE solution 1
EQUILIBRIUM_PHASES 1
    Gypsum   0 1e-3
    Calcite  0 1e-5
    Fluorite 0 1e-6
END
""", group="stress", si=["Gypsum", "Calcite", "Fluorite"],
     title="Three phases that dissolve completely",
     compared="all phase amounts go to 0, SI < 0, pH, molalities",
     steps=[step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                 add={"Gypsum": 1e-3, "Calcite": 1e-5, "Fluorite": 1e-6},
                 system=dict(components=["Ca", "C(4)", "S(6)", "F"], phases=["Gypsum", "Calcite", "Fluorite"]),
                 label="dissolve")])

# Upper limits: phreeqc.dat and pitzer.dat fit their log K / Pitzer T-dependence to ~200 C; llnl.dat tabulates to
# 300 C; sit.dat is a 25 C database with van 't Hoff / analytic extrapolation, taken to 100 C.
# wateq4f_PWN.dat is not included: IPhreeqc 3.8.7 rejects the bundled file ("Expecting log k").
T_LIMITS = {"phreeqc.dat": 200.0, "pitzer.dat": 200.0, "llnl.dat": 300.0, "sit.dat": 100.0}
for dbname, tmax in T_LIMITS.items():
    for tc in (0.01, tmax):
        tag = dbname.split(".")[0].split("_")[0]
        case(f"stress_T_{tag}_{tc:g}C", dbname, f"""
SOLUTION 1
    units mol/kgw
    temp {tc}
    pH 7 charge
    Na 0.5
    Cl 0.5
END
USE solution 1
EQUILIBRIUM_PHASES 1
    Calcite 0 1
    Gypsum 0 1
END
""", group="stress", si=["Calcite", "Gypsum", "Anhydrite"],
             title=f"{dbname} at {tc:g} C: 0.5 m NaCl with calcite and gypsum",
             compared="phase amounts, pH, I, log activities",
             note=("Above 100 C PHREEQC works at the saturation pressure of water; the package is given the same "
                   "pressure." if tc > 100 else ""),
             steps=[step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"Calcite": 1.0, "Gypsum": 1.0},
                         system=dict(components=["Na", "Cl", "Ca", "C(4)", "S(6)"], phases=["Calcite", "Gypsum"]),
                         label=f"{tc:g}C")])

case("stress_surface_dense_lowI", "phreeqc.dat", """
SOLUTION 1
    units mmol/kgw
    pH 7.0
    Na 0.1
    Cl 0.1
    Zn 0.001
END
SURFACE 1
    Hfo_sOH  5e-4  600  9.0
    Hfo_wOH  2e-2
END
USE solution 1
USE surface 1
END
""", group="stress", edl=["Hfo"], title="Hfo with 100x example 8's sites and mass at I ~ 1e-4",
     compared="pH, surface species, Zn+2, surface potential",
     steps=[step((3, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"Hfo_w": 2e-2, "Hfo_s": 5e-4},
                 system=dict(components=["Na", "Cl", "Zn"], surfaces={"Hfo": {"area": 600, "mass": 9.0}}),
                 label="batch")])

case("stress_exchange_divalent", "phreeqc.dat", """
SOLUTION 1
    units mmol/kgw
    Na 100
    Cl 100
END
EXCHANGE 1
    X 0.1
    -equilibrate 1
END
SOLUTION 2
    units mmol/kgw
    pH 7 charge
    Ca 5
    Mg 5
    Sr 1
    Ba 1
    Cl 24
END
USE solution 2
USE exchange 1
END
""", group="stress", title="Na-loaded exchanger (0.1 eq) against Ca, Mg, Sr, Ba",
     compared="all exchange species, log activities",
     steps=[step((4, "react", 1, None), base=[((3, "i_soln", None, 2), 1.0), ((2, "i_exch", None, None), 1.0, ["ex"])],
                 system=dict(components=["Na", "Ca", "Mg", "Sr", "Ba", "Cl"], exchange=["X"]), label="batch")])

case("stress_charge_minor_ion", "phreeqc.dat", """
SOLUTION 1
    units mmol/kgw
    pH 7.0
    Ca 1
    Na 1
    Cl 2.9
    F 1 charge
END
""", group="stress", title="Charge balance on F- (a minor ion)",
     compared="adjusted F total, log activities",
     steps=[step((1, "i_soln", None, 1), totals={"Ca": 1e-3, "Na": 1e-3, "Cl": 2.9e-3, "F": 1e-3},
                 system=dict(components=["Ca", "Na", "Cl", "F"]),
                 call=dict(pH=7.0, water_mass=1.0, charge_balance="F-"), label="F-charge")])


def add_gas_vanish(rows, resolve):
    r = rows[resolve((2, "react", 1, None))]
    s = rows[resolve((1, "i_soln", None, 1))]
    n0 = r["tot"].get("C(4)", 0.0) * r["water"] + r["gas"].get("CO2(g)", 0.0) - s["tot"].get("C(4)", 0.0) * s["water"]
    return {"CO2(g)": n0}


case("stress_gas_vanishing", "phreeqc.dat", """
SOLUTION 1
    units mmol/kgw
    pH 12 charge
    Na 100
END
USE solution 1
GAS_PHASE 1
    -fixed_volume
    -volume 1.0
    CO2(g) 0.01
END
""", group="stress", gases=["CO2(g)"], si=["CO2(g)"],
     title="Fixed-volume CO2 gas (0.01 atm, 1 L) absorbed by 0.1 m NaOH",
     compared="remaining gas moles (~1e-10 of the initial 4e-4), partial pressure, pH",
     note="The initial CO2(g) moles are taken from PHREEQC's carbon balance.",
     steps=[step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)], add=add_gas_vanish,
                 system=dict(components=["Na", "C(4)"], gas_phase={"members": ["CO2(g)"], "volume": 1.0}),
                 label="absorb")])


# Pressure: log K from species molar volumes (-Vm), water density and dielectric constant at P, and (Pitzer) the
# pressure-dependent Debye-Hueckel b. SOLUTION -pressure sets P; llnl.dat has no molar volumes, so its log K are
# pressure independent in PHREEQC too.
P_CASES = {"phreeqc.dat": ((25.0, (100.0, 200.0, 500.0)), (75.0, (300.0,))),
           "pitzer.dat": ((25.0, (100.0, 500.0)), (75.0, (300.0,))),
           "sit.dat": ((25.0, (200.0,)),),
           "llnl.dat": ((25.0, (200.0,)),)}
for dbname, groups in P_CASES.items():
    tag = dbname.split(".")[0]
    for tc, pressures in groups:
        for patm in pressures:
            case(f"stress_P_{tag}_{tc:g}C_{patm:g}atm", dbname, f"""
SOLUTION 1
    units mol/kgw
    temp {tc}
    pressure {patm}
    pH 7 charge
    Na 0.5
    Cl 0.5
    Ca 2e-3
    C(4) 4e-3
    S(6) 2e-3
END
USE solution 1
EQUILIBRIUM_PHASES 1
    Calcite 0 1
    Gypsum 0 1
END
""", group="stress", si=["Calcite", "Gypsum", "Anhydrite"],
                 title=f"{dbname} at {tc:g} C and {patm:g} atm: 0.5 m NaCl with calcite and gypsum",
                 compared="pH, I, log activities, phase amounts (log K from molar volumes at P)",
                 steps=[step((2, "react", 1, None), base=[((1, "i_soln", None, 1), 1.0)],
                             add={"Calcite": 1.0, "Gypsum": 1.0},
                             system=dict(components=["Na", "Cl", "Ca", "C(4)", "S(6)"], phases=["Calcite", "Gypsum"]),
                             label="equilibrium")])

# Example 22: CO2 solubility at high pressure, a fixed-volume (1 L) CO2 + H2O(g) gas phase with Peng-Robinson
# gases and pressure-dependent log K. CO2 is added 1 mol at a time.
EX22_T = {1: (25.0, 31), 2: (50.0, 29), 3: (75.0, 27), 4: (100.0, 28), 5: (150.0, 26)}
EX22 = """
SOLUTION 1
GAS_PHASE 1
    -fixed_volume
    CO2(g) 0
    H2O(g) 0
REACTION
 CO2 1; 0 31*1
INCREMENTAL_REACTIONS true
END
""" + "".join(f"""USE solution 1
USE gas_phase 1
REACTION
 CO2 1; 0 {n}*1
REACTION_TEMPERATURE {sim}
 {tc}
END
""" for sim, (tc, n) in list(EX22_T.items())[1:])
case("ex22", "phreeqc.dat", EX22, si=["CO2(g)", "H2O(g)"], gases=["CO2(g)", "H2O(g)"],
     title="CO2 solubility at high pressure: fixed-volume CO2 + H2O(g), 25-150 C",
     compared="pH, dissolved C, gas moles, gas pressure, log activities",
     note="USER_GRAPH blocks removed. The package is given PHREEQC's pressure (the gas pressure) for log K.",
     steps=[step((sim, "react", j + 1, None), base=[((1, "i_soln", None, 1), 1.0)], add={"CO2": float(j)},
                 system=dict(components=["C(4)"], gas_phase={"members": ["CO2(g)", "H2O(g)"], "volume": 1.0}),
                 label=f"{tc:g}C-{j}mol")
            for sim, (tc, n) in EX22_T.items() for j in range(1, n + 1)])


# ============================================================================ run
def main():
    # the IPhreeqc library bundled with phreeqpython 1.6.2 (its wrapper exposes no version call)
    pq_version = "3.8.7-17149"
    version = f"PHREEQC {pq_version} (IPhreeqc) via phreeqpython {importlib.metadata.version('phreeqpython')}"
    out = {"version": version, "source": "Parkhurst & Appelo (2013) USGS TM 6-A43; "
                                        "https://github.com/usgs-coupled/phreeqc3/tree/master/examples",
           "cases": []}
    for c in CASES:
        text = dump_block(c["si"], c["gases"], c["edl"]) + c["body"]
        rows = run_phreeqc(c["database"], text)

        def resolve(sel, rows=rows):
            sim, state, stp, soln = sel
            return find(rows, sim, state, stp, soln)

        steps = []
        for s in c["steps"]:
            row = resolve(s["row"])
            # [row, factor, groups]: an i_exch / i_gas row also lists the solution it was equilibrated with,
            # so only its exchange / gas amounts are taken from it
            base = [[resolve(b[0]), b[1], b[2] if len(b) > 2 else ["n", "ex", "surf", "gas"]] for b in s["base"]]
            add = s["add"](rows, resolve) if callable(s["add"]) else s["add"]
            W0 = sum(f * rows[i]["water"] for i, f, g in base if "n" in g) if base else 1.0
            totals = s["totals"]
            if isinstance(totals, list):
                r = rows[row]
                totals = {el: r["tot"].get(el, 0.0) for el in totals}
            steps.append(dict(label=s["label"], row=row, base=base, add=add, W0=W0, totals=totals,
                              system=s["system"], call=s["call"], skip_phases=s["skip_phases"]))
        used = sorted({s["row"] for s in steps} | {b[0] for s in steps for b in s["base"]})
        keep = {i: k for k, i in enumerate(used)}
        for s in steps:
            s["row"] = keep[s["row"]]
            s["base"] = [[keep[i], f, g] for i, f, g in s["base"]]
        out["cases"].append(dict(id=c["id"], group=c["group"], scope=c["scope"], title=c["title"],
                                 compared=c["compared"], note=c["note"], database=c["database"],
                                 db_additions=c["db_additions"], input=text, rows=[rows[i] for i in used],
                                 steps=steps))
        print(f"{c['id']:<32} {len(steps):3d} steps {len(rows):4d} rows")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

"""Generate ``phreeqc_peng_robinson.json``: PHREEQC 3's Peng-Robinson gas states, to check ``discopt.speciation.eos``.

Run with an interpreter that has ``phreeqpython`` (see ``generate_phreeqc_examples.py``)::

    /tmp/phreeqc-venv/bin/python tests/data/generate_phreeqc_pr.py

Each case equilibrates water with a gas phase (fixed pressure or fixed volume)
or with one gas at a fixed saturation index (``EQUILIBRIUM_PHASES``, so a pure
gas at pressure ``10**SI``), using the bundled ``phreeqc.dat``. It stores
PHREEQC's temperature, total pressure, molar volume, and per gas the moles,
partial pressure ``PR_P`` and fugacity coefficient ``PR_PHI``. The test
recomputes the coefficients from the stored composition, temperature and
pressure (or volume). The comparison does not involve ``log K``, so PHREEQC's
pressure dependence of ``log K``, which the package lacks, does not enter.
"""

import importlib.metadata
import json
import pathlib

from phreeqpython import PhreeqPython

HERE = pathlib.Path(__file__).parent
DB = HERE.parents[1] / "src" / "discopt" / "speciation" / "data" / "phreeqc.dat"
OUT = HERE / "phreeqc_peng_robinson.json"
GASES = ["CO2(g)", "CH4(g)", "N2(g)", "H2O(g)"]

PUNCH = """
SELECTED_OUTPUT 1
 -reset false
 -high_precision true
USER_PUNCH 1
 -headings tc gas_p gas_vm {heads}
 10 PUNCH TC, GAS_P, GAS_VM
 20 PUNCH {cols}
"""


def punch(gases):
    heads = " ".join(f"n_{g} p_{g} phi_{g}" for g in gases)
    cols = ", ".join(f'GAS("{g}"), PR_P("{g}"), PR_PHI("{g}")' for g in gases)
    return PUNCH.format(heads=heads, cols=cols)


def run(text):
    ip = PhreeqPython(database_directory=DB.parent, database=DB.name).ip
    status = ip.run_string(text)
    out = ip.get_selected_output_array()
    if status or len(out) < 2:
        raise RuntimeError(f"PHREEQC failed ({status}): {ip.get_error_string()}")
    head, rows = out[0], out[1:]
    return [dict(zip(head, r)) for r in rows]


def gas_phase_case(cid, kind, value, tc, comp):
    """Water with a ``-fixed_pressure`` (value = atm) or ``-fixed_volume`` (value = L) gas phase."""
    gas_lines = "\n".join(f" {g} {p}" for g, p in comp.items())
    opt = f"-fixed_pressure\n -pressure {value}" if kind == "pressure" else f"-fixed_volume\n -volume {value}"
    text = (f"SOLUTION 1\n temp {tc}\n Na 100\n Cl 100\nEND\n{punch(list(comp))}\n"
            f"USE SOLUTION 1\nGAS_PHASE 1\n {opt}\n -temperature {tc}\n{gas_lines}\nEND\n")
    row = run(text)[-1]
    gases = {g: {"moles": row[f"n_{g}"], "p": row[f"p_{g}"], "phi": row[f"phi_{g}"]} for g in comp}
    return {"id": cid, "kind": kind, "value": value, "input": text, "tc": row["tc"], "P": row["gas_p"],
            "Vm": row["gas_vm"], "gases": gases}


def pure_case(cid, gas, log_p, tc):
    """One gas at a fixed saturation index (EQUILIBRIUM_PHASES): PHREEQC's pure-gas phi at P = 10**SI."""
    text = (f"SOLUTION 1\n temp {tc}\nEND\n"
            f"SELECTED_OUTPUT 1\n -reset false\n -high_precision true\n"
            f"USER_PUNCH 1\n -headings tc phi si\n 10 PUNCH TC, PR_PHI(\"{gas}\"), SI(\"{gas}\")\n"
            f"USE SOLUTION 1\nEQUILIBRIUM_PHASES 1\n {gas} {log_p} 10\nEND\n")
    row = run(text)[-1]
    return {"id": cid, "kind": "fixed_si", "gas": gas, "log_p": log_p, "input": text, "tc": row["tc"],
            "phi": row["phi"], "si": row["si"]}


def main():
    cases = []
    for tc in (25.0, 60.0, 100.0):
        for P in (1.0, 10.0, 50.0, 100.0, 200.0):
            cases.append(gas_phase_case(f"co2_P{P:g}_T{tc:g}", "pressure", P, tc, {"CO2(g)": P}))
    for tc in (25.0, 80.0):
        for P in (1.0, 20.0, 100.0, 200.0):
            cases.append(gas_phase_case(f"mix_P{P:g}_T{tc:g}", "pressure", P, tc,
                                        {"CO2(g)": 0.4 * P, "CH4(g)": 0.4 * P, "N2(g)": 0.2 * P, "H2O(g)": 0.0}))
    for tc in (25.0, 80.0):
        for V in (20.0, 1.0, 0.2):
            cases.append(gas_phase_case(f"mixV_V{V:g}_T{tc:g}", "volume", V, tc,
                                        {"CO2(g)": 10.0, "CH4(g)": 10.0, "N2(g)": 5.0, "H2O(g)": 0.0}))
    for gas in ("CO2(g)", "CH4(g)"):
        for log_p in (-3.5, 0.0, 1.0, 1.7, 2.0):
            cases.append(pure_case(f"si_{gas}_{log_p:g}", gas, log_p, 25.0))
    # the IPhreeqc library bundled with phreeqpython 1.6.2 (its wrapper exposes no version call)
    ver = importlib.metadata.version("phreeqpython")
    data = {"version": f"PHREEQC 3.8.7-17149 (IPhreeqc) via phreeqpython {ver}", "database": DB.name,
            "cases": cases}
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()

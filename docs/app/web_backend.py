"""Browser backend for the discopt-speciation web app (runs in Pyodide).

The page loads Pyodide, the WebAssembly build of the pounce NLP solver, and
the pure-Python sources of discopt and ``discopt.speciation``. The solve path
(discopt's tape evaluator and pounce) needs neither JAX nor discopt's compiled
extension, so the calculation is exactly the one the Python package does.

This module exposes JSON functions the page calls from a Web Worker:
database contents, equilibrium, sweeps, predominance/Pourbaix grids and
log K(T). Sweeps and grids warm-start each point from the previous one.
"""

from __future__ import annotations

import json
import math
import re
from functools import reduce

import numpy as np

from discopt.speciation import PhreeqcDatabase, available_databases
from discopt.speciation import solve as S
from discopt.speciation.species import LN10, R_GAS

F_CONST = 96485.33212


# --------------------------------------------------------------------------- warm starts
_orig_run = S.run
_WARM = {"on": False, "key": None, "x": None, "active": None}


def _run(net, act, method, options=None, eps_seq=S.DEFAULT_EPS):
    """``solve.run``, starting from the previous state of the same system when sweeping."""
    L = S.layout(net, act)
    key = (L.n, tuple(L.Nph), method)
    if _WARM["on"] and _WARM["key"] == key and method != "gibbs":
        opts = dict(S.NLP_OPTIONS, **(options or {}))
        info = S.SolveInfo(method=method)
        info.notes.append("warm start from the previous point")
        try:
            x, active, ok = S.active_set_polish(net, act, set(_WARM["active"]), _WARM["x"], opts, info)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            _WARM.update(x=np.asarray(x, float), active=set(active))
            return x, active, ok, info
    x, active, ok, info = _orig_run(net, act, method, options, eps_seq)
    if _WARM["on"] and ok:
        _WARM.update(key=key, x=np.asarray(x, float), active=set(active))
    return x, active, ok, info


S.run = _run


class warm_starts:
    def __enter__(self):
        _WARM.update(on=True, key=None, x=None, active=None)

    def __exit__(self, *exc):
        _WARM.update(on=False, key=None, x=None, active=None)


# --------------------------------------------------------------------------- formulas
_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def formula_counts(formula: str) -> dict:
    """Atoms per formula unit: ``"Fe2(OH)2+4"`` -> ``{"Fe": 2, "O": 2, "H": 2}``; hydrates ``:2H2O`` included."""
    f = re.sub(r"\((?:[a-z]+|g|s|aq|am|a|c|d)\)$", "", formula.strip())  # phase tags: (g), (a), (s), (aq)
    f = re.sub(r"(?<=[\w)\]])[+-]\d*(\.\d+)?$", "", f)  # trailing charge
    total: dict = {}
    for part in f.split(":"):
        mm = re.match(r"^(\d+\.?\d*)(.*)$", part)
        mult, body = (float(mm.group(1)), mm.group(2)) if mm else (1.0, part)
        for k, v in _parse_group(body).items():
            total[k] = total.get(k, 0.0) + mult * v
    return {k: v for k, v in total.items() if v}


def _parse_group(s: str) -> dict:
    stack = [{}]
    i = 0
    while i < len(s):
        c = s[i]
        if c in "([":
            stack.append({})
            i += 1
        elif c in ")]":
            i += 1
            mm = re.match(r"\d+\.?\d*", s[i:])
            mult = float(mm.group(0)) if mm else 1.0
            i += len(mm.group(0)) if mm else 0
            top = stack.pop()
            for k, v in top.items():
                stack[-1][k] = stack[-1].get(k, 0.0) + mult * v
        else:
            mm = _ELEMENT.match(s, i)
            if not mm or not mm.group(0):
                i += 1
                continue
            num = mm.group(2)
            stack[-1][mm.group(1)] = stack[-1].get(mm.group(1), 0.0) + (float(num) if num not in ("", ".") else 1.0)
            i = mm.end()
    while len(stack) > 1:
        top = stack.pop()
        for k, v in top.items():
            stack[-1][k] = stack[-1].get(k, 0.0) + v
    return stack[0]


# --------------------------------------------------------------------------- JSON helpers
_DB: dict = {}


def _db(name) -> PhreeqcDatabase:
    if name not in _DB:
        _DB[name] = PhreeqcDatabase.load(name)
    return _DB[name]


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x) for x in v]
    if isinstance(v, (bool, np.bool_, str)) or v is None:
        return bool(v) if isinstance(v, np.bool_) else v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return str(v)


def _dumps(obj) -> str:
    return json.dumps(_clean(obj))


# --------------------------------------------------------------------------- JSON API
def ready() -> str:
    import discopt

    try:
        import pounce

        pounce_version = getattr(pounce, "__version__", "?")
    except ImportError:
        pounce_version = None
    return _dumps({"ok": True, "databases": available_databases(), "numpy": np.__version__,
                   "discopt": getattr(discopt, "__version__", "?"), "pounce": pounce_version})


def databases() -> str:
    return _dumps(available_databases())


def database_info(name: str) -> str:
    """Elements (with valence states) and phases of a database."""
    db = _db(name)
    skip = {"H", "H(0)", "H(1)", "H(+1)", "O", "O(0)", "O(-2)", "E", "Alkalinity"}
    elements: dict = {}
    for m, d in db.masters.items():
        if m in skip or d["species"] in ("H2O", "e-", "H+"):
            continue
        base = m.split("(")[0]
        elements.setdefault(base, {"name": base, "states": [], "gfw": None, "master": None})
        if "(" in m:
            elements[base]["states"].append(m)
        else:
            elements[base]["master"] = d["species"]
            elements[base]["gfw"] = d["gfw"]
    phases = []
    for nm, ent in db.phases.items():
        formula = ent.options.get("formula", nm)
        phases.append({"name": nm, "gas": ent.gas, "formula": formula, "elements": sorted(formula_counts(formula))})
    return _dumps({"name": name, "activity": type(db.default_activity()).__name__,
                   "elements": sorted(elements.values(), key=lambda e: e["name"]), "phases": phases,
                   "exchangers": sorted(db.exchange_masters)})


def system_info(cfg_json: str) -> str:
    """Aqueous species, master species and compatible phases for the chosen components."""
    cfg = json.loads(cfg_json)
    db = _db(cfg["database"])
    s = db.system(cfg["components"], phases=(), activity="ideal")
    names = {sp.name for sp in s.species}
    compat = []
    for nm, ent in db.phases.items():
        if all(o in names for o in ent.others()):
            compat.append({"name": nm, "gas": ent.gas, "formula": ent.options.get("formula", nm), "logK": ent.logK})
    aq = [{"name": sp.name, "charge": sp.charge} for sp in s.species if sp.phase == "aqueous"]
    return _dumps({"aqueous": aq, "phases": compat, "masters": dict(s.masters),
                   "has_electron": any(sp.phase == "electron" for sp in s.species)})


def _system(cfg):
    db = _db(cfg["database"])
    phases = list(cfg.get("solids") or []) + list(cfg.get("gases") or [])
    return db.system(cfg["components"], phases=phases, activity=cfg.get("activity") or "auto")


def _conditions(cfg):
    kw: dict = {"amounts": {k: float(v) for k, v in (cfg.get("amounts") or {}).items() if v not in (None, "")}}
    fixed = {k: float(v) for k, v in (cfg.get("fixed") or {}).items() if v not in (None, "")}
    if fixed:
        kw["fixed"] = fixed
    for key in ("pH", "pe"):
        if cfg.get(key) not in (None, ""):
            kw[key] = float(cfg[key])
    if cfg.get("T_C") not in (None, ""):
        kw["T"] = float(cfg["T_C"]) + 273.15
    if cfg.get("charge_balance"):
        kw["charge_balance"] = cfg["charge_balance"]
    if cfg.get("method"):
        kw["method"] = cfg["method"]
    return kw


def _with(cfg, assignments):
    """A copy of ``cfg`` with ``pH``, ``pe``, ``T_C``, ``amount:<key>`` or ``fixed:<species>`` set."""
    c = dict(cfg)
    c["amounts"], c["fixed"] = dict(cfg.get("amounts") or {}), dict(cfg.get("fixed") or {})
    for var, val in assignments:
        if var.startswith("amount:"):
            c["amounts"][var[7:]] = val
        elif var.startswith("fixed:"):
            c["fixed"][var[6:]] = val
        else:
            c[var] = val
    return c


def _equilibrate(s, cfg):
    kw = _conditions(cfg)
    amounts = kw.pop("amounts")
    return s.equilibrate(amounts, **kw)


def equilibrate(cfg_json: str) -> str:
    cfg = json.loads(cfg_json)
    s = _system(cfg)
    r = _equilibrate(s, cfg)
    d = r.to_dict()
    d.pop("gibbs_energy", None)
    d["pe"] = r.pe
    d["species_charge"] = {sp.name: sp.charge for sp in s.species if sp.phase == "aqueous"}
    d["activity_model"] = type(s.activity).__name__
    d["summary"] = r.summary()
    return _dumps(d)


def sweep(cfg_json: str, progress=None) -> str:
    """Equilibria along one variable: ``pH``, ``pe``, ``T_C``, ``amount:<key>`` or ``fixed:<species>``."""
    cfg = json.loads(cfg_json)
    s = _system(cfg)
    var, values = cfg["variable"], [float(v) for v in cfg["values"]]
    out = []
    with warm_starts():
        for i, val in enumerate(values):
            try:
                r = _equilibrate(s, _with(cfg, [(var, val)]))
                out.append({"converged": r.converged, "pH": r.pH, "pe": r.pe, "I": r.ionic_strength,
                            "molality": r.molality, "log10_activity": r.log10_activity, "solids": r.solids,
                            "SI": r.SI, "partial_pressure": r.partial_pressure, "gases": r.gases,
                            "present": r.present})
            except Exception as exc:  # noqa: BLE001
                out.append({"converged": False, "error": str(exc)})
            if progress is not None:
                progress(i + 1, len(values))
    return _dumps({"variable": var, "values": values, "points": out})


def predominance(cfg_json: str, progress=None) -> str:
    """Dominant species or solid of one element on a grid of two variables (pH-pe for Pourbaix).

    Candidates are weighted by the number of atoms of the element they hold,
    as in :func:`discopt.speciation.plot.predominance`.
    """
    cfg = json.loads(cfg_json)
    s = _system(cfg)
    el = cfg["element"]
    names = {sp.name: sp for sp in s.species}
    cands = {}
    for nm, sp in names.items():
        if sp.phase in ("aqueous", "solid"):
            c = formula_counts(nm).get(el, 0.0)
            if c:
                cands[nm] = c
    labels = list(cands)
    xs, ys = [float(v) for v in cfg["x_values"]], [float(v) for v in cfg["y_values"]]
    xv, yv = cfg["x"], cfg["y"]
    grid = []
    total, k = len(xs) * len(ys), 0
    with warm_starts():
        for j, yy in enumerate(ys):
            vals = [-1] * len(xs)
            order = range(len(xs)) if j % 2 == 0 else range(len(xs) - 1, -1, -1)  # serpentine: neighbours warm-start
            for i in order:
                try:
                    r = _equilibrate(s, _with(cfg, [(xv, xs[i]), (yv, yy)]))
                    if r.converged:
                        amt = [cands[nm] * max(r.molality.get(nm, 0.0), r.solids.get(nm, 0.0)) for nm in labels]
                        if amt and max(amt) > 0:
                            vals[i] = int(np.argmax(amt))
                except Exception:  # noqa: BLE001
                    pass
                k += 1
                if progress is not None:
                    progress(k, total)
            grid.append(vals)
    T = float(cfg.get("T_C") or 25.0) + 273.15
    return _dumps({"labels": labels, "phase": {nm: names[nm].phase for nm in labels}, "x": xs, "y": ys,
                   "grid": grid, "x_label": xv, "y_label": yv, "eh_per_pe": R_GAS * T * LN10 / F_CONST})


def logk_table(cfg_json: str) -> str:
    """log10 K(T) of the chosen reactions (as written in the database) over a temperature range."""
    cfg = json.loads(cfg_json)
    s = _system(cfg)
    T = [float(t) + 273.15 for t in cfg["T_C"]]
    wanted = set(cfg.get("phases") or [])
    out = {}
    for r in s.reactions:
        if not wanted or r.name in wanted:
            vals = []
            for t in T:
                try:
                    vals.append(r.logK_at(t))
                except Exception:  # noqa: BLE001
                    vals.append(None)
            out[r.name] = vals
    return _dumps({"T_C": [t - 273.15 for t in T], "logK": out})

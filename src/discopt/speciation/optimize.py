"""Equilibrium-constrained optimization: design, and worst-case bounds under uncertainty.

Both problems put the equilibrium conditions of :mod:`discopt.speciation.solve`
(mass action, conservation, complementarity) on one discopt model, with some
inputs promoted to variables:

* input amounts of reagents (``b = b0 + A e d``: conservation is linear in the dose);
* ``log10 K`` at 25 °C of chosen reactions (shifting ``logK(T)`` by the same amount);
* fixed log activities (``pH``, ``pe``, ``log10 P`` of a gas reservoir).

Two methods:

``"local"``
    Multi-start. Each start is an ordinary :meth:`System.equilibrate` at a
    parameter point, then a smoothed Fischer-Burmeister homotopy with the
    objective and constraints, then a smooth NLP per phase assemblage
    ("piece": present phases saturated with ``N >= 0``, absent phases
    ``N = 0`` with ``SI <= 0``), flipping phases that sit at a boundary while
    that improves the objective. A KKT point, not a proof.

``"global"``
    Complementarity as a disjunction (:func:`discopt.mpec.reformulate_gdp`) and
    discopt's spatial branch and bound, warm-started from the local answer.
    Variable bounds are tightened first (see :meth:`_Problem.variable_bounds`).
    A result is reported ``certified`` only when discopt returns status
    ``optimal`` with a certified gap *and* every variable bound used is
    rigorous. Otherwise the bound is returned as ``conditional_bound``, valid
    only under the listed assumptions.

Every answer is re-computed with an ordinary :meth:`System.equilibrate` at the
optimal parameters; disagreements are reported in ``verification``.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

import discopt.modeling as dm
from discopt.speciation import solve as S
from discopt.speciation import water as WP
from discopt.speciation.activity import EXPR, NUM
from discopt.speciation.species import T_REF

__all__ = [
    "Amount", "LogK", "LogActivity", "Quantity",
    "molality", "log10_molality", "log10_activity", "pH", "solid", "saturation_index",
    "dissolved", "log10_dissolved", "precipitated", "dissolved_fraction", "precipitated_fraction", "parameter",
    "design", "worst_case_bounds", "corner_bounds", "monte_carlo_bounds", "evaluate",
    "OptimizationResult", "BoundsResult",
]

FB_EPS = (1e-2, 1e-4, 1e-6)
SCREEN_TOL = 1e-7  # residual tolerance for accepting an NLP point as a candidate (verified afterwards)
VERIFY_TOL = 1e-5  # relative (log quantities: absolute) agreement between the model and equilibrate


# =========================================================================== parameters
@dataclass(eq=False)
class Amount:
    """Input amount (mol) of a reagent. ``composition`` maps species/element names to mol per mol of reagent,
    e.g. ``Amount("NaOH", {"Na": 1, "OH-": 1})``; default ``{name: 1}``."""

    name: str
    composition: dict | None = None


@dataclass(eq=False)
class LogK:
    """``log10 K`` at 25 °C of a reaction (a :class:`Reaction`, its name or its equation)."""

    reaction: object


@dataclass(eq=False)
class LogActivity:
    """Fixed ``log10`` activity (or ``log10 P`` of a gas) of a species, as in ``equilibrate(fixed=...)``."""

    species: str


@dataclass
class _Param:
    kind: str  # amount | logK | pH | pe | fixed
    label: str
    lo: float
    hi: float
    nominal: float
    comp: dict | None = None
    reaction: object = None
    species: str | None = None


def _find_reaction(system, key):
    if not isinstance(key, str):
        for r in system.reactions:
            if r is key:
                return r
        raise ValueError(f"reaction {key!r} is not part of this system")
    hits = [r for r in system.reactions if r.name == key or r.equation() == key]
    if not hits:
        raise KeyError(f"no reaction named {key!r}")
    return hits[0]


def _is_species_key(system, key):
    return key in system.masters or key in getattr(system, "_species", {}) or key in getattr(system, "exchange_sites", {})


def _make_param(system, key, rng, default, reagents):
    reagents = reagents or {}
    obj = key
    if isinstance(key, str):
        if key in ("pH", "pe"):
            obj = key
        elif key in reagents:
            obj = Amount(key, reagents[key])
        elif default == "amount" and _is_species_key(system, key):
            obj = Amount(key)
        else:
            try:
                obj = LogK(_find_reaction(system, key))
            except KeyError:
                if _is_species_key(system, key):
                    obj = Amount(key)
                else:
                    raise KeyError(f"{key!r} is neither a species/element nor a reaction of the system") from None
    elif hasattr(key, "stoich"):
        obj = LogK(key)

    def box(nominal=None):
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
        elif nominal is not None and isinstance(rng, (int, float)):
            lo, hi = nominal - abs(float(rng)), nominal + abs(float(rng))
        else:
            raise ValueError(f"range for {key!r} must be (lo, hi){' or a +-delta' if nominal is not None else ''}")
        if not hi >= lo:
            raise ValueError(f"empty range {rng!r} for {key!r}")
        return lo, hi

    if obj in ("pH", "pe"):
        lo, hi = box()
        return _Param(obj, obj, lo, hi, 0.5 * (lo + hi))
    if isinstance(obj, Amount):
        lo, hi = box()
        if lo < 0 or hi <= 0:
            raise ValueError(f"amount range for {obj.name!r} must satisfy 0 <= lo and hi > 0")
        comp = dict(obj.composition or {obj.name: 1.0})
        return _Param("amount", obj.name, lo, hi, 0.5 * (lo + hi), comp=comp)
    if isinstance(obj, LogK):
        r = _find_reaction(system, obj.reaction)
        nominal = float(r.logK_at(T_REF))
        lo, hi = box(nominal)
        label = key if isinstance(key, str) else (r.name or r.equation())
        return _Param("logK", str(label), lo, hi, nominal, reaction=r)
    if isinstance(obj, LogActivity):
        lo, hi = box()
        return _Param("fixed", f"log10a[{obj.species}]", lo, hi, 0.5 * (lo + hi), species=obj.species)
    raise TypeError(f"cannot interpret parameter {key!r}")


@contextmanager
def _shifted(shifts):
    """Temporarily shift ``log10 K(25 °C)`` of reactions (analytic reactions: the constant term)."""
    saved = []
    try:
        for r, d in shifts:
            saved.append((r, r.logK, r.analytic))
            if r.analytic is not None:
                r.analytic = (r.analytic[0] + d,) + tuple(r.analytic[1:])
            else:
                r.logK = r.logK + d
        yield
    finally:
        for r, lk, an in reversed(saved):
            r.logK, r.analytic = lk, an


# =========================================================================== element counts
_ELEMENTS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
)
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\))(\d*\.?\d*)")


def _parse_group(s):
    stack, pos = [{}], 0
    while pos < len(s):
        mt = _TOKEN.match(s, pos)
        if mt is None:
            return None
        if mt.group(1):
            sym = mt.group(1)
            if sym not in _ELEMENTS:
                return None
            n = float(mt.group(2)) if mt.group(2) else 1.0
            stack[-1][sym] = stack[-1].get(sym, 0.0) + n
        elif mt.group(3):
            stack.append({})
        else:
            if len(stack) == 1:
                return None
            n = float(mt.group(5)) if mt.group(5) else 1.0
            inner = stack.pop()
            for k, c in inner.items():
                stack[-1][k] = stack[-1].get(k, 0.0) + n * c
        pos = mt.end()
    return stack[0] if len(stack) == 1 and stack[0] else None


def _parse_formula(name):
    """Element counts from a formula-like name (``Al(OH)2+``, ``NdPO4:10H2O``, ``ThO2(am)``); ``None`` if not a formula."""
    s = re.sub(r"\([a-z][a-z0-9,\-]*\)", "", name.strip())
    s = re.sub(r"(?:[+-]\d*|\++|-+)$", "", s)
    out: dict = {}
    for part in s.split(":"):
        mt = re.match(r"^(\d+\.?\d*)?(.*)$", part)
        mult = float(mt.group(1)) if mt.group(1) else 1.0
        g = _parse_group(mt.group(2))
        if g is None:
            return None
        for k, c in g.items():
            out[k] = out.get(k, 0.0) + mult * c
    return out


_COUNT_CACHE: dict = {}


def _composition(system, name, _seen=None):
    key = (id(system), name)
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key]
    sp = system._resolve(name)
    comp = getattr(sp, "composition", None)
    if isinstance(comp, dict) and comp:
        out = {k: float(v) for k, v in comp.items()}
    elif sp.phase == "electron":
        out = {}
    else:
        out = _parse_formula(sp.name)
    if out is None:  # e.g. a mineral name: derive from its reaction (balanced, so elements cancel)
        seen = set() if _seen is None else _seen
        seen.add(sp.name)
        for r in system.reactions:
            if sp not in r.stoich:
                continue
            acc, ok = {}, True
            for o, c in r.stoich.items():
                if o is sp:
                    continue
                if o.name in seen:
                    ok = False
                    break
                co = _composition(system, o.name, seen)
                if co is None:
                    ok = False
                    break
                for k, n in co.items():
                    acc[k] = acc.get(k, 0.0) - c * n / r.stoich[sp]
            if ok:
                out = {k: v for k, v in acc.items() if abs(v) > 1e-12}
                break
    _COUNT_CACHE[key] = out
    return out


def _count(system, name, element):
    comp = _composition(system, name)
    if comp is None:
        raise ValueError(f"cannot determine the composition of {name!r}; pass explicit weights instead of an element")
    return comp.get(element, 0.0)


# =========================================================================== quantities
class _ModelView:
    """Quantities as expressions on a model (``o=EXPR``) or numbers at a model point (``o=NUM``)."""

    def __init__(self, prob, net, v, st, theta, o, ybounds=None):
        self.prob, self.system, self.net, self.v, self.st, self.theta, self.o = prob, prob.system, net, v, st, theta, o
        self.idx = {s.name: k for k, s in enumerate(net.species)}
        self.ypos = {k: i for i, k in enumerate(net.yidx)}
        self.aqpos = {k: j for j, k in enumerate(net.aq_idx)}
        self.ybounds_ = ybounds

    def aq_names(self):
        return [self.net.species[k].name for k in self.net.aq_idx]

    def solid_names(self):
        return [self.net.species[k].name for p, ph in enumerate(self.net.phases) if ph.kind in ("pure", "ss")
                for k in ph.members]

    def has_aq(self, name):
        return self.idx.get(name) in self.aqpos

    def molality(self, name):
        k = self.idx.get(name)
        if k is None:
            return 0.0
        if k not in self.aqpos:
            raise ValueError(f"{name!r} is not an aqueous species")
        return self.st["m"][self.aqpos[k]]

    def log10_molality(self, name):
        k = self.idx.get(name)
        if k not in self.aqpos:
            raise ValueError(f"{name!r} is not an aqueous species of this network")
        return self.v["y"][self.ypos[k]]

    def ybounds(self, name):
        i = self.ypos[self.idx[name]]
        return float(self.ybounds_[0][i]), float(self.ybounds_[1][i])

    def log10_activity(self, name):
        return self.st["la"][self.idx[name]]

    def amount(self, name):
        k = self.idx.get(name)
        return 0.0 if k is None else self.st["n"].get(k, 0.0)

    def saturation(self, name):
        return self.st["la_eq"][self.idx[name]]

    def water_mass(self):
        return self.v["W"]

    def param(self, label):
        return self.theta[self.prob.label_index(label)]

    def log10(self, x):
        return dm.log10(x) if self.o is EXPR else math.log10(x)


class _ResultView:
    def __init__(self, system, res, decisions):
        self.system, self.res, self.decisions, self.o = system, res, decisions or {}, NUM

    def aq_names(self):
        return list(self.res.molality)

    def solid_names(self):
        return list(self.res.solids)

    def has_aq(self, name):
        return name in self.res.molality

    def molality(self, name):
        return self.res.molality.get(name, 0.0)

    def log10_molality(self, name):
        m = self.res.molality.get(name, 0.0)
        return math.log10(m) if m > 0 else -math.inf

    def log10_activity(self, name):
        return self.res.log10_activity[name]

    def amount(self, name):
        r = self.res
        if name in r.molality:
            return r.molality[name] * r.water_mass
        for d in (r.solids, r.gases, r.exchange, r.surface):
            if name in d:
                return d[name]
        return 0.0

    def saturation(self, name):
        return self.res.SI[name]

    def water_mass(self):
        return self.res.water_mass

    def param(self, label):
        return self.decisions[label]

    def log10(self, x):
        return math.log10(x) if x > 0 else -math.inf


class Quantity:
    """A scalar function of the equilibrium state (and parameters), usable as objective or constraint.

    Build with :func:`molality`, :func:`dissolved`, :func:`pH`, ... and combine with ``+ - * /``.
    ``q.value(result)`` evaluates it on an :class:`~discopt.speciation.system.Result`.
    """

    def __init__(self, fn, label, *, terms=None, log=False, ratio=None):
        self._fn, self.label, self._terms, self.log, self.ratio = fn, label, terms, log, ratio

    def expr(self, view):
        return self._fn(view)

    def value(self, result, decisions=None):
        system = result._state["system"] if result._state else None
        return float(self._fn(_ResultView(system, result, decisions)))

    def lsum(self, view):
        """``[(log10 c_i, log10 m_i)]`` if this is a positive sum of molalities (or its log10), else ``None``."""
        if self._terms is None:
            return None
        return [(math.log10(c), view.log10_molality(name)) for c, name in self._terms(view) if c > 0 and view.has_aq(name)]

    def __repr__(self):
        return f"Quantity({self.label})"

    @staticmethod
    def _lift(x):
        return x if isinstance(x, Quantity) else Quantity(lambda v, c=float(x): c, repr(float(x)))

    def __add__(self, other):
        o = Quantity._lift(other)
        terms = None
        if self._terms is not None and o._terms is not None and not self.log and not o.log:
            a, b = self._terms, o._terms
            terms = lambda v: a(v) + b(v)  # noqa: E731
        return Quantity(lambda v: self._fn(v) + o._fn(v), f"({self.label} + {o.label})", terms=terms)

    __radd__ = __add__

    def __sub__(self, other):
        o = Quantity._lift(other)
        return Quantity(lambda v: self._fn(v) - o._fn(v), f"({self.label} - {o.label})")

    def __rsub__(self, other):
        return Quantity._lift(other) - self

    def __mul__(self, other):
        o = Quantity._lift(other)
        terms = None
        if not isinstance(other, Quantity) and float(other) > 0 and self._terms is not None and not self.log:
            t, c = self._terms, float(other)
            terms = lambda v: [(c * w, n) for w, n in t(v)]  # noqa: E731
        return Quantity(lambda v: self._fn(v) * o._fn(v), f"{self.label}*{o.label}", terms=terms)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = Quantity._lift(other)
        if not isinstance(other, Quantity):
            return self * (1.0 / float(other))
        return Quantity(lambda v: self._fn(v) * (1.0 / 1.0) / o._fn(v), f"{self.label}/{o.label}", ratio=(self, o))

    def __neg__(self):
        return Quantity(lambda v: -1.0 * self._fn(v), f"-{self.label}")


def molality(species) -> Quantity:
    """Molality (mol/kg water) of an aqueous species."""
    return Quantity(lambda v: v.molality(species), f"m[{species}]", terms=lambda v: [(1.0, species)])


def log10_molality(species) -> Quantity:
    return Quantity(lambda v: v.log10_molality(species), f"log10 m[{species}]", terms=lambda v: [(1.0, species)], log=True)


def log10_activity(species) -> Quantity:
    return Quantity(lambda v: v.log10_activity(species), f"log10 a[{species}]")


def pH() -> Quantity:
    return Quantity(lambda v: -1.0 * v.log10_activity("H+"), "pH")


def solid(name) -> Quantity:
    """Amount (mol) of a solid (phase member) or gas."""
    return Quantity(lambda v: v.amount(name), f"n[{name}]")


def saturation_index(name) -> Quantity:
    return Quantity(lambda v: v.saturation(name), f"SI[{name}]")


def _weights(what):
    if isinstance(what, dict):
        w = {k: float(c) for k, c in what.items()}
        return lambda v: [(c, n) for n, c in w.items() if v.has_aq(n)], "+".join(w)
    el = str(what)
    if el not in _ELEMENTS:
        raise ValueError(f"{el!r} is not an element symbol; pass a {{species: weight}} dict")
    return lambda v: [(c, n) for n in v.aq_names() if (c := _count(v.system, n, el)) > 0], el


def dissolved(what) -> Quantity:
    """Total dissolved element (mol/kg water): ``sum_i nu_i m_i`` over aqueous species, or explicit ``{species: weight}``."""
    terms, lab = _weights(what)
    return Quantity(lambda v: v.o.sum([c * v.molality(n) for c, n in terms(v)]), f"dissolved[{lab}]", terms=terms)


def log10_dissolved(what) -> Quantity:
    terms, lab = _weights(what)
    return Quantity(lambda v: v.log10(v.o.sum([c * v.molality(n) for c, n in terms(v)])), f"log10 dissolved[{lab}]",
                    terms=terms, log=True)


def _dissolved_mol(what):
    terms, lab = _weights(what)
    return Quantity(lambda v: v.water_mass() * v.o.sum([c * v.molality(n) for c, n in terms(v)]), f"mol dissolved[{lab}]")


def precipitated(element) -> Quantity:
    """Moles of an element in solids."""
    el = str(element)
    return Quantity(lambda v: v.o.sum([c * v.amount(n) for n in v.solid_names() if (c := _count(v.system, n, el)) > 0]),
                    f"precipitated[{el}]")


def dissolved_fraction(element) -> Quantity:
    d, p = _dissolved_mol(element), precipitated(element)
    q = d / (d + p)
    q.label = f"dissolved_fraction[{element}]"
    return q


def precipitated_fraction(element) -> Quantity:
    d, p = _dissolved_mol(element), precipitated(element)
    q = p / (d + p)
    q.label = f"precipitated_fraction[{element}]"
    return q


def parameter(label) -> Quantity:
    """A decision or uncertain parameter by its label (e.g. the reagent name)."""
    return Quantity(lambda v: v.param(label), str(label))


def _as_quantity(q, prob=None):
    if isinstance(q, Quantity):
        return q
    if isinstance(q, str) and prob is not None and q in prob.labels:
        return parameter(q)
    if callable(q):
        return Quantity(q, getattr(q, "__name__", "objective"))
    raise TypeError(f"cannot use {q!r} as a quantity")


# =========================================================================== results
@dataclass
class OptimizationResult:
    """Outcome of :func:`design` (and each side of :func:`worst_case_bounds`).

    ``status``: ``"optimal"`` (certified global optimum), ``"infeasible"``
    (certified: no feasible point), ``"local_optimum"``, ``"feasible"``
    (global search ran out of time; incumbent only), ``"no_feasible_point_found"``
    (local search; *not* a proof of infeasibility).
    """

    status: str
    certified: bool
    method: str
    decisions: dict
    objective: float | None  # recomputed with equilibrate
    objective_model: float | None
    bound: float | None = None  # certified dual bound (objective units)
    gap: float | None = None
    conditional_bound: float | None = None  # dual bound valid only under ``assumptions``
    result: object = None
    constraints: list = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    wall_time: float = 0.0
    info: dict = field(default_factory=dict)


@dataclass
class BoundsResult:
    """Range of a quantity over a parameter box.

    ``lower``/``upper`` are attained values (verified with equilibrate at
    ``argmin``/``argmax``): the true range contains ``[lower, upper]``.
    ``lower_bound``/``upper_bound`` are certified outer bounds (the true range
    is inside them) and are ``None`` unless proved.
    """

    quantity: str
    method: str
    lower: float | None
    upper: float | None
    argmin: dict | None
    argmax: dict | None
    lower_bound: float | None = None
    upper_bound: float | None = None
    certified: bool = False
    minimize: OptimizationResult | None = None
    maximize: OptimizationResult | None = None
    samples: np.ndarray | None = None
    wall_time: float = 0.0
    assumptions: list = field(default_factory=list)


# =========================================================================== problem
class _Problem:
    def __init__(self, system, amounts, params, eq_kwargs):
        self.system, self.params = system, params
        self.base = dict(amounts)
        self.kw = dict(eq_kwargs)
        self.labels = [p.label for p in params]
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"duplicate parameter labels {self.labels}")
        for p in params:
            if p.kind in ("pH", "pe") and self.kw.get(p.kind) is not None:
                raise ValueError(f"{p.kind} is both a parameter and a fixed input")
        self.theta0 = np.array([p.nominal for p in params], float)
        self.lo = np.array([p.lo for p in params], float)
        self.hi = np.array([p.hi for p in params], float)
        amounts0, kw0, _ = self.inputs(self.theta0)
        self.net, self.act = system._network(amounts0, **self._net_kw(kw0))
        self.L = S.layout(self.net, self.act)
        if self.L.X:
            raise NotImplementedError("non-ideal solid solutions are not supported by discopt.speciation.optimize")
        if self.L.G:
            raise NotImplementedError("Peng-Robinson gas phases are not supported by discopt.speciation.optimize; "
                                      "use gas_eos='ideal' (System) or eos='ideal' (gas phase)")
        net = self.net
        self.idx = {s.name: k for k, s in enumerate(net.species)}
        cpos = {k: j for j, k in enumerate(net.cols)}
        sid = {id(s): k for k, s in enumerate(net.species)}
        self.Ae = np.zeros((net.A.shape[0], len(params)))
        self.notes = []
        for i, p in enumerate(params):
            if p.kind == "amount":
                e = np.zeros(len(net.cols))
                for sp, c in system._amounts(p.comp).items():
                    k = sid.get(id(sp))
                    if k is None:
                        raise ValueError(f"reagent {p.label!r}: species {sp.name!r} is not in the network")
                    if k not in cpos:
                        self.notes.append(f"reagent {p.label!r}: {sp.name} has a fixed activity; its amount has no effect")
                        continue
                    e[cpos[k]] += c
                self.Ae[:, i] = net.A @ e
            elif p.kind in ("pH", "pe", "fixed"):
                name = {"pH": "H+", "pe": "e-"}.get(p.kind, p.species)
                k = self.idx.get(system._resolve(name).name)
                if k is None or k not in net.fixed:
                    raise ValueError(f"parameter {p.label!r}: {name} is not a fixed-activity species of the network")
                if k in net.fixed_logphi:
                    raise NotImplementedError(f"parameter {p.label!r}: {name} is a Peng-Robinson gas, whose fugacity "
                                              "is not linear in log10 P; use gas_eos='ideal'")
                p.species = net.species[k].name
            elif p.kind == "logK":
                if not any(rx.reaction is p.reaction for rx in list(net.rxns) + list(net.member_rxn.values())):
                    raise ValueError(f"reaction {p.label!r} is not used by this calculation")
        rows = {int(n.split("_")[1]) for n, _, _ in self._numeric_equations(self.theta0, S.initial_point(net, self.act))
                if n.startswith("conservation_")}
        for r in range(net.A.shape[0]):
            if r not in rows and np.any(self.Ae[r]):
                raise ValueError("a reagent changes a conservation law that no species can carry")

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _net_kw(kw):
        return {k: v for k, v in kw.items() if k not in ("method", "options")}

    def label_index(self, label):
        return self.labels.index(label)

    def inputs(self, theta):
        amounts = dict(self.base)
        kw = dict(self.kw)
        fixed = dict(kw.get("fixed") or {})
        shifts = []
        for p, t in zip(self.params, theta):
            t = float(t)
            if p.kind == "amount":
                for sp, c in p.comp.items():
                    amounts[sp] = amounts.get(sp, 0.0) + c * t
            elif p.kind == "logK":
                shifts.append((p.reaction, t - p.nominal))
            elif p.kind in ("pH", "pe"):
                kw[p.kind] = t
            else:
                fixed[p.species] = t
        if fixed:
            kw["fixed"] = fixed
        return amounts, kw, shifts

    def clip(self, theta):
        """``theta`` clipped to the parameter box.

        NLP solutions may sit a solver bound tolerance outside the box (observed: a dose of -1e-15 at lo = 0,
        6.00001e-3 at hi = 6e-3, logK 3.00000003 at hi = 3, -17.90000155 at lo = -17.9); those are clipped, but a
        point genuinely outside the box raises instead of being silently moved. The box is read from
        ``self.params`` so that problems built without ``__init__`` (see :func:`evaluate`) work too.
        """
        th = np.asarray(theta, float)
        lo = np.array([p.lo for p in self.params], float)
        hi = np.array([p.hi for p in self.params], float)
        slack = 1e-6 * (hi - lo) + 1e-7 * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
        if np.any(th < lo - slack) or np.any(th > hi + slack):
            raise ValueError(f"parameter point {th} lies outside the box [{lo}, {hi}]")
        return np.clip(th, lo, hi)

    def equilibrate_at(self, theta):
        theta = self.clip(theta)
        key = tuple(float(t) for t in theta)
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            amounts, kw, shifts = self.inputs(theta)
            with _shifted(shifts):
                cache[key] = self.system.equilibrate(amounts, **kw)
        return cache[key]

    def decisions(self, theta):
        # report the point that equilibrate verified (clipped to the box), not the raw NLP iterate
        return {p.label: float(t) for p, t in zip(self.params, self.clip(theta))}

    # ------------------------------------------------------------------ networks
    def net_with(self, theta, symbolic):
        base = self.net
        net = copy.copy(base)
        d = {id(p.reaction): (theta[i] - p.nominal) for i, p in enumerate(self.params) if p.kind == "logK"}

        def rep(rx):
            s = d.get(id(rx.reaction))
            return rx if s is None else dataclasses.replace(rx, logK=rx.logK + s)

        net.rxns = [rep(rx) for rx in base.rxns]
        net.member_rxn = {k: rep(rx) for k, rx in base.member_rxn.items()}
        fixed = dict(base.fixed)
        for i, p in enumerate(self.params):
            if p.kind in ("pH", "pe"):
                fixed[self.idx[p.species]] = (-1.0) * theta[i]
            elif p.kind == "fixed":
                fixed[self.idx[p.species]] = theta[i]
        net.fixed = fixed
        if not symbolic:
            net.b = base.b + self.Ae @ (np.asarray(theta, float) - self.theta0)
        return net

    def _numeric_equations(self, theta, x):
        net = self.net_with(theta, symbolic=False)
        v = S._vars(net, self.L, [float(t) for t in x])
        st = S.state(NUM, net, self.act, S._prepare(net, self.act), v)
        return S.equations(NUM, net, self.act, S._prepare(net, self.act), v, st)

    def numeric_view(self, theta, x):
        net = self.net_with(theta, symbolic=False)
        v = S._vars(net, self.L, [float(t) for t in x])
        st = S.state(NUM, net, self.act, S._prepare(net, self.act), v)
        return _ModelView(self, net, v, st, [float(t) for t in theta], NUM), net, v, st

    def x_from_result(self, res, bounds):
        """A point in this problem's layout from an equilibrate result (mapped by names)."""
        net, L = self.net, self.L
        st = res._state
        rnet, rx = st["net"], st["x"]
        rL = S.layout(rnet, st["act"])
        ridx = {s.name: k for k, s in enumerate(rnet.species)}
        rypos = {k: i for i, k in enumerate(rnet.yidx)}
        x = np.zeros(L.n)
        for i, k in enumerate(net.yidx):
            rk = ridx.get(net.species[k].name)
            x[i] = rx[rypos[rk]] if rk in rypos else bounds[0][i]
        rph = {ph.name: q for q, p in enumerate(rL.Nph) for ph in [rnet.phases[p]]}
        for q, p in enumerate(L.Nph):
            rq = rph.get(net.phases[p].name)
            x[L.ny + q] = rx[rL.ny + rq] if rq is not None else 0.0
        for name in ("W", "I", "delta"):
            a, b = getattr(L, name), getattr(rL, name)
            if a is not None:
                x[a] = rx[b] if b is not None else S.initial_point(net, self.act)[a]
        if L.psi is not None and rL.psi is not None and L.npsi == rL.npsi:
            x[L.psi : L.psi + L.npsi] = rx[rL.psi : rL.psi + rL.npsi]
        return np.clip(x, bounds[0], bounds[1])

    # ------------------------------------------------------------------ bounds
    def default_bounds(self):
        m = dm.Model("bounds")
        S.model_variables(m, self.net, self.act)
        lb, ub = [], []
        for var in m._variables:
            lb.append(np.broadcast_to(np.asarray(var.lb, float), var.shape).ravel())
            ub.append(np.broadcast_to(np.asarray(var.ub, float), var.shape).ravel())
        return np.concatenate(lb), np.concatenate(ub)

    def variable_bounds(self, log_span=None):
        """Tightened bounds on the layout variables, with the assumptions they rest on.

        Rigorous (up to a 1e-6 relative LP safety margin):

        * upper bounds on every amount from the LP ``max n_j  s.t.  A n = b(d), n >= 0``
          over the reagent box; hence ``log10 m <= log10(n_max / W_min)``, phase
          amounts, water mass, the charge-balance adjustment and ionic strength;
        * ``log10 m`` of a fixed-activity aqueous species with an ideal activity model.

        Not rigorous: ``log_span`` sets ``log10 m >= ub - log_span`` (the default
        lower bound is -200, which makes relaxations useless). It is listed in
        the returned assumptions, and a result using it is never ``certified``.
        """
        from scipy.optimize import linprog

        net, L, act = self.net, self.L, self.act
        lb, ub = self.default_bounds()
        notes, rigorous = [], True
        nc = len(net.cols)
        amt_i = [i for i, p in enumerate(self.params) if p.kind == "amount"]
        has_cb = L.delta is not None
        nv = nc + len(amt_i) + int(has_cb)
        Aeq = np.zeros((net.A.shape[0], nv))
        Aeq[:, :nc] = net.A
        for q, i in enumerate(amt_i):
            Aeq[:, nc + q] = -self.Ae[:, i]
        if has_cb:
            Aeq[:, -1] = -np.asarray(net.cb_col, float)
        beq = net.b - self.Ae[:, amt_i] @ self.theta0[amt_i] if amt_i else net.b.copy()
        vb = [(0.0, 0.0) if net.kind[k] == "virt" else (0.0, None) for k in net.cols]
        vb += [(self.lo[i], self.hi[i]) for i in amt_i]
        if has_cb:
            vb.append((float(lb[L.delta]), float(ub[L.delta])))
        # row scaling for the LP
        sc = np.maximum(np.abs(Aeq).max(axis=1), 1e-300)
        Aeq_s, beq_s = Aeq / sc[:, None], beq / sc

        def extreme(j, sign):
            c = np.zeros(nv)
            c[j] = sign
            r = linprog(c, A_eq=Aeq_s, b_eq=beq_s, bounds=vb, method="highs")
            return float(sign * r.fun) if r.status == 0 else None

        nmax = {}
        for j, k in enumerate(net.cols):
            if net.kind[k] == "virt":
                continue
            val = extreme(j, -1.0)
            if val is not None:
                nmax[k] = val * (1 + 1e-6) + 1e-12 * net.n_scale
        Wmin, Wmax = None, None
        if L.W is not None and net.water in net.cols:
            j = net.cols.index(net.water)
            lo_w, hi_w = extreme(j, 1.0), extreme(j, -1.0)
            if lo_w is not None:
                Wmin = max(float(lb[L.W]), lo_w * WP.M_W * (1 - 1e-6))
                lb[L.W] = Wmin
            if hi_w is not None:
                Wmax = min(float(ub[L.W]), hi_w * WP.M_W * (1 + 1e-6))
                ub[L.W] = Wmax
        elif L.W is None:
            Wmin = net.water_mass if net.water_mass is not None else net.W0
        if Wmin is None:
            Wmin = float(lb[L.W])
        if has_cb:
            dlo, dhi = extreme(nv - 1, 1.0), extreme(nv - 1, -1.0)
            if dlo is not None:
                lb[L.delta] = max(lb[L.delta], dlo - 1e-6 * abs(dlo) - 1e-12 * net.n_scale)
            if dhi is not None:
                ub[L.delta] = min(ub[L.delta], dhi + 1e-6 * abs(dhi) + 1e-12 * net.n_scale)
        mub = {}
        for i, k in enumerate(net.yidx):
            kd = net.kind[k]
            if k in nmax and kd in ("aq", "ex", "surf"):
                val = nmax[k] / (Wmin if kd == "aq" else 1.0)
                if val > 0:
                    ub[i] = min(ub[i], math.log10(val) + 1e-9)
                else:
                    ub[i] = min(ub[i], lb[i] + 1e-9)
            if kd == "aq":
                mub[k] = 10.0 ** ub[i]
        fixed_th = {self.idx[p.species]: (i, p) for i, p in enumerate(self.params) if p.kind in ("pH", "pe", "fixed")}
        for i, k in enumerate(net.yidx):
            if net.kind[k] == "aq" and k in net.fixed and act.is_ideal:
                if k in fixed_th:
                    q, p = fixed_th[k]
                    a, b = (-p.hi, -p.lo) if p.kind in ("pH", "pe") else (p.lo, p.hi)
                else:
                    a = b = float(net.fixed[k])
                lb[i], ub[i] = max(lb[i], a - 1e-9), min(ub[i], b + 1e-9)
                mub[k] = 10.0 ** ub[i]
        for q, p in enumerate(L.Nph):
            mem = net.phases[p].members
            if all(k in nmax for k in mem):
                ub[L.ny + q] = min(ub[L.ny + q], sum(nmax[k] for k in mem))
        if L.I is not None:
            z = {k: net.species[k].charge for k in net.aq_idx}
            if all(k in mub for k in net.aq_idx):
                ub[L.I] = min(ub[L.I], max(0.5 * sum(z[k] ** 2 * mub[k] for k in net.aq_idx) * (1 + 1e-6), lb[L.I]))
        if log_span is not None:
            raised = False
            for i, k in enumerate(net.yidx):
                if net.kind[k] in ("aq", "ex", "surf") and ub[i] - log_span > lb[i]:
                    lb[i] = ub[i] - log_span
                    raised = True
            if raised:
                rigorous = False
                notes.append(f"assumed log10 molality >= (rigorous upper bound) - {log_span:g} for every species")
        bad = np.flatnonzero(lb > ub)
        if len(bad):
            raise ValueError("variable bounds are inconsistent (the reagent box admits no nonnegative amounts)")
        return lb, ub, rigorous, notes

    def slack_intervals(self, bounds):
        """Rigorous enclosures of the complementarity slack ``s_p`` of each phase over the variable box.

        Needs an ideal activity model (``log10 a = log10 m``) and no surfaces; raises otherwise.
        """
        net, act, L = self.net, self.act, self.L
        if not act.is_ideal or net.surfaces:
            raise NotImplementedError("method='global' needs an ideal activity model and no surface complexation "
                                      "(rigorous activity-coefficient enclosures are not implemented)")
        lb, ub = bounds
        la = {}
        msum = 0.0
        for i, k in enumerate(net.yidx):
            kd = net.kind[k]
            off = 0.0
            if kd == "ex":
                sp = net.species[k]
                off = math.log10(sp.nsites / max(net.site_total.get(sp.site, 0.0), 1e-300))
            la[k] = (lb[i] + off, ub[i] + off)
            if kd == "aq":
                msum += 10.0 ** ub[i]
        if net.water is not None:
            if getattr(act, "water", None) == "phreeqc":
                lo = math.log(1.0 - 0.017 * msum) / math.log(10) if 0.017 * msum < 1 else -math.inf
            else:
                lo = -WP.M_W * msum * (1 + 1e-9) / math.log(10)
            la[net.water] = (lo, 0.0)
        pbox = {}
        for i, p in enumerate(self.params):
            if p.kind == "logK":
                pbox[id(p.reaction)] = (p.lo - p.nominal, p.hi - p.nominal)
        for k, val in net.fixed.items():
            if net.kind[k] == "const":
                la[k] = (float(val), float(val))
        out = {}
        for p in L.Nph:
            ph = net.phases[p]
            if ph.model is not None:
                raise NotImplementedError(f"method='global' does not support the non-ideal solid solution {ph.name!r}")
            leq = []
            for k in ph.members:
                rx = net.member_rxn[k]
                d = pbox.get(id(rx.reaction), (0.0, 0.0))
                lo, hi = rx.logK + d[0], rx.logK + d[1]
                for i, c in rx.nu.items():
                    if i == k:
                        continue
                    a, b = la.get(i, (-math.inf, math.inf))
                    lo, hi = lo - max(c * a, c * b), hi - min(c * a, c * b)
                lo, hi = sorted((lo / rx.nu[k], hi / rx.nu[k]))
                leq.append((lo, hi))
            if len(leq) == 1:
                sig = leq[0]
            else:
                sig = (max(a for a, _ in leq), max(b for _, b in leq) + math.log10(len(leq)))
            if ph.kind in ("pure", "ss"):
                sint = (-sig[1], -sig[0])
            else:
                sint = (math.log10(ph.pressure) - sig[1], math.log10(ph.pressure) - sig[0])
            if not (math.isfinite(sint[0]) and math.isfinite(sint[1])):
                raise NotImplementedError(f"cannot bound the saturation of phase {ph.name!r} rigorously")
            pad = 1e-9 * max(1.0, abs(sint[0]), abs(sint[1]))
            out[p] = (sint[0] - pad, sint[1] + pad)
        return out

    # ------------------------------------------------------------------ models
    def build(self, mode, objective, sense, constraints, bounds, scales, active=frozenset()):
        net, act, L = self.net, self.act, self.L
        m = dm.Model(f"speciation_{mode}")
        xs = S.model_variables(m, net, act)
        pos = 0
        for var in list(m._variables):
            size = int(np.prod(var.shape)) if var.shape else 1
            var.lb = np.asarray(bounds[0][pos : pos + size], float).reshape(var.shape)
            var.ub = np.asarray(bounds[1][pos : pos + size], float).reshape(var.shape)
            pos += size
        th = [m.continuous(f"theta_{i}", lb=float(p.lo), ub=float(p.hi)) for i, p in enumerate(self.params)]
        net2 = self.net_with(th, symbolic=True)
        ctx = S._prepare(net, act)
        v = S._vars(net2, L, xs)
        st = S.state(EXPR, net2, act, ctx, v)
        for name, expr, _ in S.equations(EXPR, net2, act, ctx, v, st):
            if name.startswith("conservation_"):
                r = int(name.split("_")[1])
                terms = [float(self.Ae[r, i]) * (th[i] - float(self.theta0[i])) for i in range(len(th)) if self.Ae[r, i]]
                if terms:
                    expr = expr - dm.sum(terms) * (1.0 / float(net.row_scale[r])) if hasattr(dm, "sum") and len(terms) > 1 \
                        else expr - terms[0] * (1.0 / float(net.row_scale[r]))
            m.subject_to(expr == 0, name=name)
        eps = None
        pairs = []
        if mode == "fb":
            eps = m.parameter("eps", value=FB_EPS[0])
        sbox = self.slack_intervals(bounds) if mode == "gdp" else None
        for p in L.Nph:
            a, s, lab = v["N"][p] * (1.0 / net.n_scale), st["s"][p], net.phases[p].name
            if mode == "fb":
                m.subject_to(a + s - dm.sqrt(a * a + s * s + 2.0 * eps * eps) == 0, name=f"fb_{lab}")
            elif mode == "piece":
                if p in active:
                    m.subject_to(s == 0, name=f"saturated_{lab}")
                else:
                    m.subject_to(v["N"][p] == 0, name=f"absent_{lab}")
                    m.subject_to(s >= 0, name=f"undersaturated_{lab}")
            elif mode == "gdp":
                from discopt.mpec import complementarity

                # lift s into a variable with a rigorous finite box so the big-M reformulation is exact
                u = m.continuous(f"slack_{p}", lb=float(max(sbox[p][0], 0.0 if sbox[p][1] >= 0 else sbox[p][0])),
                                 ub=float(sbox[p][1]))
                m.subject_to(u - s == 0, name=f"slack_def_{lab}")
                pairs.append(complementarity(a, u, name=f"cc_{lab}"))
        if pairs:
            from discopt.mpec import reformulate_gdp

            reformulate_gdp(m, pairs)
        view = _ModelView(self, net2, v, st, th, EXPR, ybounds=bounds)
        z, decode = None, None
        q = objective
        ls = q.lsum(view)
        if ls is not None and not ls:
            raise ValueError(f"objective {q.label} has no species in this network")
        if ls is not None and not (q.log and len(ls) == 1 and ls[0][0] == 0.0):
            names = [n for c, n in q._terms(view) if c > 0 and view.has_aq(n)]
            yb = [view.ybounds(n) for n in names]
            lcs = [lc for lc, _ in ls]
            zlo = max(lc + b[0] for lc, b in zip(lcs, yb))
            zhi = max(lc + b[1] for lc, b in zip(lcs, yb)) + math.log10(len(yb))
            z = m.continuous("z_objective", lb=zlo - 1e-9, ub=zhi + 1e-9)
            g = dm.sum([10.0 ** (y - z + lc) for lc, y in ls]) if len(ls) > 1 else 10.0 ** (ls[0][1] - z + ls[0][0])
            m.subject_to((g <= 1.0) if sense == "min" else (g >= 1.0), name="objective_epigraph")
            obj = z
            decode = (lambda val: val) if q.log else (lambda val: 10.0 ** val)
        else:
            sc = scales["objective"]
            obj = q.expr(view) * (1.0 / sc)
            decode = lambda val: val * sc  # noqa: E731
        (m.minimize if sense == "min" else m.maximize)(obj)
        for c, (cq, op, val) in enumerate(constraints):
            self._add_constraint(m, view, cq, op, val, scales["constraints"][c], f"spec_{c}")
        return dict(model=m, eps=eps, theta=th, z=z, decode=decode, n_layout=L.n)

    @staticmethod
    def _add_constraint(m, view, q, op, val, scale, name):
        def add(e, rhs):
            if op == "<=":
                m.subject_to(e <= rhs, name=name)
            elif op == ">=":
                m.subject_to(e >= rhs, name=name)
            else:
                m.subject_to(e == rhs, name=name)

        ls = q.lsum(view)
        if ls is not None and (q.log or val > 0):
            target = val if q.log else math.log10(val)
            if not ls:
                raise ValueError(f"constraint on {q.label}: no species in this network")
            if q.log and len(ls) == 1 and ls[0][0] == 0.0:
                add(ls[0][1], target)
                return
            g = dm.sum([10.0 ** (y + (lc - target)) for lc, y in ls]) if len(ls) > 1 else 10.0 ** (ls[0][1] + (ls[0][0] - target))
            add(g, 1.0)
        elif q.ratio is not None:
            num, den = q.ratio
            add((num.expr(view) - val * den.expr(view)) * (1.0 / scale), 0.0)
        else:
            add((q.expr(view) - val) * (1.0 / scale), 0.0)


# =========================================================================== helpers
def _spec_constraints(constraints, prob):
    out = []
    for c in constraints or ():
        if len(c) != 3 or c[1] not in ("<=", ">=", "=="):
            raise ValueError(f"constraint {c!r} must be (quantity, '<='|'>='|'==', value)")
        out.append((_as_quantity(c[0], prob), c[1], float(c[2])))
    return out


def _satisfied(q, op, val, achieved, scale):
    if achieved is None or not math.isfinite(achieved):
        return op == "<=" and achieved == -math.inf
    tol = VERIFY_TOL * (1.0 if q.log else max(abs(val), scale))
    if op == "<=":
        return achieved <= val + tol
    if op == ">=":
        return achieved >= val - tol
    return abs(achieved - val) <= tol


def _flat(m, n_layout, x_layout, theta, z):
    parts, pos = [], 0
    for var in m._variables:
        size = int(np.prod(var.shape)) if var.shape else 1
        if pos < n_layout:
            parts.append(np.asarray(x_layout[pos : pos + size], float))
        elif var.name.startswith("theta_"):
            parts.append(np.array([theta[int(var.name.split("_")[1])]], float))
        elif var.name == "z_objective" and z is not None:
            parts.append(np.array([z], float))
        else:
            lo = np.broadcast_to(np.asarray(var.lb, float), var.shape).ravel()
            hi = np.broadcast_to(np.asarray(var.ub, float), var.shape).ravel()
            parts.append(np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), np.where(np.isfinite(lo), lo, 0.0)))
        pos += size
    return np.concatenate(parts)


def _split(m, n_layout, flat, P):
    x = np.asarray(flat[:n_layout], float)
    theta = np.zeros(P)
    z = None
    pos = 0
    for var in m._variables:
        size = int(np.prod(var.shape)) if var.shape else 1
        if var.name.startswith("theta_"):
            theta[int(var.name.split("_")[1])] = flat[pos]
        elif var.name == "z_objective":
            z = float(flat[pos])
        pos += size
    return x, theta, z


class _Search:
    """Local search machinery shared by design and bounds."""

    def __init__(self, prob, objective, sense, constraints, bounds, scales, options):
        self.prob, self.q, self.sense, self.cons = prob, objective, sense, constraints
        self.bounds, self.scales = bounds, scales
        self.options = dict(S.NLP_OPTIONS) if options is None else dict(options)
        self.fb = prob.build("fb", objective, sense, constraints, bounds, scales)
        self.pieces = {}

    def objective_value(self, theta, x):
        view, *_ = self.prob.numeric_view(theta, x)
        return float(self.q.expr(view))

    def check(self, theta, x, active):
        prob, L = self.prob, self.prob.L
        view, net, v, st = prob.numeric_view(theta, x)
        ctx = S._prepare(net, prob.act)
        for _, val, _ in S.equations(NUM, net, prob.act, ctx, v, st):
            if not abs(val) <= SCREEN_TOL:
                return False
        for p in L.Nph:
            N, s = v["N"][p] / net.n_scale, st["s"][p]
            if p in active and not (abs(s) <= SCREEN_TOL and N >= -SCREEN_TOL):
                return False
            if p not in active and not (abs(N) <= SCREEN_TOL and s >= -SCREEN_TOL):
                return False
        for c, (q, op, val) in enumerate(self.cons):
            try:
                got = float(q.expr(view))
            except (ValueError, ZeroDivisionError):
                return False
            if not _satisfied(q, op, val, got, self.scales["constraints"][c]):
                return False
        return True

    def solve_piece(self, active, x, theta, z):
        key = frozenset(active)
        if key not in self.pieces:
            self.pieces[key] = self.prob.build("piece", self.q, self.sense, self.cons, self.bounds, self.scales, key)
        B = self.pieces[key]
        m = B["model"]
        r = S._local_solve(m, _flat(m, B["n_layout"], x, theta, z), self.options)
        if r.x is None or not np.all(np.isfinite(r.x)):
            return None
        xl, th, zz = _split(m, B["n_layout"], np.asarray(r.x, float), len(theta))
        if not self.check(th, xl, key):
            return None
        return dict(x=xl, theta=th, z=zz, active=set(key), obj=self.objective_value(th, xl))

    def better(self, a, b):
        if b is None:
            return True
        tol = 1e-9 * max(1.0, abs(b["obj"]))
        return a["obj"] < b["obj"] - tol if self.sense == "min" else a["obj"] > b["obj"] + tol

    def run_start(self, theta0, res0):
        prob, L = self.prob, self.prob.L
        x0 = prob.x_from_result(res0, self.bounds)
        z0 = None
        if self.fb["z"] is not None:
            try:
                val = self.q.value(res0, prob.decisions(theta0))
                z0 = val if self.q.log else (math.log10(val) if val > 0 else None)
            except (ValueError, KeyError):
                z0 = None
        m = self.fb["model"]
        flat = _flat(m, L.n, x0, theta0, z0)
        for e in FB_EPS:
            self.fb["eps"].value = float(e)
            r = S._local_solve(m, flat, self.options) if L.Nph else S._local_solve(m, flat, self.options)
            if r.x is not None and np.all(np.isfinite(r.x)):
                flat = np.asarray(r.x, float)
            if not L.Nph:
                break
        x, th, z = _split(m, L.n, flat, len(theta0))
        _, net, v, st = prob.numeric_view(th, x)
        guess = {p for p in L.Nph if v["N"][p] / net.n_scale > st["s"][p]}
        present = set(res0.present)
        start_active = {p for p in L.Nph if any(net.species[k].name in present for k in net.phases[p].members)}
        cur = None
        for act_set, xs, ts, zs in ((guess, x, th, z), (start_active, x0, theta0, z0)):
            c = self.solve_piece(act_set, xs, ts, zs)
            if c is not None and self.better(c, cur):
                cur = c
        if cur is None:
            return None
        for _ in range(2 * len(L.Nph) + 2):
            _, net, v, st = prob.numeric_view(cur["theta"], cur["x"])
            improved = False
            for p in L.Nph:
                at_edge = (v["N"][p] / net.n_scale < 1e-5) if p in cur["active"] else (st["s"][p] < 1e-5)
                if not at_edge:
                    continue
                c = self.solve_piece(cur["active"] ^ {p}, cur["x"], cur["theta"], cur["z"])
                if c is not None and self.better(c, cur):
                    cur, improved = c, True
                    break
            if not improved:
                break
        return cur


def _start_thetas(prob, n_starts, seed, extra=()):
    lo, hi, mid = prob.lo, prob.hi, 0.5 * (prob.lo + prob.hi)
    P = len(lo)
    pts = [mid]
    if P <= 3:
        pts += [np.where(np.array(bits, bool), hi, lo) for bits in itertools.product((0, 1), repeat=P)]
    rng = np.random.default_rng(seed)
    pts += [lo + rng.random(P) * (hi - lo) for _ in range(max(n_starts, 0))]
    pts += [np.asarray(e, float) for e in extra]
    out = []
    for p in pts:
        if not any(np.allclose(p, q) for q in out):
            out.append(np.asarray(p, float))
    return out


def _scales(prob, objective, constraints, res, theta):
    dec = prob.decisions(theta)
    sc = {"constraints": []}
    try:
        ov = objective.value(res, dec)
    except (ValueError, KeyError, ZeroDivisionError):
        ov = 1.0
    sc["objective"] = max(abs(ov), 1e-12) if math.isfinite(ov) else 1.0
    for q, op, val in constraints:
        if q.ratio is not None:
            try:
                den = abs(q.ratio[1].value(res, dec))
            except (ValueError, KeyError, ZeroDivisionError):
                den = 0.0
            sc["constraints"].append(max(den, 1e-12))
        else:
            try:
                got = abs(q.value(res, dec))
            except (ValueError, KeyError, ZeroDivisionError):
                got = 0.0
            sc["constraints"].append(max(abs(val), got if math.isfinite(got) else 0.0, 1e-12))
    return sc


def _verify(prob, objective, constraints, theta, obj_model, scales, active_model):
    res = prob.equilibrate_at(theta)
    dec = prob.decisions(theta)
    try:
        obj = objective.value(res, dec)
    except (ValueError, KeyError, ZeroDivisionError):
        obj = None
    rows = []
    for c, (q, op, val) in enumerate(constraints):
        try:
            got = q.value(res, dec)
        except (ValueError, KeyError, ZeroDivisionError):
            got = None
        rows.append(dict(quantity=q.label, op=op, value=val, achieved=got,
                         satisfied=bool(got is not None and _satisfied(q, op, val, got, scales["constraints"][c]))))
    ver = dict(converged=bool(res.converged), objective_model=obj_model, objective_equilibrate=obj,
               assemblage_model=sorted(active_model), assemblage_equilibrate=sorted(res.present))
    agree = res.converged and obj is not None and obj_model is not None
    if agree:
        tol = VERIFY_TOL * (1.0 if objective.log else max(abs(obj), abs(obj_model), 1e-300))
        agree = abs(obj - obj_model) <= tol or (not objective.log and abs(obj - obj_model) <= 1e-12 * scales["objective"])
    sig = {n for n, a in {**res.solids, **res.gases}.items() if a > 1e-8 * prob.net.n_scale} if res.converged else set()
    ver["significant_phases_equilibrate"] = sorted(sig)
    ver["agrees"] = bool(agree and set(active_model) == sig)
    if not ver["agrees"]:
        ver["note"] = "the equilibrium recomputed with equilibrate differs from the optimization model"
    return res, obj, rows, ver


def _phase_names(prob, active, theta=None, x=None, tol=1e-8):
    """Names of phases in ``active`` whose amount is not negligible (all of ``active`` if no point is given)."""
    net = prob.net
    if x is not None:
        _, _, v, _ = prob.numeric_view(theta, x)
        active = {p for p in active if v["N"][p] > tol * net.n_scale}
    return {net.species[k].name for p in active for k in net.phases[p].members}


def _optimize(prob, objective, sense, constraints, method, *, n_starts, seed, time_limit, gap_tolerance, log_span,
              options, extra_starts=(), start_results=None, starts=None):
    if sense not in ("min", "max"):
        raise ValueError("sense must be 'min' or 'max'")
    if method not in ("local", "global"):
        raise ValueError("method must be 'local' or 'global'")
    t0 = time.perf_counter()
    info = {"starts": 0, "local_candidates": 0}
    if starts is not None:  # explicit start points replace the defaults
        mid = 0.5 * (prob.lo + prob.hi)
        thetas = [np.array([float(st.get(lab, mid[i])) for i, lab in enumerate(prob.labels)]) for st in starts]
        start_results = None
    else:
        thetas = _start_thetas(prob, n_starts, seed, extra_starts)
    evals = list(start_results or [])
    for th in thetas:
        res = prob.equilibrate_at(th)
        if res.converged:
            evals.append((th, res))
    if not evals:
        raise RuntimeError("equilibrate failed at every start point")
    info["starts"] = len(evals)
    # local search uses only the rigorous bounds (the -200 log floor is harmless for an NLP)
    lb, ub, rigorous, notes = prob.variable_bounds(log_span if method == "global" else None)
    bounds = (lb, ub)
    if method == "global":
        prob.slack_intervals(bounds)  # fail early if the global model cannot be bounded
    scales = _scales(prob, objective, constraints, evals[0][1], evals[0][0])
    search = _Search(prob, objective, sense, constraints, bounds, scales, options)
    best = None
    for th, res in evals:
        try:
            c = search.run_start(th, res)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:  # a failed start is not fatal
            info.setdefault("start_errors", []).append(str(exc))
            continue
        if c is not None:
            info["local_candidates"] += 1
            if search.better(c, best):
                best = c
    assumptions = list(prob.notes)
    out = None
    if best is not None:
        res, obj, rows, ver = _verify(prob, objective, constraints, best["theta"], best["obj"], scales,
                                      _phase_names(prob, best["active"], best["theta"], best["x"]))
        out = OptimizationResult("local_optimum", False, "local", prob.decisions(best["theta"]), obj, best["obj"],
                                 result=res, constraints=rows, verification=ver, assumptions=assumptions, info=info)
    if method == "local":
        if out is None:
            out = OptimizationResult("no_feasible_point_found", False, "local", {}, None, None,
                                     assumptions=assumptions + ["local search: not a proof of infeasibility"], info=info)
        out.wall_time = time.perf_counter() - t0
        return out

    # ---------------------------------------------------------------- global
    B = prob.build("gdp", objective, sense, constraints, bounds, scales)
    m = B["model"]
    init = None
    if best is not None:
        flat = _flat(m, B["n_layout"], best["x"], best["theta"], best["z"])
        init, pos = {}, 0
        for var in m._variables:
            size = int(np.prod(var.shape)) if var.shape else 1
            init[var] = flat[pos : pos + size].reshape(var.shape) if var.shape else float(flat[pos])
            pos += size
    tg = time.perf_counter()
    try:
        r = m.solve(time_limit=time_limit, gap_tolerance=gap_tolerance, initial_solution=init)
    except ValueError as exc:
        if init is None or "expected length" not in str(exc):
            raise
        # discopt 0.8.0: the warm start is validated before the disjunctive reformulation appends
        # indicator variables and is then evaluated against the longer vector. Solve cold.
        info["warm_start_dropped"] = str(exc)
        r = m.solve(time_limit=max(time_limit - (time.perf_counter() - tg), 1.0), gap_tolerance=gap_tolerance)
    info["global"] = dict(status=r.status, objective=r.objective, bound=r.bound, gap=r.gap,
                          gap_certified=getattr(r, "gap_certified", None), nodes=getattr(r, "node_count", None),
                          wall_time=time.perf_counter() - tg)
    assumptions = assumptions + notes
    proven = rigorous
    if r.status == "infeasible":
        gl = OptimizationResult("infeasible" if proven else "no_feasible_point_found", proven, "global", {}, None, None,
                                assumptions=assumptions, info=info)
        if not proven:
            gl.assumptions.append("discopt proved infeasibility only within the assumed variable box")
        if out is not None:  # a verified local point exists: the box assumption cut it off
            out.method = "global"
            out.info["warning"] = "global solve reported infeasible but a verified local point exists (assumed bounds too tight)"
            out.assumptions = assumptions
            out.wall_time = time.perf_counter() - t0
            return out
        gl.wall_time = time.perf_counter() - t0
        return gl
    cand = out
    if r.x is not None and r.objective is not None:
        flat = np.concatenate([np.ravel(np.asarray(r.value(var), float)) for var in m._variables])
        xg, thg, _ = _split(m, B["n_layout"], flat, len(prob.params))
        objg = float(B["decode"](r.objective))
        _, net, v, st = prob.numeric_view(thg, xg)
        act_g = {p for p in prob.L.Nph if v["N"][p] / net.n_scale > st["s"][p]}
        res, obj, rows, ver = _verify(prob, objective, constraints, thg, objg, scales,
                                      _phase_names(prob, act_g, thg, xg))
        gcand = OptimizationResult("feasible", False, "global", prob.decisions(thg), obj, objg, result=res,
                                   constraints=rows, verification=ver, assumptions=assumptions, info=info)
        if cand is None or cand.objective is None or (obj is not None and all(rw["satisfied"] for rw in rows) and (
                (sense == "min" and obj < cand.objective) or (sense == "max" and obj > cand.objective))):
            cand = gcand
    if cand is None:
        out = OptimizationResult("no_feasible_point_found", False, "global", {}, None, None, assumptions=assumptions,
                                 info=info)
        out.wall_time = time.perf_counter() - t0
        return out
    cand.method = "global"
    cand.assumptions = assumptions
    cand.info = info
    if r.bound is not None and math.isfinite(r.bound):
        bnd = float(B["decode"](r.bound))
        certified = r.status == "optimal" and bool(getattr(r, "gap_certified", False))
        if certified and proven:
            cand.status, cand.certified, cand.bound, cand.gap = "optimal", True, bnd, r.gap
        else:
            cand.conditional_bound = bnd
            cand.gap = r.gap
            cand.status = "feasible" if cand.status != "local_optimum" else "local_optimum"
            if certified and not proven:
                cand.info["note"] = "discopt certified the gap, but only within assumed (non-rigorous) variable bounds"
    # a proof is about the model; if equilibrate disagrees the certificate does not transfer
    if cand.certified and not cand.verification.get("agrees", False):
        cand.info["warning"] = "certified model optimum does not match the recomputed equilibrium"
    cand.wall_time = time.perf_counter() - t0
    return cand


# =========================================================================== public API
def design(system, amounts, decisions, objective, sense="min", constraints=(), *, method="local", reagents=None,
           n_starts=4, seed=0, starts=None, time_limit=60.0, gap_tolerance=1e-4, log_span=None, options=None,
           **equilibrate_kwargs) -> OptimizationResult:
    """Choose inputs so an equilibrium state optimizes ``objective`` subject to ``constraints``.

    Parameters
    ----------
    system, amounts :
        As for :meth:`System.equilibrate`. Decision amounts are *added* to ``amounts``.
    decisions : dict
        ``{key: (lo, hi)}``. A key is a species/element name or reagent name
        (an input amount), ``"pH"``/``"pe"``, or an :class:`Amount`,
        :class:`LogK`, :class:`LogActivity`. Reagents with a composition can be
        given as ``reagents={"NaOH": {"Na": 1, "OH-": 1}}``.
    objective : Quantity, decision label, or callable(view) -> expression
    sense : ``"min"`` or ``"max"``
    constraints : list of ``(quantity, "<="|">="|"==", value)``
    method : ``"local"`` or ``"global"`` (see the module docstring)
    n_starts, seed, starts :
        Random start points (in addition to the box center and, for up to 3
        parameters, its corners), or an explicit list of ``{label: value}``
        start points that replaces them.
    log_span : float, optional
        Global method only: assumed width (log10 units) of every molality box
        below its rigorous upper bound. Makes relaxations useful; results
        using it are not certified.
    **equilibrate_kwargs :
        ``pH``, ``fixed``, ``activity``, ``T``, ``charge_balance``, ...
    """
    params = [_make_param(system, k, rng, "amount", reagents) for k, rng in decisions.items()]
    prob = _Problem(system, amounts, params, equilibrate_kwargs)
    q = _as_quantity(objective, prob)
    cons = _spec_constraints(constraints, prob)
    return _optimize(prob, q, sense, cons, method, n_starts=n_starts, seed=seed, time_limit=time_limit,
                     gap_tolerance=gap_tolerance, log_span=log_span, options=options, starts=starts)


def _linear_corners(prob, q, h=1e-3):
    """Corners predicted by a finite-difference linearization (min, max), and the nominal evaluation."""
    mid = 0.5 * (prob.lo + prob.hi)
    res0 = prob.equilibrate_at(mid)
    f0 = q.value(res0, prob.decisions(mid))
    g = np.zeros(len(mid))
    for i in range(len(mid)):
        step = h * max(prob.hi[i] - prob.lo[i], 1e-12)
        t = mid.copy()
        t[i] += step
        g[i] = (q.value(prob.equilibrate_at(t), prob.decisions(t)) - f0) / step
    return np.where(g > 0, prob.lo, prob.hi), np.where(g > 0, prob.hi, prob.lo), (mid, res0)


def worst_case_bounds(system, amounts, quantity, uncertain, *, method="global", reagents=None, n_starts=4, seed=0,
                      time_limit=60.0, gap_tolerance=1e-4, log_span=None, options=None,
                      **equilibrate_kwargs) -> BoundsResult:
    """Minimum and maximum of ``quantity`` over a box of uncertain inputs.

    ``uncertain`` maps reactions (objects, names or equations; strings resolve
    to reactions first) to ``(lo, hi)`` in log10 K at 25 °C or to ``+-delta``
    around the current value; ``"pH"``, amounts and :class:`LogActivity` are
    also accepted. See :func:`design` for the other arguments.
    """
    t0 = time.perf_counter()
    params = [_make_param(system, k, rng, "logK", reagents) for k, rng in uncertain.items()]
    prob = _Problem(system, amounts, params, equilibrate_kwargs)
    q = _as_quantity(quantity, prob)
    cmin, cmax, nominal = _linear_corners(prob, q)
    kw = dict(n_starts=n_starts, seed=seed, time_limit=time_limit, gap_tolerance=gap_tolerance, log_span=log_span,
              options=options)
    lo = _optimize(prob, q, "min", [], method, extra_starts=[cmin], start_results=[nominal], **kw)
    hi = _optimize(prob, q, "max", [], method, extra_starts=[cmax], start_results=[nominal], **kw)
    out = BoundsResult(q.label, method, lo.objective, hi.objective, lo.decisions or None, hi.decisions or None,
                       minimize=lo, maximize=hi, assumptions=sorted(set(lo.assumptions) | set(hi.assumptions)))
    if lo.certified and hi.certified:
        out.certified = True
    out.lower_bound = lo.bound if lo.certified else None
    out.upper_bound = hi.bound if hi.certified else None
    out.wall_time = time.perf_counter() - t0
    return out


def evaluate(system, amounts, quantity, values, *, reagents=None, **equilibrate_kwargs):
    """``(value, Result)`` of ``quantity`` at one parameter point ``values = {key: value}``."""
    params = [_make_param(system, k, (float(v), float(v)), "logK", reagents) for k, v in values.items()]
    for p, (k, v) in zip(params, values.items()):
        if p.kind == "logK":
            p.lo = p.hi = float(v)
    prob = _Problem.__new__(_Problem)
    prob.system, prob.params, prob.base, prob.kw = system, params, dict(amounts), dict(equilibrate_kwargs)
    prob.labels = [p.label for p in params]
    theta = np.array([float(v) for v in values.values()])
    res = prob.equilibrate_at(theta)
    q = _as_quantity(quantity, prob)
    return q.value(res, prob.decisions(theta)), res


def _sampled_bounds(system, amounts, quantity, uncertain, thetas_fn, method, reagents, equilibrate_kwargs):
    t0 = time.perf_counter()
    params = [_make_param(system, k, rng, "logK", reagents) for k, rng in uncertain.items()]
    prob = _Problem.__new__(_Problem)
    prob.system, prob.params, prob.base, prob.kw = system, params, dict(amounts), dict(equilibrate_kwargs)
    prob.labels = [p.label for p in params]
    prob.lo = np.array([p.lo for p in params])
    prob.hi = np.array([p.hi for p in params])
    q = _as_quantity(quantity, prob)
    vals, pts = [], []
    for th in thetas_fn(prob):
        res = prob.equilibrate_at(th)
        if not res.converged:
            continue
        vals.append(q.value(res, prob.decisions(th)))
        pts.append(th)
    if not vals:
        raise RuntimeError("equilibrate failed at every sample")
    vals = np.array(vals)
    i, j = int(np.argmin(vals)), int(np.argmax(vals))
    return BoundsResult(q.label, method, float(vals[i]), float(vals[j]), prob.decisions(pts[i]), prob.decisions(pts[j]),
                        samples=vals, wall_time=time.perf_counter() - t0,
                        assumptions=["sampling: an inner estimate of the range, never a bound"])


def corner_bounds(system, amounts, quantity, uncertain, *, reagents=None, max_dim=12, include_center=True,
                  **equilibrate_kwargs) -> BoundsResult:
    """Range of ``quantity`` over the ``2**d`` corners of the box (plus the center): the usual baseline."""
    if len(uncertain) > max_dim:
        raise ValueError(f"{len(uncertain)} parameters: 2**{len(uncertain)} corners exceeds max_dim={max_dim}")

    def pts(prob):
        out = [np.where(np.array(b, bool), prob.hi, prob.lo) for b in itertools.product((0, 1), repeat=len(prob.lo))]
        return out + ([0.5 * (prob.lo + prob.hi)] if include_center else [])

    return _sampled_bounds(system, amounts, quantity, uncertain, pts, "corners", reagents, equilibrate_kwargs)


def monte_carlo_bounds(system, amounts, quantity, uncertain, *, n_samples=1000, seed=0, reagents=None,
                       **equilibrate_kwargs) -> BoundsResult:
    """Range of ``quantity`` over uniform random samples of the box."""
    rng = np.random.default_rng(seed)

    def pts(prob):
        return [prob.lo + rng.random(len(prob.lo)) * (prob.hi - prob.lo) for _ in range(n_samples)]

    return _sampled_bounds(system, amounts, quantity, uncertain, pts, "monte_carlo", reagents, equilibrate_kwargs)

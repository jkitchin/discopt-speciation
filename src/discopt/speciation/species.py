"""Species and reaction algebra.

Species are lightweight domain objects (not discopt expressions) that overload
``+``, ``*`` and ``>>`` so a reaction network can be written as::

    s.reaction(M + L >> ML, logK=3.0)
    s.reaction(NL >> N + L, logK=-4.0)        # dissolution of a solid

Discopt expressions are produced only when a :class:`~discopt.speciation.System` is
solved.

Species phases
--------------
``aqueous``   solute; activity ``gamma m`` (molality scale)
``solvent``   water; activity from the activity model
``solid``     member of a pure phase or a solid solution
``gas``       member of a gas phase (activity = partial pressure, atm) or a fixed reservoir
``exchange``  ion-exchange species; Gaines-Thomas equivalent-fraction activity
``surface``   surface-complexation species; site mole fraction times the Boltzmann factor
``electron``  the aqueous electron ``e-`` (redox)

*Virtual* species (``e-``, exchange masters such as ``X-``) take part in
reactions and conservation laws but always have zero amount; their activity is
a free variable (``pe = -log10 a(e-)``).
"""

from __future__ import annotations

import math
import re

from discopt.speciation.water import P_REF, R_GAS, T_REF

LN10 = math.log(10.0)
PHASES = ("aqueous", "solvent", "solid", "gas", "exchange", "surface", "electron")
ATM_BAR = 1.01325

_CHARGE = re.compile(r"(\++|-+)(\d*(?:\.\d+)?)$")


def parse_charge(name: str) -> float:
    """Charge from a trailing sign suffix.

    ``"Ca+2"`` -> 2, ``"SO4-2"`` -> -2, ``"Na+"`` -> 1, ``"Ca++"`` -> 2, ``"CaSO4"`` -> 0.
    """
    m = _CHARGE.search(name)
    if not m:
        return 0
    signs, digits = m.group(1), m.group(2)
    sign = 1 if signs[0] == "+" else -1
    if digits:
        if len(signs) != 1:
            raise ValueError(f"ambiguous charge suffix in species name {name!r}")
        v = float(digits)
        return sign * (int(v) if v.is_integer() else v)
    return sign * len(signs)


class Species:
    """A chemical species in one phase.

    Parameters
    ----------
    name : str
        Unique label.
    phase : str
        One of :data:`PHASES`.
    charge : float
        Charge (aqueous, surface, virtual species); used by activity models,
        charge balance and electrostatics.
    params : dict, optional
        Activity-model parameters, e.g. ``{"a0": 4.08, "b": 0.082}`` (PHREEQC ``-gamma``).
    virtual : bool
        Zero-amount species (``e-``, ``X-``).
    site : str, optional
        Exchange or surface site this species occupies (``"X"``, ``"Hfo_w"``).
    nsites : float
        Number of exchange sites (``CaX2`` -> 2).
    """

    def __init__(self, name: str, phase: str, charge: float = 0.0, *, params=None, virtual=False,
                 site=None, nsites=1.0, composition=None):
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; choose from {PHASES}")
        self.name = name
        self.phase = phase
        self.charge = 0.0 if phase in ("solid", "gas", "solvent") else float(charge)
        self.params = dict(params or {})
        self.virtual = bool(virtual)
        self.site = site
        self.nsites = float(nsites)
        self.composition = dict(composition) if composition else None

    @property
    def is_solid(self) -> bool:
        return self.phase == "solid"

    @property
    def is_phase_member(self) -> bool:
        return self.phase in ("solid", "gas")

    def __mul__(self, coeff: float) -> "Term":
        return Term(self, float(coeff))

    __rmul__ = __mul__

    def __add__(self, other) -> "Side":
        return Side([Term(self, 1.0)]) + other

    def __rshift__(self, other) -> "Reaction":
        return Side([Term(self, 1.0)]) >> other

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.phase})"


class Term:
    """A ``(species, coefficient)`` pair produced by ``coeff * species``."""

    def __init__(self, species: Species, coeff: float = 1.0):
        self.species = species
        self.coeff = float(coeff)

    def __add__(self, other) -> "Side":
        return Side([self]) + other

    def __rshift__(self, other) -> "Reaction":
        return Side([self]) >> other


def _terms(obj) -> list[Term]:
    if isinstance(obj, Side):
        return list(obj.terms)
    if isinstance(obj, Term):
        return [obj]
    if isinstance(obj, Species):
        return [Term(obj, 1.0)]
    raise TypeError(f"cannot interpret {obj!r} as part of a reaction side")


class Side:
    """One side of a reaction: an additive collection of terms."""

    def __init__(self, terms: list[Term]):
        self.terms = list(terms)

    def __add__(self, other) -> "Side":
        return Side(self.terms + _terms(other))

    def __rshift__(self, other) -> "Reaction":
        return Reaction(_merge(self.terms), _merge(_terms(other)))


def _merge(terms: list[Term]) -> dict:
    out: dict = {}
    for t in terms:
        out[t.species] = out.get(t.species, 0.0) + t.coeff
    return out


def log_k(T: float, P: float = 1.0, *, logK=None, dH=0.0, dCp=0.0, analytic=None, dV=0.0) -> float:
    r"""Equilibrium constant at ``T`` (K) and ``P`` (atm).

    Precedence (as in PHREEQC): ``analytic`` coefficients
    :math:`A_1+A_2T+A_3/T+A_4\log_{10}T+A_5/T^2+A_6T^2`, otherwise
    :math:`\log K_r` with the enthalpy ``dH`` (J/mol) and constant heat-capacity
    ``dCp`` (J/mol/K) corrections. ``dV`` (cm3/mol) adds
    :math:`-\Delta V (P-P_r)/(RT\ln 10)`.
    """
    if analytic is not None:
        a = list(analytic) + [0.0] * (6 - len(analytic))
        val = a[0] + a[1] * T + a[2] / T + a[3] * math.log10(T) + a[4] / (T * T) + a[5] * T * T
    else:
        if logK is None:
            raise ValueError("reaction has neither logK nor analytic coefficients")
        val = float(logK)
        if T != T_REF:
            if dH:
                val -= dH / (R_GAS * LN10) * (1.0 / T - 1.0 / T_REF)
            if dCp:
                val += dCp / (R_GAS * LN10) * (T_REF / T - 1.0 + math.log(T / T_REF))
    if dV and P != 1.0:
        val -= dV * (P - 1.0) * ATM_BAR * 0.1 / (R_GAS * T * LN10)
    return float(val)


class Reaction:
    """A reversible reaction ``reactants <=> products`` with an equilibrium constant.

    ``logK`` is the base-10 equilibrium constant at ``T_REF`` for the reaction as
    written (products over reactants). Temperature/pressure dependence: ``dH``
    (J/mol, van 't Hoff), ``dCp`` (J/mol/K), ``analytic`` (PHREEQC ``-analytic``
    A1..A6, takes precedence) and ``dV`` (cm3/mol).
    """

    def __init__(self, reactants: dict, products: dict):
        self.reactants = dict(reactants)
        self.products = dict(products)
        self.logK: float | None = None
        self.dH: float = 0.0
        self.dCp: float = 0.0
        self.dV: float = 0.0
        self.analytic: tuple | None = None
        self.name: str | None = None

    @property
    def stoich(self) -> dict:
        """Net coefficients, products positive; species that cancel are dropped."""
        nu: dict = {}
        for s, c in self.reactants.items():
            nu[s] = nu.get(s, 0.0) - c
        for s, c in self.products.items():
            nu[s] = nu.get(s, 0.0) + c
        return {s: c for s, c in nu.items() if abs(c) > 1e-12}

    def species(self) -> set:
        return set(self.stoich)

    def logK_at(self, T: float, P: float = 1.0) -> float:
        """``logK`` at ``T`` (K) and ``P`` (atm); see :func:`log_k`."""
        if self.logK is None and self.analytic is None:
            raise ValueError(f"reaction {self.equation()!r} has no logK")
        return log_k(T, P, logK=self.logK, dH=self.dH, dCp=self.dCp, analytic=self.analytic, dV=self.dV)

    def equation(self) -> str:
        def side(d):
            return " + ".join(s.name if abs(c - 1) < 1e-12 else f"{c:g} {s.name}" for s, c in d.items()) or "0"

        if any(c < 0 for c in (*self.reactants.values(), *self.products.values())):
            # net-coefficient reactions (e.g. from a database): negative coefficients are reactants
            nu = self.stoich
            return f"{side({s: -c for s, c in nu.items() if c < 0})} = {side({s: c for s, c in nu.items() if c > 0})}"
        return f"{side(self.reactants)} = {side(self.products)}"

    def __repr__(self) -> str:
        k = "" if self.logK is None else f", logK={self.logK:g}"
        return f"Reaction({self.equation()!r}{k})"


__all__ = ["ATM_BAR", "P_REF", "PHASES", "R_GAS", "T_REF", "Reaction", "Side", "Species", "Term", "log_k", "parse_charge"]

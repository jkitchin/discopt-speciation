r"""Activity models for aqueous solutions.

A model supplies, for the aqueous (solute) species of a system,

* :math:`\ln\gamma_i(m, I)` per species,
* :math:`\ln a_w`, the water activity, and optionally
* :math:`g = G^{ex}/(RT\,w)`, the excess Gibbs energy per kg of water.

When a model has an excess Gibbs energy (``has_excess``), ``ln gamma`` are its
derivatives and the water activity follows from Gibbs-Duhem,

.. math:: \ln a_w = -M_w\Big(\sum_i m_i - g + \sum_i m_i\ln\gamma_i\Big),

so the model can also be used by Gibbs-energy minimization. Models with
species-specific ion sizes (Truesdell-Jones / WATEQ, B-dot) are not derivatives
of any G^ex; they use PHREEQC's :math:`a_w = 1 - 0.017\sum m_i` and support only
the mass-action solvers.

All formulas are written once against a small "ops" namespace so the same code
builds discopt expressions for the equilibrium model and evaluates numbers for
reporting and verification (:data:`NUM`, :data:`EXPR`).

Temperature enters through the Debye-Hueckel parameters (computed from water
properties when ``A``/``B`` are left as ``None``, see :mod:`discopt.speciation.water`)
and through temperature-dependent Pitzer parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import discopt.modeling as dm

from discopt.speciation import water as W
from discopt.speciation._etheta import J_and_xJprime

LN10 = math.log(10.0)


# --------------------------------------------------------------------------- ops
class _NumOps:
    sqrt = staticmethod(lambda v: math.sqrt(max(v, 0.0)))
    log = staticmethod(math.log)
    log10 = staticmethod(math.log10)
    exp = staticmethod(math.exp)
    sinh = staticmethod(math.sinh)

    @staticmethod
    def sum(xs):
        return math.fsum(xs)


class _ExprOps:
    sqrt = staticmethod(dm.sqrt)
    log = staticmethod(dm.log)
    log10 = staticmethod(dm.log10)
    exp = staticmethod(dm.exp)
    sinh = staticmethod(dm.sinh)

    @staticmethod
    def sum(xs):
        const, terms = 0.0, []
        for v in xs:
            if isinstance(v, (int, float, np.floating)):
                const += float(v)
            else:
                terms.append(v)
        if not terms:
            return const
        s = dm.sum(terms) if len(terms) > 1 else terms[0]
        return s + const if const else s


class _JaxOps:
    """JAX backend: lets ``state``/``equations`` be traced for exact Jacobians (``jax.jacfwd``)."""

    _FUNCS = ("sqrt", "log", "log10", "exp", "sinh")

    def __getattr__(self, name):
        if name in self._FUNCS:
            import jax.numpy as jnp

            return getattr(jnp, name)
        raise AttributeError(name)

    @staticmethod
    def sum(xs):
        total = 0.0
        for v in xs:
            total = total + v
        return total


NUM = _NumOps()
EXPR = _ExprOps()
JAX = _JaxOps()


@dataclass
class Context:
    """Per-solve data prepared by :meth:`ActivityModel.prepare`."""

    species: list
    z: np.ndarray
    T: float
    P: float
    A: float = 0.0  # Debye-Hueckel A (log10 units)
    B: float = 0.0  # Debye-Hueckel B (1/Angstrom)
    data: dict = field(default_factory=dict)


def _param(sp, key, overrides=None):
    if overrides and sp.name in overrides:
        v = overrides[sp.name]
        if isinstance(v, dict):
            return v.get(key)
        return {"a0": v[0], "b": v[1]}.get(key) if isinstance(v, (tuple, list)) else None
    return (getattr(sp, "params", None) or {}).get(key)


# --------------------------------------------------------------------------- base
class ActivityModel:
    name = "abstract"
    #: True for unit activity coefficients (convex Gibbs energy).
    is_ideal = False
    #: True if ln gamma depends on the ionic strength (the solver adds ``I``).
    uses_I = True
    #: True if the model derives from an excess Gibbs energy (Gibbs-Duhem water activity, method="gibbs").
    has_excess = False
    #: "gibbs-duhem" (requires has_excess) or "phreeqc" (a_w = 1 - 0.017 sum m)
    water = "gibbs-duhem"

    @property
    def needs_ionic_strength(self) -> bool:  # backwards-compatible alias
        return self.uses_I

    def dh_A(self, T, P=W.P_REF) -> float:
        A = getattr(self, "A", None)
        return W.debye_huckel_A(T, P) if A is None else float(A)

    def dh_B(self, T, P=W.P_REF) -> float:
        B = getattr(self, "B", None)
        return W.debye_huckel_B(T, P) if B is None else float(B)

    def prepare(self, species, T=W.T_REF, P=W.P_REF) -> Context:
        z = np.array([float(s.charge) for s in species])
        return Context(list(species), z, float(T), float(P), self.dh_A(T, P), self.dh_B(T, P))

    def ln_gamma(self, o, ctx: Context, m, I) -> list:
        raise NotImplementedError

    def excess_gibbs(self, o, ctx: Context, m, I):
        raise NotImplementedError(f"activity model {self.name!r} has no excess Gibbs energy")

    def ln_water_activity(self, o, ctx: Context, m, I, lng=None):
        msum = o.sum(m)
        if self.water == "phreeqc":
            return o.log(1.0 - 0.017 * msum)
        if lng is None:
            lng = self.ln_gamma(o, ctx, m, I)
        g = self.excess_gibbs(o, ctx, m, I)
        return -W.M_W * (msum - g + o.sum([mi * lg for mi, lg in zip(m, lng) if not _is_zero(lg)]))

    # numeric conveniences --------------------------------------------------
    def log10_gamma_values(self, species, m, T=W.T_REF, P=W.P_REF) -> np.ndarray:
        """``log10 gamma`` for ``species`` at molalities ``m`` (ionic strength computed)."""
        ctx = self.prepare(species, T, P)
        m = [float(v) for v in m]
        I = ionic_strength(NUM, ctx.z, m)
        return np.array(self.ln_gamma(NUM, ctx, m, I)) / LN10

    def to_spec(self):
        return self.name

    def __repr__(self):
        return f"{type(self).__name__}()"


def _is_zero(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and v == 0.0


def ionic_strength(o, z, m):
    return 0.5 * o.sum([float(zi * zi) * mi for zi, mi in zip(z, m) if zi])


def _dh_F(o, s, c):
    """``int_0^I sqrt(I')/(1+c sqrt(I')) dI' / 2`` expressed in ``s = sqrt(I)``."""
    if c == 0.0:
        return s * s * s / 3.0
    return s * s / (2.0 * c) - s / (c * c) + o.log(1.0 + c * s) / (c**3)


# --------------------------------------------------------------------------- simple models
class Ideal(ActivityModel):
    """Unit activity coefficients; ``ln a_w = -M_w sum m`` (ideal-dilute Gibbs-Duhem)."""

    name = "ideal"
    is_ideal = True
    uses_I = False
    has_excess = True

    def ln_gamma(self, o, ctx, m, I):
        return [0.0] * len(m)

    def excess_gibbs(self, o, ctx, m, I):
        return 0.0


class DebyeHuckel(ActivityModel):
    r"""Debye-Hueckel limiting law :math:`\log_{10}\gamma_i=-Az_i^2\sqrt I` (valid below ~1e-3 m)."""

    name = "debye-huckel"
    has_excess = True

    def __init__(self, A: float | None = None):
        self.A = A

    def ln_gamma(self, o, ctx, m, I):
        s = o.sqrt(I)
        return [(-LN10 * ctx.A * zi * zi) * s if zi else 0.0 for zi in ctx.z]

    def excess_gibbs(self, o, ctx, m, I):
        s = o.sqrt(I)
        return (-4.0 / 3.0 * LN10 * ctx.A) * I * s

    def to_spec(self):
        return {"model": self.name, **({"A": self.A} if self.A is not None else {})}

    def __repr__(self):
        return f"DebyeHuckel(A={self.A})"


class Davies(ActivityModel):
    r"""Davies equation :math:`\log_{10}\gamma_i=-A z_i^2(\sqrt I/(1+\sqrt I)-bI)`, neutral ``gamma = 1``.

    Reasonable to I ~ 0.1-0.5 m. ``A=None`` computes ``A(T)`` from water
    properties (0.5100 at 25 C); pass ``A=0.509`` for the common textbook value.
    Has an excess Gibbs energy, so ``method="gibbs"`` works.
    """

    name = "davies"
    has_excess = True

    def __init__(self, A: float | None = None, b: float = 0.3):
        self.A = None if A is None else float(A)
        self.b = float(b)

    def ln_gamma(self, o, ctx, m, I):
        s = o.sqrt(I)
        f = s / (1.0 + s) - self.b * I
        return [(-LN10 * ctx.A * zi * zi) * f if zi else 0.0 for zi in ctx.z]

    def excess_gibbs(self, o, ctx, m, I):
        s = o.sqrt(I)
        return (-2.0 * LN10 * ctx.A) * (I - 2.0 * s + 2.0 * o.log(1.0 + s) - 0.5 * self.b * I * I)

    def to_spec(self):
        d = {"model": "davies", "b": self.b}
        if self.A is not None:
            d["A"] = self.A
        return d

    def __repr__(self):
        return f"Davies(A={self.A}, b={self.b})"


class ExtendedDebyeHuckel(ActivityModel):
    r"""Extended Debye-Hueckel with one common ion size ``a0`` (Angstrom).

    :math:`\log_{10}\gamma_i=-Az_i^2\sqrt I/(1+Ba_0\sqrt I)`. With a common size
    the model has an excess Gibbs energy (unlike :class:`TruesdellJones`).
    """

    name = "extended-debye-huckel"
    has_excess = True

    def __init__(self, a0: float = 4.0, A: float | None = None, B: float | None = None):
        self.a0, self.A, self.B = float(a0), A, B

    def ln_gamma(self, o, ctx, m, I):
        s = o.sqrt(I)
        f = s / (1.0 + ctx.B * self.a0 * s)
        return [(-LN10 * ctx.A * zi * zi) * f if zi else 0.0 for zi in ctx.z]

    def excess_gibbs(self, o, ctx, m, I):
        return (-4.0 * LN10 * ctx.A) * _dh_F(o, o.sqrt(I), ctx.B * self.a0)

    def to_spec(self):
        d = {"model": self.name, "a0": self.a0}
        d.update({k: v for k, v in (("A", self.A), ("B", self.B)) if v is not None})
        return d


class TruesdellJones(ActivityModel):
    r"""PHREEQC's aqueous model (``phreeqc.dat``, ``wateq4f.dat``).

    * ions with an ion size ``a0`` (species parameter, PHREEQC ``-gamma a0 b``):
      :math:`\log_{10}\gamma=-Az^2\sqrt I/(1+Ba_0\sqrt I)+bI` (WATEQ / Truesdell-Jones);
    * ions without one: Davies with ``b = 0.3``;
    * neutral species: :math:`\log_{10}\gamma=bI` with the species' ``b``
      (a Setschenow coefficient), default ``neutral_b = 0.1``;
    * water: :math:`a_w = 1-0.017\sum m_i`.

    ``params`` optionally overrides species parameters: ``{"Na+": (4.08, 0.082)}``.
    """

    name = "truesdell-jones"
    water = "phreeqc"

    def __init__(self, params: dict | None = None, neutral_b: float = 0.1, A=None, B=None):
        self.params = dict(params or {})
        self.neutral_b = float(neutral_b)
        self.A, self.B = A, B

    def prepare(self, species, T=W.T_REF, P=W.P_REF):
        ctx = super().prepare(species, T, P)
        ctx.data["a0"] = [_param(s, "a0", self.params) for s in species]
        ctx.data["b"] = [_param(s, "b", self.params) for s in species]
        return ctx

    def ln_gamma(self, o, ctx, m, I):
        s = o.sqrt(I)
        out, davies = [], None
        for zi, a0, b in zip(ctx.z, ctx.data["a0"], ctx.data["b"]):
            if zi == 0:
                bb = self.neutral_b if b is None else float(b)
                out.append((LN10 * bb) * I if bb else 0.0)
            elif a0 is not None:
                v = (-LN10 * ctx.A * zi * zi) * s / (1.0 + (ctx.B * float(a0)) * s)
                out.append(v + (LN10 * float(b)) * I if b else v)
            else:
                if davies is None:
                    davies = s / (1.0 + s) - 0.3 * I
                out.append((-LN10 * ctx.A * zi * zi) * davies)
        return out

    def to_spec(self):
        d = {"model": self.name, "neutral_b": self.neutral_b}
        if self.params:
            d["params"] = {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()}
        return d


class BDot(ActivityModel):
    r"""B-dot (Helgeson 1969, LLNL/EQ3): :math:`\log_{10}\gamma=-Az^2\sqrt I/(1+Ba_0\sqrt I)+\dot B I`.

    Ion sizes from species parameter ``a0`` (default ``a0_default``); neutral
    species get :math:`\log_{10}\gamma = kI` with their ``b`` parameter
    (Setschenow), default 0. Water: PHREEQC's linear approximation.
    """

    name = "bdot"
    water = "phreeqc"

    def __init__(self, bdot: float = 0.041, a0_default: float = 4.0, A=None, B=None, params=None):
        self.bdot, self.a0_default, self.A, self.B = float(bdot), float(a0_default), A, B
        self.params = dict(params or {})

    def prepare(self, species, T=W.T_REF, P=W.P_REF):
        ctx = super().prepare(species, T, P)
        ctx.data["a0"] = [_param(s, "a0", self.params) for s in species]
        ctx.data["b"] = [_param(s, "b", self.params) for s in species]
        return ctx

    def ln_gamma(self, o, ctx, m, I):
        s = o.sqrt(I)
        out = []
        for zi, a0, b in zip(ctx.z, ctx.data["a0"], ctx.data["b"]):
            if zi == 0:
                out.append((LN10 * float(b)) * I if b else 0.0)
            else:
                a = self.a0_default if a0 is None else float(a0)
                out.append((-LN10 * ctx.A * zi * zi) * s / (1.0 + (ctx.B * a) * s) + (LN10 * self.bdot) * I)
        return out

    def to_spec(self):
        return {"model": self.name, "bdot": self.bdot, "a0_default": self.a0_default}


LLNL_TABLE = {  # LLNL_AQUEOUS_MODEL_PARAMETERS of llnl.dat (thermo.com.V8.R6)
    "temperatures": (0.01, 25.0, 60.0, 100.0, 150.0, 200.0, 250.0, 300.0),
    "dh_a": (0.4939, 0.5114, 0.5465, 0.5995, 0.6855, 0.7994, 0.9593, 1.218),
    "dh_b": (0.3253, 0.3288, 0.3346, 0.3421, 0.3525, 0.3639, 0.3766, 0.3925),
    "bdot": (0.0374, 0.041, 0.0438, 0.046, 0.047, 0.047, 0.034, 0.0),
    "co2_coefs": (-1.0312, 0.0012806, 255.9, 0.4445, -0.001606),
}


class LLNL(ActivityModel):
    r"""B-dot with the tabulated LLNL/EQ3 parameters, as PHREEQC applies it to ``llnl.dat``.

    ``A``, ``B`` and :math:`\dot B` are interpolated linearly in °C in the
    ``table`` (``LLNL_AQUEOUS_MODEL_PARAMETERS``; 0.01-300 °C). Per species,
    by the ``gamma`` parameter:

    * ``"llnl"`` (``-llnl_gamma a0``): ions
      :math:`\log_{10}\gamma=-Az^2\sqrt I/(1+Ba_0\sqrt I)+\dot B I`, neutral species 1;
    * ``"co2"`` (``-co2_llnl_gamma``): Drummond (1981),
      :math:`\ln\gamma=(c_0+c_1T+c_2/T)I-(c_3+c_4T)\,I/(I+1)`;
    * no flag: ions with an ``a0`` are treated as ``"llnl"``; other ions
      Davies (0.3); neutral species :math:`bI` with ``b`` default ``neutral_b``.

    Water: PHREEQC's :math:`a_w = 1-0.017\sum m`.
    """

    name = "llnl"
    water = "phreeqc"

    def __init__(self, table: dict | None = None, neutral_b: float = 0.1, params: dict | None = None):
        self.table = {k: tuple(float(x) for x in v) for k, v in {**LLNL_TABLE, **(table or {})}.items()}
        self.neutral_b = float(neutral_b)
        self.params = dict(params or {})

    def interpolate(self, T: float) -> tuple[float, float, float]:
        """``(A, B, bdot)`` at ``T`` (K)."""
        tc, ts = T - 273.15, self.table["temperatures"]
        if tc < ts[0] - 1e-9 or tc > ts[-1] + 1e-9:
            raise ValueError(f"T = {tc:.2f} C is outside the LLNL parameter range {ts[0]}-{ts[-1]} C")
        return tuple(float(np.interp(tc, ts, self.table[k])) for k in ("dh_a", "dh_b", "bdot"))

    def prepare(self, species, T=W.T_REF, P=W.P_REF):
        ctx = super().prepare(species, T, P)  # ctx.A (water properties) is PHREEQC's Davies A
        a, b, bdot = self.interpolate(float(T))
        c = self.table["co2_coefs"]
        d = ctx.data
        d["llnl"] = (a, b, bdot)
        d["co2"] = (c[0] + c[1] * T + c[2] / T, c[3] + c[4] * T)
        d["a0"] = [_param(s, "a0", self.params) for s in species]
        d["b"] = [_param(s, "b", self.params) for s in species]
        d["kind"] = [_param(s, "gamma", self.params) or ("llnl" if a0 is not None else None)
                     for s, a0 in zip(species, d["a0"])]
        return ctx

    def ln_gamma(self, o, ctx, m, I):
        d = ctx.data
        a, b_dh, bdot = d["llnl"]
        s = o.sqrt(I)
        out, davies = [], None
        for zi, a0, b, kind in zip(ctx.z, d["a0"], d["b"], d["kind"]):
            if kind == "co2":
                out.append(d["co2"][0] * I - d["co2"][1] * I / (I + 1.0))
            elif kind == "llnl":
                if zi == 0:
                    out.append(0.0)
                else:
                    v = (-LN10 * a * zi * zi) * s / (1.0 + (b_dh * float(a0 if a0 is not None else 4.0)) * s)
                    out.append(v + (LN10 * bdot) * I if bdot else v)
            elif zi == 0:
                bb = self.neutral_b if b is None else float(b)
                out.append((LN10 * bb) * I if bb else 0.0)
            else:
                if davies is None:
                    davies = s / (1.0 + s) - 0.3 * I
                out.append((-LN10 * ctx.A * zi * zi) * davies)
        return out

    def to_spec(self):
        d = {"model": self.name, "neutral_b": self.neutral_b}
        if self.table != {k: tuple(v) for k, v in LLNL_TABLE.items()}:
            d["table"] = {k: list(v) for k, v in self.table.items()}
        if self.params:
            d["params"] = self.params
        return d


class SIT(ActivityModel):
    r"""Specific ion interaction theory (Brønsted-Guggenheim-Scatchard).

    :math:`\log_{10}\gamma_i=-z_i^2\,A\sqrt I/(1+1.5\sqrt I)+\sum_j\varepsilon_{ij}m_j`
    with symmetric interaction coefficients ``eps={("Na+", "Cl-"): 0.03, ...}``
    (kg/mol). Has an excess Gibbs energy.
    """

    name = "sit"
    has_excess = True

    def __init__(self, eps: dict | None = None, A=None, rho: float = 1.5):
        self.eps = {tuple(sorted(k)): float(v) for k, v in (eps or {}).items()}
        self.A, self.rho = A, float(rho)

    def prepare(self, species, T=W.T_REF, P=W.P_REF):
        ctx = super().prepare(species, T, P)
        names = [s.name for s in species]
        pairs = []
        for i in range(len(names)):
            for j in range(i, len(names)):
                e = self.eps.get(tuple(sorted((names[i], names[j]))))
                if e:
                    pairs.append((i, j, e))
        ctx.data["pairs"] = pairs
        return ctx

    def ln_gamma(self, o, ctx, m, I):
        s = o.sqrt(I)
        D = ctx.A * s / (1.0 + self.rho * s)
        terms = [[] for _ in m]
        for i, j, e in ctx.data["pairs"]:
            terms[i].append(e * m[j])
            if j != i:
                terms[j].append(e * m[i])
        return [LN10 * ((-zi * zi) * D + o.sum(t)) if (zi or t) else 0.0 for zi, t in zip(ctx.z, terms)]

    def excess_gibbs(self, o, ctx, m, I):
        h = (-4.0 * ctx.A) * _dh_F(o, o.sqrt(I), self.rho)
        quad = o.sum([(0.5 if i == j else 1.0) * e * m[i] * m[j] for i, j, e in ctx.data["pairs"]])
        return LN10 * (h + quad)

    def to_spec(self):
        return {"model": self.name, "eps": [[a, b, v] for (a, b), v in self.eps.items()]}


# --------------------------------------------------------------------------- Pitzer
PITZER_KINDS = ("B0", "B1", "B2", "C0", "THETA", "LAMDA", "ZETA", "PSI")


class PitzerParameters:
    """Pitzer interaction parameters with PHREEQC's temperature dependence.

    Each entry holds up to six coefficients ``a0..a5``:

    ``P(T) = a0 + a1 (1/T - 1/Tr) + a2 ln(T/Tr) + a3 (T - Tr) + a4 (T^2 - Tr^2) + a5 (1/T^2 - 1/Tr^2)``
    """

    def __init__(self):
        self.data = {k: {} for k in PITZER_KINDS}

    def add(self, kind: str, names, coeffs) -> None:
        kind = kind.upper().replace("LAMBDA", "LAMDA")
        if kind not in self.data:
            raise ValueError(f"unknown Pitzer parameter kind {kind!r}")
        c = tuple(float(v) for v in (coeffs if isinstance(coeffs, (list, tuple)) else [coeffs]))
        self.data[kind][tuple(sorted(names))] = c + (0.0,) * (6 - len(c))

    @staticmethod
    def at(c, T: float) -> float:
        Tr = W.T_REF
        return (c[0] + c[1] * (1 / T - 1 / Tr) + c[2] * math.log(T / Tr) + c[3] * (T - Tr)
                + c[4] * (T * T - Tr * Tr) + c[5] * (1 / (T * T) - 1 / (Tr * Tr)))

    def get(self, kind, names, T):
        c = self.data[kind].get(tuple(sorted(names)))
        return None if c is None else self.at(c, T)

    def species(self) -> set:
        return {n for kind in self.data.values() for key in kind for n in key}

    def to_dict(self) -> dict:
        return {k: [[*key, *c] for key, c in v.items()] for k, v in self.data.items() if v}

    @classmethod
    def from_dict(cls, d: dict) -> "PitzerParameters":
        p = cls()
        for kind, rows in d.items():
            n = 3 if kind.upper() in ("ZETA", "PSI") else 2
            for row in rows:
                p.add(kind, row[:n], row[n:])
        return p


def _g(o, x):
    return 2.0 * (1.0 - (1.0 + x) * o.exp(-x)) / (x * x)


def _gp(o, x):
    return -2.0 * (1.0 - (1.0 + x + 0.5 * x * x) * o.exp(-x)) / (x * x)


class Pitzer(ActivityModel):
    r"""Pitzer (Harvie-Moller-Weare) ion-interaction model, as in PHREEQC's ``pitzer.dat``.

    Includes binary ``B0, B1, B2, C0``, mixing ``THETA`` with unsymmetric
    E-theta terms, ternary ``PSI``, neutral ``LAMDA`` and ``ZETA`` parameters.
    Water activity follows from the excess Gibbs energy (osmotic coefficient),
    so the model is thermodynamically consistent and ``method="gibbs"`` works.

    ``alpha`` values follow PHREEQC: ``alpha1`` = 1.4 when both charges are
    at least 2 (2.0 otherwise) and ``alpha2`` = 12. ``A_phi=None`` uses
    ``ln(10) A(T) / 3``.

    ``macinnes=True`` (PHREEQC's default) rescales individual-ion activity
    coefficients so that ``gamma(Cl-)`` equals the mean activity coefficient of
    KCl at the solution's ionic strength: ``ln gamma_i -> ln gamma_i - z_i D``.
    A charge-weighted shift cancels in every electroneutral reaction and in the
    water activity, so it only matters when an individual ion activity (pH) is
    fixed or reported. It applies when ``Cl-`` is present and K-Cl parameters exist.
    """

    name = "pitzer"
    has_excess = True
    b = 1.2

    def __init__(self, params: PitzerParameters | dict | None = None, A_phi: float | None = None, etheta: bool = True,
                 macinnes: bool = True):
        if isinstance(params, dict):
            params = PitzerParameters.from_dict(params)
        self.params = params or PitzerParameters()
        self.A_phi = A_phi
        self.etheta = bool(etheta)
        self.macinnes = bool(macinnes)

    @staticmethod
    def _alphas(zc, za):
        # PHREEQC pitzer.cpp: alpha1 = 1.4 when both |z| >= 2, else 2; alpha2 = 12 always
        # (validated against phreeqc for Ca-Cl, which has a B2 term, at 25 and 60 C)
        zc, za = abs(zc), abs(za)
        return (1.4 if (zc >= 2 and za >= 2) else 2.0), 12.0

    def prepare(self, species, T=W.T_REF, P=W.P_REF):
        ctx = super().prepare(species, T, P)
        p, names, z = self.params, [s.name for s in species], ctx.z
        cat = [i for i, v in enumerate(z) if v > 0]
        an = [i for i, v in enumerate(z) if v < 0]
        neu = [i for i, v in enumerate(z) if v == 0]
        ca = []
        for c in cat:
            for a in an:
                key = (names[c], names[a])
                b0, b1, b2, c0 = (p.get(k, key, T) for k in ("B0", "B1", "B2", "C0"))
                if all(v is None for v in (b0, b1, b2, c0)):
                    continue
                a1, a2 = self._alphas(z[c], z[a])
                C = (c0 or 0.0) / (2.0 * math.sqrt(abs(z[c] * z[a])))
                ca.append((c, a, b0 or 0.0, b1 or 0.0, b2 or 0.0, C, a1, a2))
        like = []  # (i, j, theta, unsymmetric?)
        for group in (cat, an):
            for x in range(len(group)):
                for y in range(x + 1, len(group)):
                    i, j = group[x], group[y]
                    th = p.get("THETA", (names[i], names[j]), T) or 0.0
                    uns = self.etheta and abs(z[i]) != abs(z[j])
                    if th or uns:
                        like.append((i, j, th, uns))
        psi = []
        for group, other in ((cat, an), (an, cat)):
            for x in range(len(group)):
                for y in range(x + 1, len(group)):
                    for k in other:
                        v = p.get("PSI", (names[group[x]], names[group[y]], names[k]), T)
                        if v:
                            psi.append((group[x], group[y], k, v))
        lam = []
        for n in neu:
            for i in range(len(names)):
                if z[i] == 0 and i < n:
                    continue
                v = p.get("LAMDA", (names[n], names[i]), T)
                if v:
                    lam.append((n, i, v))
        zeta = []
        for n in neu:
            for c in cat:
                for a in an:
                    v = p.get("ZETA", (names[n], names[c], names[a]), T)
                    if v:
                        zeta.append((n, c, a, v))
        Aphi = W.LN10 * ctx.A / 3.0 if self.A_phi is None else float(self.A_phi)
        kcl = None
        if self.macinnes and "Cl-" in names:
            b0, b1, b2, c0 = (p.get(k, ("K+", "Cl-"), T) for k in ("B0", "B1", "B2", "C0"))
            if not all(v is None for v in (b0, b1, b2, c0)):
                kcl = (names.index("Cl-"), b0 or 0.0, b1 or 0.0, b2 or 0.0, (c0 or 0.0) / 2.0)
        ctx.data.update(ca=ca, like=like, psi=psi, lam=lam, zeta=zeta, Aphi=Aphi, kcl=kcl)
        return ctx

    # E-theta(I) and its I-derivative for charges zi, zj (same sign)
    @staticmethod
    def _etheta(o, zi, zj, Aphi, s, I, cache):
        def Jx(k):
            if k not in cache:
                x = (6.0 * k * Aphi) * s
                cache[k] = J_and_xJprime(x)
            return cache[k]

        zi, zj = abs(zi), abs(zj)
        Jij, xJij = Jx(zi * zj)
        Jii, xJii = Jx(zi * zi)
        Jjj, xJjj = Jx(zj * zj)
        E = (zi * zj / 4.0) * (Jij - 0.5 * Jii - 0.5 * Jjj) / I
        Ep = -E / I + (zi * zj / 8.0) * (xJij - 0.5 * xJii - 0.5 * xJjj) / (I * I)
        return E, Ep

    def _common(self, o, ctx, m, I):
        d = ctx.data
        s = o.sqrt(I)
        Z = o.sum([abs(float(zi)) * mi for zi, mi in zip(ctx.z, m) if zi])
        cache = {}
        pairs = []
        for c, a, b0, b1, b2, C, a1, a2 in d["ca"]:
            x1, x2 = a1 * s, a2 * s
            B = b0 + (b1 * _g(o, x1) if b1 else 0.0) + (b2 * _g(o, x2) if b2 else 0.0)
            Bp = ((b1 * _gp(o, x1) if b1 else 0.0) + (b2 * _gp(o, x2) if b2 else 0.0)) / I
            pairs.append((c, a, B, Bp, C))
        like = []
        for i, j, th, uns in d["like"]:
            if uns:
                E, Ep = self._etheta(o, ctx.z[i], ctx.z[j], d["Aphi"], s, I, cache)
                like.append((i, j, th + E, Ep))
            else:
                like.append((i, j, th, 0.0))
        return s, Z, pairs, like

    def ln_gamma(self, o, ctx, m, I):
        d, z = ctx.data, ctx.z
        s, Z, pairs, like = self._common(o, ctx, m, I)
        Aphi, b = d["Aphi"], self.b
        fg = -Aphi * (s / (1.0 + b * s) + (2.0 / b) * o.log(1.0 + b * s))
        F = o.sum([fg] + [m[c] * m[a] * Bp for c, a, B, Bp, C in pairs if not _is_zero(Bp)]
                  + [m[i] * m[j] * Php for i, j, Ph, Php in like if not _is_zero(Php)])
        CC = o.sum([m[c] * m[a] * C for c, a, B, Bp, C in pairs if C])
        T = [[] for _ in m]
        for c, a, B, Bp, C in pairs:
            T[c].append(m[a] * (2.0 * B + (Z * C if C else 0.0)))
            T[a].append(m[c] * (2.0 * B + (Z * C if C else 0.0)))
        for i, j, Ph, Php in like:
            T[i].append(2.0 * m[j] * Ph)
            T[j].append(2.0 * m[i] * Ph)
        for i, j, k, v in d["psi"]:  # i, j like-charged; k opposite
            T[i].append(v * m[j] * m[k])
            T[j].append(v * m[i] * m[k])
            T[k].append(v * m[i] * m[j])
        for n, i, v in d["lam"]:
            if n == i:
                T[n].append(2.0 * v * m[n])
            else:
                T[n].append(2.0 * v * m[i])
                T[i].append(2.0 * v * m[n])
        for n, c, a, v in d["zeta"]:
            T[n].append(v * m[c] * m[a])
            T[c].append(v * m[n] * m[a])
            T[a].append(v * m[n] * m[c])
        out = []
        for i, zi in enumerate(z):
            terms = T[i]
            if zi:
                terms = [(zi * zi) * F] + terms + ([abs(float(zi)) * CC] if not _is_zero(CC) else [])
            out.append(o.sum(terms) if terms else 0.0)
        if d.get("kcl") is not None:
            icl, b0, b1, b2, C = d["kcl"]
            # mean ln gamma of pure KCl at molality I: f + I (2B + I B') + 3 I^2 C
            B = b0 + (b1 * _g(o, 2.0 * s) if b1 else 0.0) + (b2 * _g(o, 12.0 * s) if b2 else 0.0)
            Bp = ((b1 * _gp(o, 2.0 * s) if b1 else 0.0) + (b2 * _gp(o, 12.0 * s) if b2 else 0.0)) / I
            ln_kcl = fg + I * (2.0 * B + I * Bp) + 3.0 * C * I * I
            D = out[icl] - ln_kcl
            out = [v + float(zi) * D if zi else v for v, zi in zip(out, z)]
        return out

    def excess_gibbs(self, o, ctx, m, I):
        d = ctx.data
        s, Z, pairs, like = self._common(o, ctx, m, I)
        Aphi, b = d["Aphi"], self.b
        terms = [(-Aphi * 4.0 / b) * I * o.log(1.0 + b * s)]
        terms += [m[c] * m[a] * (2.0 * B + (Z * C if C else 0.0)) for c, a, B, Bp, C in pairs]
        terms += [2.0 * m[i] * m[j] * Ph for i, j, Ph, Php in like]
        terms += [v * m[i] * m[j] * m[k] for i, j, k, v in d["psi"]]
        terms += [(1.0 if n == i else 2.0) * v * m[n] * m[i] for n, i, v in d["lam"]]
        terms += [v * m[n] * m[c] * m[a] for n, c, a, v in d["zeta"]]
        return o.sum(terms)

    def to_spec(self):
        d = {"model": self.name, "params": self.params.to_dict(), "etheta": self.etheta, "macinnes": self.macinnes}
        if self.A_phi is not None:
            d["A_phi"] = self.A_phi
        return d


# --------------------------------------------------------------------------- helpers used by other phases
def truesdell_jones_ln_gamma(o, z, a0, b, I, A, B):
    """WATEQ ``ln gamma`` for a scalar charge ``z`` (used for exchange species)."""
    if a0 is None:
        return 0.0
    s = o.sqrt(I)
    v = (-LN10 * A * z * z) * s / (1.0 + (B * float(a0)) * s)
    return v + (LN10 * float(b)) * I if b else v


MODELS = {
    "ideal": Ideal, "davies": Davies, "debye-huckel": DebyeHuckel,
    "extended-debye-huckel": ExtendedDebyeHuckel, "truesdell-jones": TruesdellJones,
    "phreeqc": TruesdellJones, "wateq": TruesdellJones, "bdot": BDot, "b-dot": BDot, "llnl": LLNL,
    "sit": SIT, "pitzer": Pitzer,
}


def get_activity_model(spec) -> ActivityModel:
    """Model from a name (``"davies"``), a dict (``{"model": "davies", "A": 0.509}``) or an instance."""
    if isinstance(spec, ActivityModel):
        return spec
    if spec is None:
        return Ideal()
    if isinstance(spec, str):
        if spec.lower() not in MODELS:
            raise ValueError(f"unknown activity model {spec!r}; choose from {sorted(set(MODELS))}")
        return MODELS[spec.lower()]()
    if isinstance(spec, dict):
        d = dict(spec)
        kind = str(d.pop("model", "")).lower()
        if kind not in MODELS:
            raise ValueError(f"unknown activity model {spec!r}")
        if kind == "sit" and isinstance(d.get("eps"), list):
            d["eps"] = {(a, b): v for a, b, v in d["eps"]}
        if kind in ("truesdell-jones", "phreeqc", "wateq") and "params" in d:
            d["params"] = {k: tuple(v) if isinstance(v, list) else v for k, v in d["params"].items()}
        return MODELS[kind](**d)
    raise ValueError(f"unknown activity model {spec!r}")

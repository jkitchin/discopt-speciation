r"""Peng-Robinson equation of state for gases, with PHREEQC's conventions.

A gas with critical constants (``Tc`` in K, ``Pc`` in atm, acentric factor
``omega``) has the Peng-Robinson (1976) parameters

.. math:: a_i = 0.457235\,\frac{(RT_{c,i})^2}{P_{c,i}}\,\big[1 + \kappa_i(1-\sqrt{T/T_{c,i}})\big]^2,
          \qquad b_i = 0.077796\,\frac{RT_{c,i}}{P_{c,i}},

with :math:`\kappa_i = 0.37464 + 1.54226\,\omega_i - 0.26992\,\omega_i^2`, mixed by
:math:`a = \sum_{ij} x_i x_j (1-k_{ij})\sqrt{a_i a_j}` and :math:`b = \sum_i x_i b_i`.
The constants and mixing rule are PHREEQC's (``calc_PR`` in ``prep.cpp``), and
so are the default binary parameters :data:`DEFAULT_KIJ`.

The fugacity of member :math:`i` in a gas of amounts :math:`n` (mol) and volume
:math:`V` (L) is written out explicitly in :math:`(n, V)`, with
:math:`B_n=\sum_i n_i b_i`, :math:`S_i=\sum_j a_{ij} n_j`,
:math:`A_n=\sum_i n_i S_i` and :math:`D=V^2+2B_nV-B_n^2`:

.. math:: \ln f_i = \ln\frac{n_i RT}{V-B_n} + \frac{b_i n}{V-B_n} - \frac{b_i A_n V}{B_n RT\,D}
          - \Big(\frac{S_i}{\sqrt2\,B_n RT} - \frac{A_n b_i}{2\sqrt2\,B_n^2 RT}\Big)
            \ln\frac{V+(1+\sqrt2)B_n}{V+(1-\sqrt2)B_n} .

This is the usual :math:`\ln(x_i\varphi_i P)` with :math:`P` eliminated, so a
fixed-volume gas needs no cubic root. A fixed-pressure gas uses the same formula
with :math:`n\to x` and :math:`V\to V_m`, and adds the pressure equation
:math:`P(x, V_m) = P`. Among the roots of that cubic the vapor root (the
largest) is the one meant; :func:`PengRobinson.molar_volume` returns it, and the
solver starts from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from discopt.speciation.water import R_GAS

#: gas constant in L atm / (mol K)
R_LATM = R_GAS / 101.325
PR_A, PR_B = 0.457235, 0.077796
SQ2 = math.sqrt(2.0)
EOS = ("auto", "ideal", "peng-robinson")

_H2O_KIJ = {"CO2(g)": 0.19, "H2S(g)": 0.19, "H2Sg(g)": 0.19, "CH4(g)": 0.49, "Mtg(g)": 0.49, "Methane(g)": 0.49,
            "N2(g)": 0.49, "Ntg(g)": 0.49, "Ethane(g)": 0.49, "Propane(g)": 0.55}
#: PHREEQC's built-in binary interaction parameters (``calc_gas_binary_parameter``,
#: after Soreide and Whitson 1992), used for pairs that ``GAS_BINARY_PARAMETERS`` or
#: ``kij=`` do not give.
DEFAULT_KIJ = {frozenset(("H2O(g)", g)): k for g, k in _H2O_KIJ.items()}


def critical_constants(species):
    """``(Tc, Pc, omega)`` of a gas species, or ``None`` when ``Tc`` or ``Pc`` is not given."""
    p = species.params
    Tc, Pc = p.get("Tc"), p.get("Pc")
    if not Tc or not Pc or Tc <= 0 or Pc <= 0:
        return None
    return float(Tc), float(Pc), float(p.get("omega") or 0.0)


def check_eos(eos) -> str:
    eos = "auto" if eos is None else str(eos).lower().replace("_", "-")
    if eos in ("pr", "pengrobinson"):
        eos = "peng-robinson"
    if eos not in EOS:
        raise ValueError(f"unknown gas equation of state {eos!r}; choose one of {EOS}")
    return eos


@dataclass
class PengRobinson:
    """Peng-Robinson mixture of gases ``names`` with critical constants ``Tc`` (K), ``Pc`` (atm), ``omega``.

    ``kij`` maps unordered name pairs (``frozenset``) to binary interaction
    parameters :math:`k_{ij}`; missing pairs are 0.
    """

    names: list
    Tc: list
    Pc: list
    omega: list
    kij: dict = field(default_factory=dict)

    @classmethod
    def for_members(cls, members, kij=None, eos: str = "auto"):
        """Model for gas species ``members``, or ``None`` for an ideal gas.

        ``eos="auto"`` uses Peng-Robinson when every member has critical constants
        and raises when only some do (PHREEQC does not mix the two either);
        ``"peng-robinson"`` requires them; ``"ideal"`` returns ``None``.
        """
        eos = check_eos(eos)
        if eos == "ideal":
            return None
        crit = [critical_constants(m) for m in members]
        missing = [m.name for m, c in zip(members, crit) if c is None]
        if missing:
            if eos == "auto" and len(missing) == len(members):
                return None
            raise ValueError(
                f"gases {missing} have no critical constants (Tc, Pc) but "
                + ("the gas phase is Peng-Robinson" if eos == "peng-robinson"
                   else f"{sorted(m.name for m, c in zip(members, crit) if c)} do")
                + "; give Tc, Pc and omega for every member (System.critical_constants) or use eos='ideal'")
        names = [m.name for m in members]
        k = dict(DEFAULT_KIJ)
        k.update(normalize_kij(kij))
        pairs = {pair: val for pair, val in k.items() if pair <= set(names) and len(pair) == 2}
        return cls(names, [c[0] for c in crit], [c[1] for c in crit], [c[2] for c in crit], pairs)

    def subset(self, names) -> "PengRobinson":
        """The model of members ``names`` (any subset, in that order)."""
        pos = [self.names.index(n) for n in names]
        kij = {pair: k for pair, k in self.kij.items() if pair <= set(names)}
        return PengRobinson(list(names), [self.Tc[i] for i in pos], [self.Pc[i] for i in pos],
                            [self.omega[i] for i in pos], kij)

    # ------------------------------------------------------------------ parameters
    def coefficients(self, T: float):
        """``(a, b)``: the matrix :math:`(1-k_{ij})\\sqrt{a_ia_j}` (L² atm/mol²) and the vector ``b`` (L/mol) at ``T`` (K)."""
        RTc = [R_LATM * t for t in self.Tc]
        ai, bi = [], []
        for rtc, pc, tc, om in zip(RTc, self.Pc, self.Tc, self.omega):
            kk = 0.37464 + om * (1.54226 - 0.26992 * om)
            alpha = (1.0 + kk * (1.0 - math.sqrt(T / tc))) ** 2
            ai.append(PR_A * rtc * rtc / pc * alpha)
            bi.append(PR_B * rtc / pc)
        n = len(ai)
        a = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                k = self.kij.get(frozenset((self.names[i], self.names[j])), 0.0) if i != j else 0.0
                a[i, j] = (1.0 - k) * math.sqrt(ai[i] * ai[j])
        return a, np.array(bi)

    # ------------------------------------------------------------------ expressions
    def ln_fugacity(self, o, n, V, T: float) -> list:
        """:math:`\\ln f_i` (``f`` in atm) at amounts ``n`` (mol) and volume ``V`` (L); see the module docstring.

        ``o`` is an ops backend (``NUM``, ``EXPR``, ``JAX``); ``n`` and ``V`` may be
        expressions. With mole fractions and the molar volume it gives the same
        intensive result.
        """
        a, b = self.coefficients(T)
        RT = R_LATM * T
        m = len(n)
        ntot = o.sum(list(n))
        Bn = o.sum([float(b[i]) * n[i] for i in range(m)])
        S = [o.sum([float(a[i, j]) * n[j] for j in range(m)]) for i in range(m)]
        An = o.sum([n[i] * S[i] for i in range(m)])
        VB = V - Bn
        D = V * V + 2.0 * Bn * V - Bn * Bn
        lnr = o.log((V + (1.0 + SQ2) * Bn) / (V + (1.0 - SQ2) * Bn))
        out = []
        for i in range(m):
            bi = float(b[i])
            out.append(o.log(n[i] * RT / VB) + bi * ntot / VB - bi * An * V / (Bn * RT * D)
                       - (S[i] / (SQ2 * Bn * RT) - An * bi / (2.0 * SQ2 * Bn * Bn * RT)) * lnr)
        return out

    def pressure(self, o, n, V, T: float):
        """Pressure (atm) of amounts ``n`` (mol) in volume ``V`` (L)."""
        a, b = self.coefficients(T)
        m = len(n)
        Bn = o.sum([float(b[i]) * n[i] for i in range(m)])
        An = o.sum([float(a[i, j]) * n[i] * n[j] for i in range(m) for j in range(m)])
        return o.sum(list(n)) * (R_LATM * T) / (V - Bn) - An / (V * V + 2.0 * Bn * V - Bn * Bn)

    # ------------------------------------------------------------------ numbers
    def molar_volume(self, x, T: float, P: float) -> float:
        """Molar volume (L/mol) of composition ``x`` at ``T`` (K) and ``P`` (atm): the largest (vapor) root."""
        a, b = self.coefficients(T)
        x = np.asarray(x, float)
        am, bm = float(x @ a @ x), float(x @ b)
        RT = R_LATM * T
        A, B = am * P / RT**2, bm * P / RT
        roots = np.roots([1.0, -(1.0 - B), A - 3.0 * B * B - 2.0 * B, -(A * B - B * B - B**3)])
        real = [r.real for r in roots if abs(r.imag) <= 1e-10 * max(1.0, abs(r.real)) and r.real > B]
        if not real:
            raise ArithmeticError(f"no Peng-Robinson root above b at T={T} K, P={P} atm")
        return max(real) * RT / P

    def ln_phi(self, x, T: float, P: float) -> np.ndarray:
        """Fugacity coefficients :math:`\\ln\\varphi_i` of composition ``x`` at ``T`` (K) and ``P`` (atm)."""
        from discopt.speciation.activity import NUM

        x = [float(v) for v in x]
        Vm = self.molar_volume(x, T, P)
        lnf = self.ln_fugacity(NUM, [max(v, 1e-300) for v in x], Vm, T)
        return np.array([lf - math.log(max(v, 1e-300) * P) for lf, v in zip(lnf, x)])


def normalize_kij(kij) -> dict:
    """``{(a, b): k}``, ``{frozenset: k}`` or ``[[a, b, k], ...]`` -> ``{frozenset((a, b)): k}``."""
    if not kij:
        return {}
    items = kij.items() if isinstance(kij, dict) else [((a, b), k) for a, b, k in kij]
    out = {}
    for pair, k in items:
        names = [getattr(p, "name", p) for p in pair]
        if len(set(names)) != 2:
            raise ValueError(f"binary parameter needs two different gases, got {pair!r}")
        out[frozenset(names)] = float(k)
    return out


def pure_ln_phi(species, T: float, P: float) -> float:
    """:math:`\\ln\\varphi` of a pure gas with critical constants at ``T`` (K) and ``P`` (atm)."""
    model = PengRobinson.for_members([species], eos="peng-robinson")
    return float(model.ln_phi([1.0], T, P)[0])

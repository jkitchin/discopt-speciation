"""Multi-species phase containers: solid solutions, gas phases and surfaces.

Pure solids need no container: every solid that is not a member of a
:class:`SolidSolution` is its own pure phase.

Each container with complementarity (pure solid, solid solution, fixed-pressure
gas phase) is present or absent as a whole. With ``SI_j`` the log10 saturation of
member ``j`` computed as if it were pure (``log10`` partial pressure for gases),

* an ideal solid solution is present iff :math:`\\sum_j 10^{SI_j} \\ge 1`, and then
  its mole fractions are :math:`x_j = 10^{SI_j}`;
* a non-ideal solid solution (activity coefficients :math:`\\lambda_j(x)`) has
  mole fractions and undersaturation ``s`` from
  :math:`\\log_{10}(x_j\\lambda_j) = SI_j + s`, :math:`\\sum_j x_j = 1`, and is
  present iff ``s = 0``. These are the stationarity conditions of the tangent
  plane distance, so several solutions (a miscibility gap) are possible; see
  :mod:`discopt.speciation.certify`;
* an ideal gas phase at total pressure ``P`` is present iff :math:`\\sum_j 10^{SI_j} \\ge P`,
  and then :math:`P_j = 10^{SI_j}`;
* an ideal fixed-volume gas phase always exists with :math:`n_j = P_j V / RT`;
* a Peng-Robinson gas phase replaces partial pressures by fugacities
  :math:`f_j = \\varphi_j x_j P = 10^{SI_j}` (see :mod:`discopt.speciation.eos`):
  at fixed ``P`` its composition and undersaturation ``s`` solve
  :math:`\\log_{10} f_j(x, V_m) = SI_j + s`, :math:`\\sum_j x_j = 1`,
  :math:`P(x, V_m) = P`, and it is present iff ``s = 0``; at fixed ``V`` the
  amounts solve :math:`\\log_{10} f_j(n, V) = SI_j`.

Pure phases are the one-member special case of the solid solution rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SolidSolution:
    """Solid solution of solid species ``members`` (names or Species).

    Ideal unless an excess Gibbs energy is given (dimensionless, ``G_ex/RT`` per
    mole of solid solution):

    ``guggenheim=(a0, a1[, a2])``
        Binary Redlich-Kister (Guggenheim) expansion, as PHREEQC's ``-Gugg_nondim``
        with ``members = [comp1, comp2]``:
        :math:`G_{ex}/RT = x_1x_2\\,[a_0 + a_1(x_2-x_1) + a_2(x_2-x_1)^2]`
        (this sign of ``a1`` reproduces the miscibility gap PHREEQC reports for
        its example 10).
    ``margules={(i, j): W}`` (or a list of ``[i, j, W]``)
        Symmetric regular (pairwise Margules) mixing of any number of members:
        :math:`G_{ex}/RT = \\sum_{i<j} W_{ij}x_ix_j`, keys are member names.

    Activity coefficients are :math:`\\ln\\lambda_j = \\partial(nG_{ex}/RT)/\\partial n_j`.
    """

    name: str
    members: list
    guggenheim: tuple | None = None
    margules: dict | None = None
    _W: list = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.guggenheim is not None and self.margules is not None:
            raise ValueError(f"solid solution {self.name!r}: give guggenheim or margules, not both")
        if self.guggenheim is not None:
            g = tuple(float(a) for a in self.guggenheim)
            if len(self.members) != 2 or not 1 <= len(g) <= 3:
                raise ValueError(f"solid solution {self.name!r}: guggenheim needs 2 members and (a0, a1[, a2])")
            self.guggenheim = g + (0.0,) * (3 - len(g))
        if self.margules is not None:
            items = self.margules.items() if isinstance(self.margules, dict) else [((a, b), w) for a, b, w in self.margules]
            self.margules = {tuple(getattr(m, "name", m) for m in k): float(v) for k, v in items}

    @property
    def is_ideal(self) -> bool:
        return self.guggenheim is None and not self.margules

    def _pairs(self):
        if self._W is None:
            names = [getattr(m, "name", m) for m in self.members]
            W = []
            for (a, b), val in self.margules.items():
                if a not in names or b not in names or a == b:
                    raise ValueError(f"solid solution {self.name!r}: bad margules pair {(a, b)}")
                W.append((names.index(a), names.index(b), val))
            self._W = W
        return self._W

    def margules_matrix(self) -> list:
        """Symmetric matrix ``W[i][j]`` of the Margules parameters (zero diagonal), in member order."""
        n = len(self.members)
        W = [[0.0] * n for _ in range(n)]
        for i, j, w in self._pairs():
            W[i][j] += w
            W[j][i] += w
        return W

    def ln_lambda(self, o, x) -> list:
        """``ln`` activity coefficients of the members at mole fractions ``x`` (summing to 1)."""
        if self.guggenheim is not None:
            a0, a1, a2 = self.guggenheim
            x1, x2 = x
            d = x2 - x1
            return [x2 * x2 * (a0 - a1 * (3.0 * x1 - x2) + a2 * d * (x2 - 5.0 * x1)),
                    x1 * x1 * (a0 + a1 * (3.0 * x2 - x1) + a2 * d * (5.0 * x2 - x1))]
        if self.margules:
            W = self._pairs()
            g = self.excess(o, x)
            out = []
            for k in range(len(x)):
                terms = [w * x[j if i == k else i] for i, j, w in W if k in (i, j)]
                out.append(o.sum(terms) - g if terms else -g)
            return out
        return [0.0] * len(x)

    def excess(self, o, x):
        """``G_ex/RT`` per mole of solid solution."""
        if self.guggenheim is not None:
            a0, a1, a2 = self.guggenheim
            x1, x2 = x
            d = x2 - x1
            return x1 * x2 * (a0 + a1 * d + a2 * d * d)
        if self.margules:
            return o.sum([w * x[i] * x[j] for i, j, w in self._pairs()])
        return 0.0

    def excess_bound(self) -> float:
        """Upper bound on ``|d(G_ex/RT)/dx_j|`` over the simplex (used for boundary truncation error)."""
        if self.guggenheim is not None:
            return 4.0 * sum(abs(a) for a in self.guggenheim) + 1.0
        if self.margules:
            return 2.0 * sum(abs(w) for _, _, w in self._pairs()) + 1.0
        return 0.0

    def to_spec(self) -> dict:
        d = {"name": self.name, "members": [getattr(m, "name", m) for m in self.members]}
        if self.guggenheim is not None:
            d["guggenheim"] = list(self.guggenheim)
        if self.margules:
            d["margules"] = [[a, b, w] for (a, b), w in self.margules.items()]
        return d


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


@dataclass
class GasPhase:
    """Gas phase; give ``pressure`` (atm, fixed total pressure) or ``volume`` (L, fixed volume).

    ``eos`` is ``"ideal"``, ``"peng-robinson"`` or ``None`` (the system's
    ``gas_eos``, by default ``"auto"``: Peng-Robinson when every member has
    critical constants). ``kij`` gives Peng-Robinson binary interaction
    parameters, ``{(gas_i, gas_j): k}`` or ``[[gas_i, gas_j, k], ...]``; pairs it
    does not give use the database's ``GAS_BINARY_PARAMETERS`` and then PHREEQC's
    defaults (:data:`discopt.speciation.eos.DEFAULT_KIJ`).
    """

    members: list
    pressure: float | None = 1.0
    volume: float | None = None
    name: str = "gas"
    eos: str | None = None
    kij: dict | list | None = None

    def __post_init__(self):
        if self.volume is not None:
            self.pressure = None
        if (self.pressure is None) == (self.volume is None):
            raise ValueError("gas phase needs exactly one of pressure (atm) or volume (L)")
        if self.eos is not None:
            from discopt.speciation.eos import check_eos

            self.eos = check_eos(self.eos)

    def to_spec(self) -> dict:
        d = {"name": self.name, "members": [getattr(m, "name", m) for m in self.members]}
        d.update({"pressure": self.pressure} if self.volume is None else {"volume": self.volume})
        if self.eos is not None:
            d["eos"] = self.eos
        if self.kij:
            from discopt.speciation.eos import normalize_kij

            d["kij"] = [sorted(pair) + [k] for pair, k in normalize_kij(self.kij).items()]
        return d


@dataclass
class SurfaceType:
    """A sorbing surface with one or more site types.

    ``area`` (m2/g) and ``mass`` (g) are needed for the electrostatic (``edl="ddl"``,
    diffuse double layer, Dzombak and Morel 1990) model; ``edl=None`` is a
    non-electrostatic surface.
    """

    name: str
    sites: list = field(default_factory=list)
    area: float | None = None
    mass: float | None = None
    edl: str | None = "ddl"

    def __post_init__(self):
        if self.edl not in (None, "ddl"):
            raise ValueError(f"unknown surface electrostatic model {self.edl!r}; use 'ddl' or None")
        if self.edl == "ddl" and (not self.area or not self.mass):
            raise ValueError(f"surface {self.name!r}: the DDL model needs area (m2/g) and mass (g)")

    def to_spec(self) -> dict:
        return {"name": self.name, "sites": list(self.sites), "area": self.area, "mass": self.mass, "edl": self.edl}

"""Sweeps and diagrams: speciation curves, predominance (Pourbaix) and Ellingham plots.

All diagrams are built from full equilibrium calculations, so solid fields and
aqueous predominance come out of one consistent model (a Pourbaix diagram is a
predominance diagram in pH and pe with ``pe`` converted to ``Eh``).

matplotlib is imported only when a plot is drawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from discopt.speciation.species import LN10, R_GAS, T_REF

F_CONST = 96485.33212  # C/mol

__all__ = ["sweep", "Sweep", "plot_sweep", "predominance", "Predominance", "pourbaix", "eh_from_pe",
           "pe_from_eh", "ellingham"]


def eh_from_pe(pe, T: float = T_REF):
    """``Eh`` (V) from ``pe``: ``Eh = pe RT ln10 / F``."""
    return np.asarray(pe) * R_GAS * T * LN10 / F_CONST


def pe_from_eh(eh, T: float = T_REF):
    return np.asarray(eh) * F_CONST / (R_GAS * T * LN10)


def _conditions(base, variable, value):
    kw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    kw.setdefault("amounts", {})
    if callable(variable):
        extra = variable(value)
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(kw.get(k), dict):
                kw[k].update(v)
            else:
                kw[k] = v
    elif variable in ("pH", "pe", "T", "P", "water_mass"):
        kw[variable] = value
    elif variable.startswith("fixed:"):
        kw.setdefault("fixed", {})[variable[6:]] = value
    else:
        kw["amounts"][variable] = value
    return kw


def _solve(system, kw):
    kw = dict(kw)
    amounts = kw.pop("amounts")
    return system.equilibrate(amounts, **kw)


# --------------------------------------------------------------------------- 1-D sweeps
@dataclass
class Sweep:
    variable: str
    values: np.ndarray
    results: list

    @property
    def converged(self) -> np.ndarray:
        return np.array([r.converged for r in self.results])

    def get(self, quantity) -> np.ndarray:
        """Quantity along the sweep (see :func:`discopt.speciation.fit.observe`); NaN if unavailable."""
        from discopt.speciation.fit import observe

        out = []
        for r in self.results:
            try:
                out.append(observe(r, quantity))
            except (KeyError, ValueError):
                out.append(math.nan)
        return np.array(out)


def sweep(system, variable, values, **conditions) -> Sweep:
    """Equilibrate along ``values`` of ``variable``.

    ``variable`` is ``"pH"``, ``"pe"``, ``"T"``, ``"P"``, ``"fixed:<species>"`` (a
    log10 activity or partial pressure), an input-amount key (``"Cl"``,
    ``"Calcite"``), or a callable ``value -> dict`` of equilibrate keywords.
    ``conditions`` are the other keywords of :meth:`System.equilibrate`
    (``amounts=...``, ``pH=...``).
    """
    values = np.asarray(values, float)
    res = [_solve(system, _conditions(conditions, variable, v)) for v in values]
    return Sweep(variable if isinstance(variable, str) else getattr(variable, "__name__", "x"), values, res)


def plot_sweep(sw: Sweep, species=None, *, quantity: str = "molality", ax=None, log: bool = True, floor: float = 1e-12,
               solids: bool = True, xlabel=None):
    """Plot ``quantity`` (``molality``, ``log10_activity``, ``gamma``, ``SI``) of species along a sweep.

    ``species`` defaults to every aqueous species that exceeds ``floor`` somewhere.
    Solid amounts are drawn dashed on the same axis when ``solids`` is true.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    if species is None:
        names = {k for r in sw.results for k, v in getattr(r, quantity).items()
                 if k != "H2O" and (quantity != "molality" or v > floor)}
        species = sorted(names)
    for k in species:
        y = sw.get(f"{quantity}:{k}")
        ax.plot(sw.values, y, label=k)
    if solids:
        names = sorted({k for r in sw.results for k, v in r.solids.items() if v > floor})
        for k in names:
            ax.plot(sw.values, sw.get(f"solids:{k}"), "--", lw=2, label=f"{k} (s)")
    if log and quantity in ("molality",):
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(lo, floor), hi)
    ax.set_xlabel(xlabel or sw.variable)
    ax.set_ylabel(quantity if quantity != "molality" else "mol/kg (solids: mol)")
    ax.legend(fontsize="small", ncol=2)
    return ax


# --------------------------------------------------------------------------- 2-D predominance
@dataclass
class Predominance:
    x: np.ndarray
    y: np.ndarray
    labels: list  # candidate names; grid holds indices into this list (-1 = failed)
    grid: np.ndarray  # shape (len(y), len(x))
    xlabel: str
    ylabel: str
    results: list

    def plot(self, ax=None, *, cmap="Pastel1", annotate: bool = True, fontsize=9):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        ax = ax or plt.gca()
        present = sorted({int(v) for v in np.unique(self.grid) if v >= 0})
        base = plt.get_cmap(cmap)
        colors = [base(i % base.N) for i in range(len(self.labels))]
        g = np.ma.masked_less(self.grid, 0)
        ax.pcolormesh(self.x, self.y, g, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(self.labels) - 0.5,
                      shading="nearest")
        if len(self.x) > 1 and len(self.y) > 1:
            for i in present:  # one boundary per field (index contours would stack lines)
                ax.contour(self.x, self.y, (self.grid == i).astype(float), levels=[0.5], colors="k", linewidths=0.6)
        if annotate:
            X, Y = np.meshgrid(self.x, self.y)
            for i in present:
                mask = self.grid == i
                ax.text(X[mask].mean(), Y[mask].mean(), self.labels[i], ha="center", va="center", fontsize=fontsize)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        return ax


def predominance(system, candidates, x, y, *, weights=None, **conditions) -> Predominance:
    """Dominant species or phase on a 2-D grid.

    Parameters
    ----------
    candidates : list or dict
        Names of aqueous species, exchange/surface species and solids to compare;
        a dict maps each name to the number of atoms of the element of interest
        (``{"Fe+2": 1, "Fe2O3(s)": 2}``) so amounts are compared per atom.
    x, y : (variable, values)
        Axes, as for :func:`sweep` (``("pH", np.linspace(2, 12, 41))``).
    conditions
        Other keywords of :meth:`System.equilibrate`.

    Solids are compared by amount (mol per kg water), aqueous species by molality.
    """
    w = dict(candidates) if isinstance(candidates, dict) else {c: 1.0 for c in candidates}
    if weights:
        w.update(weights)
    labels = list(w)
    (xv, xs), (yv, ys) = x, y
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    grid = np.full((len(ys), len(xs)), -1, dtype=int)
    results = []
    for i, yy in enumerate(ys):
        row = []
        for j, xx in enumerate(xs):
            kw = _conditions(_conditions(conditions, xv, xx), yv, yy)
            r = _solve(system, kw)
            row.append(r)
            if not r.converged:
                continue
            amt = [w[k] * max(r.molality.get(k, 0.0), r.solids.get(k, 0.0), r.exchange.get(k, 0.0),
                              r.surface.get(k, 0.0)) for k in labels]
            if max(amt) > 0:
                grid[i, j] = int(np.argmax(amt))
        results.append(row)
    return Predominance(xs, ys, labels, grid, str(xv), str(yv), results)


def pourbaix(system, candidates, pH, pe, *, eh: bool = True, water_lines: bool = True, ax=None, T: float = T_REF,
             **conditions):
    """Pourbaix (Eh-pH) diagram: :func:`predominance` over ``pH`` and ``pe``.

    Returns ``(Predominance, ax)``. With ``eh`` the vertical axis is Eh in volts;
    ``water_lines`` draws the O2/H2O and H2O/H2 stability limits at 1 atm.
    """
    import matplotlib.pyplot as plt

    conditions.setdefault("T", T)
    pr = predominance(system, candidates, ("pH", pH), ("pe", pe), **conditions)
    ax = ax or plt.gca()
    if eh:
        pr_plot = Predominance(pr.x, eh_from_pe(pr.y, T), pr.labels, pr.grid, "pH", "Eh (V)", pr.results)
    else:
        pr_plot = pr
    pr_plot.plot(ax)
    if water_lines:
        ph = np.array([pr.x[0], pr.x[-1]])
        # O2 + 4H+ + 4e- = 2H2O, logK = 4 * 20.78 at 25 C; H2 at pe = -pH
        upper, lower = 20.78 - ph, -ph
        f = (lambda v: eh_from_pe(v, T)) if eh else (lambda v: v)
        ax.plot(ph, f(upper), "b--", lw=1)
        ax.plot(ph, f(lower), "b--", lw=1)
        ax.set_ylim(pr_plot.y[0], pr_plot.y[-1])
    return pr, ax


# --------------------------------------------------------------------------- Ellingham
def ellingham(reactions, T, *, per=None, ax=None, celsius: bool = True):
    """Ellingham diagram: standard Gibbs energy vs temperature.

    ``reactions`` maps a label to a :class:`Reaction` (``logK_at(T)`` is used) or to a
    callable ``T -> log10 K``; ``per`` maps a label to the divisor (moles of O2 in
    the reaction as written, so curves are per mol O2). ``dG = -RT ln10 logK``.
    Returns ``(T, {label: dG in kJ})`` and draws the lines.
    """
    import matplotlib.pyplot as plt

    T = np.asarray(T, float)
    out = {}
    for label, r in reactions.items():
        f = r if callable(r) else r.logK_at
        lk = np.array([f(t) for t in T])
        out[label] = -R_GAS * T * LN10 * lk / 1000.0 / float((per or {}).get(label, 1.0))
    ax = ax or plt.gca()
    for label, g in out.items():
        ax.plot(T - 273.15 if celsius else T, g, label=label)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("T (°C)" if celsius else "T (K)")
    ax.set_ylabel(r"$\Delta G^\circ$ (kJ" + ("/mol O$_2$)" if per else ")"))
    ax.legend(fontsize="small")
    return T, out

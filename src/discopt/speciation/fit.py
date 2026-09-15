"""Fit thermodynamic parameters to equilibrium measurements, with confidence intervals.

Every observation is a full equilibrium calculation (so solubility data that
cross a phase boundary are handled correctly), and the parameters are adjusted
by bounded nonlinear least squares on the weighted residuals
``(model - value) / sigma``. Uncertainty is reported two ways:

* **linearized** (Fisher information): ``cov = s2 (J^T J)^-1`` with Student-t
  intervals, where ``s2`` is the reduced chi-square (or 1 with
  ``absolute_sigma=True``, when ``sigma`` are true standard deviations);
* **profile likelihood** (:meth:`FitResult.profile`): each parameter is fixed on
  a grid and the others re-optimized; the interval is where
  ``chi2 <= chi2_min (1 + t^2 / dof)`` (or ``chi2_min + chi2_1`` with absolute
  sigma). Profiles stay honest when the model is strongly nonlinear or a
  parameter is poorly determined, where the linearized interval is not.

Example::

    from discopt.speciation import fit

    params = [fit.logK(s, "Calcite", lower=-10, upper=-6)]
    obs = [fit.Observation({"amounts": {"Calcite": 1.0}, "fixed": {"CO2(g)": lp}}, "pH", pH)
           for lp, pH in data]
    res = fit.fit_equilibrium(s, obs, params)
    print(res.summary())
    res.profile("logK Calcite").interval
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from discopt.speciation.species import T_REF

__all__ = [
    "Parameter", "Observation", "FitResult", "ProfileResult", "fit_equilibrium",
    "logK", "enthalpy", "activity_parameter", "total", "observe",
]


# --------------------------------------------------------------------------- parameters
@dataclass
class Parameter:
    """A fitted quantity.

    ``set(value)`` writes the value into the system (a reaction, species or
    activity model); ``get()`` reads the current value. ``scale`` is the typical
    magnitude, used for finite-difference steps.
    """

    name: str
    set: Callable[[float], None]
    get: Callable[[], float]
    lower: float = -math.inf
    upper: float = math.inf
    initial: float | None = None
    scale: float = 1.0

    def start(self) -> float:
        return float(self.get() if self.initial is None else self.initial)


def _reaction(system, key):
    if not isinstance(key, str):
        return key
    for r in system.reactions:
        if r.name == key or r.equation() == key:
            return r
    names = sorted(str(r.name) for r in system.reactions if r.name)
    raise KeyError(f"no reaction named {key!r} (named reactions: {names[:20]}{' ...' if len(names) > 20 else ''})")


def logK(system, reaction, *, lower=-math.inf, upper=math.inf, initial=None, name=None) -> Parameter:
    """``log10 K`` at 25 °C of a reaction (a :class:`Reaction`, its name or its equation).

    For reactions with ``-analytic`` coefficients the constant term ``A1`` is
    shifted, so the temperature dependence is kept and the parameter is still
    ``log10 K(25 °C)``.
    """
    r = _reaction(system, reaction)
    label = name or f"logK {r.name or r.equation()}"
    if r.analytic is not None:
        base = tuple(r.analytic)
        base_lk = r.logK_at(T_REF)

        def set_(v):
            r.analytic = (base[0] + (v - base_lk),) + base[1:]

        return Parameter(label, set_, lambda: r.logK_at(T_REF), lower, upper, initial)

    def set_(v):
        r.logK = float(v)

    return Parameter(label, set_, lambda: float(r.logK), lower, upper, initial)


def enthalpy(system, reaction, *, lower=-math.inf, upper=math.inf, initial=None, name=None) -> Parameter:
    """Reaction enthalpy ``dH`` in **kJ/mol** (van 't Hoff temperature dependence)."""
    r = _reaction(system, reaction)
    if r.analytic is not None:
        raise ValueError(f"reaction {r.equation()!r} uses -analytic coefficients; fit logK or set analytic=None")

    def set_(v):
        r.dH = 1000.0 * float(v)

    return Parameter(name or f"dH {r.name or r.equation()} [kJ/mol]", set_, lambda: r.dH / 1000.0, lower, upper,
                     initial, scale=10.0)


def activity_parameter(system, species, key, *, lower=-math.inf, upper=math.inf, initial=None, name=None) -> Parameter:
    """A per-species activity parameter such as ``"a0"`` or ``"b"`` (Truesdell-Jones, B-dot)."""
    sp = system[species] if isinstance(species, str) else species

    def get():
        if key not in sp.params:
            raise KeyError(f"species {sp.name!r} has no parameter {key!r}; give initial=")
        return float(sp.params[key])

    def set_(v):
        sp.params[key] = float(v)

    if initial is not None and key not in sp.params:
        sp.params[key] = float(initial)
    return Parameter(name or f"{key} {sp.name}", set_, get, lower, upper, initial)


# --------------------------------------------------------------------------- observations
def total(*names, weights=None) -> Callable:
    """Quantity: ``sum_i w_i * molality_i`` (e.g. dissolved Ca = ``Ca+2 + CaSO4 + CaHCO3+ ...``)."""
    w = list(weights) if weights is not None else [1.0] * len(names)

    def q(res):
        return sum(wi * res.molality.get(n, 0.0) for wi, n in zip(w, names))

    q.__name__ = "total(" + ",".join(names) + ")"
    return q


def observe(result, quantity) -> float:
    """Evaluate a quantity on a :class:`Result`.

    ``quantity`` is a callable ``f(result)`` or one of ``"pH"``, ``"pe"``,
    ``"ionic_strength"``, ``"water_mass"``, or ``"<kind>:<name>"`` with kind in
    ``molality``, ``log10_molality``, ``log10_activity``, ``gamma``, ``SI``,
    ``solids``, ``gases``, ``partial_pressure``, ``log10_partial_pressure``,
    ``exchange``, ``surface``, ``mole_fraction``.
    """
    if callable(quantity):
        return float(quantity(result))
    if ":" not in quantity:
        v = getattr(result, quantity)
        if v is None:
            raise ValueError(f"result has no {quantity!r}")
        return float(v)
    kind, name = quantity.split(":", 1)
    if kind == "log10_molality":
        return math.log10(result.molality[name])
    if kind == "log10_partial_pressure":
        return math.log10(result.partial_pressure[name])
    d = getattr(result, kind)
    if kind == "solids" and name not in d:
        return 0.0
    return float(d[name])


@dataclass
class Observation:
    """One measurement.

    ``conditions`` are keyword arguments of :meth:`System.equilibrate`
    (``amounts``, ``pH``, ``fixed``, ``T``, ...); ``quantity`` is what was
    measured (see :func:`observe`); ``sigma`` is its standard deviation (or a
    relative weight if ``absolute_sigma=False``).
    """

    conditions: dict
    quantity: str | Callable
    value: float
    sigma: float = 1.0
    label: str | None = None


# --------------------------------------------------------------------------- results
@dataclass
class ProfileResult:
    name: str
    values: np.ndarray
    chi2: np.ndarray
    threshold: float
    interval: tuple  # (lower, upper); +-inf where the profile never crosses the threshold
    confidence: float

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.plot(self.values, self.chi2, "o-", ms=3)
        ax.axhline(self.threshold, color="k", ls="--", lw=1, label=f"{100 * self.confidence:g}% threshold")
        for v in self.interval:
            if math.isfinite(v):
                ax.axvline(v, color="C3", lw=1)
        ax.set_xlabel(self.name)
        ax.set_ylabel(r"$\chi^2$ (others re-optimized)")
        ax.legend()
        return ax


@dataclass
class FitResult:
    parameters: dict
    standard_errors: dict
    confidence_intervals: dict
    covariance: np.ndarray
    correlation: np.ndarray
    chi2: float
    dof: int
    s2: float
    residuals: np.ndarray  # weighted, (model - value) / sigma
    predictions: np.ndarray
    jacobian: np.ndarray  # d prediction / d parameter (unweighted)
    success: bool
    message: str
    nfev: int
    confidence: float
    absolute_sigma: bool
    n_solves: int = 0  # equilibrium calculations, including those for Jacobians
    jacobian_method: str = "ift"
    jacobian_fallbacks: list = field(default_factory=list)  # (observation, reason) where IFT fell back to FD
    _ctx: dict = field(repr=False, default_factory=dict)

    @property
    def names(self) -> list:
        return list(self.parameters)

    def summary(self) -> str:
        c = 100 * self.confidence
        lines = [f"fit: {self.message} (success={self.success}, {self.nfev} model evaluations)",
                 f"chi2 = {self.chi2:.6g}, dof = {self.dof}, reduced chi2 = {self.chi2 / max(self.dof, 1):.4g}"
                 + (" (sigma treated as absolute)" if self.absolute_sigma else ""),
                 f"{'parameter':<34}{'value':>12}{'std err':>12}   {c:g}% CI (linearized)"]
        for n, v in self.parameters.items():
            lo, hi = self.confidence_intervals[n]
            lines.append(f"{n:<34}{v:>12.5g}{self.standard_errors[n]:>12.3g}   [{lo:.5g}, {hi:.5g}]")
        if len(self.parameters) > 1:
            lines.append("correlation:")
            for n, row in zip(self.parameters, self.correlation):
                lines.append(f"  {n[:30]:<30} " + " ".join(f"{x:6.3f}" for x in row))
        return "\n".join(lines)

    def predict(self, conditions: dict, quantity) -> tuple[float, float]:
        """Model value and its linearized standard error at new conditions."""
        ctx = self._ctx
        theta = np.array(list(self.parameters.values()))
        J, y0 = _model_jacobian(ctx["system"], ctx["params"], theta, [Observation(conditions, quantity, 0.0)], ctx["kw"])
        g = J[0]
        _set(ctx["params"], theta)
        return float(y0[0]), float(math.sqrt(max(g @ self.covariance @ g, 0.0)))

    def profile(self, name: str, *, n: int = 21, span: float = 4.0, bounds: tuple | None = None,
                refine: bool = False) -> ProfileResult:
        """Profile likelihood for one parameter over ``value +- span * std err`` (or ``bounds``).

        Each grid point re-optimizes the other parameters (warm-started from its
        neighbour). Interval ends are interpolated between grid points, or located
        by root finding with ``refine=True`` (more equilibrium solves).
        """
        from scipy import stats
        from scipy.optimize import brentq, least_squares

        ctx = self._ctx
        params, system, obs, kw = ctx["params"], ctx["system"], ctx["observations"], ctx["kw"]
        names = self.names
        i = names.index(name)
        best = np.array(list(self.parameters.values()))
        se = self.standard_errors[name]
        if bounds is None:
            w = span * (se if math.isfinite(se) and se > 0 else max(abs(best[i]), 1.0) * 0.1)
            bounds = (best[i] - w, best[i] + w)
        lo_b, hi_b = max(bounds[0], params[i].lower), min(bounds[1], params[i].upper)
        others = [j for j in range(len(names)) if j != i]
        y = np.array([o.value for o in obs])
        sig = np.array([o.sigma for o in obs])

        def chi2_at(v, start):
            th = best.copy()
            th[i] = v
            if not others:
                r = (_evaluate(system, params, th, obs, kw) - y) / sig
                return float(r @ r), th
            sub = [params[j] for j in others]

            last = {}

            def fun(z):
                th[others] = z
                keep = []
                out = (_evaluate(system, params, th, obs, kw, keep=keep) - y) / sig
                last.clear()
                last[tuple(np.asarray(z, float))] = keep
                return out

            def jac(z):
                if kw.get("jacobian", "ift") == "fd":
                    return _jacobian(lambda zz: fun(zz), z, sub, kw["fd_step"])
                th[others] = z
                _set(params, th)
                return _model_jacobian(system, sub, z, obs, kw, last.get(tuple(np.asarray(z, float))))[0] / sig[:, None]

            sol = least_squares(fun, start, bounds=([p.lower for p in sub], [p.upper for p in sub]), jac=jac,
                                xtol=kw["tol"], ftol=kw["tol"], gtol=kw["tol"], max_nfev=kw["max_nfev"])
            th[others] = sol.x
            return float(2.0 * sol.cost), th.copy()

        if self.absolute_sigma:
            thr = self.chi2 + stats.chi2.ppf(self.confidence, 1)
        else:
            t = stats.t.ppf(0.5 + self.confidence / 2.0, max(self.dof, 1))
            thr = self.chi2 * (1.0 + t * t / max(self.dof, 1))

        grid = np.unique(np.concatenate([np.linspace(lo_b, hi_b, n), [best[i]]]))
        chis = np.empty(len(grid))
        k0 = int(np.argmin(abs(grid - best[i])))
        cache = {}
        for direction in (range(k0, len(grid)), range(k0, -1, -1)):
            start = best[others]
            for k in direction:
                chis[k], th = chi2_at(grid[k], start)
                cache[grid[k]] = th
                start = th[others]

        def crossing(ka, kb):
            a, b = grid[ka], grid[kb]
            if not refine:  # linear interpolation of chi2 between grid points
                return a + (thr - chis[ka]) * (b - a) / (chis[kb] - chis[ka])
            k_in = ka if chis[ka] <= thr else kb
            return brentq(lambda v: chi2_at(v, cache[grid[k_in]][others])[0] - thr, a, b,
                          xtol=1e-4 * max(se if math.isfinite(se) else 0.0, 1e-8))

        lower = upper = None
        for k in range(k0, len(grid) - 1):
            if chis[k] <= thr < chis[k + 1]:
                upper = crossing(k, k + 1)
                break
        for k in range(k0, 0, -1):
            if chis[k] <= thr < chis[k - 1]:
                lower = crossing(k - 1, k)
                break
        _set(params, best)
        return ProfileResult(name, grid, chis, float(thr),
                             (-math.inf if lower is None else lower, math.inf if upper is None else upper), self.confidence)


# --------------------------------------------------------------------------- engine
def _set(params, theta):
    for p, v in zip(params, theta):
        p.set(float(v))


def _evaluate(system, params, theta, observations, kw, keep=None) -> np.ndarray:
    _set(params, theta)
    out = np.empty(len(observations))
    for k, o in enumerate(observations):
        cond = dict(o.conditions)
        amounts = cond.pop("amounts", {})
        cond.setdefault("method", kw["method"])
        if kw["activity"] is not None:
            cond.setdefault("activity", kw["activity"])
        res = system.equilibrate(amounts, **cond)
        kw["solves"] = kw.get("solves", 0) + 1
        if keep is not None:
            keep.append(res)
        if not res.converged:
            kw["failures"].append((o.label or k, dict(zip([p.name for p in params], map(float, theta)))))
            if kw["strict"]:
                raise RuntimeError(f"equilibrium did not converge for observation {o.label or k} at {theta}")
        out[k] = observe(res, o.quantity)
    return out


def _jacobian(fun, theta, params, rel_step) -> np.ndarray:
    theta = np.asarray(theta, float)
    f0 = None
    cols = []
    for j, p in enumerate(params):
        h = rel_step * max(abs(theta[j]), p.scale)
        up, dn = theta.copy(), theta.copy()
        up[j] = min(theta[j] + h, p.upper)
        dn[j] = max(theta[j] - h, p.lower)
        if up[j] == dn[j]:
            cols.append(None)
            continue
        cols.append((fun(up) - fun(dn)) / (up[j] - dn[j]))
    m = next(len(c) for c in cols if c is not None)
    return np.column_stack([np.zeros(m) if c is None else c for c in cols])


def _model_jacobian(system, params, theta, observations, kw, results=None):
    """``d observe / d theta`` (unweighted) and the model values at ``theta``.

    With ``kw["jacobian"] == "ift"`` each row comes from the implicit function
    theorem at the converged state (no extra solves); an observation whose
    sensitivity is degenerate or unavailable falls back to central differences
    of re-solves, and is recorded in ``kw["fallbacks"]``.
    """
    theta = np.asarray(theta, float)
    if kw.get("jacobian", "ift") == "fd":
        f0 = _evaluate(system, params, theta, observations, kw)
        J = _jacobian(lambda th: _evaluate(system, params, th, observations, kw), theta, params, kw["fd_step"])
        _set(params, theta)
        return J, f0
    if results is None:
        results = []
        f0 = _evaluate(system, params, theta, observations, kw, keep=results)
    else:
        _set(params, theta)
        f0 = np.array([observe(r, o.quantity) for r, o in zip(results, observations)])
    rows = []
    for k, (o, res) in enumerate(zip(observations, results)):
        reason = None
        try:
            S = res.sensitivity(params, [o.quantity])
            if S.ok:
                rows.append(S.jacobian[0])
                continue
            reason = "; ".join(S.issues) or "non-finite sensitivity"
        except Exception as e:  # noqa: BLE001 - any failure falls back to differences of re-solves
            reason = f"{type(e).__name__}: {e}"
        kw.setdefault("fallbacks", []).append((o.label or k, reason))
        one = [o]
        rows.append(_jacobian(lambda th: _evaluate(system, params, th, one, kw), theta, params, kw["fd_step"])[0])
        _set(params, theta)
    return np.array(rows), f0


def fit_equilibrium(system, observations, parameters, *, method: str = "complementarity", activity=None,
                    confidence: float = 0.95, absolute_sigma: bool = False, jacobian: str = "ift",
                    fd_step: float = 1e-4, tol: float = 1e-10, max_nfev: int = 200, strict: bool = False) -> FitResult:
    """Least-squares fit of ``parameters`` to ``observations``.

    Parameters
    ----------
    system : System
        Modified in place: on return it holds the best-fit values.
    observations : list of Observation
    parameters : list of Parameter
        From :func:`logK`, :func:`enthalpy`, :func:`activity_parameter` or a
        custom :class:`Parameter` (any setter, e.g. a Pitzer coefficient).
    method, activity
        Passed to :meth:`System.equilibrate` unless an observation overrides them.
    confidence : float
        Level of the reported intervals.
    absolute_sigma : bool
        ``True`` if the ``sigma`` are real standard deviations; otherwise they
        are relative weights and the covariance is scaled by the reduced chi-square.
    jacobian : {"ift", "fd"}
        How ``d prediction / d parameter`` is computed. ``"ift"`` (default) uses
        exact implicit-function-theorem sensitivities at each converged state
        (:func:`sensitivity.sensitivities`, no extra solves); observations whose
        sensitivity is degenerate (e.g. a phase exactly at its saturation
        boundary) fall back to central differences and are listed in
        ``FitResult.jacobian_fallbacks``. ``"fd"`` differences re-solves throughout.
    fd_step : float
        Relative central-difference step for the ``"fd"`` path and fallbacks
        (equilibrium solves converge to ~1e-8, so steps much below 1e-5 amplify solver noise).
    strict : bool
        Raise if an equilibrium calculation fails to converge (default: warn).
    """
    from scipy import stats
    from scipy.optimize import least_squares

    obs = list(observations)
    params = list(parameters)
    if not obs or not params:
        raise ValueError("need at least one observation and one parameter")
    if len({p.name for p in params}) != len(params):
        raise ValueError("parameter names must be unique")
    y = np.array([o.value for o in obs], float)
    sig = np.array([o.sigma for o in obs], float)
    if np.any(sig <= 0):
        raise ValueError("sigma must be positive")
    if jacobian not in ("ift", "fd"):
        raise ValueError(f"jacobian must be 'ift' or 'fd', not {jacobian!r}")
    kw = dict(method=method, activity=activity, fd_step=fd_step, tol=tol, max_nfev=max_nfev, strict=strict, failures=[],
              jacobian=jacobian, fallbacks=[], solves=0)
    x0 = np.array([p.start() for p in params])
    lb = np.array([p.lower for p in params])
    ub = np.array([p.upper for p in params])
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError("initial values must lie within the bounds")
    nfev = [0]
    last = {}  # theta -> results of the latest model evaluation, reused by the IFT Jacobian

    def model(th):
        nfev[0] += 1
        keep = []
        out = _evaluate(system, params, th, obs, kw, keep=keep)
        last.clear()
        last[tuple(np.asarray(th, float))] = keep
        return out

    def resid(th):
        return (model(th) - y) / sig

    def jac(th):
        if jacobian == "fd":
            return _jacobian(resid, th, params, fd_step)
        return _model_jacobian(system, params, th, obs, kw, last.get(tuple(np.asarray(th, float))))[0] / sig[:, None]

    sol = least_squares(resid, x0, bounds=(lb, ub), jac=jac, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev)
    theta = sol.x
    pred = model(theta)
    J = _model_jacobian(system, params, theta, obs, kw, last.get(tuple(np.asarray(theta, float))))[0]
    Jw = J / sig[:, None]
    r = (pred - y) / sig
    chi2 = float(r @ r)
    n, p = len(obs), len(params)
    dof = n - p
    s2 = 1.0 if absolute_sigma else (chi2 / dof if dof > 0 else math.nan)
    try:
        cov = s2 * np.linalg.pinv(Jw.T @ Jw)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), math.nan)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    q = stats.t.ppf(0.5 + confidence / 2.0, dof) if (dof > 0 and not absolute_sigma) else stats.norm.ppf(0.5 + confidence / 2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(se, se)
    cond = np.linalg.cond(Jw) if Jw.size else math.inf
    if cond > 1e10:
        warnings.warn(f"sensitivity matrix is ill-conditioned (cond={cond:.1e}): some parameters are not "
                      "identifiable from these data; use FitResult.profile to check", stacklevel=2)
    elif len(se) > 1:
        off = np.abs(corr - np.eye(len(se)))
        if np.nanmax(off) > 0.99:
            i, j = np.unravel_index(np.nanargmax(off), off.shape)
            warnings.warn(f"parameters {params[i].name!r} and {params[j].name!r} are strongly correlated "
                          f"(r={corr[i, j]:.3f}): their individual intervals are wide; use FitResult.profile "
                          "or add more informative data", stacklevel=2)
    if kw["failures"]:
        warnings.warn(f"{len(kw['failures'])} equilibrium calculations did not converge during the fit", stacklevel=2)
    fallbacks = list(kw["fallbacks"])
    _set(params, theta)
    names = [pp.name for pp in params]
    return FitResult(
        parameters=dict(zip(names, map(float, theta))),
        standard_errors=dict(zip(names, map(float, se))),
        confidence_intervals={nm: (float(t - q * e), float(t + q * e)) for nm, t, e in zip(names, theta, se)},
        covariance=cov, correlation=corr, chi2=chi2, dof=dof, s2=float(s2), residuals=r, predictions=pred,
        jacobian=J, success=bool(sol.success), message=str(sol.message), nfev=nfev[0], confidence=confidence,
        absolute_sigma=absolute_sigma, n_solves=int(kw["solves"]), jacobian_method=jacobian,
        jacobian_fallbacks=fallbacks,
        _ctx=dict(system=system, params=params, observations=obs, kw=kw),
    )

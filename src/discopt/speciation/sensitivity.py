r"""Exact sensitivities of an equilibrium state (implicit function theorem).

At a converged state the square system of the solved phase assemblage
(:func:`discopt.speciation.solve.square_equations`: mass action, conservation,
activity model, present phases saturated ``s_p = 0``, absent phases ``N_p = 0``)
satisfies ``F(x, theta) = 0``, so

.. math:: \frac{dx}{d\theta} = -F_x^{-1} F_\theta,\qquad
          \frac{dq}{d\theta} = q_x \frac{dx}{d\theta} + q_\theta .

* ``F_x`` and ``q_x`` (built-in quantity kinds) are exact: JAX forward mode
  through ``state``/``equations``.
* ``F_theta`` and ``q_theta`` come from rebuilding the numeric network at
  ``theta +- h`` (no solve) and differencing the residuals at the fixed state
  ``x``. The conservation rows are expressed in the base network's basis and
  scaling, so the residuals are **affine** in ``logK``, ``dH`` (van 't Hoff),
  the analytic constant ``A1``, fixed log activities, ``pH``/``pe`` and input
  totals (except exchange/surface site totals, which enter logarithmically).
  For those the difference is exact up to rounding (tests check that ``h = 1e-5``
  and ``h = 0.1`` agree to ~1e-10). For nonlinear parameters (ion sizes, Pitzer
  coefficients, temperature) the difference is noise-free, and a Richardson
  step makes it ``O(h^4)``; ``SensitivityResult.linear`` says which case applied.
* Arbitrary callable quantities are differenced along ``(x + t dx/dtheta_j,
  theta_j + t)`` on results rebuilt from the state (no solve): noise-free, ``O(h^4)``.

Degeneracies are reported in ``issues`` rather than hidden: a phase with both
``N ~ 0`` and ``s ~ 0`` (at a saturation boundary the derivative is one-sided;
``left``/``right`` hold both one-sided values when a single phase is at its
boundary, and ``jacobian`` is NaN where they differ), and a singular or
ill-conditioned ``F_x``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from discopt.speciation import solve as _solve
from discopt.speciation.activity import JAX, _NumOps, ionic_strength

__all__ = ["SensitivityResult", "sensitivities"]

REL_STEP = 1e-5
BOUNDARY_TOL = 1e-7
COND_LIMIT = 1e12


class _PlainOps(_NumOps):
    """Float ops that are not ``NUM``: residual rows are then scaled exactly as in the JAX/EXPR models."""


_PLAIN = _PlainOps()


class _StructureChanged(RuntimeError):
    pass


@dataclass
class SensitivityResult:
    """Derivatives ``d quantity / d parameter`` at an equilibrium state.

    ``jacobian[i, j]`` is ``d quantities[i] / d parameters[j]``; ``dx`` is the
    derivative of the solver's state vector (see :func:`solve.layout`).
    ``issues`` is empty when every derivative is a well-defined two-sided one.
    """

    parameters: list
    quantities: list
    values: np.ndarray
    jacobian: np.ndarray
    dx: np.ndarray
    condition: float
    linear: list
    boundary: list = field(default_factory=list)
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    issues: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and bool(np.all(np.isfinite(self.jacobian)))

    def get(self, quantity, parameter) -> float:
        """One entry, by labels (or positions)."""
        i = quantity if isinstance(quantity, int) else self.quantities.index(_label(quantity))
        j = parameter if isinstance(parameter, int) else self.parameters.index(getattr(parameter, "name", parameter))
        return float(self.jacobian[i, j])

    def summary(self) -> str:
        w = max([len(q) for q in self.quantities] + [8])
        lines = [f"{'':<{w}}" + "".join(f"{p[:14]:>16}" for p in self.parameters)]
        for q, row in zip(self.quantities, self.jacobian):
            lines.append(f"{q:<{w}}" + "".join(f"{v:>16.6g}" for v in row))
        lines.append(f"cond(F_x, scaled) = {self.condition:.3g}")
        lines += [f"issue: {s}" for s in self.issues]
        return "\n".join(lines)


def _label(q) -> str:
    return q if isinstance(q, str) else getattr(q, "__name__", repr(q))


# --------------------------------------------------------------------------- parameters
@dataclass
class _Spec:
    name: str
    value: float
    step: float
    lower: float
    at: Callable  # at(value, fn) -> fn(net, act) with the system set to value


def _find_key(d: dict, name: str):
    for k in d:
        if getattr(k, "name", k) == name:
            return k
    return None


def _resolve(system, amounts, call, p, n_scale, rel) -> _Spec:
    if hasattr(p, "set") and hasattr(p, "get"):
        v0 = float(p.get())

        def at(v, fn):
            p.set(float(v))
            try:
                return fn(*system._network(amounts, **call))
            finally:
                p.set(v0)

        scale = float(getattr(p, "scale", 1.0) or 1.0)
        return _Spec(p.name, v0, rel * max(abs(v0), scale), -math.inf, at)
    if not isinstance(p, str):
        raise TypeError(f"cannot interpret parameter {p!r}: give a fit.Parameter or a string spec")
    kind, _, name = p.partition(":")
    if kind == "total" and name:
        key = _find_key(amounts, name)
        key = name if key is None else key
        v0 = float(amounts.get(key, 0.0))

        def at(v, fn):
            a = dict(amounts)
            a[key] = float(v)
            return fn(*system._network(a, **call))

        return _Spec(p, v0, rel * (abs(v0) if v0 > 0 else n_scale), 0.0, at)
    if kind == "fixed" and name:
        fixed = dict(call.get("fixed") or {})
        key = _find_key(fixed, name)
        if key is None:
            raise KeyError(f"{name!r} has no fixed activity in this calculation (fixed: {[getattr(k, 'name', k) for k in fixed]})")
        v0 = float(fixed[key])

        def at(v, fn):
            return fn(*system._network(amounts, **dict(call, fixed={**fixed, key: float(v)})))

        return _Spec(p, v0, rel, -math.inf, at)
    if p in ("pH", "pe", "T"):
        v0 = call.get(p)
        if v0 is None:
            if p != "T":
                raise KeyError(f"{p} was not fixed in this calculation")
            v0 = system.T
        v0 = float(v0)

        def at(v, fn):
            return fn(*system._network(amounts, **dict(call, **{p: float(v)})))

        return _Spec(p, v0, rel * max(abs(v0), 1.0), 0.0 if p == "T" else -math.inf, at)
    raise ValueError(f"unknown parameter spec {p!r}; use a fit.Parameter, 'total:<key>', 'fixed:<species>', 'pH', 'pe' or 'T'")


def _aligned(base, L, net, act):
    """``net`` in the conservation basis and scaling of ``base``; raises if the structure differs."""
    extra = base.phases[len(net.phases):]
    if extra and all(p.copy_of is not None for p in extra):  # second copies added by phase splitting
        net = replace(net, phases=list(net.phases) + list(extra))
    same = (len(net.species) == len(base.species) and all(a is b for a, b in zip(net.species, base.species))
            and net.kind == base.kind and net.yidx == base.yidx and net.cols == base.cols
            and [(p.kind, p.members) for p in net.phases] == [(p.kind, p.members) for p in base.phases]
            and len(net.rxns) == len(base.rxns) and [S.members for S in net.surfaces] == [S.members for S in base.surfaces]
            and (net.cb is None) == (base.cb is None) and _solve.layout(net, act) == L)
    if not same:
        raise _StructureChanged("the perturbation changes the structure of the system (species pruned or added)")
    return replace(net, A=base.A, b=base.A @ net.amt, row_scale=base.row_scale, n_scale=base.n_scale, cb_col=base.cb_col)


# --------------------------------------------------------------------------- quantities
def _builtin(q, net, active):
    """``fn(o, v, st)`` for quantity kinds with a closed form in the state; ``None`` otherwise."""
    if not isinstance(q, str):
        return None
    names = {s.name: k for k, s in enumerate(net.species)}
    ypos = {k: i for i, k in enumerate(net.yidx)}
    if ":" not in q:
        if q == "pH" and "H+" in names:
            k = names["H+"]
            return lambda o, v, st: -st["la"][k]
        if q == "pe":
            k = next((i for i, s in enumerate(net.species) if s.phase == "electron"), None)
            return None if k is None else (lambda o, v, st: -st["la"][k])
        if q == "ionic_strength":
            z = [net.species[k].charge for k in net.aq_idx]
            return lambda o, v, st: v["I"] if v["I"] is not None else ionic_strength(o, z, st["m"])
        if q == "water_mass":
            return lambda o, v, st: v["W"]
        return None
    kind, name = q.split(":", 1)
    k = names.get(name)
    if k is None:
        return None
    kd, sp = net.kind[k], net.species[k]
    if kind == "molality" and kd == "aq":
        return lambda o, v, st: 10.0 ** v["y"][ypos[k]]
    if kind == "log10_molality" and kd == "aq":
        return lambda o, v, st: v["y"][ypos[k]]
    if kind == "log10_activity" and kd in ("aq", "water", "ex", "surf", "virt", "const"):
        return lambda o, v, st: st["la"][k]
    if kind == "gamma" and kd == "aq":
        return lambda o, v, st: 10.0 ** (st["la"][k] - v["y"][ypos[k]])
    if kind == "SI" and kd == "member":
        return lambda o, v, st: st["la_eq"][k]
    if kind == "SI" and kd == "const":
        return lambda o, v, st: net.fixed[k]
    if (kind == "solids" and kd == "member" and sp.is_solid) or (kind == "gases" and kd == "member" and sp.phase == "gas"):
        return lambda o, v, st: st["n"][k]
    if kind in ("exchange", "surface") and kd in ("ex", "surf"):
        return lambda o, v, st: st["n"][k]
    if kind == "mole_fraction" and kd == "member" and net.phases[net.member_phase[k]].kind == "ss":
        p = net.member_phase[k]
        return lambda o, v, st: 10.0 ** (st["la_eq"][k] - st["sigma"][p])
    return None  # e.g. partial pressures (assemblage-dependent): numeric path on rebuilt results


def _rows(o, net, act, L, x, qfuns):
    """Square residuals with every phase row written as ``s_p``, then the built-in quantities."""
    rows = list(_solve.square_equations(o, net, act, set(L.Nph), x))
    if qfuns:
        v = _solve._vars(net, L, [x[i] for i in range(L.n)])
        st = _solve.state(o, net, act, _solve._prepare(net, act), v)
        rows += [fn(o, v, st) for fn in qfuns]
    return rows


def _jax():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _eval_rows(net, act, L, x, qfuns) -> np.ndarray:
    try:
        r = _rows(_PLAIN, net, act, L, [float(t) for t in x], qfuns)
        if len(r) == L.n + len(qfuns):
            return np.array([float(t) for t in r])
    except (ValueError, OverflowError, ZeroDivisionError):
        pass
    # float evaluation dropped a term (exact zeros) or hit a domain error: evaluate as JAX does
    jax, jnp = _jax()
    r = _rows(JAX, net, act, L, jnp.asarray(np.asarray(x, float)), qfuns)
    return np.array([float(t) for t in r])


def _jac_rows(net, act, L, x, qfuns) -> np.ndarray:
    jax, jnp = _jax()

    def f(xx):
        return jnp.stack([jnp.asarray(r, dtype=jnp.float64) for r in _rows(JAX, net, act, L, xx, qfuns)])

    return np.asarray(jax.jacfwd(f)(jnp.asarray(np.asarray(x, float))))


# --------------------------------------------------------------------------- differences
def _stencil(spec: _Spec, ev, f0=None):
    """``d ev(theta)/d theta`` at ``spec.value`` and whether ``ev`` is affine in theta (to rounding)."""
    v0, h = spec.value, spec.step
    if v0 - h < spec.lower:
        f0 = np.asarray(ev(v0) if f0 is None else f0, float)
        f1, f2 = (np.asarray(ev(v0 + c * h), float) for c in (1.0, 2.0))
        lin = np.all(np.abs(f2 - 2.0 * f1 + f0) <= 1e-10 * (np.abs(f0) + np.abs(f1) + np.abs(f2)) + 1e-300)
        return (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * h), bool(lin)
    fp, fm = (np.asarray(ev(v0 + c * h), float) for c in (1.0, -1.0))
    gp, gm = (np.asarray(ev(v0 + c * h / 2.0), float) for c in (1.0, -1.0))
    d1, d2 = (fp - fm) / (2.0 * h), (gp - gm) / h
    lin = bool(np.all(np.abs(d1 - d2) <= 1e-8 * (np.abs(d1) + np.abs(d2)) + 1e-300))
    if not lin:
        return (4.0 * d2 - d1) / 3.0, False
    # affine: the difference is exact for any step, and rounding (eps |F| / h) shrinks with a larger one
    H = 1e3 * h
    if v0 - H >= spec.lower:
        try:
            dH = (np.asarray(ev(v0 + H), float) - np.asarray(ev(v0 - H), float)) / (2.0 * H)
        except (_StructureChanged, ValueError, OverflowError, ZeroDivisionError):
            dH = None
        if dH is not None and np.all(np.abs(dH - d2) <= 1e-7 * np.abs(dH) + 1e-9 * np.max(np.abs(fp - fm)) / h + 1e-300):
            return dH, True
    return d2, True


def _solve_ift(Gx, Gt, L, net, assemblage):
    """``dx/dtheta`` for an assemblage and the scaled condition number of ``F_x`` (``None`` if singular)."""
    n, ne = L.n, L.n - len(L.Nph)
    Fx, Ft = Gx[:n].copy(), Gt[:n].copy()
    for q, p in enumerate(L.Nph):
        if p not in assemblage:
            Fx[ne + q] = 0.0
            Fx[ne + q, L.ny + q] = 1.0 / net.n_scale
            Ft[ne + q] = 0.0
    rmax = np.max(np.abs(Fx), axis=1)
    if np.any(rmax == 0) or not np.all(np.isfinite(Fx)):
        return None, math.inf
    M = Fx / rmax[:, None]
    cmax = np.max(np.abs(M), axis=0)
    if np.any(cmax == 0):
        return None, math.inf
    M = M / cmax[None, :]
    cond = float(np.linalg.cond(M))
    if not math.isfinite(cond) or cond > 1e15:
        return None, cond
    bad = ~np.all(np.isfinite(Ft), axis=0)
    rhs = -np.nan_to_num(Ft) / rmax[:, None]
    dx = np.linalg.solve(M, rhs) / cmax[:, None]
    dx[:, bad] = np.nan
    return dx, cond


# --------------------------------------------------------------------------- main
def sensitivities(result, parameters, quantities, *, rel_step: float = REL_STEP, boundary_tol: float = BOUNDARY_TOL,
                  cond_limit: float = COND_LIMIT) -> SensitivityResult:
    """Implicit-function-theorem derivatives of ``quantities`` with respect to ``parameters`` at ``result``.

    Parameters
    ----------
    result : Result
        From :meth:`System.equilibrate` (it carries the solver state).
    parameters : list
        :class:`fit.Parameter` objects (``logK``, ``enthalpy``, ``activity_parameter``,
        custom setters), or specs: ``"total:<amounts key>"`` (input amount),
        ``"fixed:<species>"`` (fixed log10 activity / partial pressure), ``"pH"``,
        ``"pe"``, ``"T"``. The system is restored after each perturbation.
    quantities : list
        As in :func:`fit.observe`: strings or callables ``f(result)``.
    rel_step : float
        Relative step for the residual differences in ``theta`` (no solves, so
        no solver noise). Residuals affine in the parameter give exact results
        for any step.
    boundary_tol : float
        A phase with ``|N|/n_scale`` and ``|s|`` both below this is at its
        saturation boundary.
    cond_limit : float
        Scaled condition number of ``F_x`` above which the result is flagged.

    Returns
    -------
    SensitivityResult
        ``jacobian`` (quantities x parameters); ``issues`` lists degeneracies.
        ``F_x`` is differentiated with JAX (``jax_enable_x64`` is switched on).
    """
    from discopt.speciation import fit as _fit
    from discopt.speciation.system import _make_result

    stt = getattr(result, "_state", None)
    if not stt:
        raise ValueError("result carries no solver state; compute it with System.equilibrate")
    if isinstance(parameters, str) or hasattr(parameters, "get") and hasattr(parameters, "set"):
        parameters = [parameters]
    if isinstance(quantities, str) or callable(quantities):
        quantities = [quantities]
    params, quants = list(parameters), list(quantities)
    system, net, act = stt["system"], stt["net"], stt["act"]
    x = np.asarray(stt["x"], float)
    active = set(stt["active"])
    call = {k: v for k, v in stt["call"].items() if k not in ("method", "options")}
    method = stt["call"].get("method", result.method)
    amounts = dict(stt["amounts"])
    L = _solve.layout(net, act)
    ne, P, nq = L.n - len(L.Nph), len(params), len(quants)
    issues: list = []
    if not result.converged:
        issues.append("the equilibrium state did not converge")

    specs = [_resolve(system, amounts, call, p, net.n_scale, rel_step) for p in params]
    qf_all = [_builtin(q, net, active) for q in quants]
    bidx = [i for i, f in enumerate(qf_all) if f is not None]
    qfuns = [qf_all[i] for i in bidx]
    G0 = _eval_rows(net, act, L, x, qfuns)
    if len(G0) != L.n + len(qfuns):
        raise RuntimeError(f"internal: {len(G0) - len(qfuns)} residuals for {L.n} unknowns")
    Gx = _jac_rows(net, act, L, x, qfuns)

    Gt = np.full((L.n + len(qfuns), P), np.nan)
    linear = []
    for j, sp in enumerate(specs):
        fn = lambda nt, ac: _eval_rows(_aligned(net, L, nt, ac), ac, L, x, qfuns)
        try:
            base = np.asarray(sp.at(sp.value, fn))
            if np.max(np.abs(base[:L.n] - G0[:L.n]), initial=0.0) > 1e-9 * (1.0 + np.max(np.abs(G0[:L.n]), initial=0.0)):
                issues.append(f"parameter {sp.name!r}: the system no longer matches this result (parameters changed "
                              "after equilibrate?)")
            Gt[:, j], lin = _stencil(sp, lambda v, _sp=sp, _fn=fn: _sp.at(v, _fn), base)
        except (_StructureChanged, ValueError, KeyError) as e:
            issues.append(f"parameter {sp.name!r}: {e}")
            lin = False
        linear.append(lin)

    def dq(dx, assemblage):
        out = np.full((nq, P), np.nan)
        if dx is None:
            return out
        B = Gx[L.n:] @ np.nan_to_num(dx) + Gt[L.n:]
        B[:, ~np.all(np.isfinite(dx), axis=0)] = np.nan
        for r, i in enumerate(bidx):
            out[i] = B[r]
        for i, q in enumerate(quants):
            if qf_all[i] is not None:
                continue
            for j, sp in enumerate(specs):
                if not np.all(np.isfinite(dx[:, j])):
                    continue

                def ev(v, _q=q, _sp=sp, _dxj=dx[:, j]):
                    def g(nt, ac):
                        nt = _aligned(net, L, nt, ac)
                        xt = x + (v - _sp.value) * _dxj
                        r = _make_result(system, nt, ac, xt, assemblage, True, result.info, method)
                        return float(_fit.observe(r, _q))

                    return _sp.at(v, g)

                try:
                    out[i, j] = _stencil(sp, ev)[0]
                except (_StructureChanged, ValueError, KeyError, ZeroDivisionError, OverflowError) as e:
                    issues.append(f"quantity {_label(q)!r}, parameter {sp.name!r}: {e}")
        return out

    dx, cond = _solve_ift(Gx, Gt, L, net, active)
    if dx is None:
        issues.append(f"F_x is singular for the assemblage {sorted(net.phases[p].name for p in active)} "
                      f"(scaled cond {cond:.2g}); derivatives are undefined")
    elif cond > cond_limit:
        issues.append(f"F_x is ill-conditioned (scaled cond {cond:.2g}); expect ~{cond * 2.2e-16:.1g} relative error")
    J = dq(dx, active)

    boundary = []
    for q, p in enumerate(L.Nph):
        N, s = float(x[L.ny + q]) / net.n_scale, float(G0[ne + q])
        if abs(N) <= boundary_tol and abs(s) <= boundary_tol:
            boundary.append((q, p, N, s))
    left = right = None
    if len(boundary) == 1:
        q, p, N, s = boundary[0]
        name = net.phases[p].name
        issues.append(f"phase {name!r} is at its saturation boundary (N/n_scale={N:.2g}, s={s:.2g}): derivatives "
                      "are one-sided (see left/right)")
        dx_in, _ = _solve_ift(Gx, Gt, L, net, active | {p})
        dx_out, _ = _solve_ift(Gx, Gt, L, net, active - {p})
        if dx_in is None or dx_out is None:
            issues.append(f"F_x is singular with {name!r} {'present' if dx_in is None else 'absent'}")
        else:
            J_in = dq(dx_in, active | {p})
            J_out = dq(dx_out, active - {p})
            left, right = np.full((nq, P), np.nan), np.full((nq, P), np.nan)
            for j in range(P):
                dN = dx_in[L.ny + q, j]
                up_in = dN >= 0  # increasing theta keeps the phase (N grows) or dissolves it
                right[:, j] = J_in[:, j] if up_in else J_out[:, j]
                left[:, j] = J_out[:, j] if up_in else J_in[:, j]
                same = np.allclose(J_in[:, j], J_out[:, j], rtol=1e-8, atol=1e-14, equal_nan=True)
                J[:, j] = J_in[:, j] if same else np.nan
    elif len(boundary) > 1:
        names = [net.phases[p].name for _, p, _, _ in boundary]
        issues.append(f"phases {names} are all at their saturation boundaries: derivatives are one-sided and "
                      "depend on the direction of change")
        J[:] = np.nan

    values = np.full(nq, np.nan)
    for i, q in enumerate(quants):
        try:
            values[i] = _fit.observe(result, q)
        except (ValueError, KeyError, ZeroDivisionError):
            pass
    return SensitivityResult(
        parameters=[s.name for s in specs], quantities=[_label(q) for q in quants], values=values, jacobian=J,
        dx=np.full((L.n, P), np.nan) if dx is None else dx, condition=cond, linear=linear,
        boundary=[net.phases[p].name for _, p, _, _ in boundary], left=left, right=right, issues=issues,
    )

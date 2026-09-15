r"""Equilibrium solvers.

All methods compute the same state: mass action for every reaction,
conservation of the components, the activity model, and for every phase that
may be absent (pure solid, solid solution, fixed-pressure gas phase) the
complementarity condition

.. math:: N_p \ge 0,\qquad s_p \ge 0,\qquad N_p\,s_p = 0,

with ``N_p`` the amount of the phase and ``s_p`` its undersaturation
(``-SI`` for a pure solid, :math:`-\log_{10}\sum_j 10^{SI_j}` for a solid
solution, :math:`\log_{10}P - \log_{10}\sum_j P_j` for a gas phase). The
methods differ in how the phase assemblage is found:

``"complementarity"`` (default)
    Log-variable mass-action model with a smoothed Fischer-Burmeister equation
    per phase, :math:`a+s-\sqrt{a^2+s^2+2\varepsilon^2}=0`, :math:`a=N/n_s`,
    driven through an :math:`\varepsilon` homotopy from a phase-free speciation.
``"gibbs"``
    Minimize the (transformed) Gibbs energy in linear amounts subject to
    conservation. Needs an activity model with an excess Gibbs energy
    (ideal, Debye-Hueckel, Davies, extended DH, SIT, Pitzer). For ideal
    activities the program is convex, so a local solve is global.
``"enumerate"``
    Square solves for every subset of present phases; keep the one satisfying
    complementarity (lowest G if several). Exponential in the number of phases.

Every method finishes with :func:`active_set_polish`: a square solve for the
chosen assemblage, repaired (drop a negative phase / add the most supersaturated
one) until complementarity holds to tolerance.

Unknowns (in order): log10 molality of aqueous species, log10 mol of exchange
and surface species, log10 activity of virtual species (``e-``, ``X-``); phase
amounts ``N``; for each non-ideal solid solution, the log10 mole fractions of
its members and its undersaturation ``s``; water mass ``W`` (kg, when water is a
species and its mass is free); ionic strength ``I``; surface potentials ``psi`` (V, diffuse double
layer); the charge-balance adjustment ``delta`` (mol).
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np

import discopt.modeling as dm
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.solvers.nlp_backend import get_nlp_solver

from discopt.speciation import water as WP
from discopt.speciation.activity import EXPR, NUM, ActivityModel, Ideal, _is_zero, ionic_strength, truesdell_jones_ln_gamma
from discopt.speciation.network import Network
from discopt.speciation.species import ATM_BAR

LN10 = math.log(10.0)
LOG_LB, LOG_UB = -200.0, 3.0
VIRT_LB, VIRT_UB = -100.0, 100.0
I_LB, I_UB = 1e-14, 50.0
PSI_B = 2.0
S_B = 1e3  # bound on the undersaturation variable of a non-ideal solid solution (log10 units)
DEFAULT_EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10)
NLP_OPTIONS = {"tol": 1e-12, "max_iter": 3000, "print_level": 0}
TOL = 1e-8  # residual tolerance (scaled balances, log10 mass action, complementarity)
METHODS = ("complementarity", "gibbs", "enumerate")


@dataclass
class SolveInfo:
    method: str
    stages: list = field(default_factory=list)  # (label, status, iterations)
    active_set_iterations: int = 0
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------- layout
@dataclass
class Layout:
    ny: int
    Nph: list  # phase indices with an amount variable (complementarity phases)
    W: int | None
    I: int | None
    psi: int | None
    npsi: int
    delta: int | None
    n: int
    X: dict = field(default_factory=dict)  # non-ideal solid-solution phase -> start of its (log10 x..., s) block

    def N(self, x):
        return np.asarray(x[self.ny : self.ny + len(self.Nph)], float)


def layout(net: Network, act: ActivityModel) -> Layout:
    ny = net.na
    Nph = [p for p, ph in enumerate(net.phases) if ph.complementarity]
    pos = ny + len(Nph)
    X = {}
    for p in Nph:
        if net.phases[p].model is not None:
            X[p], pos = pos, pos + len(net.phases[p].members) + 1
    W = I = psi = delta = None
    if net.water is not None and net.water_mass is None:
        W, pos = pos, pos + 1
    if _needs_I(net, act):
        I, pos = pos, pos + 1
    npsi = len(net.surfaces)
    if npsi:
        psi, pos = pos, pos + npsi
    if net.cb is not None:
        delta, pos = pos, pos + 1
    return Layout(ny, Nph, W, I, psi, npsi, delta, pos, X)


def _needs_I(net, act) -> bool:
    charged = any(net.species[k].charge for k in net.aq_idx)
    return charged and (act.uses_I or bool(net.surfaces))


def _prepare(net, act):
    return act.prepare(net.aq, net.T, net.P * ATM_BAR)


def _vars(net, L, x) -> dict:
    """Split a vector (numbers or discopt scalars) into named parts."""
    N = [None] * len(net.phases)
    for q, p in enumerate(L.Nph):
        N[p] = x[L.ny + q]
    X = {}
    for p, a in L.X.items():
        nm = len(net.phases[p].members)
        X[p] = ([x[a + j] for j in range(nm)], x[a + nm])
    return {
        "X": X,
        "y": [x[i] for i in range(L.ny)],
        "N": N,
        "W": x[L.W] if L.W is not None else (net.water_mass if net.water_mass is not None else net.W0),
        "I": x[L.I] if L.I is not None else None,
        "psi": [x[L.psi + i] for i in range(L.npsi)],
        "delta": x[L.delta] if L.delta is not None else None,
    }


# --------------------------------------------------------------------------- state
def state(o, net: Network, act: ActivityModel, ctx, v: dict) -> dict:
    """Activities, amounts and phase quantities, as expressions (``o=EXPR``) or numbers (``o=NUM``)."""
    sp, kind = net.species, net.kind
    ypos = {k: i for i, k in enumerate(net.yidx)}
    y, W = v["y"], v["W"]
    aq = net.aq_idx
    m = [10.0 ** y[ypos[k]] for k in aq]
    I = v["I"]
    if I is None:
        I = ionic_strength(o, ctx.z, m) if (act.uses_I or net.surfaces) and np.any(ctx.z) else 0.0
    lng = act.ln_gamma(o, ctx, m, I)
    surf_of = {k: q for q, S in enumerate(net.surfaces) for k in S.members}
    coef_psi = WP.FARADAY / (WP.R_GAS * net.T * LN10)

    la, n = {}, {}
    for j, k in enumerate(aq):
        yi = y[ypos[k]]
        la[k] = yi if _is_zero(lng[j]) else yi + lng[j] * (1.0 / LN10)
        n[k] = m[j] if (isinstance(W, (int, float)) and W == 1.0) else m[j] * W
    for k in ypos:
        s, kd, yi = sp[k], kind[k], y[ypos[k]]
        if kd == "ex":
            T = max(net.site_total.get(s.site, 0.0), 1e-300)
            val = yi + math.log10(s.nsites / T)
            if not act.is_ideal and s.params.get("a0") is not None and not _is_zero(I):
                val = val + truesdell_jones_ln_gamma(o, s.nsites, s.params["a0"], s.params.get("b"), I, ctx.A, ctx.B) / LN10
            la[k], n[k] = val, 10.0**yi
        elif kd == "surf":
            T = max(net.site_total.get(s.site, 0.0), 1e-300)
            val = yi - math.log10(T)
            if k in surf_of and s.charge:
                val = val + (s.charge * coef_psi) * v["psi"][surf_of[k]]
            la[k], n[k] = val, 10.0**yi
        elif kd == "virt":
            la[k], n[k] = yi, 0.0
    if net.water is not None:
        la[net.water] = act.ln_water_activity(o, ctx, m, I, lng) * (1.0 / LN10)
        n[net.water] = W * (1.0 / WP.M_W)
    for k, val in net.fixed.items():
        if kind[k] == "const":
            la[k] = val

    la_eq = {}
    for k, rx in net.member_rxn.items():
        rest = [c * la[i] for i, c in rx.nu.items() if i != k]
        la_eq[k] = (rx.logK - o.sum(rest)) * (1.0 / rx.nu[k])
    nph_ = len(net.phases)
    s_ph, sig_ph, x_ph, nph = [None] * nph_, [None] * nph_, [None] * nph_, [None] * nph_
    for p, ph in enumerate(net.phases):
        leq = [la_eq[k] for k in ph.members]
        if ph.model is not None:  # non-ideal: composition and s are unknowns (see equations)
            lx, sv = v["X"][p]
            frac, sig = [10.0**e for e in lx], -sv
        elif len(leq) == 1:
            sig, frac = leq[0], [1.0]
        else:
            sig = _log10_sum_pow10(o, leq)
            frac = [10.0 ** (e - sig) for e in leq]
        sig_ph[p], x_ph[p] = sig, frac
        if ph.kind == "gas_v":
            RTV = 101.325 * ph.volume / (WP.R_GAS * net.T)
            nph[p] = [RTV * 10.0**e for e in leq]
            for k, a in zip(ph.members, nph[p]):
                n[k] = a
            continue
        s_ph[p] = (-sig) if ph.kind in ("pure", "ss") else (math.log10(ph.pressure) - sig)
        N = v["N"][p]
        nph[p] = [N if (isinstance(f, float) and f == 1.0) else N * f for f in frac]
        for k, a in zip(ph.members, nph[p]):
            n[k] = n[k] + a if k in n else a  # a second copy of a phase adds to the same species
    return {"m": m, "I": I, "lng": lng, "la": la, "n": n, "la_eq": la_eq, "s": s_ph, "sigma": sig_ph,
            "x": x_ph, "nph": nph}


def _log10_sum_pow10(o, vals):
    if o is NUM:
        mx = max(vals)
        return mx + math.log10(math.fsum(10.0 ** (e - mx) for e in vals))
    return o.log10(o.sum([10.0**e for e in vals]))


def equations(o, net, act, ctx, v, st) -> list:
    """``(name, value, kind)`` for every equation except complementarity; values vanish at equilibrium."""
    out = []
    ncol = [st["n"][k] for k in net.cols]
    for r in range(net.A.shape[0]):
        terms = [float(net.A[r, j]) * ncol[j] for j in range(len(ncol)) if net.A[r, j] and not _is_zero(ncol[j])]
        if v["delta"] is not None and net.cb_col[r]:
            terms.append(float(-net.cb_col[r]) * v["delta"])
        if not terms:
            if abs(net.b[r]) > 0:
                raise ValueError("a conservation law has a nonzero total but no species that can carry it")
            continue
        out.append((f"conservation_{r}", (o.sum(terms) - float(net.b[r])) * (1.0 / float(net.row_scale[r])), "balance"))
    la = st["la"]
    for j, rx in enumerate(net.rxns):
        out.append((f"mass_action_{j}", o.sum([c * la[k] for k, c in rx.nu.items()]) - rx.logK, "mass_action"))
    for k, val in net.fixed.items():
        if net.kind[k] != "const":
            out.append((f"fixed_{net.species[k].name}", la[k] - val, "mass_action"))
    if v["I"] is not None:
        Ic = ionic_strength(o, ctx.z, st["m"])
        scale = max(v["I"], 1e-12) if o is NUM else 1.0
        out.append(("ionic_strength", (v["I"] - Ic) * (1.0 / scale) if o is NUM else v["I"] - Ic, "ionic_strength"))
    if v["delta"] is not None:
        q = [net.species[k].charge * st["n"][k] for k in st["n"]
             if net.species[k].charge and net.kind[k] in ("aq", "ex", "surf")]
        out.append(("charge_balance", o.sum(q) * (1.0 / net.n_scale), "charge_balance"))
    if net.surfaces:
        gc = WP.gouy_chapman_coefficient(net.T, net.P * ATM_BAR)
        sq = o.sqrt(st["I"])
        for i, S in enumerate(net.surfaces):
            q = o.sum([net.species[k].charge * st["n"][k] for k in S.members if net.species[k].charge])
            ddl = (S.area * S.mass * gc / WP.FARADAY) * sq * o.sinh((WP.FARADAY / (2.0 * WP.R_GAS * net.T)) * v["psi"][i])
            sites = {net.species[k].site for k in S.members}
            T = max(sum(net.site_total.get(site, 0.0) for site in sites), 1e-30)
            out.append((f"surface_charge_{S.name}", (q - ddl) * (1.0 / T), "surface"))
    for p, (lx, sv) in v["X"].items():
        # non-ideal solid solution: log10(x_j lambda_j) = SI_j + s, sum x_j = 1 (stationary tangent-plane distance)
        ph = net.phases[p]
        xs = st["x"][p]
        lnl = ph.model.ln_lambda(o, ph.fractions(xs))
        for j, k in enumerate(ph.members):
            val = lx[j] + lnl[ph.mpos[j]] * (1.0 / LN10) - st["la_eq"][k] - sv
            out.append((f"ss_{ph.name}_{net.species[k].name}", val, "mass_action"))
        out.append((f"ss_{ph.name}_sum", o.sum(xs) - 1.0, "balance"))
    return out


# --------------------------------------------------------------------------- numeric evaluation
def evaluate(net: Network, act: ActivityModel, x) -> tuple[dict, dict]:
    """Numeric ``(vars, state)`` at ``x``."""
    L = layout(net, act)
    x = [float(t) for t in x]
    v = _vars(net, L, x)
    return v, state(NUM, net, act, _prepare(net, act), v)


def residuals(net: Network, act: ActivityModel, x) -> dict:
    L = layout(net, act)
    ctx = _prepare(net, act)
    xv = [float(t) for t in x]
    v = _vars(net, L, xv)
    st = state(NUM, net, act, ctx, v)
    out = {"balance": 0.0, "mass_action": 0.0, "complementarity": 0.0}
    for _, val, kind in equations(NUM, net, act, ctx, v, st):
        out[kind] = max(out.get(kind, 0.0), abs(val))
    for p in L.Nph:
        out["complementarity"] = max(out["complementarity"], abs(min(v["N"][p] / net.n_scale, st["s"][p])))
    return out


def phase_status(net, act, x):
    """``(N, s)`` per complementarity phase (in ``layout.Nph`` order)."""
    v, st = evaluate(net, act, x)
    L = layout(net, act)
    return np.array([v["N"][p] for p in L.Nph]), np.array([st["s"][p] for p in L.Nph])


def saturation_indices(net, act, x) -> dict:
    """Per phase-member species index: ``log10`` saturation (SI for solids, log10 partial pressure for gases)."""
    return {k: float(val) for k, val in evaluate(net, act, x)[1]["la_eq"].items()}


def initial_point(net: Network, act: ActivityModel) -> np.ndarray:
    L = layout(net, act)
    x = np.zeros(L.n)
    amt = dict(zip(net.cols, net.amt))
    W0 = net.W0
    count: dict = {}
    for k in net.yidx:
        if net.kind[k] in ("ex", "surf"):
            count[net.species[k].site] = count.get(net.species[k].site, 0) + 1
    m0 = []
    for i, k in enumerate(net.yidx):
        s, kd = net.species[k], net.kind[k]
        if kd == "aq":
            x[i] = net.fixed[k] if k in net.fixed else math.log10(max(amt.get(k, 0.0) / W0, 1e-10))
            m0.append(10.0 ** x[i])
        elif kd in ("ex", "surf"):
            x[i] = math.log10(max(net.site_total.get(s.site, 0.0) / count[s.site], 1e-30))
        else:
            x[i] = -4.0 if s.phase == "electron" else 0.0
    if L.W is not None:
        x[L.W] = W0
    if L.I is not None:
        z = [net.species[k].charge for k in net.aq_idx]
        x[L.I] = max(ionic_strength(NUM, z, m0), I_LB)
    for p, a in L.X.items():
        nm = len(net.phases[p].members)
        x[a : a + nm] = -math.log10(nm)
    return x


def _eliminate(A, b, col):
    """Combine rows so only one has a nonzero ``col`` entry, and drop that row."""
    rows = np.flatnonzero(np.abs(col) > 1e-12)
    if not len(rows):
        return A, b
    p = rows[np.argmax(np.abs(col[rows]))]
    A, b = A.copy(), b.copy()
    for r in rows:
        if r != p:
            f = col[r] / col[p]
            A[r] -= f * A[p]
            b[r] -= f * b[p]
    keep = [r for r in range(A.shape[0]) if r != p]
    return A[keep], b[keep]


def consistent_point(net: Network, act: ActivityModel, max_iter: int = 200):
    """Phase-free ideal speciation that satisfies mass action and conservation exactly.

    Mass action fixes ``log10 m`` up to the conserved directions, so
    ``ln n = c + B^T lam`` with ``B`` the (water-free) conservation rows over the
    variable species. The totals then follow from minimizing the convex dual
    ``sum(n) - b.lam`` by damped Newton; the charge-balance adjustment is found
    by a secant iteration around it. Returns ``None`` when the system has
    exchange or surface species, or the iteration fails.
    """
    if any(net.kind[k] in ("ex", "surf") for k in net.yidx) or not net.rxns:
        return None
    cpos = {k: j for j, k in enumerate(net.cols)}
    var = [k for k in net.yidx if k in cpos and net.kind[k] in ("aq", "virt")]
    vpos = {k: i for i, k in enumerate(var)}
    M = np.zeros((len(net.rxns), len(var)))
    rhs = np.zeros(len(net.rxns))
    for j, rx in enumerate(net.rxns):
        rhs[j] = rx.logK
        for k, c in rx.nu.items():
            if k in vpos:
                M[j, vpos[k]] = c
            elif k in net.fixed:
                rhs[j] -= c * net.fixed[k]
            elif net.kind[k] != "water":
                return None
    y_p, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    if np.max(np.abs(M @ y_p - rhs), initial=0.0) > 1e-8:
        return None

    cbc = np.zeros(net.A.shape[0]) if net.cb_col is None else net.cb_col
    Ab = np.column_stack([net.A, net.b, cbc])
    if net.water is not None and net.water in cpos:
        Ab, _ = _eliminate(Ab, np.zeros(Ab.shape[0]), net.A[:, cpos[net.water]])
    A, b, cb = Ab[:, :-2], Ab[:, -2], Ab[:, -1]
    B = A[:, [cpos[k] for k in var]]
    aq = np.array([net.kind[k] == "aq" for k in var])
    if not aq.any():
        return None
    Baq = B[:, aq]
    c = math.log(net.W0) + LN10 * y_p
    scale = np.maximum(np.maximum(np.abs(b), np.abs(Baq).max(axis=1) * net.n_scale), 1e-300)

    x0 = initial_point(net, act)
    ln_start = np.array([math.log(net.W0) + LN10 * x0[net.yidx.index(k)] for k in var])
    lam0, *_ = np.linalg.lstsq(B.T[aq], (ln_start - c)[aq], rcond=None)

    def amounts(lam):
        return np.exp(np.clip(c[aq] + Baq.T @ lam, -700.0, 60.0))

    def dual(bt, lam):
        phi = lambda l: amounts(l).sum() / net.n_scale - bt @ l / net.n_scale
        for _ in range(max_iter):
            n = amounts(lam)
            g = (Baq @ n - bt) / scale
            if np.max(np.abs(g), initial=0.0) < 1e-11:
                return lam, n
            H = (Baq * n) @ Baq.T
            H += np.eye(len(lam)) * 1e-14 * max(np.trace(H), 1e-300)
            step = -np.linalg.lstsq(H, Baq @ n - bt, rcond=None)[0]
            big = np.max(np.abs(step), initial=0.0)
            if big > 10.0:
                step *= 10.0 / big
            f0, t = phi(lam), 1.0
            while t > 1e-10 and not phi(lam + t * step) <= f0 + 1e-12 * abs(f0):
                t *= 0.5
            lam = lam + t * step
        return None

    def charge(n):
        q = sum(net.species[k].charge * ni for k, ni in zip(np.array(var)[aq], n))
        return q + sum(net.species[k].charge * net.W0 * 10.0 ** v for k, v in net.fixed.items()
                       if net.kind[k] == "aq")

    delta = 0.0
    out = dual(b, lam0)
    if out is None:
        return None
    lam, n = out
    if net.cb is not None:
        slope = float(net.species[net.cb].charge)
        q = charge(n)
        # only a starting point: stop at the roundoff level of the charge sum and let the NLP polish
        for _ in range(60):
            if abs(q) <= 1e-9 * net.n_scale:
                break
            step = -q / slope
            for _ in range(30):
                trial = dual(b + cb * (delta + step), lam)
                if trial is not None:
                    break
                step *= 0.5
            else:
                break
            q1 = charge(trial[1])
            if q1 != q:
                slope = (q1 - q) / step
            if abs(q1) >= abs(q):
                break
            delta, q, lam, n = delta + step, q1, trial[0], trial[1]

    L = layout(net, act)
    x = x0.copy()
    lnv = c + B.T @ lam
    for i, k in enumerate(var):
        y = (lnv[i] - math.log(net.W0)) / LN10
        lb, ub = (LOG_LB, LOG_UB) if net.kind[k] == "aq" else (VIRT_LB, VIRT_UB)
        x[net.yidx.index(k)] = min(max(y, lb), ub)
    for k, v in net.fixed.items():
        if k in net.yidx:
            x[net.yidx.index(k)] = v
    if L.delta is not None:
        x[L.delta] = delta
    if L.I is not None:
        m = [10.0 ** x[net.yidx.index(k)] for k in net.aq_idx]
        x[L.I] = min(max(ionic_strength(NUM, [net.species[k].charge for k in net.aq_idx], m), I_LB), I_UB)
    return x


def convert(net, act_from, x, act_to) -> np.ndarray:
    """Map a solution vector between the layouts of two activity models."""
    La, Lb = layout(net, act_from), layout(net, act_to)
    if La == Lb:
        return np.asarray(x, float).copy()
    out = np.zeros(Lb.n)
    out[: Lb.ny + len(Lb.Nph)] = x[: La.ny + len(La.Nph)]
    for name in ("W", "delta"):
        a, b = getattr(La, name), getattr(Lb, name)
        if a is not None and b is not None:
            out[b] = x[a]
    if La.psi is not None:
        out[Lb.psi : Lb.psi + Lb.npsi] = x[La.psi : La.psi + La.npsi]
    for p, a in La.X.items():
        nm = len(net.phases[p].members) + 1
        out[Lb.X[p] : Lb.X[p] + nm] = x[a : a + nm]
    if Lb.I is not None:
        v, st = evaluate(net, act_from, x)
        z = [net.species[k].charge for k in net.aq_idx]
        out[Lb.I] = min(max(ionic_strength(NUM, z, st["m"]), I_LB), I_UB)
    return out


# --------------------------------------------------------------------------- model
def build_model(net: Network, act: ActivityModel, mode: str, active=frozenset()):
    """Mass-action model.

    ``mode``: ``"speciation"`` (all phases absent), ``"square"`` (phases in
    ``active`` saturated, others absent; amounts free in sign so a wrong
    assemblage shows up as a negative amount), or ``"fb"`` (smoothed
    Fischer-Burmeister complementarity; also returns the ``eps`` parameter).
    """
    m = dm.Model(f"thermo_{mode}")
    L = layout(net, act)
    ctx = _prepare(net, act)
    xs = model_variables(m, net, act, free_sign=(mode == "square"))
    eps = m.parameter("eps", value=DEFAULT_EPS[0]) if mode == "fb" else None

    v = _vars(net, L, xs)
    st = state(EXPR, net, act, ctx, v)
    for name, expr, _ in equations(EXPR, net, act, ctx, v, st):
        m.subject_to(expr == 0, name=name)
    for p in L.Nph:
        Np, s = v["N"][p], st["s"][p]
        label = net.phases[p].name
        if mode == "speciation" or (mode == "square" and p not in active):
            m.subject_to(Np == 0, name=f"absent_{label}")
        elif mode == "square":
            m.subject_to(s == 0, name=f"saturated_{label}")
        else:
            a = Np * (1.0 / net.n_scale)
            m.subject_to(a + s - dm.sqrt(a * a + s * s + 2.0 * eps * eps) == 0, name=f"complementarity_{label}")
    m.minimize(0.0)
    return m, eps


def model_variables(m, net: Network, act: ActivityModel, free_sign: bool = False) -> list:
    """Create the unknowns of :func:`layout` on discopt model ``m``; returns them as a flat list.

    Phase amounts are ``>= 0`` unless ``free_sign``. Use with :func:`_vars`,
    :func:`state` and :func:`equations` to build equilibrium models with extra
    decision variables or parameters.
    """
    L = layout(net, act)
    xs: list = []
    groups = [("log10_m", ("aq",), LOG_LB, LOG_UB), ("log10_n", ("ex", "surf"), LOG_LB, LOG_UB),
              ("log10_a", ("virt",), VIRT_LB, VIRT_UB)]
    for name, kinds, lb, ub in groups:
        cnt = sum(net.kind[k] in kinds for k in net.yidx)
        if cnt:
            var = m.continuous(name, shape=(cnt,), lb=lb, ub=ub)
            xs += [var[i] for i in range(cnt)]
    nb = 10.0 * max(net.n_scale, 1.0)
    if L.Nph:
        N = m.continuous("N", shape=(len(L.Nph),), lb=(-nb if free_sign else 0.0), ub=nb)
        xs += [N[i] for i in range(len(L.Nph))]
    for p in L.X:
        nm = len(net.phases[p].members)
        lx = m.continuous(f"log10_x_{p}", shape=(nm,), lb=LOG_LB, ub=0.0)
        xs += [lx[i] for i in range(nm)]
        xs.append(m.continuous(f"s_{p}", lb=-S_B, ub=S_B))
    if L.W is not None:
        xs.append(m.continuous("W", lb=1e-4 * net.W0, ub=1e2 * net.W0))
    if L.I is not None:
        xs.append(m.continuous("I", lb=I_LB, ub=I_UB))
    if L.npsi:
        psi = m.continuous("psi", shape=(L.npsi,), lb=-PSI_B, ub=PSI_B)
        xs += [psi[i] for i in range(L.npsi)]
    if L.delta is not None:
        xs.append(m.continuous("delta", lb=-nb, ub=nb))
    if len(xs) != L.n:
        raise RuntimeError(f"internal: created {len(xs)} variables, layout has {L.n}")
    return xs


def square_equations(o, net: Network, act: ActivityModel, active, x) -> list:
    """All residuals of the square system for assemblage ``active`` (phase indices) at ``x``.

    Phases in ``active`` are saturated (``s = 0``), the other complementarity
    phases absent (``N = 0``). With ``o=JAX`` the list can be stacked and
    differentiated; the system is square, so its Jacobian gives implicit-function
    sensitivities of a converged state.
    """
    L = layout(net, act)
    ctx = _prepare(net, act)
    v = _vars(net, L, [x[i] for i in range(L.n)])
    st = state(o, net, act, ctx, v)
    out = [val for _, val, _ in equations(o, net, act, ctx, v, st)]
    for p in L.Nph:
        out.append(st["s"][p] if p in active else v["N"][p] * (1.0 / net.n_scale))
    return out


def _local_solve(model, x0, options):
    ev = make_evaluator(model)
    lb, ub = ev.variable_bounds
    if len(x0) != len(lb):
        raise RuntimeError(f"internal: x0 has {len(x0)} entries, model has {len(lb)} variables")
    x0 = np.clip(np.asarray(x0, float), lb, ub)
    return get_nlp_solver("auto")(ev, x0, options=options)


def _ok(net, act, x, skip=("complementarity",)) -> bool:
    r = residuals(net, act, x)
    return all(val <= TOL for k, val in r.items() if k not in skip)


def seed_compositions(net, act, x) -> np.ndarray:
    """Start each non-ideal solid solution at its lowest tangent-plane-distance local minimum at ``x``.

    The stationarity equations ``x_j lambda_j = 10^(SI_j + s)`` have several roots
    inside a miscibility gap; Newton's method from the barycenter can stall
    between them (local infeasibility). The lowest local minimum of
    ``D(x) = sum x_j (ln x_j + ln lambda_j - ln10 SI_j)`` is the physical root.
    """
    L = layout(net, act)
    if not L.X:
        return x
    from scipy.optimize import minimize

    x = np.asarray(x, float).copy()
    _, st = evaluate(net, act, x)
    for p, a in L.X.items():
        ph = net.phases[p]
        nm = len(ph.members)
        SI = np.array([float(st["la_eq"][k]) for k in ph.members])

        def D(xv, ph=ph, SI=SI):
            xv = np.clip(xv, 1e-300, 1.0)
            return float(np.dot(xv, np.log(xv) - LN10 * SI) + ph.model.excess(NUM, ph.fractions(list(xv))))

        ideal = np.exp(LN10 * (SI - SI.max()))
        starts = [ideal / ideal.sum()] + [np.where(np.arange(nm) == j, 1.0 - (nm - 1) * 1e-4, 1e-4) for j in range(nm)]
        best = None
        for x0 in starts:
            out = minimize(D, x0, method="SLSQP", bounds=[(1e-300, 1.0)] * nm,
                           constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1.0}],
                           options={"ftol": 1e-14, "maxiter": 200})
            xv = np.clip(out.x, 1e-300, 1.0)
            xv = xv / xv.sum()
            if best is None or D(xv) < D(best):
                best = xv
        lx = np.maximum(np.log10(best), LOG_LB)
        lnl = ph.model.ln_lambda(NUM, ph.fractions(list(best)))
        x[a : a + nm] = lx
        x[a + nm] = float(np.clip(np.mean([lx[j] + lnl[ph.mpos[j]] / LN10 - SI[j] for j in range(nm)]), -S_B, S_B))
    return x


# --------------------------------------------------------------------------- stages
def solve_speciation(net, act, options, info, x0=None) -> np.ndarray:
    model, _ = build_model(net, act, "speciation")
    start = seed_compositions(net, act, initial_point(net, act) if x0 is None else x0)
    res = _local_solve(model, start, options)
    info.stages.append(("speciation", res.status.name, res.iterations))
    if res.x is not None and _ok(net, act, res.x):
        return np.asarray(res.x, float)
    if x0 is None:
        xc = consistent_point(net, act)
        if xc is not None:
            info.notes.append("speciation restarted from a mass-action-consistent ideal point")
            rc = _local_solve(model, seed_compositions(net, act, xc), options)
            info.stages.append(("speciation (consistent start)", rc.status.name, rc.iterations))
            if rc.x is not None and _ok(net, act, rc.x):
                return np.asarray(rc.x, float)
    if not act.is_ideal and x0 is None:
        info.notes.append("speciation started from the ideal-activity speciation")
        xi = solve_speciation(net, Ideal(), options, info)
        res = _local_solve(model, convert(net, Ideal(), xi, act), options)
        info.stages.append(("speciation (from ideal)", res.status.name, res.iterations))
        if res.x is not None:
            return np.asarray(res.x, float)
    if res.x is None:
        raise RuntimeError(f"speciation solve failed ({res.status.name})")
    return np.asarray(res.x, float)


def solve_square(net, act, active, starts, options, info):
    """Square solve for a fixed assemblage (phase indices); tries each start in turn."""
    model, _ = build_model(net, act, "square", frozenset(active))
    for x0 in starts:
        if x0 is None:
            continue
        res = _local_solve(model, x0, options)
        info.stages.append((f"square{sorted(active)}", res.status.name, res.iterations))
        if res.x is None:
            continue
        x = np.asarray(res.x, float)
        if _ok(net, act, x):
            return x
    return None


def active_set_polish(net, act, active, x0, options, info, fallback=None):
    """Square solve for ``active``; repair until complementarity holds."""
    L = layout(net, act)
    active = set(active)
    x = x0
    seen = set()
    for it in range(4 * len(L.Nph) + 4):
        info.active_set_iterations = it + 1
        key = frozenset(active)
        if key in seen:
            info.notes.append("active-set repair cycled")
            break
        seen.add(key)
        xs = solve_square(net, act, active, [x, fallback], options, info)
        if xs is None:
            info.notes.append(f"square solve failed for active set {sorted(active)}")
            if not active:
                break
            # Often a singular set (e.g. two polymorphs of one formula): drop
            # the least abundant phase and repair from there.
            N, _ = phase_status(net, act, x)
            pos = {p: q for q, p in enumerate(L.Nph)}
            active.remove(min(active, key=lambda p: N[pos[p]]))
            continue
        x = xs
        N, s = phase_status(net, act, x)
        pos = {p: q for q, p in enumerate(L.Nph)}
        neg = [p for p in active if N[pos[p]] < -TOL * net.n_scale]
        if neg:
            active.remove(min(neg, key=lambda p: N[pos[p]]))
            continue
        sup = [p for p in L.Nph if p not in active and s[pos[p]] < -TOL]
        if sup:
            active.add(min(sup, key=lambda p: s[pos[p]]))
            continue
        x = x.copy()
        for q, p in enumerate(L.Nph):  # clean round-off
            x[L.ny + q] = max(N[q], 0.0) if p in active else 0.0
        return x, active, True
    return x, active, False


def _assemblage_guess(net, act, x) -> set:
    L = layout(net, act)
    N, s = phase_status(net, act, x)
    return {p for q, p in enumerate(L.Nph) if N[q] / net.n_scale > s[q]}


def solve_complementarity(net, act, options, info, eps_seq=DEFAULT_EPS):
    x_spec = solve_speciation(net, act, options, info)
    L = layout(net, act)
    if not L.Nph:
        return x_spec, set(), _ok(net, act, x_spec)
    x = x_spec
    model, eps = build_model(net, act, "fb")
    for e in eps_seq:
        eps.value = float(e)
        res = _local_solve(model, x, options)
        info.stages.append((f"fb eps={e:g}", res.status.name, res.iterations))
        if res.x is not None and np.all(np.isfinite(res.x)):
            x = np.asarray(res.x, float)
    x, active, ok = active_set_polish(net, act, _assemblage_guess(net, act, x), x, options, info, fallback=x_spec)
    if not ok and not act.is_ideal:
        # Activity corrections rarely change the assemblage: start the repair
        # from the (easier) ideal-activity equilibrium.
        info.notes.append("retrying from the ideal-activity equilibrium")
        ideal = Ideal()
        xi, ai, oki = solve_complementarity(net, ideal, options, info, eps_seq)
        if oki:
            x, active, ok = active_set_polish(net, act, ai, convert(net, ideal, xi, act), options, info, fallback=x_spec)
    return x, active, ok


# --------------------------------------------------------------------------- Gibbs
def _gibbs_check(net, act):
    if not act.has_excess:
        raise NotImplementedError(
            f"method='gibbs' needs an activity model with an excess Gibbs energy; {act.name!r} "
            "(species-specific ion sizes) has none. Use method='complementarity', or an ideal, "
            "Debye-Hueckel, Davies, extended Debye-Hueckel, SIT or Pitzer model"
        )
    if net.surfaces:
        raise NotImplementedError("method='gibbs' does not support electrostatic surfaces; use 'complementarity'")
    if net.water is not None and net.water_mass is not None:
        raise NotImplementedError("method='gibbs' needs a free water mass when water is a species")


def _gibbs_terms(o, net, act, ctx, amounts: dict, W, sc: float = 1.0):
    """G/(RT sc) from species amounts (mol, divided by ``sc``); shared by the model and :func:`gibbs_energy`."""
    sp, kind = net.species, net.kind
    colpos = {k: j for j, k in enumerate(net.cols)}
    lnsc = math.log(sc)
    lnW = o.log(W) if not isinstance(W, (int, float)) else math.log(W)
    terms = []
    aq = net.aq_idx
    m = [amounts[k] * (sc / W) for k in aq]
    for k in aq:
        c = amounts[k]
        mu = -LN10 * net.fixed[k] if k in net.fixed else float(net.mu0[colpos[k]])
        terms.append(c * (mu + lnsc + o.log(c) - lnW - 1.0))
    if not act.is_ideal and np.any(ctx.z):
        I = ionic_strength(o, ctx.z, m)
        terms.append((W * (1.0 / sc)) * act.excess_gibbs(o, ctx, m, I))
    elif not act.is_ideal:
        g = act.excess_gibbs(o, ctx, m, 0.0)
        if not _is_zero(g):
            terms.append((W * (1.0 / sc)) * g)
    if net.water is not None:
        terms.append((W * (1.0 / (WP.M_W * sc))) * float(net.mu0[colpos[net.water]]))
    for k in net.yidx:
        if kind[k] in ("ex", "surf"):
            s, c = sp[k], amounts[k]
            T = net.site_total[s.site]
            terms.append(c * (float(net.mu0[colpos[k]]) + math.log(s.nsites * sc / T) + o.log(c) - 1.0))
    for p, ph in enumerate(net.phases):
        # amounts keyed (phase, species) take precedence: several copies of one solid solution
        cs = [amounts[(p, k)] if (p, k) in amounts else amounts[k] for k in ph.members]
        mus = [float(net.mu0[colpos[k]]) for k in ph.members]
        if ph.kind == "pure":
            terms.append(cs[0] * mus[0])
        elif ph.kind in ("ss", "gas_p"):
            tot = o.sum(cs)
            lntot = o.log(tot)
            lnP = math.log(ph.pressure) if ph.kind == "gas_p" else 0.0
            terms += [c * (mu + lnP + o.log(c) - lntot) for c, mu in zip(cs, mus)]
            if ph.model is not None and not (isinstance(tot, (int, float)) and tot <= 0.0):
                terms.append(tot * ph.model.excess(o, ph.fractions([c / tot for c in cs])))
        else:
            k0 = math.log(sc * WP.R_GAS * net.T / (101.325 * ph.volume))
            terms += [c * (mu + k0 + o.log(c) - 1.0) for c, mu in zip(cs, mus)]
    return o.sum(terms)


def solve_gibbs(net, act, options, info):
    _gibbs_check(net, act)
    if any(ph.copy_of is not None for ph in net.phases):
        raise NotImplementedError("method='gibbs' does not take networks with copies of a phase")
    if any(ph.model is not None for ph in net.phases):
        info.notes.append("non-ideal solid solution: the Gibbs program is non-convex, its local minimum may be metastable")
    x_spec = solve_speciation(net, act, options, info)
    L = layout(net, act)
    if not L.Nph and not any(ph.kind == "gas_v" for ph in net.phases):
        return x_spec, set(), _ok(net, act, x_spec)
    sc = net.n_scale
    ctx = _prepare(net, act)
    v0, st0 = evaluate(net, act, x_spec)
    amt_idx = [k for k in net.yidx if net.kind[k] in ("aq", "ex", "surf")] + [k for ph in net.phases for k in ph.members]
    m = dm.Model("thermo_gibbs")
    c = m.continuous("c", shape=(len(amt_idx),), lb=0.0, ub=1e3)
    Wv = m.continuous("W", lb=1e-4 * net.W0, ub=1e2 * net.W0) if L.W is not None else net.W0
    d = m.continuous("delta", lb=-1e3, ub=1e3) if net.cb is not None else None
    amounts = {k: c[i] for i, k in enumerate(amt_idx)}
    full = dict(amounts)
    for k in net.yidx:
        if net.kind[k] == "virt":
            full[k] = 0.0
    for r in range(net.A.shape[0]):
        terms = []
        for j, k in enumerate(net.cols):
            if not net.A[r, j] or net.kind[k] == "virt":
                continue
            amt = Wv * (1.0 / (WP.M_W * sc)) if k == net.water else full[k]
            terms.append(float(net.A[r, j]) * amt)
        if d is not None and net.cb_col[r]:
            terms.append(float(-net.cb_col[r]) * d)
        if terms:
            m.subject_to((dm.sum(terms) * sc - float(net.b[r])) * (1.0 / float(net.row_scale[r])) == 0)
    if d is not None:
        q = [net.species[k].charge * amounts[k] for k in amt_idx if net.species[k].charge]
        m.subject_to(dm.sum(q) == 0)
    m.minimize(_gibbs_terms(EXPR, net, act, ctx, amounts, Wv, sc))

    x0 = [max(float(st0["n"][k]) / sc, 1e-8) if net.kind[k] in ("aq", "ex", "surf") else 1e-6 for k in amt_idx]
    if L.W is not None:
        x0.append(float(v0["W"]))
    if d is not None:
        x0.append(0.0)
    opts = dict(options, bound_relax_factor=0.0, honor_original_bounds="yes")
    res = _local_solve(m, np.array(x0), opts)
    info.stages.append(("gibbs", res.status.name, res.iterations))
    if res.x is None:
        raise RuntimeError(f"Gibbs minimization failed ({res.status.name})")
    xg = np.asarray(res.x, float)
    ng = {k: xg[i] * sc for i, k in enumerate(amt_idx)}
    Wg = float(xg[len(amt_idx)]) if L.W is not None else net.W0
    x = np.zeros(L.n)
    for i, k in enumerate(net.yidx):
        kd = net.kind[k]
        if kd == "aq":
            x[i] = math.log10(max(ng[k] / Wg, 1e-200))
        elif kd in ("ex", "surf"):
            x[i] = math.log10(max(ng[k], 1e-200))
    active = set()
    for q, p in enumerate(L.Nph):
        tot = sum(ng[k] for k in net.phases[p].members)
        x[L.ny + q] = tot
        if tot > 1e-9 * sc:
            active.add(p)
    for p, a in L.X.items():
        mem = net.phases[p].members
        tot = sum(ng[k] for k in mem)
        for j, k in enumerate(mem):
            x[a + j] = math.log10(max(ng[k] / tot, 1e-200)) if tot > 1e-12 * sc else -math.log10(len(mem))
    if L.W is not None:
        x[L.W] = Wg
    if L.delta is not None:
        x[L.delta] = float(xg[-1]) * sc
    x = _fill_virtual(net, act, x)
    if L.I is not None:
        v, st = evaluate(net, act, x)
        x[L.I] = min(max(ionic_strength(NUM, [net.species[k].charge for k in net.aq_idx], st["m"]), I_LB), I_UB)
    return active_set_polish(net, act, active, x, options, info, fallback=x_spec)


def _fill_virtual(net, act, x):
    """Log activities of virtual species from the mass-action equations (least squares)."""
    virt = [i for i, k in enumerate(net.yidx) if net.kind[k] == "virt"]
    if not virt or not net.rxns:
        return x
    x = x.copy()
    for i in virt:
        x[i] = 0.0
    _, st = evaluate(net, act, x)
    vk = [net.yidx[i] for i in virt]
    M = np.array([[rx.nu.get(k, 0.0) for k in vk] for rx in net.rxns])
    r = np.array([sum(c * st["la"][k] for k, c in rx.nu.items()) - rx.logK for rx in net.rxns])
    sol, *_ = np.linalg.lstsq(M, -r, rcond=None)
    for i, val in zip(virt, sol):
        x[i] = float(np.clip(val, VIRT_LB, VIRT_UB))
    return x


def gibbs_energy(net: Network, x, act: ActivityModel) -> float | None:
    """Transformed G/RT of the state ``x``; ``None`` if the model has no excess Gibbs energy.

    ``mu0`` is fixed from ``logK`` only up to the conservation laws, so compare
    G only between states of one system with the same inputs.
    """
    if not act.has_excess or net.surfaces or (net.water is not None and net.water_mass is not None):
        return None
    v, st = evaluate(net, act, x)
    amounts = {k: max(float(val), 0.0) for k, val in st["n"].items()}
    for p, ph in enumerate(net.phases):
        for k, val in zip(ph.members, st["nph"][p]):
            amounts[(p, k)] = max(float(val), 0.0)
    ctx = _prepare(net, act)

    class _Safe(type(NUM)):
        log = staticmethod(lambda t: math.log(t) if t > 0 else 0.0)

    return float(_gibbs_terms(_Safe(), net, act, ctx, amounts, float(v["W"]), 1.0))


def solve_enumerate(net, act, options, info, max_phases=12):
    L = layout(net, act)
    if len(L.Nph) > max_phases:
        raise ValueError(f"method='enumerate' is limited to {max_phases} phases (got {len(L.Nph)})")
    x_spec = solve_speciation(net, act, options, info)
    valid = []
    for size in range(len(L.Nph) + 1):
        for S in itertools.combinations(L.Nph, size):
            xs = solve_square(net, act, set(S), [x_spec], options, info)
            if xs is None:
                continue
            N, s = phase_status(net, act, xs)
            okS = all(N[q] >= -TOL * net.n_scale for q, p in enumerate(L.Nph) if p in S)
            okU = all(s[q] >= -TOL for q, p in enumerate(L.Nph) if p not in S)
            if okS and okU:
                valid.append((set(S), xs))
    if not valid:
        info.notes.append("no assemblage satisfied the equilibrium conditions")
        return x_spec, set(), False
    if len(valid) > 1:
        if act.has_excess and gibbs_energy(net, valid[0][1], act) is not None:
            info.notes.append(f"{len(valid)} assemblages satisfy complementarity; kept lowest G")
            valid.sort(key=lambda t: gibbs_energy(net, t[1], act))
        else:
            warnings.warn(f"{len(valid)} phase assemblages satisfy the equilibrium conditions", stacklevel=3)
    S, xs = valid[0]
    return active_set_polish(net, act, S, xs, options, info, fallback=x_spec)


def run(net: Network, act: ActivityModel, method: str, options=None, eps_seq=DEFAULT_EPS):
    opts = dict(NLP_OPTIONS, **(options or {}))
    info = SolveInfo(method=method)
    if method == "complementarity":
        x, active, ok = solve_complementarity(net, act, opts, info, eps_seq)
    elif method == "gibbs":
        x, active, ok = solve_gibbs(net, act, opts, info)
    elif method == "enumerate":
        x, active, ok = solve_enumerate(net, act, opts, info)
    else:
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
    return x, active, ok, info

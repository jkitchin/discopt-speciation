r"""Certificates of global phase stability, and phase splitting.

A computed equilibrium satisfies local conditions: mass action, conservation and
complementarity. Those are also satisfied by metastable states, for example a
non-ideal solid solution inside its miscibility gap. The state is the *global*
minimum of the Gibbs energy when, in addition, no phase can form with a lower
energy than the computed solution implies. For a candidate phase with member
mole fractions :math:`x` this is the tangent plane distance (TPD)

.. math:: D(x) = \sum_j x_j \ln x_j + \frac{G_{ex}(x)}{RT} - \ln 10 \sum_j x_j\,SI_j \;\ge\; 0
          \quad\text{for all } x \text{ in the simplex,}

with :math:`SI_j` the log10 saturation of member :math:`j` as a pure phase in the
computed solution (Baker, Pierce and Luks 1982; Michelsen 1982). If it holds for
every phase, then for any other state satisfying conservation,
:math:`G \ge \sum_c \mu_c b_c = G^*`, because every phase's Gibbs energy lies on
or above the tangent plane of the component potentials :math:`\mu_c` fixed by the
computed solution. That requires the aqueous solution (with exchangers and
surfaces) to have a convex Gibbs energy. This is exact for ideal activities and
is an assumption for the other models.

How the minimum of :math:`D` is bounded:

* pure phases, ideal solid solutions and ideal gas phases: :math:`\min D = \ln 10\,s`
  exactly, with :math:`s` the undersaturation;
* non-ideal solid solutions: :math:`D` is non-convex, so its minimum is enclosed
  by interval branch and bound over the simplex, in outward-rounded interval
  arithmetic. Each box uses the better of the natural and the mean-value
  (centered) enclosures. Compositions with some :math:`x_j < \varepsilon` are
  left out and covered by a proven margin: moving :math:`x` a fraction
  :math:`t = n\varepsilon` toward the barycenter changes the convex part by at
  most :math:`t(\ln n + \ln 10\,\mathrm{range}(SI))` and :math:`G_{ex}` by at
  most :math:`2tL`, with :math:`L` a Lipschitz bound of :math:`G_{ex}`.

:func:`stabilize` repeats certify-and-repair. An unstable absent phase is added
at the TPD minimizer. An unstable phase that is already present gets a second
copy there, so both sides of a miscibility gap coexist. The assemblage is then
re-solved.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from discopt.speciation import solve as _solve
from discopt.speciation.activity import NUM

LN10 = math.log(10.0)
TOL = 1e-6
MAX_BOXES = 50_000


# --------------------------------------------------------------------------- interval arithmetic
def _dn(v: float) -> float:
    return math.nextafter(v, -math.inf)


def _up(v: float) -> float:
    return math.nextafter(v, math.inf)


class Iv:
    """Closed interval with outward-rounded ``+ - *`` (enough for the polynomial excess models)."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = float(lo)
        self.hi = float(lo if hi is None else hi)

    @staticmethod
    def of(v) -> "Iv":
        return v if isinstance(v, Iv) else Iv(v)

    def __add__(self, o):
        o = Iv.of(o)
        return Iv(_dn(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __sub__(self, o):
        o = Iv.of(o)
        return Iv(_dn(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, o):
        return Iv.of(o) - self

    def __mul__(self, o):
        o = Iv.of(o)
        p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Iv(_dn(min(p)), _up(max(p)))

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"[{self.lo:.17g}, {self.hi:.17g}]"


def _widen(lo: float, hi: float) -> Iv:
    # libm log is accurate to ~1 ulp; 1e-15 relative is several ulps
    return Iv(lo - 1e-15 * (1.0 + abs(lo)), hi + 1e-15 * (1.0 + abs(hi)))


def _log(a: Iv) -> Iv:
    return _widen(math.log(a.lo), math.log(a.hi))


def _xlogx(a: Iv) -> Iv:
    f0, f1 = a.lo * math.log(a.lo), a.hi * math.log(a.hi)
    lo = -1.0 / math.e if a.lo <= 1.0 / math.e <= a.hi else min(f0, f1)
    return _widen(lo, max(f0, f1))


class _IvOps:
    @staticmethod
    def sum(xs):
        total = Iv(0.0)
        for v in xs:
            total = total + v
        return total


_IVO = _IvOps()


# --------------------------------------------------------------------------- TPD of one phase
class _TPD:
    """``D`` over the simplex of ``ph.members`` in reduced coordinates ``z`` (``x_n = 1 - sum z``)."""

    def __init__(self, ph, SI, eps):
        self.ph, self.n, self.eps = ph, len(ph.members), eps
        self.a = [-LN10 * float(s) for s in SI]
        self.hex = self.hexmid = None
        if getattr(ph.model, "margules", None) is not None and self.n >= 3:
            # Hessian of sum W_ab x_a x_b in z (x_last = 1 - sum z) is constant: W_ij - W_i,last - W_j,last
            Wm, mp = ph.model.margules_matrix(), ph.mpos
            m, last = self.n - 1, mp[-1]
            self.hexmid = [[Wm[mp[i]][mp[j]] - Wm[mp[i]][last] - Wm[mp[j]][last] for j in range(m)] for i in range(m)]
            self.hex = [[_widen(v, v) for v in row] for row in self.hexmid]

    def ideal_dual_bound(self, xs) -> float:
        """Lower bound on ``sum_j x_j (ln x_j + a_j)`` over ``x_j in xs[j]``, ``sum x_j = 1`` (Lagrangian dual).

        For any multiplier ``nu``, ``-nu + sum_j min_{x in [l_j, u_j]} (x ln x + (a_j + nu) x)``
        is a lower bound (weak duality). The inner minimum is ``-exp(-1 - a_j - nu)`` at the
        stationary point, or an end point value when the stationary point is outside.
        ``nu`` comes from a few active-set iterations and only affects tightness. Faces
        where some ``x_j -> 0``, including the last member's face, are handled exactly.
        """
        n = self.n
        lo, hi = [v.lo for v in xs], [v.hi for v in xs]
        if sum(lo) > 1.0 + 1e-12:
            return math.inf
        if sum(hi) < 1.0 - 1e-12:
            return math.inf
        clamp = [None] * n
        nu = 0.0
        for _ in range(n + 1):
            free = [j for j in range(n) if clamp[j] is None]
            rem = 1.0 - sum(clamp[j] for j in range(n) if clamp[j] is not None)
            if not free or rem <= 0.0:
                break
            e = [-1.0 - self.a[j] for j in free]
            top = max(e)
            nu = top + math.log(sum(math.exp(v - top) for v in e)) - math.log(rem)
            changed = False
            for j in free:
                xj = math.exp(min(-1.0 - self.a[j] - nu, 700.0))
                if xj < lo[j]:
                    clamp[j], changed = lo[j], True
                elif xj > hi[j]:
                    clamp[j], changed = hi[j], True
            if not changed:
                break
        total = Iv(-nu)
        for j in range(n):
            arg = -1.0 - self.a[j] - nu
            xstar = math.exp(min(arg, 700.0))
            if lo[j] * (1.0 - 1e-9) <= xstar <= hi[j] * (1.0 + 1e-9):
                inner = -_up(xstar * (1.0 + 1e-14))  # global minimum of the 1-D convex function, rounded down
            else:
                h = [(_xlogx(Iv(b)) + Iv(b) * (self.a[j] + Iv(nu))).lo for b in (lo[j], hi[j])]
                inner = min(h)
            total = total + Iv(inner)
        return total.lo

    def _grad_point(self, x) -> np.ndarray:
        lnl = self.ph.model.ln_lambda(NUM, self.ph.fractions(list(x)))
        mp = self.ph.mpos
        t = [math.log(x[i]) + lnl[mp[i]] + self.a[i] for i in range(self.n)]
        return np.array([t[i] - t[-1] for i in range(self.n - 1)])

    def convex_bound(self, box, xs) -> float | None:
        """Lower bound on a box where ``D`` is provably convex: the tangent plane at the box minimizer.

        Convexity: the interval Hessian is ``Hc +- Hr`` (midpoint and radius matrices), and every
        symmetric matrix in it is positive definite when ``lambda_min(Hc) > ||Hr||_2``
        (Weyl's inequality). ``||Hr||_2`` is bounded by the Frobenius norm and the eigenvalue
        by a backward-error margin. The bound is nearly exact, which is what proves
        ``D >= 0`` at a tangent point.
        """
        if self.hex is None:
            return None
        m = self.n - 1
        inv = [Iv(_dn(1.0 / xj.hi), _up(1.0 / xj.lo)) for xj in xs]
        cinv = np.array([0.5 * (v.lo + v.hi) for v in inv])
        rinv = np.array([0.5 * (v.hi - v.lo) for v in inv])
        Hc = np.array(self.hexmid) + np.diag(cinv[:-1]) + cinv[-1]
        Hr = (np.diag(rinv[:-1]) + rinv[-1]
              + np.array([[0.5 * (v.hi - v.lo) for v in row] for row in self.hex]))
        # the midpoints themselves carry rounding: 1e-15 relative per entry is several ulps
        Hr = Hr + 1e-15 * (np.abs(Hc) + 1.0)
        norm_r = math.sqrt(float((Hr * Hr).sum())) * (1.0 + 1e-12)
        lam = float(np.linalg.eigvalsh(Hc)[0])
        if lam - norm_r <= 8.0 * m * np.finfo(float).eps * float(np.abs(Hc).sum()):
            return None
        lo, hi = np.array([b.lo for b in box]), np.array([b.hi for b in box])
        p = 0.5 * (lo + hi)
        for _ in range(8):  # projected Newton; any p in the box gives a valid bound, a good p a tight one
            x = np.append(p, 1.0 - p.sum())
            H = np.array(self.hexmid) + np.diag(1.0 / x[:-1]) + 1.0 / x[-1]
            p = np.clip(p - np.linalg.solve(H, self._grad_point(x)), lo, hi)
        px = [Iv(v) for v in p]
        xp = px + [1.0 - _IVO.sum(px)]
        if xp[-1].lo < self.eps:
            return None
        off = [Iv(_dn(b.lo - v), _up(b.hi - v)) for b, v in zip(box, p)]
        return (self.value(xp) + _IVO.sum([g * o for g, o in zip(self.gradient(xp), off)])).lo

    def fractions(self, box):
        """Interval mole fractions over a box; ``None`` if the box misses the truncated simplex."""
        rest = 1.0 - _IVO.sum(box)
        if rest.hi < self.eps:
            return None, False
        xn = Iv(max(rest.lo, self.eps), min(rest.hi, 1.0))
        return list(box) + [xn], rest.lo >= self.eps

    def value(self, xs) -> Iv:
        terms = [_xlogx(xj) + xj * aj for xj, aj in zip(xs, self.a)]
        return _IVO.sum(terms) + self.ph.model.excess(_IVO, self.ph.fractions(xs))

    def gradient(self, xs) -> list:
        # d/dz_i = (ln x_i + ln lambda_i + a_i) - (ln x_n + ln lambda_n + a_n): the ln lambda differences are
        # the reduced derivatives of G_ex on the simplex (lambda_i = d(n G_ex)/dn_i)
        lnl = self.ph.model.ln_lambda(_IVO, self.ph.fractions(xs))
        mp = self.ph.mpos
        last = _log(xs[-1]) + lnl[mp[-1]] + self.a[-1]
        return [(_log(xs[i]) + lnl[mp[i]] + self.a[i]) - last for i in range(self.n - 1)]

    def point(self, x) -> float:
        val = sum(xj * math.log(xj) + xj * aj for xj, aj in zip(x, self.a))
        return float(val + self.ph.model.excess(NUM, self.ph.fractions(list(x))))

    def lower(self, box, consider) -> float | None:
        xs, inside = self.fractions(box)
        if xs is None:
            return None
        lb = self.value(xs).lo
        c = [0.5 * (b.lo + b.hi) for b in box]
        if 1.0 - sum(c) < self.eps:  # box straddles the last member's face: only the enclosures without a center
            ex_lo = self.ph.model.excess(_IVO, self.ph.fractions(xs)).lo
            return max(lb, self.ideal_dual_bound(xs) + ex_lo)
        consider(c + [1.0 - sum(c)])
        cx = [Iv(v) for v in c]
        xc = cx + [1.0 - _IVO.sum(cx)]
        off = [Iv(_dn(b.lo - v), _up(b.hi - v)) for b, v in zip(box, c)]
        # the ideal part sum x (ln x + a) is convex: its tangent plane at the center underestimates it everywhere
        fc = _IVO.sum([_xlogx(xj) + xj * aj for xj, aj in zip(xc, self.a)])
        lnn = _log(xc[-1])
        gid = [(_log(xc[i]) + self.a[i]) - (lnn + self.a[-1]) for i in range(self.n - 1)]
        ideal = (fc + _IVO.sum([g * o for g, o in zip(gid, off)])).lo
        mp = self.ph.mpos
        if inside:  # mean-value forms need the whole box (and the segments to its center) feasible
            dc = self.value(xc)
            cen = dc + _IVO.sum([g * o for g, o in zip(self.gradient(xs), off)])
            lb = max(lb, cen.lo)
            lnl = self.ph.model.ln_lambda(_IVO, self.ph.fractions(xs))
            gex = [lnl[mp[i]] - lnl[mp[-1]] for i in range(self.n - 1)]
            ex = self.ph.model.excess(_IVO, self.ph.fractions(xc)) + _IVO.sum([g * o for g, o in zip(gex, off)])
            ex_lo = max(ex.lo, self.ph.model.excess(_IVO, self.ph.fractions(xs)).lo)
            cb = self.convex_bound(box, xs)
            if cb is not None:
                lb = max(lb, cb)
        else:
            ex_lo = self.ph.model.excess(_IVO, self.ph.fractions(xs)).lo
        return max(lb, max(ideal, self.ideal_dual_bound(xs)) + ex_lo)


def tpd_minimum(ph, SI, tol: float = TOL, max_boxes: int = MAX_BOXES, seeds=()) -> dict:
    """Rigorous enclosure of ``min D`` for non-ideal solid solution ``ph`` (a network ``PhaseInfo``).

    Returns ``lower`` (a lower bound valid over the whole simplex), ``upper`` (``D``
    at the best composition found), ``x`` (that composition, in ``ph.members``
    order), ``boxes`` and ``complete`` (False if the box budget ran out first).
    """
    n = len(ph.members)
    if n == 1:
        d = -LN10 * float(SI[0])
        return dict(lower=d, upper=d, x=[1.0], boxes=0, complete=True)
    Lg = ph.model.excess_bound()
    rng = LN10 * (max(SI) - min(SI))
    eps = min(1e-10, 0.01 * tol / (n * (math.log(n) + rng + 2.0 * Lg + 1.0)))
    margin = n * eps * (math.log(n) + rng + 2.0 * Lg)
    f = _TPD(ph, SI, eps)
    best = [math.inf, None]

    def consider(x):
        x = np.asarray(x, float)
        if np.all(x >= eps):
            val = f.point(x)
            if val < best[0]:
                best[0], best[1] = val, x.copy()

    SIa = np.asarray(SI, float)
    ideal = np.exp(LN10 * (SIa - SIa.max()))
    starts = ([np.full(n, 1.0 / n), ideal / ideal.sum()]
              + [np.where(np.arange(n) == j, 1.0 - (n - 1) * 1e-6, 1e-6) for j in range(n)])
    for s in list(seeds) + starts:
        s = np.clip(np.asarray(s, float), 2 * eps, 1.0)
        consider(s / s.sum())
        # a good incumbent early prunes most of the tree: polish each start locally (any feasible point is an
        # upper bound, so this needs no rigor)
        loc = np.clip(_local_tpd_min(ph, SI, s / s.sum()), 2 * eps, 1.0)
        consider(loc / loc.sum())
    thr = -0.1 * tol + margin  # boxes whose bound reaches thr hold no D < -tol/10 (a tenfold safety factor)

    root = [Iv(eps, 1.0 - (n - 1) * eps) for _ in range(n - 1)]
    heap, tick, boxes, discarded = [], 0, 1, math.inf
    lb0 = f.lower(root, consider)
    heapq.heappush(heap, (lb0, tick, root))
    complete = True
    while heap:
        lb, _, box = heap[0]
        if lb >= min(thr, best[0]):
            break
        if best[0] < thr and best[0] - lb <= max(tol, 1e-3 * abs(best[0])):
            break  # an unstable composition is located to useful accuracy
        if boxes >= max_boxes:
            complete = False
            break
        heapq.heappop(heap)
        widths = [b.hi - b.lo for b in box]
        i = int(np.argmax(widths))
        if widths[i] < 1e-14:
            discarded = min(discarded, lb)
            complete = False
            continue
        mid = 0.5 * (box[i].lo + box[i].hi)
        for part in (Iv(box[i].lo, mid), Iv(mid, box[i].hi)):
            child = list(box)
            child[i] = part
            boxes += 1
            cl = f.lower(child, consider)
            if cl is None:
                continue
            if cl >= min(thr, best[0]):
                discarded = min(discarded, cl)
                continue
            tick += 1
            heapq.heappush(heap, (cl, tick, child))
    lower = min([discarded] + [h[0] for h in heap]) - margin
    return dict(lower=float(lower), upper=float(best[0]), x=[float(v) for v in best[1]], boxes=boxes,
                complete=complete)


# --------------------------------------------------------------------------- certificates
@dataclass
class PhaseStability:
    """Bounds on the minimum tangent plane distance of one phase (dimensionless, per mol of phase)."""

    phase: str
    kind: str
    present: bool
    lower_bound: float
    upper_bound: float
    composition: dict  # member -> mole fraction at the best composition found
    method: str  # "exact" or "interval branch and bound"
    status: str  # "stable" | "unstable" | "inconclusive"
    boxes: int = 0
    index: int = field(default=-1, repr=False)


@dataclass
class Certificate:
    """Outcome of a global stability test of a computed equilibrium.

    ``status`` is ``"certified"`` (every phase's TPD is provably ``>= -tol``),
    ``"unstable"`` (some phase has a composition with ``D < -tol``) or
    ``"inconclusive"``. ``bool(certificate)`` is True only when certified.
    """

    status: str
    tol: float
    phases: list
    assumptions: list
    notes: list = field(default_factory=list)
    rounds: int = 0

    def __bool__(self) -> bool:
        return self.status == "certified"

    @property
    def lower_bound(self) -> float:
        """Smallest proven lower bound on any phase's TPD."""
        return min((q.lower_bound for q in self.phases), default=math.inf)

    def summary(self) -> str:
        lines = [f"stability: {self.status} (tol {self.tol:g}; {self.rounds} phase-splitting rounds)"]
        for q in self.phases:
            lines.append(f"  {q.phase:<26}{'present' if q.present else 'absent':>8}  min D in "
                         f"[{q.lower_bound:.3e}, {q.upper_bound:.3e}]  {q.status} ({q.method})")
        lines += [f"  assumes: {a}" for a in self.assumptions]
        lines += [f"  note: {t}" for t in self.notes]
        return "\n".join(lines)


def _status(lower, upper, tol) -> str:
    if lower >= -tol:
        return "stable"
    return "unstable" if upper < -tol else "inconclusive"


def certify_state(net, act, x, active, *, converged: bool = True, tol: float = TOL,
                  max_boxes: int = MAX_BOXES) -> Certificate:
    """Global stability certificate of state ``x`` with assemblage ``active`` (phase indices)."""
    _, st = _solve.evaluate(net, act, x)
    assumptions = [
        "one aqueous phase (with exchangers and surfaces) whose Gibbs energy is convex in its amounts"
        + ("" if act.is_ideal else f" (exact for ideal activities, assumed for {act.name!r})"),
        f"the computed solution activities (solved to residual {_solve.TOL:g}) are exact",
    ]
    phases, notes = [], []
    done = set()
    for p, ph in enumerate(net.phases):
        if not ph.complementarity:
            continue
        root = p if ph.copy_of is None else ph.copy_of
        if root in done:
            continue
        done.add(root)
        group = [q for q, other in enumerate(net.phases) if q == root or other.copy_of == root]
        present = any(q in active for q in group)
        names = [net.species[k].name for k in ph.members]
        if ph.model is None:
            d = LN10 * float(st["s"][p])
            comp = dict(zip(names, (float(v) for v in st["x"][p])))
            phases.append(PhaseStability(ph.name, ph.kind, present, d, d, comp, "exact", _status(d, d, tol), 0, p))
            continue
        SI = [float(st["la_eq"][k]) for k in ph.members]
        seeds = [np.array([float(v) for v in st["x"][q]]) for q in group]
        out = tpd_minimum(ph, SI, tol, max_boxes, seeds)
        if not out["complete"]:
            notes.append(f"{ph.name}: box budget ({max_boxes}) exhausted")
        phases.append(PhaseStability(ph.name, ph.kind, present, out["lower"], out["upper"],
                                     dict(zip(names, out["x"])), "interval branch and bound",
                                     _status(out["lower"], out["upper"], tol), out["boxes"], root))
    if any(q.status == "unstable" for q in phases):
        status = "unstable"
    elif any(q.status == "inconclusive" for q in phases):
        status = "inconclusive"
    else:
        status = "certified"
    if not converged:
        notes.append("the equilibrium solve did not converge; the bounds refer to an inexact state")
        status = "inconclusive" if status == "certified" else status
    return Certificate(status, tol, phases, assumptions, notes)


# --------------------------------------------------------------------------- phase splitting
def _remap(net, act, x, phases, src):
    """Network with ``phases`` (``src[i]``: old index of new phase ``i``, or None) and ``x`` in its layout."""
    new = dataclasses.replace(net, phases=phases)
    La, Lb = _solve.layout(net, act), _solve.layout(new, act)
    x = np.asarray(x, float)
    out = np.zeros(Lb.n)
    out[: La.ny] = x[: La.ny]
    posA = {p: q for q, p in enumerate(La.Nph)}
    for qb, pb in enumerate(Lb.Nph):
        if src[pb] is not None:
            out[Lb.ny + qb] = x[La.ny + posA[src[pb]]]
    for pb, b in Lb.X.items():
        nm = len(phases[pb].members)
        if src[pb] is not None:
            a = La.X[src[pb]]
            out[b : b + nm + 1] = x[a : a + nm + 1]
        else:
            out[b : b + nm] = -math.log10(nm)
    for name in ("W", "I", "delta"):
        if getattr(La, name) is not None:
            out[getattr(Lb, name)] = x[getattr(La, name)]
    if La.npsi:
        out[Lb.psi : Lb.psi + Lb.npsi] = x[La.psi : La.psi + La.npsi]
    return new, out


def _add_copy(net, act, x, p):
    ph = net.phases[p]
    k = 1 + sum(q.copy_of == p for q in net.phases)
    copy = dataclasses.replace(ph, name=f"{ph.name} ({k + 1})", members=list(ph.members), copy_of=p)
    return _remap(net, act, x, list(net.phases) + [copy], list(range(len(net.phases))) + [None])


def _drop_absent_copies(net, act, x, active):
    keep = [p for p, ph in enumerate(net.phases) if ph.copy_of is None or p in active]
    if len(keep) == len(net.phases):
        return net, x, active
    new, out = _remap(net, act, x, [net.phases[p] for p in keep], keep)
    return new, out, {keep.index(p) for p in active}


def _seed(net, act, x, p, comp, amount=None):
    L = _solve.layout(net, act)
    x = np.asarray(x, float).copy()
    if p in L.X:
        a, nm = L.X[p], len(comp)
        x[a : a + nm] = np.log10(np.clip(comp, 1e-200, 1.0))
        x[a + nm] = 0.0
    if amount is not None:
        x[L.ny + L.Nph.index(p)] = amount
    return x


def _local_tpd_min(ph, SI, x0):
    """Local minimizer of the TPD from ``x0`` (a starting composition for the other side of a gap)."""
    from scipy.optimize import minimize

    f = _TPD(ph, SI, 1e-14)
    n = len(x0)
    out = minimize(lambda x: f.point(np.clip(x, 1e-14, 1.0)), np.asarray(x0, float), method="SLSQP",
                   bounds=[(1e-14, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
                   options={"ftol": 1e-14, "maxiter": 500})
    x = np.clip(out.x, 1e-14, 1.0)
    return x / x.sum()


def _split(net, act, x, p, target, xm):
    """Seed present phase ``p`` and its new copy ``target`` on the two sides of a miscibility gap.

    The copy goes to the TPD minimizer ``xm``; ``p`` goes to the local TPD minimum on the
    opposite side of its current composition ``x0``; the amount is divided by the lever rule.
    """
    L = _solve.layout(net, act)
    _, st = _solve.evaluate(net, act, x)
    ph = net.phases[p]
    x0 = np.array([float(v) for v in st["x"][p]])
    SI = [float(st["la_eq"][k]) for k in ph.members]
    xm = np.asarray(xm, float)
    ref = np.clip(x0 + (x0 - xm), 1e-6, 1.0)
    x1 = _local_tpd_min(ph, SI, ref / ref.sum())
    if np.abs(x1 - xm).max() < 1e-3:
        x1 = x0
    d = xm - x1
    theta = float(np.clip(np.dot(x0 - x1, d) / max(np.dot(d, d), 1e-300), 0.05, 0.95))
    N = float(x[L.ny + L.Nph.index(p)])
    x = _seed(net, act, x, p, x1, (1.0 - theta) * N)
    return _seed(net, act, x, target, xm, theta * N)


def stabilize(net, act, x, active, ok, options, info, *, tol: float = TOL, max_rounds: int = 6,
              max_boxes: int = MAX_BOXES):
    """Certify ``x``; while a phase is unstable, add it (or a copy) at its TPD minimizer and re-solve.

    Returns ``(net, x, active, ok, certificate)``. ``net`` may have extra phase copies.
    """
    tried = set()
    rounds = 0
    while True:
        cert = certify_state(net, act, x, active, converged=ok, tol=tol, max_boxes=max_boxes)
        cert.rounds = rounds
        if cert.status != "unstable" or not ok:
            break
        if rounds >= max_rounds:
            cert.notes.append(f"still unstable after {rounds} phase-splitting rounds")
            break
        worst = min((q for q in cert.phases if q.status == "unstable"), key=lambda q: q.upper_bound)
        p = worst.index
        ph = net.phases[p]
        comp = [worst.composition[net.species[k].name] for k in ph.members]
        key = (p, tuple(round(c, 4) for c in comp))
        if key in tried:
            cert.notes.append(f"phase splitting cycled on {ph.name}")
            break
        tried.add(key)
        rounds += 1
        group = [q for q, other in enumerate(net.phases) if q == p or other.copy_of == p]
        free = [q for q in group if q not in active]
        if free:
            target = free[0]
            x = _seed(net, act, x, target, comp)
        elif ph.model is not None:
            net, x = _add_copy(net, act, x, p)
            target = len(net.phases) - 1
            present = max((q for q in group if q in active), key=lambda q: _solve.layout(net, act).Nph.index(q))
            x = _split(net, act, x, present, target, comp)
        else:
            cert.notes.append(f"{ph.name} is present but reported unstable")
            break
        info.notes.append(f"stability: {ph.name} unstable (min D <= {worst.upper_bound:.3g}); adding "
                          f"{net.phases[target].name} at x = {np.round(comp, 6).tolist()}")
        x, active, ok = _solve.active_set_polish(net, act, set(active) | {target}, x, options, info)
    net, x, active = _drop_absent_copies(net, act, x, active)
    return net, x, active, ok, cert


def stability_certificate(result, tol: float = TOL, max_boxes: int = MAX_BOXES) -> Certificate:
    """Global stability certificate for a :class:`~discopt.speciation.Result` (no re-solve)."""
    s = result._state
    if s is None:
        raise ValueError("result has no stored state; compute it with System.equilibrate")
    return certify_state(s["net"], s["act"], s["x"], s["active"], converged=result.converged, tol=tol,
                         max_boxes=max_boxes)

"""Numeric assembly of a reaction network for one equilibrium calculation.

Given declared species, reactions, phases, input amounts and fixed activities,
this builds everything the solvers need:

* the variable species (log10 molality for aqueous species, log10 mol for
  exchange/surface species, log10 activity for virtual species), water, and the
  phases with complementarity;
* reactions split into mass-action equations and one "phase reaction" per phase
  member, with ``logK(T, P)``;
* the conservation matrix ``A`` (left null space of the stoichiometric matrix,
  computed by exact RREF so the rows stay sparse and element-like) and the
  conserved totals ``b``. Virtual species are columns with zero amount, so
  e.g. exchange-site and electron balances come out automatically;
* standard chemical potentials ``mu0/RT`` consistent with every ``logK``
  (``sum_i nu_i mu0_i = -ln(10) logK``, fixed-activity species moved into the
  constant), used by Gibbs minimization.

Species that the conservation laws force to zero (a same-sign conserved row with
zero total, e.g. an Mg complex when no Mg was added) are pruned, together with
the reactions they appear in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from discopt.speciation.phases import GasPhase, SolidSolution
from discopt.speciation.species import Reaction, Species
from discopt.speciation.water import M_W

LN10 = math.log(10.0)


@dataclass
class Rxn:
    nu: dict  # species index -> coefficient (all species, fixed included)
    logK: float  # at (T, P)
    logK_eff: float  # fixed-activity species moved into the constant
    reaction: Reaction
    member: int | None = None  # phase-member species index for phase reactions


@dataclass
class PhaseInfo:
    kind: str  # "pure" | "ss" | "gas_p" | "gas_v"
    name: str
    members: list  # species indices
    pressure: float | None = None
    volume: float | None = None
    #: non-ideal solid solution: the :class:`SolidSolution` with the excess model, else None
    model: object = None
    #: positions of ``members`` in ``model.members`` (members pruned from the network are absent)
    mpos: list | None = None
    #: index of the phase this one is a second copy of (a miscibility gap), else None
    copy_of: int | None = None
    #: Peng-Robinson gas phase: the :class:`~discopt.speciation.eos.PengRobinson` model, else None (ideal)
    eos: object = None

    @property
    def complementarity(self) -> bool:
        return self.kind != "gas_v"

    def fractions(self, xs):
        """Full model-member mole-fraction list from the fractions of ``members`` (pruned members are 0)."""
        full = [0.0] * len(self.model.members)
        for q, xv in zip(self.mpos, xs):
            full[q] = xv
        return full


@dataclass
class SurfaceInfo:
    name: str
    area: float
    mass: float
    members: list  # species indices of surface species on this surface


@dataclass
class Network:
    species: list  # active species
    kind: list  # per species: aq | water | ex | surf | virt | member | const
    fixed: dict  # species index -> log10 activity
    yidx: list  # species with a log variable: aqueous, exchange, surface, virtual (in this order)
    rxns: list  # mass-action reactions (member is None)
    member_rxn: dict  # member species index -> Rxn
    phases: list  # PhaseInfo (members' phase index in member_phase)
    member_phase: dict
    cols: list  # conserved species indices
    A: np.ndarray
    b: np.ndarray
    row_scale: np.ndarray
    amt: np.ndarray  # input amounts over cols
    n_scale: float
    mu0: np.ndarray  # over cols
    T: float
    P: float
    water: int | None = None
    water_mass: float | None = None  # fixed water mass (kg) or None if variable
    W0: float = 1.0
    site_total: dict = field(default_factory=dict)
    surfaces: list = field(default_factory=list)
    cb: int | None = None
    cb_col: np.ndarray | None = None
    pruned: list = field(default_factory=list)
    #: fixed-pressure gases with a Peng-Robinson model: species index -> log10 fugacity coefficient
    #: (``fixed`` then holds the log10 fugacity)
    fixed_logphi: dict = field(default_factory=dict)

    # -- conveniences ------------------------------------------------------
    @property
    def aq_idx(self) -> list:
        return [k for k in self.yidx if self.kind[k] == "aq"]

    @property
    def aq(self) -> list:
        return [self.species[k] for k in self.aq_idx]

    @property
    def solids(self) -> list:
        return [self.species[k] for ph in self.phases for k in ph.members if self.species[k].is_solid]

    @property
    def na(self) -> int:
        """Number of log variables (they come first in the solution vector)."""
        return len(self.yidx)

    @property
    def ns(self) -> int:
        """Number of phases."""
        return len(self.phases)

    @property
    def charges(self) -> np.ndarray:
        return np.array([self.species[k].charge for k in self.aq_idx])

    def index(self, name) -> int:
        name = getattr(name, "name", name)
        for k, s in enumerate(self.species):
            if s.name == name:
                return k
        raise KeyError(name)


def rref_left_nullspace(N: np.ndarray, order: list[int]) -> np.ndarray:
    """Rows ``A`` with ``A @ N = 0``, from exact RREF of ``N.T``.

    ``order`` permutes species columns before elimination: pivots are taken
    left-first, so species placed *last* become the free "components" whose
    conserved totals the rows represent.
    """
    nsp, nr = N.shape
    if nr == 0:
        return np.eye(nsp)
    rows = [[Fraction(float(N[order[j], i])).limit_denominator(10**6) for j in range(nsp)] for i in range(nr)]
    piv: list[int] = []
    r = 0
    for c in range(nsp):
        p = next((i for i in range(r, nr) if rows[i][c] != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        pv = rows[r][c]
        rows[r] = [v / pv for v in rows[r]]
        for i in range(nr):
            if i != r and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [a - f * bb for a, bb in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
        if r == nr:
            break
    free = [c for c in range(nsp) if c not in piv]
    basis = np.zeros((len(free), nsp))
    for q, f in enumerate(free):
        v = [Fraction(0)] * nsp
        v[f] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -rows[i][f]
        for j in range(nsp):
            basis[q, order[j]] = float(v[j])
    return basis


def validate_structure(species: list, reactions: list, fixed=()) -> None:
    """Structural checks independent of input amounts. Raises ``ValueError``."""
    declared = set(species)
    fixed = set(fixed)
    for rxn in reactions:
        if rxn.logK is None and rxn.analytic is None:
            raise ValueError(f"reaction {rxn.equation()!r} has no logK")
        unknown = [s.name for s in rxn.species() if s not in declared]
        if unknown:
            raise ValueError(f"reaction {rxn.equation()!r} uses undeclared species {unknown}")
        members = [s for s in rxn.stoich if s.is_phase_member and s not in fixed]
        if len(members) > 1:
            raise ValueError(
                f"reaction {rxn.equation()!r} involves {len(members)} solids/gases; write each one's "
                "dissolution/formation from aqueous species only"
            )
    for s in species:
        if s.is_phase_member and s not in fixed:
            k = [r for r in reactions if s in r.stoich]
            if len(k) != 1:
                what = "solid" if s.is_solid else "gas"
                raise ValueError(
                    f"{what} {s.name!r} appears in {len(k)} reactions; each {what} needs exactly one "
                    "reaction (written against aqueous species)"
                )
    if sum(s.phase == "solvent" for s in species) > 1:
        raise ValueError("at most one solvent (water) species")


def build_network(species: list, reactions: list, amounts: dict, fixed: dict, T: float, *, P: float = 1.0,
                  phases=(), surfaces=(), water_mass=None, charge_balance=None, suppress=(), eos=None) -> Network:
    """Assemble arrays, pruning species forced to zero by conservation.

    ``amounts`` and ``fixed`` are keyed by :class:`Species`; ``phases`` holds
    :class:`SolidSolution`/:class:`GasPhase` with Species members; ``surfaces``
    holds :class:`SurfaceType`; ``suppress`` lists species to remove (e.g.
    metastable phases); ``eos`` maps ``id(gas phase)`` to its Peng-Robinson
    model (absent: ideal).
    """
    suppress = set(suppress)
    species = [s for s in species if s not in suppress]
    reactions = [r for r in reactions if not (r.species() & suppress)]
    validate_structure(species, reactions, fixed)
    for s in fixed:
        if s.is_solid:
            raise ValueError(f"fixed activity given for solid {s.name!r}; fix gases or aqueous species only")
        if s.phase == "solvent":
            raise ValueError("the activity of water cannot be fixed")
    for s, v in amounts.items():
        # an exchanger master (X-) takes the exchanger's total sites; other virtual species have no amount
        if s.virtual and v and s.phase != "exchange":
            raise ValueError(f"{s.name!r} is a virtual species and cannot be given an amount")
    container = {}
    for ph in phases:
        for mbr in ph.members:
            if mbr in container:
                raise ValueError(f"{mbr.name!r} belongs to two phases")
            container[mbr] = ph
    for s in species:
        if s.phase == "gas" and s not in fixed and s not in container:
            raise ValueError(f"gas {s.name!r} is neither in a gas phase nor at a fixed partial pressure")
        if s.phase == "gas" and s in container and not isinstance(container[s], GasPhase):
            raise ValueError(f"gas {s.name!r} is in a solid solution")
        if s.is_solid and s in container and not isinstance(container[s], SolidSolution):
            raise ValueError(f"solid {s.name!r} is in a gas phase")
    if charge_balance is not None and charge_balance in fixed:
        raise ValueError(f"charge-balance species {charge_balance.name!r} has a fixed activity")

    keep = list(species)
    rxns = list(reactions)
    pruned: list = []
    while True:
        net = _assemble(keep, rxns, amounts, fixed, T, P, container, surfaces, water_mass, charge_balance, eos or {})
        absent = _forced_zero(net)
        if not absent:
            net.pruned = pruned
            return net
        pruned.extend(absent)
        gone = set(absent)
        rxns = [r for r in rxns if not (r.species() & gone)]
        with_rxn = {s for r in rxns for s in r.species()}
        extra = [s for s in keep if s.is_phase_member and s not in fixed and s not in gone and s not in with_rxn]
        pruned.extend(extra)
        gone |= set(extra)
        keep = [s for s in keep if s not in gone]
        if charge_balance is not None and charge_balance in gone:
            raise ValueError(f"charge-balance species {charge_balance.name!r} is absent from the system")


def _kind(s: Species, fixed) -> str:
    if s.phase == "aqueous":
        return "aq"
    if s.phase == "solvent":
        return "water"
    if s in fixed:
        return "const"
    if s.virtual:
        return "virt"
    return {"exchange": "ex", "surface": "surf", "solid": "member", "gas": "member"}.get(s.phase, "virt")


def _assemble(keep, rxns, amounts, fixed, T, P, container, surfaces, water_mass, cb, eos) -> Network:
    kinds = {s: _kind(s, fixed) for s in keep}
    order_kind = {"aq": 0, "ex": 1, "surf": 2, "virt": 3}
    yspecies = sorted((s for s in keep if kinds[s] in order_kind), key=lambda s: order_kind[kinds[s]])
    waters = [s for s in keep if kinds[s] == "water"]
    members = [s for s in keep if kinds[s] == "member"]
    consts = [s for s in keep if kinds[s] == "const"]
    if not any(kinds[s] == "aq" for s in keep):
        raise ValueError("system has no aqueous species")
    ordered = yspecies + waters + members + consts
    idx = {s: i for i, s in enumerate(ordered)}
    kind = [kinds[s] for s in ordered]
    water = idx[waters[0]] if waters else None
    wfixed = None if water is None else (None if water_mass is None else float(water_mass))
    W0 = 1.0 if water_mass is None else float(water_mass)
    fixed_i = {idx[s]: float(v) for s, v in fixed.items() if s in idx}

    # conserved columns: every non-fixed species except water with a fixed mass
    cols = [idx[s] for s in yspecies if s not in fixed]
    if water is not None and wfixed is None:
        cols.append(water)
    cols += [idx[s] for s in members]
    cpos = {k: j for j, k in enumerate(cols)}

    N = np.zeros((len(cols), len(rxns)))
    out_rxns, member_rxn = [], {}
    for j, r in enumerate(rxns):
        lk = r.logK_at(T, P)
        nu = {idx[s]: c for s, c in r.stoich.items()}
        eff = lk - sum(c * fixed_i[k] for k, c in nu.items() if k in fixed_i)
        for k, c in nu.items():
            if k in cpos:
                N[cpos[k], j] = c
        if not np.any(N[:, j]):
            raise ValueError(f"reaction {r.equation()!r} involves only fixed-activity species")
        mem = [k for k in nu if kind[k] == "member"]
        rx = Rxn(nu, lk, eff, r, mem[0] if mem else None)
        if mem:
            member_rxn[mem[0]] = rx
        else:
            out_rxns.append(rx)
    if len(rxns) and np.linalg.matrix_rank(N) < len(rxns):
        raise ValueError(
            "reactions are linearly dependent (one is a combination of others); remove the redundant "
            "reaction(s) so every logK is independent"
        )

    phases, member_phase, seen = [], {}, {}
    for s in members:
        ph = container.get(s)
        if ph is None:
            member_phase[idx[s]] = len(phases)
            phases.append(PhaseInfo("pure", s.name, [idx[s]]))
            continue
        if id(ph) not in seen:
            seen[id(ph)] = len(phases)
            if isinstance(ph, GasPhase):
                kind_ph = "gas_p" if ph.volume is None else "gas_v"
                phases.append(PhaseInfo(kind_ph, ph.name, [], ph.pressure, ph.volume, eos=eos.get(id(ph))))
            else:
                phases.append(PhaseInfo("ss", ph.name, []))
        p = seen[id(ph)]
        phases[p].members.append(idx[s])
        member_phase[idx[s]] = p
    for obj in {id(ph): ph for ph in container.values()}.values():
        if id(obj) not in seen or not isinstance(obj, SolidSolution):
            continue
        info = phases[seen[id(obj)]]
        order = {getattr(m, "name", m): q for q, m in enumerate(obj.members)}
        info.members.sort(key=lambda k: order[ordered[k].name])  # Guggenheim parameters depend on member order
        if not obj.is_ideal:
            info.model = obj
            info.mpos = [order[ordered[k].name] for k in info.members]
    for obj in {id(ph): ph for ph in container.values()}.values():
        if id(obj) not in seen or not isinstance(obj, GasPhase) or eos.get(id(obj)) is None:
            continue
        info = phases[seen[id(obj)]]
        info.eos = info.eos.subset([ordered[k].name for k in info.members])  # members pruned from the network drop out

    amt = np.zeros(len(cols))
    for j, k in enumerate(cols):
        s = ordered[k]
        if kind[k] == "water":
            amt[j] = W0 / M_W + float(amounts.get(s, 0.0))
        else:
            amt[j] = float(amounts.get(s, 0.0))
    if np.any(amt < 0):
        raise ValueError("input amounts must be non-negative")
    # components (free columns) = species with the largest inputs: virtual species
    # and phase members first, then the rest by ascending input (water last).
    first = [j for j, k in enumerate(cols) if kind[k] in ("virt", "member")]
    rest = sorted((j for j in range(len(cols)) if j not in first), key=lambda j: (amt[j], j))
    A = rref_left_nullspace(N, first + rest)
    b = A @ amt
    row_scale = np.maximum(np.maximum(np.abs(b), np.abs(A) @ amt), 1e-30)

    if len(rxns):
        logK_eff = np.array([rx.logK_eff for rx in out_rxns + list(member_rxn.values())])
        Nord = np.column_stack([N[:, rxns.index(rx.reaction)] for rx in out_rxns + list(member_rxn.values())])
        mu0, *_ = np.linalg.lstsq(Nord.T, -LN10 * logK_eff, rcond=None)
    else:
        mu0 = np.zeros(len(cols))
    nonwater = [j for j, k in enumerate(cols) if kind[k] != "water"]
    wcol = cpos.get(water) if water is not None else None
    brows = [r for r in range(A.shape[0]) if wcol is None or A[r, wcol] == 0]
    cand = [float(np.max(amt[nonwater])) if nonwater else 0.0, float(np.max(np.abs(b[brows]))) if brows else 0.0]
    n_scale = max(max(cand), 1e-30)
    # a balance with zero total and no input (e.g. the proton balance of a
    # mineral dissolving in pure water) has no natural size: use the system's amount scale
    row_scale[row_scale <= 1e-30] = n_scale

    site_total: dict = {}
    for s in keep:
        if s.phase == "exchange":
            ns = 1.0 if s.virtual else s.nsites
            site_total[s.site] = site_total.get(s.site, 0.0) + ns * float(amounts.get(s, 0.0))
        elif s.phase == "surface":
            site_total[s.site] = site_total.get(s.site, 0.0) + float(amounts.get(s, 0.0))
    surf_info = []
    for st in surfaces:
        if st.edl is None:
            continue
        mem = [idx[s] for s in keep if s.phase == "surface" and s.site in st.sites]
        if mem:
            surf_info.append(SurfaceInfo(st.name, float(st.area), float(st.mass), mem))

    cb_i = cb_col = None
    if cb is not None:
        cb_i = idx[cb]
        if cb_i not in cpos:
            raise ValueError(f"charge-balance species {cb.name!r} is not a conserved species")
        if cb.charge == 0:
            raise ValueError(f"charge-balance species {cb.name!r} is neutral")
        cb_col = A[:, cpos[cb_i]].copy()

    return Network(
        species=ordered, kind=kind, fixed=fixed_i, yidx=[idx[s] for s in yspecies], rxns=out_rxns,
        member_rxn=member_rxn, phases=phases, member_phase=member_phase, cols=cols, A=A, b=b,
        row_scale=row_scale, amt=amt, n_scale=n_scale, mu0=mu0, T=float(T), P=float(P), water=water,
        water_mass=wfixed, W0=W0, site_total=site_total, surfaces=surf_info, cb=cb_i, cb_col=cb_col,
    )


def _forced_zero(net: Network) -> list:
    out = []
    wcol = net.cols.index(net.water) if net.water is not None and net.water in net.cols else None
    for r in range(net.A.shape[0]):
        if net.b[r] != 0.0:
            continue
        if wcol is not None and net.A[r, wcol] != 0:
            continue
        if net.cb_col is not None and net.cb_col[r] != 0:
            continue
        nz = net.A[r][np.abs(net.A[r]) > 0]
        if nz.size and (np.all(nz > 0) or np.all(nz < 0)):
            out.extend(net.species[net.cols[j]] for j in np.flatnonzero(np.abs(net.A[r]) > 0))
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

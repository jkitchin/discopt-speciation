"""User-facing container: declare species, reactions and phases, then equilibrate."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from discopt.speciation import solve as _solve
from discopt.speciation.activity import NUM, ActivityModel, get_activity_model, ionic_strength
from discopt.speciation.network import build_network, validate_structure
from discopt.speciation.phases import GasPhase, SolidSolution, SurfaceType
from discopt.speciation.species import T_REF, Reaction, Species, parse_charge

_ARROWS = ("<=>", "<->", "=", "->")
_COEFF = re.compile(r"^(\d+(?:\.\d*)?|\.\d+)(?:\s+|\*)(.+)$")


def parse_equation(equation: str) -> tuple[list, list]:
    """Split ``"2 A + B <=> C"`` into ``[(2.0, "A"), (1.0, "B")], [(1.0, "C")]``.

    ``+`` separators must be surrounded by whitespace so ion names such as
    ``"H+"`` or ``"Ca+2"`` stay intact. Coefficients are separated from names by
    whitespace or ``*``. Accepted arrows: ``<=>``, ``<->``, ``=``, ``->``.
    """
    for arrow in _ARROWS:
        if f" {arrow} " in f" {equation} ":
            parts = re.split(rf"\s{re.escape(arrow)}\s|^{re.escape(arrow)}\s|\s{re.escape(arrow)}$", equation.strip())
            break
    else:
        raise ValueError(f"no reaction arrow (<=>, =) surrounded by spaces in {equation!r}")
    if len(parts) != 2:
        raise ValueError(f"expected exactly one arrow in {equation!r}")

    def side(text):
        text = text.strip()
        if not text:
            return []
        out = []
        for tok in re.split(r"\s+\+\s+", text):
            tok = tok.strip()
            mm = _COEFF.match(tok)
            coeff, name = (float(mm.group(1)), mm.group(2).strip()) if mm else (1.0, tok)
            if not name or " " in name:
                raise ValueError(f"cannot parse term {tok!r} in {equation!r}")
            out.append((coeff, name))
        return out

    return side(parts[0]), side(parts[1])


@dataclass
class Result:
    """Equilibrium state.

    Dicts are keyed by species name. Amounts are mol for the system defined by
    the inputs (1 kg of water initially, so input amounts are molalities);
    ``molality`` is mol/kg of the equilibrium water mass. ``SI`` is the log10
    saturation of each solid (``log10(Q/K)`` per formula unit; ``0`` when present
    as a pure phase) and the log10 partial pressure (atm) of each gas.
    """

    converged: bool
    method: str
    molality: dict
    log10_activity: dict
    gamma: dict
    solids: dict
    SI: dict
    present: list
    ionic_strength: float
    charge_imbalance: float
    #: transformed G/RT (models with an excess Gibbs energy). mu0 is fixed from
    #: logK only up to the conservation laws, so compare G only between states
    #: of one system with the same inputs.
    gibbs_energy: float | None
    residuals: dict
    pruned: list = field(default_factory=list)
    info: object = None
    water_mass: float = 1.0
    pH: float | None = None
    pe: float | None = None
    gases: dict = field(default_factory=dict)
    partial_pressure: dict = field(default_factory=dict)
    exchange: dict = field(default_factory=dict)
    surface: dict = field(default_factory=dict)
    potential: dict = field(default_factory=dict)
    mole_fraction: dict = field(default_factory=dict)
    charge_balance_adjustment: float | None = None
    #: per solid solution (and second copies across a miscibility gap): present, amount (mol), mole_fraction
    solid_solutions: dict = field(default_factory=dict)
    #: global stability certificate (:class:`~discopt.speciation.stability.Certificate`), when computed
    certificate: object = None
    #: internal: what produced this state (system, network, activity model, solution vector,
    #: assemblage, equilibrate arguments); used by sensitivities and certificates
    _state: dict | None = field(default=None, repr=False, compare=False)

    def __getitem__(self, name: str) -> float:
        for d in (self.molality, self.solids, self.gases, self.exchange, self.surface):
            if name in d:
                return d[name]
        raise KeyError(name)

    def sensitivity(self, parameters, quantities):
        """Exact derivatives of ``quantities`` w.r.t. ``parameters`` at this state; see :func:`sensitivity.sensitivities`."""
        from discopt.speciation.sensitivity import sensitivities

        return sensitivities(self, parameters, quantities)

    def summary(self) -> str:
        head = f"converged={self.converged} method={self.method} I={self.ionic_strength:.4g}"
        if self.pH is not None:
            head += f" pH={self.pH:.4f}"
        if self.pe is not None:
            head += f" pe={self.pe:.4f}"
        head += f" charge_imbalance={self.charge_imbalance:.3g}"
        if self.water_mass != 1.0:
            head += f" water={self.water_mass:.6g} kg"
        if self.gibbs_energy is not None:
            head += f" G/RT={self.gibbs_energy:.8g}"
        lines = [head, f"{'aqueous':<20}{'molality':>14}{'log10 a':>12}"]
        for k, v in self.molality.items():
            lines.append(f"{k:<20}{v:>14.6e}{self.log10_activity[k]:>12.4f}")
        for title, d in (("exchange", self.exchange), ("surface", self.surface)):
            if d:
                lines.append(f"{title:<20}{'mol':>14}{'log10 a':>12}")
                for k, v in d.items():
                    lines.append(f"{k:<20}{v:>14.6e}{self.log10_activity.get(k, float('nan')):>12.4f}")
        if self.solids:
            lines.append(f"{'solid':<20}{'amount':>14}{'SI':>12}")
            for k, v in self.solids.items():
                lines.append(f"{k:<20}{v:>14.6e}{self.SI[k]:>12.4f}")
        if self.gases or self.partial_pressure:
            lines.append(f"{'gas':<20}{'amount':>14}{'log10 P':>12}")
            for k, p in self.partial_pressure.items():
                amount = f"{self.gases[k]:>14.6e}" if k in self.gases and math.isfinite(self.gases[k]) else f"{'fixed':>14}"
                lines.append(f"{k:<20}{amount}{math.log10(p) if p > 0 else -math.inf:>12.4f}")
        for k, v in self.potential.items():
            lines.append(f"surface {k}: psi = {v:.6g} V")
        for name, d in self.solid_solutions.items():
            if d["present"]:
                comp = ", ".join(f"{k} {v:.4g}" for k, v in d["mole_fraction"].items())
                lines.append(f"solid solution {name}: {d['amount']:.6e} mol, x: {comp}")
        lines.append("residuals: " + ", ".join(f"{k}={v:.1e}" for k, v in self.residuals.items()))
        if self.certificate is not None:
            lines.append(self.certificate.summary())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serializable view (for agents and logging)."""
        def clean(v):
            return None if isinstance(v, float) and not math.isfinite(v) else v

        return {
            "converged": self.converged,
            "method": self.method,
            "molality": dict(self.molality),
            "log10_activity": {k: clean(v) for k, v in self.log10_activity.items()},
            "gamma": dict(self.gamma),
            "solids": dict(self.solids),
            "SI": {k: clean(v) for k, v in self.SI.items()},
            "present": list(self.present),
            "ionic_strength": self.ionic_strength,
            "charge_imbalance": self.charge_imbalance,
            "gibbs_energy": self.gibbs_energy,
            "residuals": dict(self.residuals),
            "pruned": list(self.pruned),
            "water_mass": self.water_mass,
            "pH": self.pH,
            "pe": self.pe,
            "gases": dict(self.gases),
            "partial_pressure": dict(self.partial_pressure),
            "exchange": dict(self.exchange),
            "surface": dict(self.surface),
            "potential": dict(self.potential),
            "mole_fraction": dict(self.mole_fraction),
            "charge_balance_adjustment": self.charge_balance_adjustment,
            "solid_solutions": {k: dict(v, mole_fraction=dict(v["mole_fraction"])) for k, v in self.solid_solutions.items()},
            "stability": None if self.certificate is None else self.certificate.status,
            "notes": list(getattr(self.info, "notes", [])),
        }


class System:
    """Species, reactions and phases of a chemical equilibrium problem.

    Examples
    --------
    >>> s = System("complexation")
    >>> M, L, ML = s.aqueous("M"), s.aqueous("L"), s.aqueous("ML")
    >>> N, NL = s.aqueous("N"), s.solid("NL(s)")
    >>> _ = s.reaction(M + L >> ML, logK=3.0)
    >>> _ = s.reaction("NL(s) = N + L", logK=-4.0)
    >>> res = s.equilibrate({M: 1e-3, L: 2e-2, N: 2e-2})   # doctest: +SKIP

    Each solid and gas needs exactly one reaction linking it to non-phase
    species; reactions must be linearly independent. Only ratios of ``logK``
    matter, so a reaction doubled with doubled ``logK`` gives the same equilibrium.
    """

    def __init__(self, name: str = "system", T: float = T_REF, activity="ideal"):
        self.name = name
        self.T = float(T)
        self.activity: ActivityModel = get_activity_model(activity)
        self._species: dict[str, Species] = {}
        self.reactions: list[Reaction] = []
        self.solid_solutions: list[SolidSolution] = []
        self.gas_phases: list[GasPhase] = []
        self.surface_types: list[SurfaceType] = []
        self.exchange_sites: dict[str, Species] = {}
        #: element or valence state -> species carrying its input amount (``{"Ca": "Ca+2"}``)
        self.masters: dict[str, str] = {}

    # ------------------------------------------------------------------ declare
    def _add(self, sp: Species) -> Species:
        if sp.name in self._species:
            raise ValueError(f"species {sp.name!r} already declared")
        self._species[sp.name] = sp
        return sp

    def aqueous(self, name: str, charge: float | None = None, *, a0=None, b=None, **params) -> Species:
        """Aqueous species; charge defaults to the name's suffix (``"SO4-2"`` -> -2).

        ``a0`` (ion size, Angstrom) and ``b`` are Truesdell-Jones parameters
        (PHREEQC ``-gamma a0 b``); other keyword parameters are stored for
        activity models.
        """
        p = dict(params)
        if a0 is not None:
            p["a0"] = float(a0)
        if b is not None:
            p["b"] = float(b)
        return self._add(Species(name, "aqueous", parse_charge(name) if charge is None else charge, params=p))

    def solvent(self, name: str = "H2O") -> Species:
        """Water as a species: its activity comes from the activity model and its mass may change."""
        return self._add(Species(name, "solvent"))

    def solid(self, name: str) -> Species:
        """A solid: a pure phase (unit activity, may be absent) unless placed in a solid solution."""
        return self._add(Species(name, "solid"))

    def gas(self, name: str) -> Species:
        """A gas; put it in a :meth:`gas_phase` or fix its log10 partial pressure (atm) in ``equilibrate``."""
        return self._add(Species(name, "gas"))

    def electron(self, name: str = "e-") -> Species:
        """The aqueous electron for redox reactions (``pe = -log10 a(e-)``)."""
        return self._add(Species(name, "electron", -1.0, virtual=True))

    def exchange_site(self, name: str = "X", charge: float = -1.0) -> Species:
        """Declare an exchanger ``X``; returns the virtual master species ``X-``."""
        sym = "-" if charge == -1 else f"{charge:+g}"
        sp = self._add(Species(f"{name}{sym}", "exchange", charge, virtual=True, site=name))
        self.exchange_sites[name] = sp
        return sp

    def exchange(self, name: str, site: str | None = None, nsites: float | None = None, *, a0=None, b=None,
                 charge: float | None = None) -> Species:
        """Exchange species such as ``"CaX2"`` (Gaines-Thomas convention).

        ``nsites`` defaults to the number following the site name (``CaX2`` -> 2).
        ``a0``, ``b`` give an optional Truesdell-Jones activity coefficient.
        """
        if site is None:
            cands = [s for s in self.exchange_sites if s in name]
            if not cands:
                raise ValueError(f"cannot infer the exchange site of {name!r}; declare it with exchange_site()")
            site = max(cands, key=len)
        if nsites is None:
            mm = re.search(re.escape(site) + r"(\d+(?:\.\d+)?)", name)
            nsites = float(mm.group(1)) if mm else 1.0
        p = {k: float(v) for k, v in (("a0", a0), ("b", b)) if v is not None}
        z = 0.0 if charge is None else charge
        return self._add(Species(name, "exchange", z, params=p, site=site, nsites=nsites))

    def surface_type(self, name: str, sites=None, area: float | None = None, mass: float | None = None,
                     edl: str | None = "ddl") -> SurfaceType:
        """Declare a sorbing surface, e.g. ``surface_type("Hfo", ["Hfo_w", "Hfo_s"], area=600, mass=0.09)``."""
        st = SurfaceType(name, list(sites or [name]), area, mass, edl)
        self.surface_types.append(st)
        return st

    def surface(self, name: str, charge: float | None = None, site: str | None = None) -> Species:
        """Surface species such as ``"Hfo_wOH2+"``; the site is the longest declared site prefix."""
        if site is None:
            cands = [s for st in self.surface_types for s in st.sites if name.startswith(s)]
            if not cands:
                raise ValueError(f"no declared surface site is a prefix of {name!r}; call surface_type() first")
            site = max(cands, key=len)
        return self._add(Species(name, "surface", parse_charge(name) if charge is None else charge, site=site))

    def solid_solution(self, name: str, members, *, guggenheim=None, margules=None) -> SolidSolution:
        """Solid solution of declared solids; ideal unless ``guggenheim`` or ``margules`` is given.

        ``guggenheim=(a0, a1[, a2])``: binary, PHREEQC ``-Gugg_nondim`` with
        ``members=[comp1, comp2]``. ``margules={(name_i, name_j): W}``: regular
        mixing, ``G_ex/RT = sum W_ij x_i x_j``. See :class:`SolidSolution`.
        """
        ss = SolidSolution(name, [self._resolve(m) for m in members], guggenheim=guggenheim, margules=margules)
        for m in ss.members:
            if not m.is_solid:
                raise ValueError(f"{m.name!r} is not a solid")
        self.solid_solutions.append(ss)
        return ss

    def gas_phase(self, members, pressure: float | None = 1.0, volume: float | None = None, name: str = "gas") -> GasPhase:
        """Ideal gas phase at fixed total ``pressure`` (atm) or fixed ``volume`` (L)."""
        gp = GasPhase([self._resolve(m) for m in members], pressure, volume, name)
        for m in gp.members:
            if m.phase != "gas":
                raise ValueError(f"{m.name!r} is not a gas")
        self.gas_phases.append(gp)
        return gp

    @property
    def species(self) -> list[Species]:
        return list(self._species.values())

    def __contains__(self, name) -> bool:
        return getattr(name, "name", name) in self._species

    def __getitem__(self, name: str) -> Species:
        try:
            return self._species[name]
        except KeyError:
            raise KeyError(f"unknown species {name!r}; declared: {list(self._species)}") from None

    def _resolve(self, key) -> Species:
        if isinstance(key, Species):
            if self._species.get(key.name) is not key:
                raise ValueError(f"species {key.name!r} does not belong to system {self.name!r}")
            return key
        return self[key]

    def reaction(self, rxn, logK: float | None = None, dH: float = 0.0, name: str | None = None, *,
                 analytic=None, dCp: float = 0.0, dV: float = 0.0) -> Reaction:
        """Add a reaction with its base-10 equilibrium constant.

        ``rxn`` is a DSL reaction (``A + B >> C``), an equation string
        (``"A + B = C"``), or a dict of net coefficients (products positive).
        Temperature dependence: ``dH`` (J/mol), ``dCp`` (J/mol/K) or PHREEQC
        ``analytic`` coefficients (A1..A6); ``dV`` (cm3/mol) for pressure.
        """
        if isinstance(rxn, str):
            left, right = parse_equation(rxn)
            r = Reaction({}, {})
            for c, nm in left:
                r.reactants[self[nm]] = r.reactants.get(self[nm], 0.0) + c
            for c, nm in right:
                r.products[self[nm]] = r.products.get(self[nm], 0.0) + c
        elif isinstance(rxn, dict):
            r = Reaction({}, {self._resolve(k): float(v) for k, v in rxn.items()})
        elif isinstance(rxn, Reaction):
            r = rxn
            for sp in r.species():
                self._resolve(sp)
        else:
            raise TypeError(f"cannot interpret {rxn!r} as a reaction")
        if not r.stoich:
            raise ValueError("reaction has no net stoichiometry")
        if logK is None and analytic is None:
            raise ValueError("give logK or analytic coefficients")
        r.logK = None if logK is None else float(logK)
        r.analytic = None if analytic is None else tuple(float(a) for a in analytic)
        r.dH, r.dCp, r.dV = float(dH), float(dCp), float(dV)
        r.name = name
        self.reactions.append(r)
        return r

    def validate(self) -> None:
        """Structural checks (undeclared species, solids/gases without exactly one reaction)."""
        fixed_gases = [s for s in self.species if s.phase == "gas" and not any(s in g.members for g in self.gas_phases)]
        validate_structure(self.species, self.reactions, fixed_gases)

    # ------------------------------------------------------------------ solve
    def _amounts(self, amounts: dict) -> dict:
        amt: dict = {}
        for k, v in amounts.items():
            if isinstance(k, str) and k not in self._species:
                if k in self.masters:
                    k = self.masters[k]
                elif k in self.exchange_sites:  # "X": total exchange sites (mol eq)
                    k = self.exchange_sites[k]
            sp = self._resolve(k)
            amt[sp] = amt.get(sp, 0.0) + float(v)
        return amt

    def equilibrate(
        self,
        amounts: dict,
        *,
        fixed: dict | None = None,
        pH: float | None = None,
        pe: float | None = None,
        method: str = "complementarity",
        activity=None,
        T: float | None = None,
        P: float = 1.0,
        water_mass: float | None = None,
        charge_balance=None,
        suppress=None,
        options: dict | None = None,
        certify: bool | None = None,
    ) -> Result:
        """Compute chemical equilibrium.

        Parameters
        ----------
        amounts : dict
            Input amounts (mol; per kg of initial water) keyed by species, name, or
            element/valence name in :attr:`masters`. Only conserved totals matter.
        fixed : dict, optional
            Species held at a fixed ``log10`` activity (aqueous species, electron)
            or ``log10`` partial pressure in atm (gases): an open reservoir.
        pH, pe : float, optional
            Shorthands for ``fixed={"H+": -pH}`` and ``fixed={"e-": -pe}``.
        method : {"complementarity", "gibbs", "enumerate"}
            See :mod:`discopt.speciation.solve`.
        activity : str | dict | ActivityModel, optional
            Overrides the system's activity model.
        T, P : float, optional
            Temperature (K) and pressure (atm).
        water_mass : float, optional
            Initial kg of water (default 1). With a solvent species the water mass
            is a variable (water is produced/consumed) unless ``water_mass`` is given.
        charge_balance : species or name, optional
            Adjust the total of this ion so the solution is electroneutral
            (like PHREEQC's ``charge``); the adjustment is ``charge_balance_adjustment``.
        suppress : list, optional
            Species (e.g. metastable polymorphs) to leave out of this calculation.
            Suppressed solids cannot form, but ``Result.SI`` still reports their
            saturation index in the computed solution (``> 0``: supersaturated).
        options : dict, optional
            Extra NLP solver options.
        certify : bool, optional
            Test global stability (tangent plane distance of every phase) and, while
            some phase is unstable, add it or a second copy of it at the unstable
            composition and re-solve. The result is in ``Result.certificate``; see
            :mod:`discopt.speciation.stability`. Default: only when the system has a
            non-ideal solid solution, the case where the local conditions can
            accept a metastable state.
        """
        call = dict(fixed=fixed, pH=pH, pe=pe, activity=activity, T=T, P=P, water_mass=water_mass,
                    charge_balance=charge_balance, suppress=suppress)
        net, act = self._network(amounts, **call)
        x, active, ok, info = _solve.run(net, act, method, options)
        cert = None
        if certify or (certify is None and any(ph.model is not None for ph in net.phases)):
            from discopt.speciation.stability import stabilize

            opts = dict(_solve.NLP_OPTIONS, **(options or {}))
            net, x, active, ok, cert = stabilize(net, act, x, active, ok, opts, info)
        res = _make_result(self, net, act, x, active, ok, info, method)
        res.certificate = cert
        res._state = dict(system=self, net=net, act=act, x=np.asarray(x, float), active=set(active),
                          amounts=dict(amounts), call=dict(call, method=method, options=options))
        return res

    def _network(self, amounts: dict, *, fixed=None, pH=None, pe=None, activity=None, T=None, P=1.0,
                 water_mass=None, charge_balance=None, suppress=None):
        """Build the numeric network and activity model for one calculation, without solving."""
        act = self.activity if activity is None else get_activity_model(activity)
        T = self.T if T is None else float(T)
        fixed_r = {self._resolve(k): float(v) for k, v in (fixed or {}).items()}
        if pH is not None:
            fixed_r[self["H+"]] = -float(pH)
        if pe is not None:
            e = next((s for s in self.species if s.phase == "electron"), None)
            if e is None:
                raise ValueError("pe given but the system has no electron species")
            fixed_r[e] = -float(pe)
        amt = self._amounts(amounts)
        cb = None if charge_balance is None else self._resolve(charge_balance)
        sup = [self._resolve(k) for k in (suppress or [])]
        phases = [p for p in self.solid_solutions + self.gas_phases]
        phases = _drop_suppressed(phases, set(sup))
        wm = water_mass
        has_solvent = any(s.phase == "solvent" for s in self.species)
        net = build_network(self.species, self.reactions, amt, fixed_r, T, P=P, phases=phases,
                            surfaces=self.surface_types, water_mass=wm if (wm is not None and has_solvent) else None,
                            charge_balance=cb, suppress=sup)
        if not has_solvent and wm is not None:
            net.W0 = float(wm)
        return net, act

    # ------------------------------------------------------------------ spec
    def to_spec(self) -> dict:
        from discopt.speciation.spec import system_to_spec

        return system_to_spec(self)

    @classmethod
    def from_spec(cls, spec) -> "System":
        from discopt.speciation.spec import system_from_spec

        return system_from_spec(spec)

    def __repr__(self) -> str:
        counts: dict = {}
        for s in self.species:
            counts[s.phase] = counts.get(s.phase, 0) + 1
        parts = ", ".join(f"{v} {k}" for k, v in counts.items())
        return f"System({self.name!r}, {parts}, {len(self.reactions)} reactions)"


def _drop_suppressed(phases, sup):
    out = []
    for ph in phases:
        mem = [m for m in ph.members if m not in sup]
        if not mem:
            continue
        if len(mem) == len(ph.members):
            out.append(ph)
        elif isinstance(ph, GasPhase):
            out.append(GasPhase(mem, ph.pressure, ph.volume, ph.name))
        elif ph.guggenheim is not None:
            raise ValueError(f"cannot suppress a member of the binary non-ideal solid solution {ph.name!r}")
        else:
            names = {m.name for m in mem}
            marg = {k: w for k, w in (ph.margules or {}).items() if set(k) <= names} or None
            out.append(SolidSolution(ph.name, mem, margules=marg))
    return out


def _suppressed_si(system, solid, log10_a, T, P) -> float:
    """SI of a solid that is not in the network, from the aqueous activities; ``-inf`` if undefined."""
    for r in system.reactions:
        nu = r.stoich
        if solid not in nu or any(sp.phase in ("solid", "gas") for sp in nu if sp is not solid):
            continue
        others = {sp: c for sp, c in nu.items() if sp is not solid}
        las = {sp: log10_a.get(sp.name) for sp in others}
        if any(la is None or not math.isfinite(la) for la in las.values()):
            return -math.inf
        iap = sum(c * las[sp] for sp, c in others.items())
        return (iap - r.logK_at(T, P)) / -nu[solid]
    return -math.inf


def _make_result(system, net, act, x, active, ok, info, method) -> Result:
    v, st = _solve.evaluate(net, act, x)
    res = _solve.residuals(net, act, x)
    converged = bool(ok) and max(res.values()) <= _solve.TOL
    idx = {s: k for k, s in enumerate(net.species)}
    ypos = {k: i for i, k in enumerate(net.yidx)}
    la, n = st["la"], st["n"]
    W = float(v["W"])
    z = [net.species[k].charge for k in net.aq_idx]
    I_val = float(v["I"]) if v["I"] is not None else ionic_strength(NUM, z, st["m"])

    molality, log10_a, gamma, solids, SI = {}, {}, {}, {}, {}
    gases, pp, exch, surf, mf = {}, {}, {}, {}, {}
    for s in system.species:
        k = idx.get(s)
        if s.phase == "aqueous":
            if k is None:
                molality[s.name], log10_a[s.name], gamma[s.name] = 0.0, -math.inf, 1.0
                continue
            y = float(x[ypos[k]])
            molality[s.name] = 10.0**y
            log10_a[s.name] = float(la[k])
            gamma[s.name] = 10.0 ** (float(la[k]) - y)
        elif s.phase == "solvent":
            if k is not None:
                log10_a[s.name] = float(la[k])
        elif s.phase == "solid":
            if k is None:
                solids[s.name], SI[s.name] = 0.0, -math.inf  # suppressed or pruned; see _suppressed_si below
            else:
                solids[s.name] = float(n[k])
                SI[s.name] = float(st["la_eq"][k])
        elif s.phase == "gas":
            if k is None:
                if any(s in g.members for g in system.gas_phases):
                    gases[s.name], pp[s.name], SI[s.name] = 0.0, 0.0, -math.inf
            elif net.kind[k] == "const":
                pp[s.name] = 10.0 ** net.fixed[k]
                SI[s.name] = net.fixed[k]
            else:
                gases[s.name] = float(n[k])
                SI[s.name] = float(st["la_eq"][k])
                ph = net.phases[net.member_phase[k]]
                if ph.kind == "gas_v" or net.member_phase[k] in active:
                    pp[s.name] = 10.0 ** SI[s.name]
                else:
                    pp[s.name] = 0.0
        elif s.phase in ("exchange", "surface") and not s.virtual:
            d = exch if s.phase == "exchange" else surf
            d[s.name] = 0.0 if k is None or net.kind[k] == "const" else float(n[k])
            if k is not None:
                log10_a[s.name] = float(la[k])
        elif k is not None:  # virtual species
            log10_a[s.name] = float(la[k])
    # Solids left out of the network (suppress=) still have a saturation index in this
    # solution, as in a PHREEQC SOLUTION calculation: SI = (sum_i nu_i log a_i - log K) / -nu_solid.
    for s in system.species:
        if s.phase == "solid" and idx.get(s) is None:
            SI[s.name] = _suppressed_si(system, s, log10_a, net.T, net.P)
    ss, groups = {}, {}
    for p, ph in enumerate(net.phases):
        if ph.kind == "ss":
            xs = [float(t) for t in st["x"][p]]
            amount = float(v["N"][p]) if p in active else 0.0
            ss[ph.name] = {"present": p in active, "amount": amount,
                           "mole_fraction": {net.species[k].name: xv for k, xv in zip(ph.members, xs)}}
            groups.setdefault(p if ph.copy_of is None else ph.copy_of, []).append((p in active, amount, ph.name))
    for lst in groups.values():  # species mole fractions from the most abundant copy
        mf.update(ss[max(lst)[2]]["mole_fraction"])
    charge = sum(system[k].charge * val for k, val in molality.items())
    present = [net.phases[p].name for p in sorted(active)]
    pH = -log10_a["H+"] if "H+" in log10_a and math.isfinite(log10_a["H+"]) else None
    e = next((s for s in system.species if s.phase == "electron"), None)
    pe = -log10_a[e.name] if e is not None and e.name in log10_a else None
    potential = {S.name: float(v["psi"][i]) for i, S in enumerate(net.surfaces)}
    return Result(
        converged=converged, method=method, molality=molality, log10_activity=log10_a, gamma=gamma,
        solids=solids, SI=SI, present=present, ionic_strength=float(I_val), charge_imbalance=float(charge),
        gibbs_energy=_solve.gibbs_energy(net, x, act), residuals=res,
        pruned=[s.name for s in net.pruned], info=info, water_mass=W, pH=pH, pe=pe, gases=gases,
        partial_pressure=pp, exchange=exch, surface=surf, potential=potential, mole_fraction=mf, solid_solutions=ss,
        charge_balance_adjustment=None if v["delta"] is None else float(v["delta"]),
    )


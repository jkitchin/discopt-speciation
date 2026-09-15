"""Declarative (dict / YAML / JSON) specification of an equilibrium problem.

The schema is strict (unknown keys are rejected) so that typos from humans or
agents fail loudly instead of being silently ignored::

    name: gypsum
    activity: davies            # or "ideal", or {model: davies, A: 0.509, b: 0.3}
    temperature: 298.15
    species:
      - {name: Ca+2, a0: 5.0, b: 0.165}   # phase defaults to aqueous; charge from the name
      - {name: SO4-2}
      - {name: CaSO4}
      - {name: Gypsum, phase: solid}
    reactions:
      - {equation: "Ca+2 + SO4-2 = CaSO4", logK: 2.36, dH: 6900}
      - {equation: "Gypsum = Ca+2 + SO4-2", analytic: [68.2401, 0.0, -3221.51, -25.0627]}
    problem:                    # optional: a default calculation
      amounts: {Ca+2: 0.0324, SO4-2: 0.025}
      pH: 7

A system can also start from a PHREEQC database (a bundled name or a path) and
add or override data::

    database: {file: phreeqc.dat, components: [Ca, "S(6)"], phases: [Gypsum, Anhydrite]}
    overrides:
      Gypsum: {logK: -4.60}
    species:
      - {name: CaSO4_custom}
    reactions:
      - {equation: "Ca+2 + SO4-2 = CaSO4_custom", logK: 0.5}
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from discopt.speciation.species import T_REF


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


PhaseName = Literal["aqueous", "solvent", "solid", "gas", "exchange", "surface", "electron"]


class SpeciesSpec(_Strict):
    name: str
    phase: PhaseName = "aqueous"
    charge: Optional[float] = None
    a0: Optional[float] = Field(default=None, description="ion size (Angstrom), Truesdell-Jones / B-dot")
    b: Optional[float] = Field(default=None, description="Truesdell-Jones b / neutral-species Setschenow coefficient")
    params: dict[str, Union[float, str]] = Field(default_factory=dict, description="other activity-model parameters")
    site: Optional[str] = Field(default=None, description="exchange or surface site")
    nsites: Optional[float] = Field(default=None, description="exchange sites per formula (CaX2 -> 2)")

    @model_validator(mode="after")
    def _checks(self):
        if self.phase in ("solid", "gas", "solvent") and self.charge not in (None, 0):
            raise ValueError(f"{self.phase} {self.name!r} cannot carry a charge")
        if self.phase not in ("exchange", "surface") and (self.site is not None or self.nsites is not None):
            raise ValueError(f"'site'/'nsites' apply only to exchange and surface species ({self.name!r})")
        return self


class ReactionSpec(_Strict):
    equation: Optional[str] = None
    stoich: Optional[dict[str, float]] = Field(default=None, description="net coefficients, products positive")
    logK: Optional[float] = Field(default=None, description="log10 K at 25 C for the reaction as written")
    dH: float = Field(default=0.0, description="reaction enthalpy, J/mol (van 't Hoff)")
    dCp: float = Field(default=0.0, description="reaction heat capacity, J/mol/K")
    dV: float = Field(default=0.0, description="reaction volume, cm3/mol")
    analytic: Optional[list[float]] = Field(default=None, description="PHREEQC -analytic A1..A6 (overrides logK)")
    name: Optional[str] = None

    @model_validator(mode="after")
    def _one_form(self):
        if (self.equation is None) == (self.stoich is None):
            raise ValueError("give exactly one of 'equation' or 'stoich'")
        if self.logK is None and self.analytic is None:
            raise ValueError("give 'logK' or 'analytic'")
        return self


class OverrideSpec(_Strict):
    """Replace the thermodynamic data of an existing (e.g. database) reaction, found by name."""

    logK: Optional[float] = None
    dH: Optional[float] = None
    dCp: Optional[float] = None
    analytic: Optional[list[float]] = None


class DaviesSpec(_Strict):
    model: Literal["davies"]
    A: Optional[float] = None
    b: float = 0.3


class IdealSpec(_Strict):
    model: Literal["ideal"]


class ModelSpec(BaseModel):
    """Any other activity model: ``{model: sit, eps: [[Na+, Cl-, 0.03]]}``, ``{model: pitzer, params: ...}``."""

    model_config = ConfigDict(extra="allow")
    model: Literal["debye-huckel", "extended-debye-huckel", "truesdell-jones", "phreeqc", "wateq", "bdot", "b-dot",
                   "sit", "pitzer"]


ActivitySpec = Union[
    Literal["ideal", "davies", "debye-huckel", "extended-debye-huckel", "truesdell-jones", "phreeqc", "wateq",
            "bdot", "b-dot", "sit", "pitzer"],
    DaviesSpec, IdealSpec, ModelSpec,
]


class ExchangeSiteSpec(_Strict):
    name: str = "X"
    charge: float = -1.0


class SurfaceSpec(_Strict):
    name: str
    sites: Optional[list[str]] = None
    area: Optional[float] = Field(default=None, description="specific area, m2/g")
    mass: Optional[float] = Field(default=None, description="g of sorbent")
    edl: Optional[Literal["ddl"]] = "ddl"


class SolidSolutionSpec(_Strict):
    name: str
    members: list[str]
    guggenheim: Optional[list[float]] = Field(default=None, description="binary [a0, a1, a2], PHREEQC -Gugg_nondim")
    margules: Optional[list[tuple[str, str, float]]] = Field(default=None, description="[[member_i, member_j, W], ...]")


class GasPhaseSpec(_Strict):
    members: list[str]
    pressure: Optional[float] = 1.0
    volume: Optional[float] = None
    name: str = "gas"


class DatabaseSpec(_Strict):
    file: str = Field(description="bundled database name (phreeqc.dat, pitzer.dat, wateq4f_PWN.dat) or a path")
    components: list[str]
    phases: Union[Literal["all"], list[str]] = Field(default_factory=list)
    exchange: Optional[list[str]] = None
    surfaces: Optional[dict[str, dict]] = None
    gas_phase: Optional[dict] = None
    solid_solutions: Optional[dict[str, Union[list[str], dict]]] = None


class ProblemSpec(_Strict):
    amounts: dict[str, float]
    fixed: dict[str, float] = Field(default_factory=dict, description="species -> log10 activity (gases: log10 atm)")
    pH: Optional[float] = None
    pe: Optional[float] = None
    method: Literal["complementarity", "gibbs", "enumerate"] = "complementarity"
    activity: Optional[ActivitySpec] = None
    temperature: Optional[float] = None
    pressure: float = 1.0
    water_mass: Optional[float] = None
    charge_balance: Optional[str] = None
    suppress: list[str] = Field(default_factory=list)


class SystemSpec(_Strict):
    name: str = "system"
    temperature: float = T_REF
    activity: Optional[ActivitySpec] = Field(default=None, description="default: ideal (database systems: auto)")
    database: Optional[DatabaseSpec] = None
    species: list[SpeciesSpec] = Field(default_factory=list)
    reactions: list[ReactionSpec] = Field(default_factory=list)
    overrides: dict[str, OverrideSpec] = Field(default_factory=dict)
    exchange_sites: list[ExchangeSiteSpec] = Field(default_factory=list)
    surfaces: list[SurfaceSpec] = Field(default_factory=list)
    solid_solutions: list[SolidSolutionSpec] = Field(default_factory=list)
    gas_phases: list[GasPhaseSpec] = Field(default_factory=list)
    masters: dict[str, str] = Field(default_factory=dict, description="input-amount alias -> species")
    problem: Optional[ProblemSpec] = None

    @model_validator(mode="after")
    def _names(self):
        names = [s.name for s in self.species]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate species names: {sorted(dup)}")
        if self.database is None and not self.species:
            raise ValueError("give 'species' (or a 'database')")
        return self


def _activity(a):
    return a if isinstance(a, str) else a.model_dump()


def system_from_spec(spec) -> "System":
    from discopt.speciation.system import System

    if not isinstance(spec, SystemSpec):
        spec = SystemSpec.model_validate(spec)
    if spec.database is not None:
        from discopt.speciation.database import PhreeqcDatabase

        d = spec.database
        s = PhreeqcDatabase.load(d.file).system(
            d.components, d.phases, exchange=d.exchange, surfaces=d.surfaces, gas_phase=d.gas_phase,
            solid_solutions=d.solid_solutions, activity="auto" if spec.activity is None else _activity(spec.activity),
            T=spec.temperature, name=spec.name)
    else:
        s = System(spec.name, T=spec.temperature, activity="ideal" if spec.activity is None else _activity(spec.activity))
    for ex in spec.exchange_sites:
        s.exchange_site(ex.name, ex.charge)
    for st in spec.surfaces:
        s.surface_type(st.name, st.sites, st.area, st.mass, st.edl)
    for sp in spec.species:
        p = dict(sp.params)
        if sp.phase == "aqueous":
            s.aqueous(sp.name, sp.charge, a0=sp.a0, b=sp.b, **p)
        elif sp.phase == "solid":
            s.solid(sp.name)
        elif sp.phase == "gas":
            s.gas(sp.name)
        elif sp.phase == "solvent":
            s.solvent(sp.name)
        elif sp.phase == "electron":
            s.electron(sp.name)
        elif sp.phase == "exchange":
            s.exchange(sp.name, sp.site, sp.nsites, a0=sp.a0, b=sp.b, charge=sp.charge)
        else:
            s.surface(sp.name, sp.charge, sp.site)
    for r in spec.reactions:
        s.reaction(r.equation if r.equation is not None else r.stoich, logK=r.logK, dH=r.dH, name=r.name,
                   analytic=r.analytic, dCp=r.dCp, dV=r.dV)
    by_name = {r.name: r for r in s.reactions if r.name}
    for name, o in spec.overrides.items():
        if name not in by_name:
            raise ValueError(f"override for unknown reaction {name!r}")
        r = by_name[name]
        if o.analytic is not None:
            r.analytic = tuple(o.analytic)
        elif o.logK is not None:
            r.analytic = None
        if o.logK is not None:
            r.logK = o.logK
        if o.dH is not None:
            r.dH = o.dH
        if o.dCp is not None:
            r.dCp = o.dCp
    for ss in spec.solid_solutions:
        s.solid_solution(ss.name, ss.members, guggenheim=ss.guggenheim, margules=ss.margules)
    for gp in spec.gas_phases:
        s.gas_phase(gp.members, gp.pressure, gp.volume, gp.name)
    s.masters.update(spec.masters)
    s.validate()
    return s


def system_to_spec(system) -> dict:
    """A self-contained spec (database systems are written out species by species)."""
    species = []
    for sp in system.species:
        if sp.phase == "exchange" and sp.virtual:
            continue  # recreated by exchange_sites
        d = {"name": sp.name, "phase": sp.phase}
        if sp.phase in ("aqueous", "exchange", "surface"):
            d["charge"] = sp.charge
        params = dict(sp.params)
        if sp.phase in ("aqueous", "exchange"):
            for k in ("a0", "b"):
                if k in params:
                    d[k] = params.pop(k)
        if params and sp.phase == "aqueous":
            d["params"] = params
        if sp.phase in ("exchange", "surface"):
            d["site"] = sp.site
        if sp.phase == "exchange":
            d["nsites"] = sp.nsites
        species.append(d)
    reactions = []
    for r in system.reactions:
        d = {"stoich": {s.name: c for s, c in r.stoich.items()}}
        if r.logK is not None:
            d["logK"] = r.logK
        if r.analytic is not None:
            d["analytic"] = list(r.analytic)
        for k in ("dH", "dCp", "dV"):
            if getattr(r, k):
                d[k] = getattr(r, k)
        if r.name:
            d["name"] = r.name
        reactions.append(d)
    spec = {
        "name": system.name, "temperature": system.T, "activity": system.activity.to_spec(),
        "species": species, "reactions": reactions,
    }
    if system.exchange_sites:
        spec["exchange_sites"] = [{"name": k, "charge": v.charge} for k, v in system.exchange_sites.items()]
    if system.surface_types:
        spec["surfaces"] = [{"name": st.name, "sites": list(st.sites), "area": st.area, "mass": st.mass, "edl": st.edl}
                            for st in system.surface_types]
    if system.solid_solutions:
        spec["solid_solutions"] = [ss.to_spec() for ss in system.solid_solutions]
    if system.gas_phases:
        spec["gas_phases"] = [{"members": [m.name for m in g.members], "pressure": g.pressure, "volume": g.volume,
                               "name": g.name} for g in system.gas_phases]
    if system.masters:
        spec["masters"] = {k: getattr(v, "name", v) for k, v in system.masters.items()}
    SystemSpec.model_validate(spec)
    return spec


def load_spec(source) -> SystemSpec:
    """Validate a dict, a YAML/JSON string, or a path to a YAML/JSON file."""
    if isinstance(source, SystemSpec):
        return source
    if isinstance(source, dict):
        return SystemSpec.model_validate(source)
    text = str(source)
    if "\n" not in text and text.endswith((".yaml", ".yml", ".json")):
        with open(text) as f:
            text = f.read()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("specification must be a mapping")
    return SystemSpec.model_validate(data)


def from_yaml(source) -> "System":
    return system_from_spec(load_spec(source))


def solve_spec(source, **overrides) -> dict:
    """Validate a spec with a ``problem`` section, solve it, return a JSON-able dict.

    ``overrides`` replace fields of ``problem`` (e.g. ``method="gibbs"``). This is
    the entry point intended for agents and scripts; it never raises on a
    non-converged solve (check ``"converged"``), but does raise on invalid input.
    """
    spec = load_spec(source)
    if spec.problem is None and "amounts" not in overrides:
        raise ValueError("spec has no 'problem' section (amounts to equilibrate)")
    base = spec.problem.model_dump() if spec.problem else {}
    prob = ProblemSpec.model_validate({**base, **overrides})
    system = system_from_spec(spec)
    res = system.equilibrate(
        prob.amounts, fixed=prob.fixed, pH=prob.pH, pe=prob.pe, method=prob.method,
        activity=None if prob.activity is None else _activity(prob.activity), T=prob.temperature, P=prob.pressure,
        water_mass=prob.water_mass, charge_balance=prob.charge_balance, suppress=prob.suppress or None,
    )
    return res.to_dict()

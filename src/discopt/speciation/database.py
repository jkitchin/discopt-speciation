"""Read PHREEQC thermodynamic databases and build :class:`~discopt.speciation.System` objects.

Supported keyword blocks: ``SOLUTION_MASTER_SPECIES``, ``SOLUTION_SPECIES``,
``PHASES``, ``EXCHANGE_MASTER_SPECIES``, ``EXCHANGE_SPECIES``,
``SURFACE_MASTER_SPECIES``, ``SURFACE_SPECIES`` and ``PITZER``; other blocks
(``RATES``, ``MEAN_GAMMAS``, ...) are skipped. Per reaction the parser reads
``log_k``, ``delta_h`` (with units), ``-analytic`` and ``-gamma``, and per gas
the Peng-Robinson critical constants ``-T_c``, ``-P_c`` and ``-Omega``, with
``GAS_BINARY_PARAMETERS``. Molar volumes and diffusion coefficients are ignored
(so pressure corrections of ``log K`` are not modelled).

Example::

    db = PhreeqcDatabase.from_file("phreeqc.dat")
    s = db.system(["Ca", "S(6)", "Na", "Cl"], phases=["Gypsum", "Anhydrite"])
    res = s.equilibrate({"Ca": 0.02, "S(6)": 0.02, "Na": 0.1, "Cl": 0.1}, charge_balance="Cl-")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from discopt.speciation.activity import LLNL, SIT, PitzerParameters, Pitzer, TruesdellJones, get_activity_model
from discopt.speciation.species import T_REF, parse_charge

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def available_databases() -> list[str]:
    """Names of the PHREEQC databases bundled with the package (see ``data/README.md``)."""
    return sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".dat"))

KEYWORDS = {
    "SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES", "PHASES", "EXCHANGE_MASTER_SPECIES", "EXCHANGE_SPECIES",
    "SURFACE_MASTER_SPECIES", "SURFACE_SPECIES", "PITZER", "SIT", "RATES", "END", "MEAN_GAMMAS", "MEAN",
    "LLNL_AQUEOUS_MODEL_PARAMETERS", "NAMED_EXPRESSIONS", "CALCULATE_VALUES", "ISOTOPES", "ISOTOPE_RATIOS",
    "ISOTOPE_ALPHAS", "SOLUTION", "SOLUTION_SPREAD", "EQUILIBRIUM_PHASES", "KINETICS", "SELECTED_OUTPUT",
    "USER_PUNCH", "USER_PRINT", "TITLE", "KNOBS", "DATABASE", "GAS_BINARY_PARAMETERS", "PRINT", "GAS_PHASE",
}
_UNITS = {"kcal": 4184.0, "kcal/mol": 4184.0, "kj": 1000.0, "kj/mol": 1000.0, "j": 1.0, "j/mol": 1.0, "": 1000.0}
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_ATTACHED = re.compile(r"^(\d+\.?\d*|\.\d+)([A-Za-z(\[].*)$")
_OPTION = re.compile(r"^-[A-Za-z_]")  # "-log_k", but not the reaction "- 4 H+ + ... = O2"


@dataclass
class Entry:
    """A reaction from a species, exchange, surface or phase block."""

    name: str
    nu: dict  # species name -> coefficient (products positive), defined species/phase included
    logK: float | None = None
    dH: float = 0.0
    analytic: tuple | None = None
    a0: float | None = None
    b: float | None = None
    gas: bool = False
    options: dict = field(default_factory=dict)

    @property
    def identity(self) -> bool:
        return not any(abs(c) > 1e-12 for c in self.nu.values())

    def others(self) -> set:
        return {k for k, c in self.nu.items() if k != self.name and abs(c) > 1e-12}


def _is_num(tok: str) -> bool:
    return bool(_NUM.match(tok))


def parse_phreeqc_equation(text: str) -> tuple[list, list]:
    """``"Ca+2 + 2X- = CaX2"`` -> ``([(1, "Ca+2"), (2, "X-")], [(1, "CaX2")])``.

    Terms are separated by whitespace-delimited ``+``/``-``; a ``-`` separator
    negates the following term. Coefficients may be attached (``2H+``).
    """
    if text.count("=") != 1:
        raise ValueError(f"cannot parse reaction {text!r}")
    lhs, rhs = text.split("=")

    def side(s):
        toks = s.split()
        out, sign, coef = [], 1.0, None
        for t in toks:
            if t == "+":
                sign = 1.0
                continue
            if t == "-":
                sign = -1.0
                continue
            if _is_num(t) and coef is None:
                coef = float(t)
                continue
            mm = _ATTACHED.match(t)
            if mm and coef is None:
                c, name = float(mm.group(1)), mm.group(2)
            else:
                c, name = (1.0 if coef is None else coef), t
            out.append((sign * c, name))
            sign, coef = 1.0, None
        return out

    return side(lhs), side(rhs)


class PhreeqcDatabase:
    """Parsed PHREEQC database."""

    def __init__(self, text: str):
        self.masters: dict[str, dict] = {}  # element/valence -> {"species", "alk", "gfw"}
        self.species: dict[str, Entry] = {}
        self.phases: dict[str, Entry] = {}
        self.exchange_masters: dict[str, str] = {}
        self.exchange_species: dict[str, Entry] = {}
        self.surface_masters: dict[str, str] = {}
        self.surface_species: dict[str, Entry] = {}
        self.pitzer: PitzerParameters | None = None
        self.llnl: dict[str, list] = {}  # LLNL_AQUEOUS_MODEL_PARAMETERS: temperatures, dh_a, dh_b, bdot, co2_coefs
        self.sit: dict[tuple, float] = {}  # SIT -epsilon: (species, species) -> kg/mol
        self.sit_unsupported: list = []  # temperature-dependent or -epsilon1/-epsilon2 entries (not applied)
        self.named_expressions: dict[str, Entry] = {}
        #: GAS_BINARY_PARAMETERS: {frozenset((gas_i, gas_j)): k_ij}
        self.gas_binary_parameters: dict = {}
        self._parse(text)

    @classmethod
    def from_file(cls, path) -> "PhreeqcDatabase":
        with open(path, encoding="latin-1") as f:
            return cls(f.read())

    @classmethod
    def load(cls, name: str = "phreeqc.dat") -> "PhreeqcDatabase":
        """Load a bundled database by name (see :func:`available_databases`) or a file path."""
        path = os.path.join(DATA_DIR, name if name.endswith(".dat") else name + ".dat")
        if os.path.exists(path):
            return cls.from_file(path)
        if os.path.exists(name):
            return cls.from_file(name)
        raise FileNotFoundError(f"no bundled database {name!r}; bundled: {available_databases()}")

    # ------------------------------------------------------------------ parsing
    def _parse(self, text: str) -> None:
        block = None
        current: Entry | None = None
        phase_name = None
        pitzer_kind = llnl_key = sit_kind = None
        for raw in text.splitlines():
            line = raw.split("#")[0].rstrip()
            if not line.strip():
                continue
            first = line.split()[0]
            if not raw[0].isspace() and first in KEYWORDS:
                block, current, phase_name, pitzer_kind, llnl_key, sit_kind = first, None, None, None, None, None
                if block == "PITZER" and self.pitzer is None:
                    self.pitzer = PitzerParameters()
                continue
            if block == "SOLUTION_MASTER_SPECIES":
                t = line.split()
                if len(t) >= 2:
                    alk = float(t[2]) if len(t) > 2 and _is_num(t[2]) else 0.0
                    gfw = float(t[-1]) if _is_num(t[-1]) else None
                    self.masters[t[0]] = {"species": t[1], "alk": alk, "gfw": gfw}
            elif block == "EXCHANGE_MASTER_SPECIES":
                t = line.split()
                if len(t) >= 2:
                    self.exchange_masters[t[0]] = t[1]
            elif block == "SURFACE_MASTER_SPECIES":
                t = line.split()
                if len(t) >= 2:
                    self.surface_masters[t[0]] = t[1]
            elif block in ("SOLUTION_SPECIES", "EXCHANGE_SPECIES", "SURFACE_SPECIES"):
                store = {"SOLUTION_SPECIES": self.species, "EXCHANGE_SPECIES": self.exchange_species,
                         "SURFACE_SPECIES": self.surface_species}[block]
                for part in line.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    if "=" in part and not _OPTION.match(part):
                        lhs, rhs = parse_phreeqc_equation(part)
                        name = rhs[0][1]
                        current = Entry(name, _net(lhs, rhs))
                        current.nu[name] = current.nu.get(name, 0.0)
                        store[name] = current
                    elif current is not None:
                        _option(current, part)
            elif block == "PHASES":
                for part in line.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    if "=" in part and not _OPTION.match(part):
                        if phase_name is None:
                            continue
                        lhs, rhs = parse_phreeqc_equation(part)
                        c0, _formula = lhs[0]
                        nu = _net(lhs[1:], rhs)
                        nu = {k: v / c0 for k, v in nu.items()}
                        nu[phase_name] = -1.0
                        current = Entry(phase_name, nu, gas=phase_name.endswith("(g)"))
                        current.options["formula"] = _formula
                        self.phases[phase_name] = current
                        phase_name = None
                    elif _OPTION.match(part) or part.split()[0].lower() in ("log_k", "delta_h", "vm", "analytic",
                                                                                "t_c", "p_c", "omega"):
                        if current is not None:
                            _option(current, part)
                    else:
                        phase_name, current = part.split()[0], None
            elif block == "NAMED_EXPRESSIONS":
                part = line.strip()
                if _OPTION.match(part) or part.split()[0].lower() in ("log_k", "delta_h", "analytic"):
                    if current is not None:
                        _option(current, part)
                else:
                    current = Entry(part.split()[0], {})
                    self.named_expressions[current.name] = current
            elif block == "GAS_BINARY_PARAMETERS":
                t = line.split()
                if len(t) >= 3 and _is_num(t[2]):
                    self.gas_binary_parameters[frozenset(t[:2])] = float(t[2])
            elif block == "LLNL_AQUEOUS_MODEL_PARAMETERS":
                t = line.split()
                if _OPTION.match(t[0]):
                    llnl_key = t[0][1:].lower()
                    self.llnl[llnl_key] = []
                    t = t[1:]
                if llnl_key is not None:
                    self.llnl[llnl_key] += [float(v) for v in t if _is_num(v)]
            elif block == "SIT":
                t = line.split()
                if _OPTION.match(t[0]):
                    sit_kind = t[0][1:].lower()
                    continue
                if len(t) >= 3 and _is_num(t[2]):
                    if sit_kind in ("epsilon", "epsilon1", "epsilon2"):
                        if sit_kind != "epsilon" or len([v for v in t[3:] if _is_num(v) and float(v)]) > 0:
                            self.sit_unsupported.append((sit_kind, t[0], t[1]))
                        if sit_kind == "epsilon":
                            self.sit[(t[0], t[1])] = float(t[2])
            elif block == "PITZER":
                t = line.split()
                if t[0].startswith("-"):
                    pitzer_kind = t[0][1:].upper()
                    continue
                if pitzer_kind in ("B0", "B1", "B2", "C0", "THETA", "LAMDA", "LAMBDA", "ZETA", "PSI"):
                    nn = 3 if pitzer_kind in ("ZETA", "PSI") else 2
                    names, nums = t[:nn], [float(v) for v in t[nn:] if _is_num(v)]
                    self.pitzer.add(pitzer_kind, names, nums)
        self._apply_named_expressions()

    def _apply_named_expressions(self) -> None:
        """``-add_logk NAME coef``: add ``coef`` times the named ``log_k``, ``delta_h`` and ``-analytic``."""
        for ent in (*self.species.values(), *self.phases.values()):
            for nm, coef in ent.options.pop("add_logk", []):
                if nm not in self.named_expressions:
                    raise KeyError(f"{ent.name}: -add_logk refers to unknown NAMED_EXPRESSIONS entry {nm!r}")
                ne = self.named_expressions[nm]
                ent.logK = (ent.logK or 0.0) + coef * (ne.logK or 0.0)
                ent.dH += coef * ne.dH
                if ent.analytic or ne.analytic:
                    a = list(ent.analytic or ()) + [0.0] * 6
                    n = list(ne.analytic or ()) + [0.0] * 6
                    ent.analytic = tuple(x + coef * y for x, y in zip(a[:6], n[:6]))

    # ------------------------------------------------------------------ building systems
    def default_activity(self):
        """The database's own aqueous model: Pitzer, LLNL B-dot, SIT, or Truesdell-Jones."""
        if self.pitzer is not None:
            return Pitzer(self.pitzer)
        if self.llnl:
            return LLNL(table={k: v for k, v in self.llnl.items()
                               if k in ("temperatures", "dh_a", "dh_b", "bdot", "co2_coefs")})
        if self.sit:
            return SIT(eps=self.sit)
        return TruesdellJones()

    def master_key(self, name: str) -> str:
        """Database spelling of an element or valence state (``"C(4)"`` -> ``"C(+4)"``)."""
        if name in self.masters:
            return name
        mm = re.fullmatch(r"(\w+)\(([-+]?\d+(?:\.\d+)?)\)", name)
        if mm:
            for key in self.masters:
                km = re.fullmatch(r"(\w+)\(([-+]?\d+(?:\.\d+)?)\)", key)
                if km and km.group(1) == mm.group(1) and float(km.group(2)) == float(mm.group(2)):
                    return key
        raise KeyError(f"{name!r} is not in SOLUTION_MASTER_SPECIES")

    def expand(self, components) -> list[str]:
        """Element names -> all their master entries (``"S"`` -> ``S``, ``S(6)``, ``S(-2)``)."""
        out = []
        for c in components:
            if "(" in c:
                out.append(self.master_key(c))
            elif c in self.masters or any(m.startswith(c + "(") for m in self.masters):
                out += [m for m in self.masters if m == c or m.startswith(c + "(")]
            else:
                raise KeyError(f"element {c!r} is not in SOLUTION_MASTER_SPECIES")
        return list(dict.fromkeys(out))

    def system(self, components, phases=(), *, exchange=None, surfaces=None, gas_phase=None,
               solid_solutions=None, activity="auto", T: float = T_REF, name: str = "phreeqc", gas_eos: str = "auto"):
        """Build a :class:`~discopt.speciation.System` with every species the components allow.

        Parameters
        ----------
        components : list of str
            Elements (``"Ca"``: all valence states) or valence states (``"S(6)"``).
            ``H+`` and ``H2O`` are always included. Redox species appear only when
            several valence states of an element are requested.
        phases : list of str or "all"
            Phases (minerals, gases) to consider. Gases must be in ``gas_phase``
            or given a fixed partial pressure when equilibrating.
        exchange : list of str, optional
            Exchanger names (``["X"]``).
        surfaces : dict, optional
            ``{"Hfo": {"area": 600, "mass": 0.09, "edl": "ddl"}}``.
        gas_phase : dict, optional
            ``{"members": ["CO2(g)"], "pressure": 1.0}`` or with ``"volume"``; optional
            ``"eos"`` and ``"kij"`` as in :meth:`System.gas_phase`.
        solid_solutions : dict, optional
            ``{"name": [member phases]}`` (ideal) or
            ``{"name": {"members": [...], "guggenheim": [a0, a1]}}`` (or ``"margules"``).
        activity : "auto" or a model spec
            ``"auto"``: Pitzer for a database with a ``PITZER`` block, else Truesdell-Jones.
        gas_eos : {"auto", "ideal", "peng-robinson"}
            ``"auto"``: gases with ``-T_c``/``-P_c`` follow Peng-Robinson, as in PHREEQC.
        """
        from discopt.speciation.system import System

        if activity == "auto":
            act = self.default_activity()
        else:
            act = get_activity_model(activity)
        s = System(name, T=T, activity=act, gas_eos=gas_eos)
        s.gas_kij = dict(self.gas_binary_parameters)

        requested = self.expand(components)
        master_sp = {self.masters[m]["species"] for m in requested}
        admitted = set(master_sp) | {"H+", "H2O"}
        # llnl.dat writes redox reactions with O2(aq) (or H2) instead of e-: unless O(0)/H(0) is
        # requested, those species act like the electron and are added only as intermediates.
        carriers = {v["species"] for k, v in self.masters.items()
                    if re.fullmatch(r"[OH]\(\+?0\)", k)} - master_sp - {"H+", "H2O"}
        redox = {"e-"} | carriers
        e_ok = {sp for sp in master_sp if sp in self.species and (self.species[sp].others() & redox)}
        changed = True
        while changed:
            changed = False
            for nm, ent in self.species.items():
                if nm in admitted or ent.identity:
                    continue
                oth = ent.others()
                if oth & redox and nm not in e_ok:
                    continue
                if all(o in admitted or o in redox for o in oth):
                    admitted.add(nm)
                    changed = True
        rxns = []
        uses_e = False
        used_carriers = set()
        for nm, ent in self.species.items():
            if nm not in admitted or ent.identity:
                continue
            oth = ent.others()
            if oth & redox and nm not in e_ok:
                continue
            if all(o in admitted or o in redox for o in oth):
                rxns.append(ent)
                uses_e |= "e-" in oth
                used_carriers |= oth & carriers
        for c in sorted(used_carriers):
            admitted.add(c)
            rxns.append(self.species[c])
            uses_e |= "e-" in self.species[c].others()
        # declare solution species
        for nm in [n for n in self.species if n in admitted] + sorted(admitted - set(self.species)):
            if nm == "H2O":
                s.solvent("H2O")
            elif nm == "e-":
                continue
            else:
                ent = self.species.get(nm)
                flag = {"gamma": ent.options["gamma"]} if ent and "gamma" in ent.options else {}
                s.aqueous(nm, parse_charge(nm), a0=ent.a0 if ent else None, b=ent.b if ent else None, **flag)
        if uses_e:
            s.electron("e-")
        for ent in rxns:
            _add_reaction(s, ent)
        for el in requested:
            sp = self.masters[el]["species"]
            if sp in s:
                s.masters[el] = sp
                base = el.split("(")[0]
                if base in self.masters and self.masters[base]["species"] == sp:
                    s.masters[base] = sp
        for c in components:  # the caller's spelling too ("C(4)" for "C(+4)")
            if "(" in c and c not in s.masters:
                sp = self.masters[self.master_key(c)]["species"]
                if sp in s:
                    s.masters[c] = sp

        # exchangers
        for X in (exchange or []):
            if X not in self.exchange_masters:
                raise KeyError(f"exchanger {X!r} is not in EXCHANGE_MASTER_SPECIES")
            xm = self.exchange_masters[X]
            s.exchange_site(X, parse_charge(xm))
            for nm, ent in self.exchange_species.items():
                if ent.identity or xm not in ent.nu:
                    continue
                oth = ent.others() - {xm}
                if all(o in s for o in oth):
                    s.exchange(nm, site=X, nsites=abs(ent.nu[xm]), a0=ent.a0, b=ent.b)
                    _add_reaction(s, ent)

        # surfaces
        for surf, cfg in (surfaces or {}).items():
            cfg = dict(cfg or {})
            sites = [site for site in self.surface_masters if site.split("_")[0] == surf]
            if not sites:
                raise KeyError(f"surface {surf!r} has no SURFACE_MASTER_SPECIES")
            s.surface_type(surf, sites, cfg.get("area"), cfg.get("mass"), cfg.get("edl", "ddl"))
            mspecies = {self.surface_masters[site]: site for site in sites}
            for sp, site in mspecies.items():
                s.surface(sp, site=site)
                s.masters[site] = sp
            for nm, ent in self.surface_species.items():
                if ent.identity or nm in s:
                    continue
                ms = [m for m in mspecies if m in ent.nu]
                if len(ms) != 1:
                    continue
                oth = ent.others() - {ms[0]}
                if all(o in s for o in oth):
                    s.surface(nm, site=mspecies[ms[0]])
                    _add_reaction(s, ent)

        # phases
        names = list(self.phases) if phases == "all" else list(phases)
        gas_members = set((gas_phase or {}).get("members", []))
        ss_members = {m for mem in (solid_solutions or {}).values()
                      for m in (mem["members"] if isinstance(mem, dict) else mem)}
        wanted = list(dict.fromkeys(names + sorted(gas_members) + sorted(ss_members)))
        for ph in wanted:
            if ph not in self.phases:
                raise KeyError(f"phase {ph!r} is not in PHASES")
            ent = self.phases[ph]
            oth = ent.others()
            if not all(o in s for o in oth):
                if phases != "all":
                    missing = sorted(o for o in oth if o not in s)
                    raise ValueError(f"phase {ph!r} needs species {missing} that the components do not provide")
                continue
            if ent.gas:
                sp = s.gas(ph)
                tc, pc = ent.options.get("t_c"), ent.options.get("p_c")
                if tc and pc and tc > 0 and pc > 0:
                    s.critical_constants(sp, tc, pc, ent.options.get("omega") or 0.0)
            else:
                s.solid(ph)
            _add_reaction(s, ent)
        if gas_phase:
            gp = dict(gas_phase)
            s.gas_phase(gp.pop("members"), **gp)
        for ssname, mem in (solid_solutions or {}).items():
            if isinstance(mem, dict):  # {"members": [...], "guggenheim": [a0, a1]} or "margules"
                kw = dict(mem)
                s.solid_solution(ssname, kw.pop("members"), **kw)
            else:
                s.solid_solution(ssname, mem)
        return s


def _net(lhs, rhs) -> dict:
    nu: dict = {}
    for c, n in lhs:
        nu[n] = nu.get(n, 0.0) - c
    for c, n in rhs:
        nu[n] = nu.get(n, 0.0) + c
    return nu


def _option(ent: Entry, part: str) -> None:
    t = part.split()
    key = t[0].lstrip("-").lower()
    vals = t[1:]
    nums = [float(v) for v in vals if _is_num(v)]
    if key in ("log_k", "logk", "lk"):
        ent.logK = nums[0] if nums else 0.0
    elif key in ("delta_h", "deltah"):
        unit = next((v for v in vals if not _is_num(v)), "").lower()
        ent.dH = (nums[0] if nums else 0.0) * _UNITS.get(unit, 1000.0)
    elif key in ("analytic", "analytical", "analytical_expression", "a_e", "ae"):
        ent.analytic = tuple(nums)
    elif key in ("gamma",):
        if len(nums) >= 2:
            ent.a0, ent.b = nums[0], nums[1]
        elif nums:
            ent.a0 = nums[0]
    elif key in ("llnl_gamma",):
        ent.a0 = nums[0] if nums else None
        ent.options["gamma"] = "llnl"
    elif key in ("co2_llnl_gamma", "co2_gamma"):
        ent.options["gamma"] = "co2"
    elif key in ("add_logk", "add_log_k"):
        ent.options.setdefault("add_logk", []).append((vals[0], nums[0] if len(nums) and len(vals) > 1 else 1.0))
    elif key in ("t_c", "p_c", "omega"):
        ent.gas = True
        ent.options[key] = nums[0] if nums else None
    else:
        ent.options[key] = vals


def _add_reaction(system, ent: Entry) -> None:
    nu = {k: c for k, c in ent.nu.items() if abs(c) > 1e-12}
    logK = ent.logK if (ent.logK is not None or ent.analytic is not None) else 0.0
    system.reaction(nu, logK=logK, dH=ent.dH, analytic=ent.analytic, name=ent.name)

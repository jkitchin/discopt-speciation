"""discopt.speciation: aqueous chemical equilibrium on top of discopt.

Declare species, reactions (with ``logK``) and phases, then equilibrate. Phases
obey complementarity (present and saturated, or absent and undersaturated), so
the computed state is a true thermodynamic equilibrium: it does not depend on
how reactions are scaled or which inert candidate phases are listed.
"""

from discopt.speciation import fit, plot
from discopt.speciation import optimize  # noqa: F401  (equilibrium-constrained design and worst-case bounds)
from discopt.speciation.activity import (
    LLNL,
    SIT,
    ActivityModel,
    BDot,
    Davies,
    DebyeHuckel,
    ExtendedDebyeHuckel,
    Ideal,
    Pitzer,
    PitzerParameters,
    TruesdellJones,
    get_activity_model,
)
from discopt.speciation.database import PhreeqcDatabase, available_databases
from discopt.speciation.phases import GasPhase, SolidSolution, SurfaceType
from discopt.speciation.species import Reaction, Species, log_k, parse_charge
from discopt.speciation.spec import SystemSpec, from_yaml, load_spec, solve_spec
from discopt.speciation.stability import Certificate, stability_certificate
from discopt.speciation.system import Result, System, parse_equation

__version__ = "0.2.0"

__all__ = [
    "ActivityModel", "Ideal", "DebyeHuckel", "Davies", "ExtendedDebyeHuckel", "TruesdellJones", "BDot", "LLNL", "SIT",
    "Pitzer", "PitzerParameters", "get_activity_model",
    "GasPhase", "SolidSolution", "SurfaceType",
    "Reaction", "Species", "log_k", "parse_charge",
    "System", "Result", "parse_equation",
    "SystemSpec", "from_yaml", "load_spec", "solve_spec",
    "PhreeqcDatabase", "available_databases", "fit", "plot",
]

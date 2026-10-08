# PHREEQC databases

Five [PHREEQC](references.md#parkhurst2013) databases are bundled:

```python
from discopt.speciation import PhreeqcDatabase, available_databases

available_databases()   # ['llnl.dat', 'phreeqc.dat', 'pitzer.dat', 'sit.dat', 'wateq4f_PWN.dat']
db = PhreeqcDatabase.load("phreeqc.dat")
db = PhreeqcDatabase.from_file("/path/to/minteq.v4.dat")  # any other PHREEQC-format file
db = PhreeqcDatabase(text)                                # or a string
```

| file | activity model (`activity="auto"`) | use for |
|------|----------------|---------|
| [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) | Truesdell-Jones / Davies (ion association) | fresh and brackish water, I ≲ 0.5–1 m |
| [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat) | Pitzer | brines and evaporites, to ~6 m and ~200 °C; fewer elements and no redox |
| [`wateq4f_PWN.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat) | Truesdell-Jones | natural waters, trace metals, more minerals |
| [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) | LLNL B-dot | about 1300 aqueous species and 1200 phases, including all rare earths; 0–300 °C; I ≲ 1 m |
| [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) | SIT | [ThermoChimie](references.md#giffaut2014) v12a: Th, U, Np, Pu, Am, and other elements relevant to radioactive waste; brines to ~3–4 m; mostly 25 °C |

**Provenance.**
- The first three files are the public-domain USGS databases, copied from
  [phreeqc-python](references.md#vitens) (Apache-2.0).
- [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) (LLNL `thermo.com.V8.R6.230`) and [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) ([ThermoChimie](references.md#giffaut2014),
  ANDRA–NWS–ONDRAF) are the copies distributed with PHREEQC 3. ThermoChimie
  has its own terms of use.

Citations and terms are listed in [`src/discopt/speciation/data/README.md`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/README.md).

**Automatic model.** `db.default_activity()` picks the model that the file
defines, and `db.system(activity="auto")` uses it:
- Pitzer if the file has a `PITZER` block;
- else LLNL B-dot if it has `LLNL_AQUEOUS_MODEL_PARAMETERS`;
- else SIT if it has a `SIT` block;
- else Truesdell-Jones.

**Rare earths and thorium.**
- [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) has hydroxides, carbonates, fluorides, oxides and hydrated
  phosphates (`LaPO4:10H2O`, rhabdophane-like) for every lanthanide, and
  `Thorianite` (ThO₂).
- [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) has ThO₂ in five crystallinities (`ThO2(cr)`, `ThO2(mcr)`,
  `ThO2(aged)`, `ThO2(fresh)`, `ThO2(coll)`), Th phosphate, sulfate, fluoride
  and Na–Th carbonate, and the NEA-reviewed Th hydrolysis and carbonate
  complexes with SIT coefficients. It has no lanthanides other than Eu and Ho.
- Neither database has monazite, xenotime, bastnäsite or thorite. Add them as
  reactions, as shown in notebook 06.

## What is read

| block | read |
|-------|------|
| `SOLUTION_MASTER_SPECIES` | element and valence names, master species |
| `SOLUTION_SPECIES` | reaction, `log_k`, `delta_h` (with units), `-analytic`, `-gamma a0 b` |
| `PHASES` | dissolution reaction, `log_k`, `delta_h`, `-analytic`; gases are the names ending in `(g)`, with `-T_c`, `-P_c`, `-Omega` |
| `GAS_BINARY_PARAMETERS` | Peng–Robinson $k_{ij}$ per gas pair |
| `EXCHANGE_MASTER_SPECIES`, `EXCHANGE_SPECIES` | [Gaines–Thomas](references.md#gaines1953) exchange reactions, `-gamma` |
| `SURFACE_MASTER_SPECIES`, `SURFACE_SPECIES` | surface complexation reactions ([diffuse double layer](references.md#dzombak1990)) |
| `PITZER` | `-B0 -B1 -B2 -C0 -THETA -LAMDA -ZETA -PSI` with temperature coefficients |
| `LLNL_AQUEOUS_MODEL_PARAMETERS` | temperature table, Debye–Hückel A and B, ḃ, and the CO₂ coefficients; species flags `-llnl_gamma a0` and `-co2_llnl_gamma` |
| `NAMED_EXPRESSIONS` | `log_k`, `delta_h` and `-analytic` of named expressions, added to species through `-add_logk NAME coef` |
| `SIT` | `-epsilon` pairs, stored in `db.sit` (constant values; temperature terms are ignored) |

**Temperature.** With `-analytic`, $\log K(T)$ uses the analytic expression.
Otherwise it uses van 't Hoff with `delta_h`, which is PHREEQC's precedence.

**Gases.** The Peng–Robinson critical constants `-T_c`, `-P_c` and `-Omega`
are read, as is `GAS_BINARY_PARAMETERS`. Gases that have them are non-ideal, as
in PHREEQC; see [Concepts](concepts.md#unknowns-and-equations).

**Ignored.** Molar volumes (`-Vm`) and the pressure dependence of $\log K$ are
ignored, and so are diffusion coefficients, `-dw` and `RATES`. Results at high
pressure therefore differ from PHREEQC.

**Redox in [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) and [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat).** These files write many redox
reactions with dissolved O₂ instead of the electron, for example
`Cl- + 0.5 O2 = ClO-`. O₂(aq) is then included automatically as a redox
carrier. PHREEQC's `SOLUTION` defaults to pe 4, which fixes O₂(aq). Here the
O(0) total is balanced instead, unless you fix `pe` or O₂. Trace oxidized
species (such as ClO⁻ below 1e-24 m) can therefore differ from a default
PHREEQC run.

## Building a system

```python
s = db.system(
    ["Na", "Ca", "Cl", "C(4)"],                 # elements or valence states to include
    phases=["Calcite", "Gypsum", "CO2(g)"],     # candidate phases (or "all")
    exchange=["X"],                             # optional exchanger(s)
    surfaces={"Hfo": {"area": 600, "mass": 0.09}},
    gas_phase={"members": ["CO2(g)", "H2O(g)"], "volume": 1.0},   # or "pressure": 1.0
    solid_solutions={"Ca-Sr carbonate": ["Calcite", "Strontianite"]},
    activity="auto",                            # Pitzer for pitzer.dat, Truesdell-Jones otherwise
    T=298.15,
)
```

**Components.** Each component pulls in every species whose reaction involves
only included components.

**Redox.** A plain element name such as `"Fe"` includes all its valence states.
The electron is added when redox couples are present, so `pe` can be fixed.
A valence state such as `"Fe(2)"` restricts the species to that state.

**Input amounts.** Amounts can be keyed by element or valence name, by species,
by phase, or by exchanger name (`"X"` = total exchange capacity, mol eq):

```python
r = s.equilibrate({"Na": 0.01, "Ca": 0.005, "Cl": 0.02, "Calcite": 1.0, "X": 0.01}, pH=7)
r = s.equilibrate({...}, charge_balance="Cl-")       # adjust Cl- to electroneutrality
```

## Inspecting and changing data

The parsed data are plain Python objects:

```python
db.phases["Calcite"]              # Entry(name, nu, logK, dH, analytic, ...)
db.species["CaHCO3+"].logK
db.pitzer.data["B0"][("Cl-", "Na+")]
```

The `System` built from a database is an ordinary system. Its reactions carry
the database name, so you can add species and reactions, or override
constants, before equilibrating. See [custom-data.md](custom-data.md).

## Differences from PHREEQC to be aware of

**Pitzer.** Individual-ion activity coefficients use [MacInnes](references.md#macinnes1919) scaling, as
PHREEQC does. Mean activities do not depend on the scaling.

**Water.**
- Water activity and the water mass follow PHREEQC's conventions.
- Water is produced or consumed by reactions unless `water_mass` is fixed.
- Water in the reactions is left out of the mass balances when `water_mass` is
  given.

**Surfaces.** Only the [diffuse double layer](references.md#dzombak1990) model is implemented. CD-MUSIC,
explicit diffuse layer composition and the no-electrostatics option are not.

**Solid solutions.** They are ideal. Guggenheim/Lippmann non-ideal solid
solutions are not implemented yet.

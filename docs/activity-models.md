# Activity models

Pass a model to `System(activity=...)`, or per calculation with
`equilibrate(activity=...)`. It can be a name, a dict or an instance.

| model | name | parameters | $G^{ex}$ (`method="gibbs"`) | typical range |
|-------|------|-----------|------|---------------|
| `Ideal()` | `ideal` | none | yes | dilute, I ≲ 1e-3 m |
| `DebyeHuckel(A=None)` | `debye-huckel` | A | yes | I ≲ 1e-3 m |
| `Davies(A=None, b=0.3)` | `davies` | A, b | yes | I ≲ 0.1–0.5 m |
| `ExtendedDebyeHuckel(a0=4)` | `extended-debye-huckel` | one common a0; A, B | yes | I ≲ 0.1 m |
| `TruesdellJones(params=None, neutral_b=0.1)` | `truesdell-jones` | species `a0`, `b` | no | I ≲ 1 m ([phreeqc.dat](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat)) |
| `BDot(bdot=0.041, a0_default=4)` | `bdot` | species `a0`, neutral `b`; `bdot` | no | I ≲ 1 m ([LLNL/EQ3](references.md#delany1991)) |
| `LLNL(table=None, neutral_b=0.1)` | `llnl` | species `a0` and `gamma` flag; A, B, ḃ tables over T | no | I ≲ 1 m, 0–300 °C ([`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat)) |
| `SIT(eps={...})` | `sit` | $\varepsilon_{ij}$ (kg/mol) | yes | I ≲ 3–4 m ([NEA TDB](references.md#grenthe1992)) |
| `Pitzer(params)` | `pitzer` | B0, B1, B2, C0, THETA, LAMDA, ZETA, PSI | yes | to saturation of most salts |

**[Debye–Hückel](references.md#debye1923) constants.** `A` and `B` default to values computed from the
[density](references.md#wagner2002) and [dielectric constant](references.md#bradley1979) of water at the calculation temperature. Pass
numbers to reproduce textbook calculations.

**Water activity.**
- `Ideal`: $\ln a_w = -M_w\sum m$.
- [Truesdell-Jones](references.md#truesdell1974) and [B-dot](references.md#helgeson1969): [PHREEQC's](references.md#parkhurst2013) approximation
  $a_w = 1 - 0.017\sum m$.
- Models with an excess Gibbs energy: the Gibbs–Duhem-consistent value (the
  osmotic coefficient for [Pitzer](references.md#pitzer1973a)).

## Truesdell-Jones (PHREEQC)

**Equations.**
- Ions with an ion size: $\log_{10}\gamma = -Az^2\sqrt I/(1+Ba_0\sqrt I) + bI$.
- Ions without `a0`: [Davies](references.md#davies1938) ($b=0.3$).
- Neutral species: $\log_{10}\gamma = bI$, with $b = 0.1$ unless the species
  sets it.

**Where the parameters live.** Per-species values are `Species.params["a0"]`
and `["b"]`, which come from `-gamma a0 b` in a database. Global overrides can
be passed as `TruesdellJones(params={"Na+": (4.0, 0.075)})`.

## LLNL B-dot (`llnl.dat`)

This is [PHREEQC's](references.md#parkhurst2013) `LLNL_AQUEOUS_MODEL_PARAMETERS` model, selected
automatically for [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat).

**Equations.**
- Species with `-llnl_gamma a0`:
  $\log_{10}\gamma = -Az^2\sqrt I/(1+Ba_0\sqrt I) + \dot b I$.
- Neutral species marked `-co2_llnl_gamma` (CO₂, O₂, …):
  $\ln\gamma = (c_0 + c_1T + c_2/T)\,I - (c_3 + c_4T)\,I/(I+1)$.
- Other ions: [Davies](references.md#davies1938). Other neutral species: $\log_{10}\gamma = 0.1\,I$.
- Water: $a_w = 1 - 0.017\sum m$.

**Temperature.** $A$, $B$ and $\dot b$ are interpolated linearly in the
table of 8 temperatures from 0.01 to 300 °C. Outside that range `LLNL` raises
`ValueError`, as PHREEQC does.

**Validation.** Against PHREEQC, log activities agree within 1.3e-4 at 0.1 m
and 1 m NaCl with Th and carbonate, at 25, 60 and 150 °C.

## SIT

$\log_{10}\gamma_i = -z_i^2 \dfrac{A\sqrt I}{1+1.5\sqrt I} + \sum_j \varepsilon_{ij} m_j$,
with symmetric $\varepsilon$ values keyed by ion pair:
`SIT(eps={("Na+", "Cl-"): 0.03, ("Ca+2", "Cl-"): 0.14})`. Missing pairs are
zero.

**From [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat).** The database's `SIT` block (606 `-epsilon` pairs from
[ThermoChimie](references.md#giffaut2014)) is read into `db.sit`, and `db.default_activity()` returns
`SIT(eps=db.sit)`. Water activity is the Gibbs–Duhem value, which matches
PHREEQC's osmotic-coefficient expression. Against PHREEQC, log activities
agree within 2e-4 at 0.1, 1 and 3 m NaCl with Th and carbonate.

## Pitzer

This is the [Harvie–Møller–Weare](references.md#harvie1984) form as implemented in PHREEQC's
[`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat):

- **Unsymmetric mixing:** [E-θ terms](references.md#pitzer1975), with PHREEQC's integral for $J(x)$.
- **Constants:** $\alpha_1 = 1.4$ for 2–2 and higher pairs, 2.0 otherwise;
  $\alpha_2 = 12$.
- **Temperature:** each parameter varies with $T$ through the six-coefficient
  form above.
- **[MacInnes](references.md#macinnes1919) scaling:** individual-ion activities are scaled so that
  $\gamma_{K^+}=\gamma_{Cl^-}$ in KCl solutions of the same ionic strength
  (`macinnes=True`, PHREEQC's default). This matters only for single-ion
  quantities such as pH.

```python
from discopt.speciation import Pitzer, PitzerParameters, PhreeqcDatabase
act = PhreeqcDatabase.load("pitzer.dat").pitzer      # PitzerParameters from the database
act.add("B0", ["Na+", "Cl-"], 0.0770)                # modify or extend
s.equilibrate(..., activity=Pitzer(act))
```

**Validation.** Against PHREEQC, log activities agree within 5e-5 at 1 and
4 m NaCl with Ca and SO4, at 25 and 60 °C. See [validation.md](validation.md).

## Choosing

- **Fresh water or soils, with many minerals and redox:** [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) or
  [`wateq4f_PWN.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat) with Truesdell-Jones.
- **Brines or evaporites:** [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat).
- **Radionuclides, thorium and actinides (NEA/ThermoChimie data):** [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat)
  with SIT.
- **Many elements, rare earths, or temperatures to 300 °C:** [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) with
  the LLNL B-dot model.
- **Global optimality certificates, or `method="gibbs"`:** a model with
  $G^{ex}$ (Davies, SIT, Pitzer).

# Concepts

## Unknowns and equations

For a system with aqueous species $i$, phases $p$ and reactions $r$:

| condition | equation |
|-----------|----------|
| mass action | $\sum_i \nu_{ri}\log_{10} a_i = \log_{10} K_r(T,P)$ |
| conservation | $A\,n = b$, where the rows of $A$ span the left null space of the stoichiometric matrix |
| complementarity | $N_p \ge 0,\quad \mathrm{SI}_p \le 0,\quad N_p\,\mathrm{SI}_p = 0$ |
| activity | $a_i = \gamma_i(m, T)\,m_i$; water activity from the same model |

**Conservation.** The conserved quantities are not typed in by hand.
Conservation laws are exactly the combinations of amounts that no reaction can
change. They are computed from the reaction network, so element, charge and
site balances follow automatically. Input amounts may be given for any
species, e.g. `{"Calcite": 1.0}` or `{"Ca": 0.01}`, because only the conserved
totals matter.

**Phases and solid solutions.** Pure solids, ideal solid solutions and gas
phases at fixed pressure use the complementarity rule for the phase as a whole.
A solid solution is present iff $\sum_j 10^{\mathrm{SI}_j} \ge 1$. A gas phase
is present iff the sum of partial pressures reaches the total pressure.

**Fixed-volume gases.** A fixed-volume gas phase always exists, with
$n_j = P_j V / RT$ for an ideal gas.

**Non-ideal gases.** A gas's activity is its fugacity
$f_j = \varphi_j x_j P$, and its $\mathrm{SI}$ is $\log_{10} f_j$. Gases with
critical constants ($T_c$, $P_c$, $\omega$) follow the
[Peng–Robinson](references.md#peng1976) equation of state, with PHREEQC's
constants and mixing rule. The database gases get them from `-T_c`, `-P_c`
and `-Omega`; see [custom data](custom-data.md#gases-with-critical-constants)
for your own.
- **Fixed pressure:** the phase is present iff
  $\sum_j f_j/\varphi_j(x) = P$ can be met. The composition, the molar volume (the
  vapor root) and the undersaturation are solved with the rest of the state.
- **Fixed volume:** the amounts solve $f_j(n, V) = 10^{\mathrm{SI}_j}$, and the
  pressure follows.
- **Fixed partial pressure** (`fixed={"CO2(g)": 1.7}`): the value is $\log_{10} P$,
  and the fugacity is $\varphi(T, P)\,P$ of the pure gas, as in PHREEQC's
  `EQUILIBRIUM_PHASES`.
- `System(gas_eos="ideal")`, or `eos="ideal"` on a gas phase, turns this off.
  `Result.fugacity_coefficient` and `Result.gas_phases` report $\varphi$,
  pressure, volume and composition.

**Open systems.** A fixed activity or partial pressure (`fixed={"CO2(g)": -3.5}`,
`pH=7`, `pe=4`) removes that species from the conservation laws.

## Pressure

Above 1 atm every $\log K$ is corrected with the volume change of its reaction,
as in PHREEQC ([Appelo, Parkhurst and Post 2014](references.md#appelo2014)):

$$\log K(T, P) = \log K(T) - \frac{\Delta V_r (P - 1\,\mathrm{atm})}{RT\ln 10},
\qquad \Delta V_r = \sum_i \nu_{ri} V_i(T, P, I).$$

- **Aqueous species** with `-Vm` have a SUPCRT-type volume plus, for ions, a
  Debye–Hückel term $\tfrac12 z^2 A_v\sqrt I/(1 + bB\sqrt I)$ and a term in $I$.
  $\log K$ therefore depends on the ionic strength above 1 atm, and is solved
  with it.
- **Water** has $18.016/\rho_w(T, P)$; **minerals** their `-Vm`; **gases**
  none (their non-ideality is in the fugacity coefficient).
- **Water properties** follow the pressure too: density and compressibility
  (PHREEQC's fit to Wagner and Pruss 2002), and the dielectric constant, so the
  Debye–Hückel $A$ and $B$ change. The [Pitzer](references.md#pitzer1973a) model
  lowers its $b = 1.2$ for ions with $|z| = 1$ and $2$, as PHREEQC does.
- **llnl.dat** has no molar volumes, so its $\log K$ do not depend on pressure,
  again as in PHREEQC.

**Which pressure.** `equilibrate(P=...)` sets it. Without `P`, it is the pressure
of a fixed-pressure gas phase, else of a fixed-volume gas phase (found together
with the equilibrium), else the largest fixed partial pressure of a
Peng–Robinson gas, else 1 atm. It is never below the vapor pressure of the
solution $p_{sat}(T)\,a_w$, which matters above 100 °C. `Result.pressure`
reports it.

## Why complementarity

**The wrong formulations.** A common approach treats every candidate solid as
present, or minimizes a sum of squared saturation indices. Both give answers
that change when:

- a reaction is multiplied by two;
- an inert polymorph is added;
- a supersaturated phase is listed that should have precipitated.

**Why complementarity is right.** The complementarity conditions are the
Karush–Kuhn–Tucker conditions of Gibbs energy minimization. Any state that
satisfies them is a true equilibrium. It does not depend on how the reactions
were written. The evidence is in [`examples/02-validation.ipynb`](https://github.com/jkitchin/discopt-speciation/blob/main/examples/02-validation.ipynb).

**Metastable polymorphs.** Metastable phases such as aragonite next to calcite
are correctly absent, because they would have $\mathrm{SI} < 0$ once the
stable phase fixes the solution composition. To compute a *metastable*
equilibrium, drop the stable phase with
`equilibrate(..., suppress=["Calcite"])`.

## Methods

`System.equilibrate(method=...)` accepts three methods:

- **`"complementarity"`** (default)
  1. Speciation with all phases absent.
  2. A smoothed [Fischer–Burmeister](references.md#fischer1992) homotopy on the complementarity conditions.
  3. An active-set polish that removes phases with negative amounts and adds
     supersaturated ones until the conditions hold exactly.
- **`"gibbs"`**
  - Minimizes the Gibbs energy directly.
  - Needs an activity model with an excess Gibbs energy (ideal, [Davies](references.md#davies1938), [SIT](references.md#grenthe1992),
    [Pitzer](references.md#pitzer1973a)). [Truesdell-Jones](references.md#truesdell1974) and [B-dot](references.md#helgeson1969) have none.
  - Its local minimum can be metastable when a non-ideal solid solution makes
    the problem non-convex. Global optimality is checked separately, for any
    method, by the stability certificate below.
- **`"enumerate"`**
  - Solves every phase assemblage and keeps the ones that satisfy all
    conditions.
  - Exponential in the number of phases (limit 12), but proves uniqueness.

## Global stability

Complementarity is necessary for equilibrium. It is sufficient only when the
Gibbs energy is convex. A non-ideal solid solution breaks convexity. Inside
its miscibility gap a single homogeneous solid satisfies every local
condition, but two coexisting solids have lower energy.

`equilibrate` then runs a tangent-plane stability test on every phase. If a
phase is unstable, it adds the phase, or a second copy of it, at the unstable
composition and re-solves. It attaches a `Result.certificate`: an
interval-arithmetic lower bound on each phase's tangent plane distance. The
bound proves that no phase, at any composition, has lower Gibbs energy than
the computed state implies. See [stability](stability.md).

## Sensitivities

At a converged state the equations of its phase assemblage (present phases
saturated, absent ones at zero amount) form a square system $F(x,\theta)=0$.
By the implicit function theorem

$$\frac{dx}{d\theta} = -F_x^{-1}F_\theta,\qquad
\frac{dq}{d\theta} = q_x\frac{dx}{d\theta} + q_\theta .$$

- $F_x$ is exact (JAX forward mode). No extra solves are needed.
- $F_\theta$ comes from rebuilding the network at $\theta\pm h$ at fixed $x$.
  The residuals are affine in `logK`, `dH`, fixed activities and input
  totals, so this is exact up to rounding.
- A phase with $N_p = 0$ and $\mathrm{SI}_p = 0$ at once sits on a kink.
  Its derivatives are one-sided and are reported as such.

See `Result.sensitivity` and [fitting](fitting.md#sensitivities).

## Units and conventions

| quantity | unit / convention |
|----------|-------------------|
| amounts | mol for 1 kg of initial water (i.e. molalities) unless `water_mass` is given |
| temperature | K |
| pressure | atm |
| `dH` | J/mol |
| `dCp` | J/mol/K |
| `dV` | cm³/mol |
| `logK` | base 10, at 25 °C and 1 atm, for the reaction as written (products over reactants) |
| pressure | atm; `Result.pressure` is the value used |
| `SI` | $\log_{10}(Q/K)$ per formula unit as written; for a gas, $\log_{10}$ of its fugacity (atm) |
| `pe` | $-\log_{10} a_{e^-}$; $E_h = pe \cdot RT\ln 10/F$ |

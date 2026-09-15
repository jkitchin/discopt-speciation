# Global stability and miscibility gaps

The local equilibrium conditions are mass action, conservation and
complementarity. They are the first-order optimality conditions of Gibbs
energy minimization. With ideal solid solutions and a convex aqueous model,
any state satisfying them is the global minimum.

A **non-ideal solid solution** makes the Gibbs energy non-convex. Inside its
*miscibility gap* one homogeneous solid satisfies every local condition, but
two solids of different composition have lower energy. [PHREEQC](references.md#parkhurst2013) handles binary
gaps with a [dedicated calculation](references.md#glynn1991). `discopt.speciation` instead proves global
stability for any number of members, and repairs the state when the proof fails.

## Declaring non-ideal solid solutions

```python
# PHREEQC -Gugg_nondim: G_ex/RT = x1 x2 [a0 + a1 (x2 - x1) + a2 (x2 - x1)^2]
s.solid_solution("Ca(x)Sr(1-x)CO3", ["Aragonite", "Strontianite"], guggenheim=[3.43, -1.82])

# regular (pairwise Margules) mixing of any number of members: G_ex/RT = sum W_ij x_i x_j
s.solid_solution("(A,B,C)L", ["AL", "BL", "CL"], margules={("AL", "BL"): 3.5, ("AL", "CL"): 0.5})
```

YAML specs take `guggenheim: [a0, a1]` or `margules: [[AL, BL, 3.5], ...]` on
a `solid_solutions` entry. `PhreeqcDatabase.system(solid_solutions=...)`
accepts `{"name": {"members": [...], "guggenheim": [a0, a1]}}`.

The member activities are $x_j\lambda_j$ with
$\ln\lambda_j = \partial(nG_{ex}/RT)/\partial n_j$. The solver carries each
non-ideal phase's composition as unknowns and imposes
$\log_{10}(x_j\lambda_j) = \mathrm{SI}_j + s$ and $\sum_j x_j = 1$, where $s$
is the undersaturation in the complementarity condition.

## The certificate

Let the computed state fix the log10 saturation index $\mathrm{SI}_j$ of every
phase member. The **tangent plane distance** of a phase at composition $x$ is

$$D(x) = \sum_j x_j\left(\ln x_j - \ln 10\,\mathrm{SI}_j\right) + \frac{G_{ex}(x)}{RT}.$$

It is the Gibbs energy of forming one mole of that phase from the solution,
measured from the tangent plane of the component chemical potentials. Suppose
$D \ge 0$ for every phase and every $x$. Then no other state that satisfies
conservation has lower Gibbs energy, and the state is the global minimum
([Baker, Pierce and Luks 1982](references.md#baker1982); [Michelsen 1982](references.md#michelsen1982)).

`Result.certificate` bounds $\min_x D$ for each phase:

| phase | bound |
|-------|-------|
| pure solid, ideal solid solution, ideal gas phase | exact: $\min D = \ln 10\; s$ |
| non-ideal solid solution | interval branch and bound over the composition simplex |

The branch and bound computes rigorous lower bounds on boxes of the simplex.
It works in outward-rounded [interval arithmetic](references.md#moore2009) and uses the better of the
natural and mean-value enclosures on each box. Near-zero mole fractions,
where $\ln x$ is unbounded, are handled with a proven truncation margin. A
box is discarded once its bound proves $D \ge -\mathrm{tol}/10$.

`certificate.status` takes one of three values:

- `"certified"`: every lower bound is $\ge -\mathrm{tol}$ (default $10^{-6}$).
- `"unstable"`: some composition has $D < -\mathrm{tol}$.
- `"inconclusive"`: the box budget ran out, or the solve did not converge.

```python
r = s.equilibrate({"A+": 1.7, "B+": 0.3, "L-": 2.0})
print(r.certificate.summary())
# stability: certified (tol 1e-06; 1 phase-splitting rounds)
#   SS    present  min D in [-4.5e-07, -1.7e-16]  stable (interval branch and bound)
#   assumes: one aqueous phase ... whose Gibbs energy is convex in its amounts
#   assumes: the computed solution activities (solved to residual 1e-08) are exact
r.solid_solutions   # {"SS": {..., "mole_fraction": {"AL": 0.0707, ...}}, "SS (2)": {...}}
```

### What the certificate assumes

- **The aqueous phase is convex.** Exchangers and surfaces count as part of it.
  This is exact for ideal activities. Davies, SIT, Pitzer and the other
  models are not guaranteed convex at high ionic strength. The certificate
  lists this as an assumption rather than proving it.
- **The computed activities are exact.** The solve reaches residuals of about
  $10^{-8}$, which is small compared with the tolerance of $10^{-6}$ on $D$.

### Cost and limits

Pure and ideal phases are certified exactly and cost nothing. For a non-ideal
phase, branch and bound works on a box of $n-1$ mole fractions, so the cost
grows quickly with the number of members $n$. Typical cost for random
Margules parameters $W_{ij} \in [0, 4]$ (default `max_boxes=50_000`):

| members | boxes | time | result |
|---|---|---|---|
| 2 (binary) | ~10² | ~0.01 s | certified |
| 3 | ~10³ | 0.1–0.2 s | certified |
| 4 | ~1.4·10⁴ | ~1.7 s | certified |
| 5–6 | > 2·10⁵ | > 20 s | usually `inconclusive` |

With five or more members the lower bounds near the simplex faces are too
loose to close the gap in a reasonable budget. The result is then reported as
`inconclusive` with a note that the box budget ran out. It is never reported
as `certified`. Raise `max_boxes` in `stability_certificate(result, max_boxes=...)`
to try harder, or pass `certify=False` to skip the test.

## Phase splitting

`equilibrate(certify=None)` certifies by default whenever the system has a
non-ideal solid solution. `certify=True` forces the test, and `certify=False`
skips it. When a phase is unstable:

1. an absent phase is added at its minimizer of $D$;
2. a present phase gets a second copy, `"Name (2)"`, at the minimizer. The
   original moves to the local minimum of $D$ on the other side, and the
   amount is split by the lever rule;
3. the assemblage is re-solved and certified again, for up to 6 rounds.

`stability_certificate(result)` certifies an existing result without
re-solving. For example, `certify=False` shows the metastable state the local
conditions accept.

## Validation

[PHREEQC example 10](references.md#parkhurst2013-examples) (aragonite–strontianite, $a_0=3.43$, $a_1=-1.82$) reports
a miscibility gap at $x_{\mathrm{Sr}} = 0.0048032$ and $0.857863$. For every
bulk composition in the gap, the phase split reproduces both values to six
digits. [`tests/test_certify.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_certify.py) also covers:

- symmetric regular solutions, where the binodal is known in closed form;
- starts at the spinodal point $x = 0.5$ and between spinodal and binodal;
- $W$ at and just above the critical value 2, where $D$ is flat to fourth order;
- members at $10^{-30}$ mole fraction;
- a ternary gap;
- an absent phase that is undersaturated at its stationary composition but
  supersaturated near a pure end member;
- the [Davies](references.md#davies1938) model, and the `gibbs` and `enumerate` methods.

## References

- Baker, L. E., Pierce, A. C., and Luks, K. D. (1982). Gibbs energy analysis
  of phase equilibria. *SPE Journal* 22(5), 731–742.
  [doi:10.2118/9806-PA](https://doi.org/10.2118/9806-PA)
- Michelsen, M. L. (1982). The isothermal flash problem. Part I. Stability.
  *Fluid Phase Equilibria* 9(1), 1–19.
  [doi:10.1016/0378-3812(82)85001-2](https://doi.org/10.1016/0378-3812(82)85001-2)
- Glynn, P. D. (1991). MBSSAS: a code for the computation of Margules
  parameters and equilibrium relations in binary solid-solution aqueous-solution
  systems. *Computers & Geosciences* 17(7), 907–966.
  [doi:10.1016/0098-3004(91)90090-Z](https://doi.org/10.1016/0098-3004(91)90090-Z)
- Moore, R. E., Kearfott, R. B., and Cloud, M. J. (2009). *Introduction to
  Interval Analysis*. SIAM. [doi:10.1137/1.9780898717716](https://doi.org/10.1137/1.9780898717716)

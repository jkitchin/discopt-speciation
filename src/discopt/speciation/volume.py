r"""Molar volumes and the pressure dependence of ``log K``, with PHREEQC's equations.

Above 1 atm PHREEQC corrects every equilibrium constant with the volume change of
its reaction (``k_calc`` in ``prep.cpp``):

.. math:: \log K(T, P) = \log K(T) - \frac{\Delta V_r\,(P - P_0)}{RT\ln 10},
          \qquad \Delta V_r = \sum_i \nu_{ri} V_i(T, P, I),

for :math:`P > P_0 = 1` atm only. The molar volumes are (``calc_vm``; Appelo,
Parkhurst and Post 2014):

* aqueous species with ``-Vm a1 a2 a3 a4 W b i1 i2 i3 i4``: a SUPCRT term
  :math:`a_1 + a_2/(2600+p) + (a_3 + a_4/(2600+p))/(T-228) - W\,Q_{Born}`
  (``p`` in bar, the ``a`` and ``W`` scaled from cal to cm³ as PHREEQC does),
  plus, for ions, :math:`\tfrac12 z^2 A_v \sqrt I/(1 + b\,B\sqrt I)` and
  :math:`(i_1 + i_2/(T-228) + i_3 (T-228))\,I^{i_4}`;
* water: :math:`18.016/\rho_w(T, P)`;
* minerals: their ``-Vm`` (cm³/mol); gases and species without data: 0.

The ionic-strength terms make ``log K`` depend on ``I`` above 1 atm; the solver
keeps them as expressions in ``I``. ``A_v``, ``B``, :math:`Q_{Born}` and
:math:`\rho_w` come from :func:`discopt.speciation.water.water_state`.
"""

from __future__ import annotations

import math

from discopt.speciation.water import water_state

LN10 = math.log(10.0)
R_KJ = 0.0083147  # kJ/(mol K), PHREEQC's R_KJ_DEG_MOL
P0_PA = 1.01325e5
#: PHREEQC's scale factors for a1..a4 (cal -> cm3) and W
_SCALE = (41.84004e-1, 41.84004e2, 41.84004, 41.84004e4, 1e5)


def aqueous_vm_params(values) -> tuple:
    """PHREEQC ``-Vm a1 a2 a3 a4 W b i1 i2 i3 i4`` (as written in the database) -> scaled 10-tuple."""
    v = [float(t) for t in values][:10]
    if not v:
        raise ValueError("-Vm needs at least one number")
    out = v + [0.0] * (10 - len(v))
    if len(v) < 10:
        out[9] = 1.0  # i4 defaults to 1
    for k, s in enumerate(_SCALE):
        out[k] *= s
    return tuple(out)


def pressure_factor(T: float, P: float) -> float:
    """``c`` in ``log K -= c * dV`` (dV in cm3/mol): PHREEQC applies it only above 1 atm (``P`` in atm)."""
    dp = P * P0_PA - P0_PA
    return dp * 1e-9 / (LN10 * R_KJ * T) if dp > 0 else 0.0


def species_volume(sp, ws):
    """``(V0, ion_terms)`` of a species: the I-independent volume (cm3/mol) and the ionic-strength terms.

    ``ion_terms`` is ``[(coefficient, b), ...]`` for :math:`c\\sqrt I/(1+bB\\sqrt I)` (``b`` None: no
    denominator) and ``[(coefficient, exponent)]`` for :math:`c I^e`, as ``("dh", c, b)`` and ``("pow", c, e)``.
    ``ws`` is a :class:`~discopt.speciation.water.WaterState`. ``None`` when the species has no volume.
    """
    if sp.phase == "solvent":
        return 18.016 / ws.rho, []
    vm = sp.params.get("Vm")
    if vm is None:
        return None
    if sp.phase != "aqueous":
        return float(vm if not isinstance(vm, (list, tuple)) else vm[0]), []
    a1, a2, a3, a4, W, b, i1, i2, i3, i4 = aqueous_vm_params(vm if isinstance(vm, (list, tuple)) else [vm])
    pb_s, TK_s = 2600.0 + ws.P * 1.01325, ws.T - 273.15 + 45.15
    v0 = a1 + a2 / pb_s + (a3 + a4 / pb_s) / TK_s - W * ws.QBrn
    terms = []
    if sp.charge:
        z2 = sp.charge * sp.charge
        terms.append(("dh", z2 * 0.5 * ws.Av, None if b < 1e-5 else b * ws.B))
        if i1 or i2 or i3:
            terms.append(("pow", i1 + i2 / TK_s + i3 * TK_s, i4))
    return v0, terms


def uses_volumes(species) -> bool:
    """Molar volumes apply when some species carries ``Vm`` (then water's volume counts too)."""
    return any(s.params.get("Vm") is not None for s in species)


def reaction_volume(stoich: dict, ws):
    """``(dV0, terms)`` of a reaction ``{species: nu}``: I-independent ``dV`` (cm3/mol) and ``nu``-weighted terms."""
    dv0, terms = 0.0, []
    for sp, nu in stoich.items():
        v = species_volume(sp, ws)
        if v is None:
            continue
        dv0 += nu * v[0]
        terms += [(kind, nu * c, p) for kind, c, p in v[1]]
    return dv0, terms


def ionic_terms(o, terms, I):
    """Sum of the ionic-strength volume terms at ``I`` (expression or number)."""
    if not terms:
        return 0.0
    s = o.sqrt(I)
    out = []
    for kind, c, p in terms:
        if kind == "dh":
            out.append(c * s if p is None else c * s / (1.0 + p * s))
        elif p == 1.0:
            out.append(c * I)
        else:
            out.append(c * o.exp(p * o.log(I)))
    return o.sum(out)


def reaction_logK(rxn, T: float, P: float, I: float, species, aw: float = 1.0) -> float:
    """Numeric ``log K`` of ``rxn`` at ``T`` (K), ``P`` (atm) and ionic strength ``I``, volumes included."""
    from discopt.speciation.activity import NUM

    lk = rxn.logK_at(T, P)
    c = pressure_factor(T, P)
    if not c or not uses_volumes(species):
        return lk
    ws = water_state(T, P, aw)
    dv0, terms = reaction_volume(rxn.stoich, ws)
    return lk - c * (dv0 + (float(ionic_terms(NUM, terms, I)) if I > 0 else 0.0))

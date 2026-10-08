"""Properties of liquid water used by the activity and electrostatic models.

The correlations are those used by PHREEQC 3, so Debye-Hueckel parameters
agree with it to about 1e-5:

* density: Wagner and Pruss (2002), eq. 2.6, on the saturation curve, plus
  PHREEQC's pressure polynomial (``calc_rho_0``), with the compressibility;
* relative dielectric constant: Bradley and Pitzer (1979);
* Debye-Hueckel ``A`` (log10 units) and ``B`` (1/Angstrom) from those, and the
  volume terms ``A_v`` and ``Q_Born`` of ``calc_dielectrics`` (see
  :mod:`discopt.speciation.volume`).

Temperatures are in K. Pressures are in bar, except in :func:`water_state` and
:func:`saturation_pressure` (atm, as PHREEQC).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

T_REF = 298.15
P_REF = 1.01325  # bar
M_W = 0.01801528  # kg/mol
R_GAS = 8.314462618  # J/(mol K)
FARADAY = 96485.33212  # C/mol
AVOGADRO = 6.02214076e23
EPS0 = 8.8541878128e-12  # F/m
LN10 = math.log(10.0)


def saturation_pressure(T: float, aw: float = 1.0) -> float:
    """Vapor pressure of water (atm) at ``T`` (K), times the water activity ``aw`` (PHREEQC's ``p_sat``)."""
    return math.exp(11.6702 - 3816.44 / (T - 46.13)) * min(aw, 1.0)


def _rho_kappa(T: float, P_atm: float, aw: float = 1.0):
    """PHREEQC ``calc_rho_0``: density (kg/L) and compressibility (1/atm) at ``T`` (K) and ``P_atm``.

    The pressure is raised to the saturation pressure (times ``aw``) when below it.
    Above 350 C the correlations are evaluated at 350 C, as in PHREEQC.
    """
    T = min(T, 623.15)
    th = 1.0 - T / 647.096
    b = (1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352, -6.7469445e5)
    ex = (1 / 3, 2 / 3, 5 / 3, 16 / 3, 43 / 3, 110 / 3)
    rho_sat = 322.0 * (1.0 + sum(bi * th**e for bi, e in zip(b, ex)))
    tc = T - 273.15
    p0 = 5.1880000e-02 + tc * (-4.1885519e-04 + tc * (6.6780748e-06 + tc * (-3.6648699e-08 + tc * 8.3501912e-11)))
    p1 = -6.0251348e-06 + tc * (3.6696407e-07 + tc * (-9.2056269e-09 + tc * (6.7024182e-11 + tc * -1.5947241e-13)))
    p2 = -2.2983596e-09 + tc * (-4.0133819e-10 + tc * (1.2619821e-11 + tc * (-9.8952363e-14 + tc * 2.3363281e-16)))
    p3 = 7.0517647e-11 + tc * (6.8566831e-12 + tc * (-2.2829750e-13 + tc * (1.8113313e-15 + tc * -4.2475324e-18)))
    psat = saturation_pressure(T, aw)
    pa = max(P_atm, psat) - (psat - 1e-6)
    rho = max(rho_sat + pa * (p0 + pa * (p1 + pa * (p2 + math.sqrt(pa) * p3))), 0.01)
    kappa = (p0 + pa * (2 * p1 + pa * (3 * p2 + math.sqrt(pa) * 3.5 * p3))) / rho
    return rho / 1e3, kappa


def density(T: float, P: float = P_REF) -> float:
    """Density of liquid water (kg/L) at ``T`` (K) and ``P`` (bar), as PHREEQC (Wagner and Pruss 2002 + pressure)."""
    if T >= 647.096:
        raise ValueError(f"T={T} K is above the critical temperature of water")
    return _rho_kappa(T, P / 1.01325)[0]


def dielectric_constant(T: float, P: float = P_REF) -> float:
    """Relative permittivity of water (Bradley and Pitzer 1979); ``P`` in bar."""
    u = (3.4279e2, -5.0866e-3, 9.469e-7, -2.0525, 3.1159e3, -1.8289e2, -8.0325e3, 4.2142e6, 2.1417)
    d1000 = u[0] * math.exp(T * (u[1] + T * u[2]))
    c = u[3] + u[4] / (u[5] + T)
    b = u[6] + u[7] / T + u[8] * T
    return d1000 + c * math.log((b + P) / (b + 1000.0))


def _e2_DkT(T: float, P: float) -> float:
    # q_e^2 / (4 pi eps0 eps_r k_B T) in cgs units (cm)
    return 1.671008e-3 / (dielectric_constant(T, P) * T)


def debye_huckel_A(T: float = T_REF, P: float = P_REF) -> float:
    """Debye-Hueckel ``A`` for ``log10 gamma`` in (kg/mol)^0.5 (0.5100 at 25 C)."""
    e2 = _e2_DkT(T, P)
    kappa = math.sqrt(8.0 * math.pi * 6.02252e23 * e2 * density(T, P) / 1e3)
    return kappa * e2 / (2.0 * LN10)


def debye_huckel_B(T: float = T_REF, P: float = P_REF) -> float:
    """Debye-Hueckel ``B`` in 1/(Angstrom (mol/kg)^0.5) (0.3285 at 25 C)."""
    e2 = _e2_DkT(T, P)
    return math.sqrt(8.0 * math.pi * 6.02252e23 * e2 * density(T, P) / 1e3) / 1e8


@dataclass(frozen=True)
class WaterState:
    """Water properties PHREEQC uses for molar volumes (``calc_rho_0`` and ``calc_dielectrics``).

    ``P`` is the pressure used (atm, at least the saturation pressure times ``aw``);
    ``rho`` in g/cm3, ``kappa`` in 1/atm, ``B`` in 1/(Angstrom (mol/kg)^0.5), ``Av``
    (the Debye-Hueckel volume slope) in (cm3/mol)(mol/kg)^-0.5, ``QBrn`` the Born
    volume function (times 41.84 cal -> cm3).
    """

    T: float
    P: float
    rho: float
    kappa: float
    eps: float
    B: float
    Av: float
    QBrn: float


@lru_cache(maxsize=4096)
def water_state(T: float, P_atm: float, aw: float = 1.0) -> WaterState:
    """PHREEQC's water properties at ``T`` (K) and ``P_atm`` (atm); see :class:`WaterState`."""
    Pa = max(P_atm, saturation_pressure(T, aw))
    rho, kappa = _rho_kappa(T, P_atm, aw)
    Tt = min(T, 623.15)
    u = (3.4279e2, -5.0866e-3, 9.469e-7, -2.0525, 3.1159e3, -1.8289e2, -8.0325e3, 4.2142e6, 2.1417)
    d1000 = u[0] * math.exp(Tt * (u[1] + Tt * u[2]))
    c = u[3] + u[4] / (u[5] + Tt)
    b = u[6] + u[7] / Tt + u[8] * Tt
    pb = Pa * 1.01325
    eps = d1000 + c * math.log((b + pb) / (b + 1e3))
    e2 = 1.671008e-3 / (eps * Tt)
    B_cm = math.sqrt(8 * 3.14159265358979 * 6.02252e23 * e2 * rho / 1e3)
    Av = B_cm * e2 * 0.0820597 * 1e3 * Tt * (c / (b + pb) * 1.01325 / eps - kappa / 3.0)
    QBrn = c / (b + pb) / eps / eps * 41.84004
    return WaterState(float(T), float(Pa), rho, kappa, eps, B_cm / 1e8, Av, QBrn)


def A_phi(T: float = T_REF, P: float = P_REF) -> float:
    """Osmotic Debye-Hueckel coefficient ``A_phi = ln(10) A / 3`` (natural log units)."""
    return LN10 * debye_huckel_A(T, P) / 3.0


def gouy_chapman_coefficient(T: float = T_REF, P: float = P_REF) -> float:
    """``sqrt(8000 eps_r eps0 R T)`` so that ``sigma = coef sqrt(I) sinh(F psi / 2RT)`` in C/m^2."""
    return math.sqrt(8000.0 * dielectric_constant(T, P) * EPS0 * R_GAS * T)

"""Properties of liquid water used by the activity and electrostatic models.

The correlations are those used by PHREEQC 3, so Debye-Hueckel parameters
agree with it to about 1e-5:

* density along the saturation curve: Wagner and Pruss (2002), eq. 2.6;
* relative dielectric constant: Bradley and Pitzer (1979);
* Debye-Hueckel ``A`` (log10 units) and ``B`` (1/Angstrom) from those.

Temperatures are in K and pressures in bar.
"""

from __future__ import annotations

import math

T_REF = 298.15
P_REF = 1.01325  # bar
M_W = 0.01801528  # kg/mol
R_GAS = 8.314462618  # J/(mol K)
FARADAY = 96485.33212  # C/mol
AVOGADRO = 6.02214076e23
EPS0 = 8.8541878128e-12  # F/m
LN10 = math.log(10.0)


def density(T: float) -> float:
    """Density of liquid water (kg/L) on the saturation curve (Wagner and Pruss 2002)."""
    th = 1.0 - T / 647.096
    if th <= 0:
        raise ValueError(f"T={T} K is above the critical temperature of water")
    b = (1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352, -6.7469445e5)
    ex = (1 / 3, 2 / 3, 5 / 3, 16 / 3, 43 / 3, 110 / 3)
    return 0.322 * (1.0 + sum(bi * th**e for bi, e in zip(b, ex)))


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
    kappa = math.sqrt(8.0 * math.pi * 6.02252e23 * e2 * density(T) / 1e3)
    return kappa * e2 / (2.0 * LN10)


def debye_huckel_B(T: float = T_REF, P: float = P_REF) -> float:
    """Debye-Hueckel ``B`` in 1/(Angstrom (mol/kg)^0.5) (0.3285 at 25 C)."""
    e2 = _e2_DkT(T, P)
    return math.sqrt(8.0 * math.pi * 6.02252e23 * e2 * density(T) / 1e3) / 1e8


def A_phi(T: float = T_REF, P: float = P_REF) -> float:
    """Osmotic Debye-Hueckel coefficient ``A_phi = ln(10) A / 3`` (natural log units)."""
    return LN10 * debye_huckel_A(T, P) / 3.0


def gouy_chapman_coefficient(T: float = T_REF, P: float = P_REF) -> float:
    """``sqrt(8000 eps_r eps0 R T)`` so that ``sigma = coef sqrt(I) sinh(F psi / 2RT)`` in C/m^2."""
    return math.sqrt(8000.0 * dielectric_constant(T, P) * EPS0 * R_GAS * T)

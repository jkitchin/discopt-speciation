import math

import numpy as np
import pytest

from discopt.speciation import System, fit, plot


def gypsum_system(logK=-4.58, pair=2.30):
    s = System("gypsum")
    Ca, SO4, CaSO4, Na = s.aqueous("Ca+2"), s.aqueous("SO4-2"), s.aqueous("CaSO4"), s.aqueous("Na+")
    s.reaction(Ca + SO4 >> CaSO4, logK=pair, name="pair")
    s.reaction(s.solid("Gypsum") >> Ca + SO4, logK=logK, name="gypsum")
    return s


DISSOLVED_CA = fit.total("Ca+2", "CaSO4")


def synthetic(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = gypsum_system()
    obs = []
    for na2so4 in (0.0, 0.01, 0.03, 0.1):
        cond = {"amounts": {"Gypsum": 1.0, "Na+": 2 * na2so4, "SO4-2": na2so4}}
        r = truth.equilibrate(cond["amounts"])
        y = DISSOLVED_CA(r)
        obs.append(fit.Observation(cond, DISSOLVED_CA, y * (1 + noise * rng.normal()), sigma=max(noise, 1e-3) * y))
    return obs


def test_fit_recovers_parameters():
    s = gypsum_system(logK=-4.3, pair=2.0)
    params = [fit.logK(s, "gypsum", lower=-6, upper=-3), fit.logK(s, "pair", lower=0, upper=4)]
    res = fit.fit_equilibrium(s, synthetic(), params, absolute_sigma=True)
    assert res.success
    assert res.parameters["logK gypsum"] == pytest.approx(-4.58, abs=1e-4)
    assert res.parameters["logK pair"] == pytest.approx(2.30, abs=1e-3)
    assert s.reactions[1].logK == pytest.approx(-4.58, abs=1e-4)  # system holds the fit
    assert res.covariance.shape == (2, 2) and abs(res.correlation[0, 1]) <= 1
    assert "logK gypsum" in res.summary()


def test_confidence_interval_and_profile():
    s = gypsum_system(logK=-4.4)
    params = [fit.logK(s, "gypsum", lower=-6, upper=-3)]
    res = fit.fit_equilibrium(s, synthetic(noise=0.02, seed=3), params, absolute_sigma=True)
    lo, hi = res.confidence_intervals["logK gypsum"]
    se = res.standard_errors["logK gypsum"]
    assert lo < res.parameters["logK gypsum"] < hi and se > 0
    assert hi - lo == pytest.approx(2 * 1.959964 * se, rel=1e-6)
    pr = res.profile("logK gypsum", n=9)
    # one nearly linear parameter: profile and linearized intervals agree
    assert pr.interval[0] == pytest.approx(lo, abs=0.25 * se)
    assert pr.interval[1] == pytest.approx(hi, abs=0.25 * se)
    y, sy = res.predict({"amounts": {"Gypsum": 1.0}}, DISSOLVED_CA)
    assert y > 0 and sy > 0


def test_observe_and_errors():
    s = gypsum_system()
    r = s.equilibrate({"Gypsum": 1.0})
    assert fit.observe(r, "SI:Gypsum") == pytest.approx(0.0, abs=1e-8)
    assert fit.observe(r, "log10_molality:Ca+2") == pytest.approx(math.log10(r.molality["Ca+2"]))
    with pytest.raises(KeyError):
        fit.logK(s, "missing")
    with pytest.raises(ValueError):
        fit.fit_equilibrium(s, [], [fit.logK(s, "gypsum")])


def test_sweep_and_predominance():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = gypsum_system()
    sw = plot.sweep(s, "SO4-2", [0.0, 0.05, 0.2], amounts={"Ca+2": 0.05})
    assert sw.converged.all()
    g = sw.get("solids:Gypsum")
    assert g[0] == 0.0 and g[-1] > 0
    plot.plot_sweep(sw)
    pr = plot.predominance(s, ["Ca+2", "CaSO4", "Gypsum"], ("Ca+2", [0.001, 0.1]), ("SO4-2", [0.001, 0.1]))
    assert pr.labels[pr.grid[0, 0]] in ("Ca+2", "CaSO4") and pr.labels[pr.grid[1, 1]] == "Gypsum"
    pr.plot()
    T, dG = plot.ellingham({"gypsum": s.reactions[1]}, [298.15, 350.0])
    assert dG["gypsum"][0] == pytest.approx(-8.314462618 * 298.15 * math.log(10) * -4.58 / 1000, rel=1e-4)
    assert plot.pe_from_eh(plot.eh_from_pe(10.0)) == pytest.approx(10.0)
    plt.close("all")

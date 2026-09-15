# Reference data for the tests

| file | contents | source |
|------|----------|--------|
| `phreeqc_llnl_sit.json` | Th and CO₂ in NaCl at pH 3, 25–150 °C: log activities, ionic strength, water activity | PHREEQC 3 through phreeqpython 1.6.2, on the bundled `llnl.dat` and `sit.dat` (see the `source` field) |
| `phreeqc_activity.json` | activity-model comparisons | PHREEQC 3 through phreeqpython; the version is not recorded (see the `source` field) |
| `phreeqc_features.json` | charge balance, exchange, DDL surface, Fe redox, calcite–CO₂, fixed-volume gas; each entry stores its PHREEQC input | PHREEQC 3 through phreeqpython 1.6.2. The database is not recorded in the file; `tests/test_phreeqc.py` compares against `phreeqc.dat` |
| `minteq_cases.json` | the three cases of Laliwala et al. (2025), with `minteq.v4` constants and PHREEQC results (ideal and activity-corrected) | constants: `minteq.v4.dat` as distributed with PHREEQC, derived from MINTEQA2 (Allison et al. 1991). The PHREEQC version of those runs is not recorded |
| `phreeqc_examples.json` | PHREEQC example problems 1–8, 10, 11, 14, 19 (the equilibrium parts) and adversarial stress cases: the exact PHREEQC input, every selected-output row (species moles, log activities, SI, phase/exchange/surface/gas/solid-solution amounts, ψ, pH, pe, I, water mass) and the per-step translation used by `tests/test_phreeqc_examples.py` | PHREEQC 3.8.7 (IPhreeqc 3.8.7-17149) through phreeqpython 1.6.2, on the bundled `phreeqc.dat`, `pitzer.dat`, `llnl.dat` and `sit.dat`. Written by `generate_phreeqc_examples.py` (needs phreeqpython, which is not a project dependency). Inputs adapted from the PHREEQC 3 example files (Parkhurst and Appelo 2013) |
| `notebook_reference.json` | counterexample, `Case2_davies` and scan values | Pyomo/IPOPT active-set solutions from the review notebook `torres-mass-action-review.ipynb`, which is not part of this repository |

`examples/data/` holds copies of `minteq_cases.json` and `notebook_reference.json` for the notebooks.

Full citations are in [`docs/references.md`](../../docs/references.md).

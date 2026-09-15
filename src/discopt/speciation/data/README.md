# Bundled PHREEQC databases

| file | activity model | source |
|------|----------------|--------|
| `phreeqc.dat` | Truesdell-Jones / Davies (ion association) | PHREEQC 3 distribution |
| `pitzer.dat` | Pitzer (brines, to ~200 °C) | PHREEQC 3 distribution |
| `wateq4f_PWN.dat` | Truesdell-Jones (WATEQ4F with PWN additions) | PHREEQC / PWN (Netherlands) |

The files are unmodified copies of the databases in phreeqpython 1.6.2
(Vitens, Apache-2.0, <https://github.com/Vitens/phreeqpython>), which
redistributes the databases of PHREEQC. PHREEQC and its databases were written
by D.L. Parkhurst and C.A.J. Appelo at the U.S. Geological Survey and are in
the public domain as works of the U.S. Government
(<https://www.usgs.gov/software/phreeqc-version-3>).

Please cite the data source when you publish results computed with these files:

- Parkhurst, D.L., and Appelo, C.A.J., 2013, Description of input and examples
  for PHREEQC version 3, U.S. Geological Survey Techniques and Methods, book 6,
  chap. A43, 497 p. <https://doi.org/10.3133/tm6A43>
- For `pitzer.dat`: Plummer, L.N., Parkhurst, D.L., Fleming, G.W., and Dunkle,
  S.A., 1988, A computer program incorporating Pitzer's equations for
  calculation of geochemical reactions in brines, USGS WRI Report 88-4153.
  <https://doi.org/10.3133/wri884153>. The PHREEQC 3 revisions to 200 °C are
  described by Appelo, C.A.J., 2015, *Applied Geochemistry* 55, 62–71.
  <https://doi.org/10.1016/j.apgeochem.2014.11.007>
- For `wateq4f_PWN.dat`: Ball, J.W., and Nordstrom, D.K., 1991, User's manual
  for WATEQ4F, USGS Open-File Report 91-183. <https://doi.org/10.3133/ofr91183>

## `llnl.dat` and `sit.dat`

| file | activity model | source |
|------|----------------|--------|
| `llnl.dat` | LLNL B-dot, 0–300 °C | PHREEQC 3 distribution (2024 copy) |
| `sit.dat` | SIT | PHREEQC 3 distribution (2024 copy) of ThermoChimie v12a |

Both are unmodified copies of the files distributed with PHREEQC 3.

- **`llnl.dat`.** From `thermo.com.V8.R6.230`, prepared by Jim Johnson at
  Lawrence Livermore National Laboratory in Geochemist's Workbench format. It
  was converted to PHREEQC format by Greg Anderson with help from David
  Parkhurst. The file notes that it is "relatively untested", and that each
  species has a valid temperature range that is not enforced. Staying inside
  those ranges is the user's responsibility.
- **`sit.dat`.** The ThermoChimie database, version 12a (2023), of ANDRA
  (France), NWS (UK) and ONDRAF/NIRAS (Belgium), <https://www.thermochimie-tdb.com>.
  It was generated with XCheck v5.2.0. D. Parkhurst modified its redox states
  and the gram formula weights of S(6) and Si in May 2024. ThermoChimie data
  are provided under the ThermoChimie terms of use: cite ThermoChimie, and
  check the project website before redistributing the data in modified form.

**Citation.** When you publish results computed with these files, cite:
- Parkhurst and Appelo (2013), above.
- For `sit.dat`: Giffaut, E., et al., 2014, *Andra thermodynamic database for
  performance assessment: ThermoChimie*, Applied Geochemistry 49, 225–236.
  <https://doi.org/10.1016/j.apgeochem.2014.05.007>
- For `llnl.dat`: Delany, J.M., and Lundeen, S.R., 1991, *The LLNL
  thermochemical data base: revised data and file format for the EQ3/6
  package*, Lawrence Livermore National Laboratory, UCID-21658.
  <https://www.osti.gov/biblio/138193>

## What is read

`discopt.speciation` reads only part of each file:
- `log_k`, `delta_h`, `-analytic`, `-gamma`;
- `-llnl_gamma`, `-co2_llnl_gamma`, `-add_logk`;
- the `NAMED_EXPRESSIONS`, `LLNL_AQUEOUS_MODEL_PARAMETERS`, `PITZER` and
  `SIT` blocks.

Molar volumes, Peng-Robinson gas parameters, diffusion coefficients and
`RATES` are ignored. See `docs/databases.md`.

All references, including the activity models, are in `docs/references.md`.

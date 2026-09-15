# References

This page lists the software, databases, models and papers that `discopt.speciation` implements or relies on. For each, it says where the package uses it. DOIs were checked against Crossref. Entries without a DOI give a stable library or publisher link.

## Citing results computed with this package

Cite:
- the package and discopt;
- the database you used (see [Databases](#databases));
- the activity model, if the result depends on it (see [Activity models](#activity-models)).

For example, a calculation with [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) should cite PHREEQC ([Parkhurst and Appelo 2013](#parkhurst2013)), ThermoChimie ([Giffaut et al. 2014](#giffaut2014)) and the [SIT references](#activity-models).

## Software

- <a id="kitchin"></a>Kitchin, J. *discopt* (modeling and optimization in Python). <https://github.com/jkitchin/discopt>
  — the NLP modeling and solver layer used by every method.
- <a id="parkhurst2013"></a>Parkhurst, D.L., and Appelo, C.A.J. (2013). *Description of input and examples for PHREEQC version 3 — a computer program for speciation, batch-reaction, one-dimensional transport, and inverse geochemical calculations.* U.S. Geological Survey Techniques and Methods, book 6, chap. A43. <https://doi.org/10.3133/tm6A43>; software: <https://www.usgs.gov/software/phreeqc-version-3>
  — database format and conventions ([`database.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/database.py)), the Truesdell–Jones, LLNL, SIT and Pitzer conventions, and the validation results.
- <a id="vitens"></a>Vitens. *phreeqpython* 1.6.2 (Apache-2.0). <https://github.com/Vitens/phreeqpython>
  — the source of [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat), [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat) and [`wateq4f_PWN.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat), and the PHREEQC engine used to generate the reference data in [`tests/data/`](https://github.com/jkitchin/discopt-speciation/tree/main/tests/data).

## Databases

| file | cite |
|------|------|
| [`phreeqc.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/phreeqc.dat) | [Parkhurst and Appelo (2013)](#parkhurst2013) |
| [`pitzer.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/pitzer.dat) | [Plummer et al. (1988)](#plummer1988); [Appelo (2015)](#appelo2015) |
| [`wateq4f_PWN.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/wateq4f_PWN.dat) | [Ball and Nordstrom (1991)](#ball1991) |
| [`llnl.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/llnl.dat) | [Delany and Lundeen (1991)](#delany1991) |
| [`sit.dat`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/data/sit.dat) | [Giffaut et al. (2014)](#giffaut2014); ThermoChimie v12a |
| [`minteq.v4.dat`](https://github.com/usgs-coupled/phreeqc3/blob/master/database/minteq.v4.dat) (tests only) | [Allison et al. (1991)](#allison1991) |

- <a id="plummer1988"></a>Plummer, L.N., Parkhurst, D.L., Fleming, G.W., and Dunkle, S.A. (1988). *A computer program incorporating Pitzer's equations for calculation of geochemical reactions in brines.* USGS Water-Resources Investigations Report 88-4153. <https://doi.org/10.3133/wri884153>
- <a id="appelo2015"></a>Appelo, C.A.J. (2015). Principles, caveats and improvements in databases for calculating hydrogeochemical reactions in saline waters from 0 to 200 °C and 1 to 1000 atm. *Applied Geochemistry* 55, 62–71. <https://doi.org/10.1016/j.apgeochem.2014.11.007>
- <a id="ball1991"></a>Ball, J.W., and Nordstrom, D.K. (1991). *User's manual for WATEQ4F, with revised thermodynamic data base and test cases for calculating speciation of major, trace, and redox elements in natural waters.* USGS Open-File Report 91-183. <https://doi.org/10.3133/ofr91183>
- <a id="delany1991"></a>Delany, J.M., and Lundeen, S.R. (1991). *The LLNL thermochemical data base: revised data and file format for the EQ3/6 package.* Lawrence Livermore National Laboratory, UCID-21658. <https://www.osti.gov/biblio/138193>
- <a id="giffaut2014"></a>Giffaut, E., Grivé, M., Blanc, P., Vieillard, P., Colàs, E., Gailhanou, H., et al. (2014). Andra thermodynamic database for performance assessment: ThermoChimie. *Applied Geochemistry* 49, 225–236. <https://doi.org/10.1016/j.apgeochem.2014.05.007>; data (v12a) and terms of use: <https://www.thermochimie-tdb.com>
- <a id="allison1991"></a>Allison, J.D., Brown, D.S., and Novo-Gradac, K.J. (1991). *MINTEQA2/PRODEFA2, a geochemical assessment model for environmental systems: version 3.0 user's manual.* EPA/600/3-91/021. <https://www.osti.gov/biblio/5673069>
  — origin of the `minteq.v4` constants used in the validation cases.

## Activity models

[`activity.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/activity.py):

- **Debye–Hückel** (`DebyeHuckel`, `ExtendedDebyeHuckel`, and the long-range term of every model): <a id="debye1923"></a>Debye, P., and Hückel, E. (1923). Zur Theorie der Elektrolyte. I. Gefrierpunktserniedrigung und verwandte Erscheinungen. *Physikalische Zeitschrift* 24, 185–206.
- **Davies** (`Davies`, and the fallback for ions without size parameters in `TruesdellJones` and `LLNL`):
  - <a id="davies1938"></a>Davies, C.W. (1938). The extent of dissociation of salts in water. Part VIII. *Journal of the Chemical Society*, 2093. <https://doi.org/10.1039/JR9380002093>
  - <a id="davies1962"></a>Davies, C.W. (1962). *Ion Association.* Butterworths, London.
- **Truesdell–Jones** (`TruesdellJones`, also used for exchange species): <a id="truesdell1974"></a>Truesdell, A.H., and Jones, B.F. (1974). WATEQ, a computer program for calculating chemical equilibria of natural waters. *Journal of Research of the U.S. Geological Survey* 2(2), 233–248. <https://pubs.usgs.gov/publication/70232379>
- **B-dot** (`BDot`, `LLNL`): <a id="helgeson1969"></a>Helgeson, H.C. (1969). Thermodynamics of hydrothermal systems at elevated temperatures and pressures. *American Journal of Science* 267, 729–804. <https://doi.org/10.2475/ajs.267.7.729>
- **CO₂(aq) activity coefficient** (`-co2_llnl_gamma` in `LLNL`): <a id="drummond1981"></a>Drummond, S.E. (1981). *Boiling and mixing of hydrothermal fluids: chemical effects on mineral precipitation.* PhD thesis, Pennsylvania State University.
- **Salting-out of neutral species** (the *bI* term): <a id="setschenow1889"></a>Setschenow, J. (1889). Über die Konstitution der Salzlösungen auf Grund ihres Verhaltens zu Kohlensäure. *Zeitschrift für Physikalische Chemie* 4, 117–125. <https://doi.org/10.1515/zpch-1889-0409>
- **Specific ion interaction theory** (`SIT`):
  - <a id="bronsted1922"></a>Brønsted, J.N. (1922). Studies on solubility. IV. The principle of the specific interaction of ions. *Journal of the American Chemical Society* 44, 877–898. <https://doi.org/10.1021/ja01426a001>
  - <a id="scatchard1936"></a>Scatchard, G. (1936). Concentrated solutions of strong electrolytes. *Chemical Reviews* 19, 309–327. <https://doi.org/10.1021/cr60064a008>
  - <a id="guggenheim1955"></a>Guggenheim, E.A., and Turgeon, J.C. (1955). Specific interaction of ions. *Transactions of the Faraday Society* 51, 747. <https://doi.org/10.1039/TF9555100747>
  - <a id="ciavatta1980"></a>Ciavatta, L. (1980). The specific interaction theory in evaluating ionic equilibria. *Annali di Chimica (Rome)* 70, 551–567.
  - <a id="grenthe1992"></a>Grenthe, I., Fuger, J., Konings, R.J.M., Lemire, R.J., Muller, A.B., Nguyen-Trung, C., and Wanner, H. (1992). *Chemical Thermodynamics of Uranium.* NEA-OECD, North-Holland. <https://www.oecd-nea.org/jcms/pl_13704>
    — the NEA TDB form, with the 1.5√I denominator used by ThermoChimie.
- **Pitzer** (`Pitzer`, [`_etheta.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/_etheta.py)):
  - <a id="pitzer1973a"></a>Pitzer, K.S. (1973). Thermodynamics of electrolytes. I. Theoretical basis and general equations. *Journal of Physical Chemistry* 77, 268–277. <https://doi.org/10.1021/j100621a026>
  - <a id="pitzer1973b"></a>Pitzer, K.S., and Mayorga, G. (1973). Thermodynamics of electrolytes. II. Activity and osmotic coefficients for strong electrolytes with one or both ions univalent. *Journal of Physical Chemistry* 77, 2300–2308. <https://doi.org/10.1021/j100638a009>
  - <a id="pitzer1974"></a>Pitzer, K.S., and Kim, J.J. (1974). Thermodynamics of electrolytes. IV. Activity and osmotic coefficients for mixed electrolytes. *Journal of the American Chemical Society* 96, 5701–5707. <https://doi.org/10.1021/ja00825a004>
  - <a id="pitzer1975"></a>Pitzer, K.S. (1975). Thermodynamics of electrolytes. V. Effects of higher-order electrostatic terms. *Journal of Solution Chemistry* 4, 249–265. <https://doi.org/10.1007/BF00646562>
    — the unsymmetrical mixing terms (E-θ).
  - <a id="harvie1984"></a>Harvie, C.E., Møller, N., and Weare, J.H. (1984). The prediction of mineral solubilities in natural waters: the Na-K-Mg-Ca-H-Cl-SO4-OH-HCO3-CO3-CO2-H2O system to high ionic strengths at 25 °C. *Geochimica et Cosmochimica Acta* 48, 723–751. <https://doi.org/10.1016/0016-7037(84)90098-X>
  - <a id="macinnes1919"></a>MacInnes, D.A. (1919). The activities of the ions of strong electrolytes. *Journal of the American Chemical Society* 41, 1086–1092. <https://doi.org/10.1021/ja02228a006>
    — the convention for individual-ion activity coefficients.

## Water properties

[`water.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/water.py):

- <a id="wagner2002"></a>Wagner, W., and Pruß, A. (2002). The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance for general and scientific use. *Journal of Physical and Chemical Reference Data* 31, 387–535. <https://doi.org/10.1063/1.1461829>
  — density on the saturation curve.
- <a id="bradley1979"></a>Bradley, D.J., and Pitzer, K.S. (1979). Thermodynamics of electrolytes. 12. Dielectric properties of water and Debye–Hückel parameters to 350 °C and 1 kbar. *Journal of Physical Chemistry* 83, 1599–1603. <https://doi.org/10.1021/j100475a009>
  — dielectric constant and Debye–Hückel parameters.

## Phases, exchange and surfaces

[`phases.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/phases.py), [`species.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/species.py):

- <a id="gaines1953"></a>Gaines, G.L., and Thomas, H.C. (1953). Adsorption studies on clay minerals. II. A formulation of the thermodynamics of exchange adsorption. *Journal of Chemical Physics* 21, 714–718. <https://doi.org/10.1063/1.1698996>
  — exchange activities as equivalent fractions.
- <a id="dzombak1990"></a>Dzombak, D.A., and Morel, F.M.M. (1990). *Surface Complexation Modeling: Hydrous Ferric Oxide.* Wiley, New York. ISBN 978-0-471-63731-8.
  — the diffuse double layer model (`SurfaceType(edl="ddl")`).
- Ideal solid solutions, ideal gas phases, and $\log K(T,P)$ from van 't Hoff, ΔCp, ΔV or analytic expressions follow the PHREEQC conventions ([Parkhurst and Appelo 2013](#parkhurst2013)).

## Solution methods

[`solve.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/solve.py):

- <a id="fischer1992"></a>Fischer, A. (1992). A special Newton-type optimization method. *Optimization* 24, 269–284. <https://doi.org/10.1080/02331939208843795>
  — the Fischer–Burmeister complementarity function.
- <a id="kanzow1996"></a>Kanzow, C. (1996). Some noninterior continuation methods for linear complementarity problems. *SIAM Journal on Matrix Analysis and Applications* 17, 851–868. <https://doi.org/10.1137/S0895479894273134>
  — the smoothed function and the ε-homotopy.
- <a id="smith1982"></a>Smith, W.R., and Missen, R.W. (1982). *Chemical Reaction Equilibrium Analysis: Theory and Algorithms.* Wiley, New York.
  — Gibbs energy minimization (`method="gibbs"`) and the stoichiometric formulation.
- <a id="leal2017"></a>Leal, A.M.M., Kulik, D.A., Smith, W.R., and Saar, M.O. (2017). An overview of computational methods for chemical equilibrium and kinetic calculations for geochemical and reactive transport modeling. *Pure and Applied Chemistry* 89, 597–643. <https://doi.org/10.1515/pac-2016-1107>
  — a review of mass-action and Gibbs minimization methods.

## Global stability

[Stability](stability.md):

- <a id="baker1982"></a>Baker, L.E., Pierce, A.C., and Luks, K.D. (1982). Gibbs energy analysis of phase equilibria. *Society of Petroleum Engineers Journal* 22(5), 731–742. <https://doi.org/10.2118/9806-PA>
  — the tangent plane criterion for global stability.
- <a id="michelsen1982"></a>Michelsen, M.L. (1982). The isothermal flash problem. Part I. Stability. *Fluid Phase Equilibria* 9(1), 1–19. <https://doi.org/10.1016/0378-3812(82)85001-2>
  — the tangent plane distance stability test.
- <a id="glynn1991"></a>Glynn, P.D. (1991). MBSSAS: a code for the computation of Margules parameters and equilibrium relations in binary solid-solution aqueous-solution systems. *Computers & Geosciences* 17(7), 907–966. <https://doi.org/10.1016/0098-3004(91)90090-Z>
  — binary miscibility gaps, as computed by PHREEQC.
- <a id="moore2009"></a>Moore, R.E., Kearfott, R.B., and Cloud, M.J. (2009). *Introduction to Interval Analysis.* SIAM, Philadelphia. <https://doi.org/10.1137/1.9780898717716>
  — the interval arithmetic in the certificate.

## Parameter fitting

[`fit.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/fit.py):

- <a id="venzon1988"></a>Venzon, D.J., and Moolgavkar, S.H. (1988). A method for computing profile-likelihood-based confidence intervals. *Applied Statistics* 37, 87–94. <https://doi.org/10.2307/2347496>
- <a id="raue2009"></a>Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U., and Timmer, J. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25, 1923–1929. <https://doi.org/10.1093/bioinformatics/btp358>

## Diagrams

[`plot.py`](https://github.com/jkitchin/discopt-speciation/blob/main/src/discopt/speciation/plot.py):

- <a id="pourbaix1974"></a>Pourbaix, M. (1966). *Atlas of Electrochemical Equilibria in Aqueous Solutions.* Pergamon Press, Oxford. Reprinted by NACE, Houston (1974).
- <a id="ellingham1944"></a>Ellingham, H.J.T. (1944). Reducibility of oxides and sulphides in metallurgical processes. *Journal of the Society of Chemical Industry* 63(5), 125–133. <https://doi.org/10.1002/jctb.5000630501>

## Validation cases and examples

- <a id="laliwala2025"></a>Laliwala, C.A., Amusat, O.O., and Torres, A.I. (2025). An optimization-based law of mass action precipitation/dissolution model. *Systems and Control Transactions* 4, 2140–2145 (Proceedings of ESCAPE 35). <https://doi.org/10.69997/sct.132742>
  — the three validation cases in notebook 02, [`tests/test_minteq.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_minteq.py) and [`tests/test_counterexamples.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_counterexamples.py).
- <a id="parkhurst2013-examples"></a>Parkhurst, D.L., and Appelo, C.A.J. (2013), chapter "Examples" of *Description of input and examples for PHREEQC version 3* (USGS Techniques and Methods 6-A43, <https://doi.org/10.3133/tm6A43>); input files distributed with PHREEQC 3 (<https://github.com/usgs-coupled/phreeqc3/tree/master/examples>)
  — examples 1–8, 10, 11, 14, 17 (forward part) and 19 in [`tests/test_phreeqc_examples.py`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/test_phreeqc_examples.py) ([Validation](validation.md#phreeqc-example-problems)).
- The provenance of each reference data file is recorded in [`tests/data/README.md`](https://github.com/jkitchin/discopt-speciation/blob/main/tests/data/README.md).

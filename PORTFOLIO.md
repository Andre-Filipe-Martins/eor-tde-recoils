# Portfolio Summary: EoR TDE Recoils

This repository contains research code developed for my MSc thesis in Astrophysics. The project models whether recoiling massive black-hole remnants and their external tidal disruption events (TDEs) could contribute meaningfully to the hydrogen-ionising energy budget during the Epoch of Reionisation (EoR).

Although the scientific application is astrophysical, the project also demonstrates a complete data-analysis workflow: literature-based model construction, Monte Carlo simulation, numerical integration, performance-aware sampling, and reproducible post-processing.

## My contribution

I designed and implemented the full analysis pipeline in Python. The work included:

- fitting a high-redshift galaxy stellar mass function using simulation-calibrated data;
- generating Monte Carlo catalogues of black-hole merger remnants across the EoR;
- implementing empirical scaling relations for black-hole mass, galaxy size, halo mass, and host-galaxy potentials;
- modelling recoil trajectories in composite NFW + Hernquist potentials;
- estimating external TDE counts with time-dependent depletion;
- converting TDE counts into an ionising-energy budget;
- computing an independent photon-counting reionisation requirement;
- comparing the simulated contribution with the required EoR energy budget.

## Technical skills demonstrated

This project demonstrates experience with:

- Python-based scientific programming;
- numerical modelling and Monte Carlo methods;
- data cleaning and transformation;
- statistical binning and interpolation;
- performance-aware sampling and rescaling;
- reproducible analysis pipelines;
- structured post-processing of generated catalogues;
- scientific visualisation and tabular reporting;
- translating research literature into testable computational models.

## Why this is relevant for data-analysis roles

The workflow follows the same logic as many applied data-analysis problems:

1. define a quantitative question;
2. gather and interpret external data sources;
3. build a reproducible processing pipeline;
4. implement numerical and statistical models;
5. optimise performance for large synthetic datasets;
6. validate assumptions and compare outputs against a target requirement;
7. communicate results clearly through figures, tables, and written interpretation.

The project therefore reflects both research independence and practical data-analysis skills.

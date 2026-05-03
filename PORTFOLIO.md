# Portfolio Summary: EoR TDE Recoils

This repository contains research code developed for my MSc thesis in Astrophysics. The project models whether recoiling massive black-hole remnants and their external tidal disruption events (TDEs) could contribute meaningfully to the hydrogen-ionising energy budget during the Epoch of Reionisation (EoR).

The main scientific result is astrophysical, but the project is also a data-analysis and scientific-computing case study. It combines literature-based modelling, Monte Carlo simulation, numerical integration, performance-aware sampling, and reproducible post-processing.

## Project objective

The aim was to build an end-to-end computational pipeline that connects:

- high-redshift galaxy population models;
- empirical black-hole and galaxy scaling relations;
- black-hole merger remnants and recoil velocities;
- external TDE rate estimates;
- ionising-energy production;
- comparison with the energy required for hydrogen reionisation.

The calculation was designed as an optimistic upper-bound test: if the channel remains small under favourable assumptions, it is unlikely to be cosmologically significant.

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

## Data-analysis and technical skills demonstrated

This project demonstrates experience with:

- Python-based scientific programming;
- numerical modelling and Monte Carlo methods;
- data cleaning and transformation;
- statistical binning and interpolation;
- performance-aware sampling and rescaling;
- reproducible analysis pipelines;
- structured post-processing of large generated catalogues;
- scientific visualisation and tabular reporting;
- translating research literature into testable computational models.

## Pipeline structure

The main scripts are:

| File | Role |
|---|---|
| `physics_relations.py` | Shared physical relations and empirical scaling laws. |
| `gsmf_fitting.py` | Galaxy stellar mass function fitting. |
| `ratio_scan_sample_gen.py` | Monte Carlo catalogue generation for the ratio scan. |
| `ratio_scan_vcent.py` | Kick-ratio optimisation. |
| `simulation.py` | Main Monte Carlo simulation. |
| `external_tde_ionizing_energy.py` | External TDE energy-budget calculation. |
| `igm_reionization_requirement.py` | Reionisation energy-requirement calculation. |
| `gsmf_required_for_ftde_target.py` | Diagnostic GSMF boost calculation. |

## Performance and reproducibility

The pipeline was designed to handle large generated catalogues without storing them directly in the repository. To keep the analysis tractable, capped Monte Carlo samples are used in some stages and then rescaled to physical merger-event targets. This reduces runtime and storage costs while preserving the population-level quantities needed for the thesis analysis.

The scripts also use fixed random seeds where appropriate and avoid storing generated Parquet catalogues, figures, and spreadsheets in version control.

## Why this project is relevant for data-analysis roles

Although the project is astrophysical, the workflow is similar to many applied data-analysis problems:

1. define a quantitative question;
2. gather and interpret external data sources;
3. build a reproducible processing pipeline;
4. implement numerical/statistical models;
5. optimise performance for large synthetic datasets;
6. validate assumptions and compare outputs against a target requirement;
7. communicate results clearly through figures, tables, and written interpretation.

The project therefore reflects both research independence and practical data-analysis skills.

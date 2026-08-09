# Portfolio Summary: EoR TDE Recoils

This repository contains the scientific Python pipeline developed for my MSc Astrophysics research project on whether tidal disruption events (TDEs) around recoiling massive black holes could contribute meaningfully to cosmic hydrogen reionisation.

The scientific application is astrophysical, but the project is also a complete computational data-analysis workflow: it combines heterogeneous literature inputs, numerical modelling, Monte Carlo sampling, weighted population estimates, optimisation, large intermediate datasets, reproducible post-processing, and scientific communication.

## Project outcome

The current major-plus-minor merger calculation predicts approximately $4\times10^{12}$ external TDEs over the adopted Epoch of Reionisation interval, corresponding to roughly $3\times10^{65},\mathrm{erg}$ of hydrogen-ionising energy. This is only about **0.06 per cent** of the estimated ionising-energy requirement for the same comoving volume.

The main scientific conclusion is therefore a negative but useful one: even when uncertain assumptions are deliberately chosen to favour the recoil-TDE channel, its global contribution to reionisation remains negligible.

## What I built

I designed and implemented the analysis pipeline in Python. The work included:

* fitting a high-redshift galaxy stellar mass function to simulation-calibrated data;
* constructing explicit primary/secondary galaxy merger pairs rather than sampling remnant properties independently;
* combining an observationally calibrated major-merger rate with a mass-ratio-dependent extension to minor mergers;
* integrating physical event targets over progenitor mass and merger mass ratio while grouping the final population by descendant stellar mass;
* mapping progenitor galaxies to central black holes and computing post-merger remnant properties;
* implementing empirical galaxy-size, halo-mass, black-hole-mass, and host-potential relations;
* modelling radial recoil trajectories in composite NFW + Hernquist gravitational potentials;
* scanning the kick parameter $V_{\mathrm{kick}}/V_{\mathrm{cent}}$ to optimise external TDE production by descendant mass bin;
* estimating time-dependent TDE production from the stellar cluster retained by the recoiling remnant;
* converting external TDE counts into a hydrogen-ionising energy budget;
* independently calculating the photon-counting reionisation requirement for the diffuse IGM;
* comparing the simulated contribution with the required EoR energy budget.

## Data and numerical engineering

A large part of the project involved making the calculation computationally practical while preserving the physical population normalisation. Examples include:

* **Weighted Monte Carlo sampling:** very large physical event populations are represented by capped samples of up to 100,000 events per simulation context, with each row carrying the corresponding physical weight.
* **Shared population catalogues:** the final recoil/TDE simulation reuses the same galaxy-pair catalogues as the optimisation scan, preventing two independent Monte Carlo populations from being mixed.
* **Vectorised numerical calculations:** array-based sampling and TDE calculations reduce Python-level loops in the expensive stages.
* **Parallel execution:** independent redshift snapshots can be processed across multiple worker processes while BLAS/OpenMP thread counts are controlled to avoid CPU oversubscription.
* **Efficient intermediate storage:** large event catalogues are stored in compressed Parquet files and unnecessary numerical columns are downcast where appropriate.
* **Version validation:** generated catalogues contain a model-version identifier so incompatible pipeline generations fail explicitly instead of being combined silently.
* **Reproducibility:** independent runs use deterministic seed conventions and downstream physical totals are constructed consistently from the stored event weights.

## Technical skills demonstrated

The project demonstrates practical experience with:

* Python scientific programming;
* NumPy and vectorised numerical computing;
* pandas and structured data pipelines;
* SciPy-based fitting, interpolation, and numerical integration;
* Astropy cosmology calculations;
* Monte Carlo simulation and weighted sampling;
* numerical optimisation and parameter scans;
* multiprocessing and performance-aware computation;
* Parquet-based intermediate datasets;
* JSON and Excel output generation;
* Matplotlib scientific visualisation;
* translating peer-reviewed research into a testable computational model;
* validating assumptions and communicating uncertainty in model results.

## Why this is relevant beyond astrophysics

The workflow follows the same pattern as many applied data-analysis and modelling projects:

1. define a quantitative question and measurable target;
2. identify and reconcile multiple external data/model inputs;
3. transform those inputs into a reproducible computational pipeline;
4. design a sampling strategy for a population too large to represent directly;
5. optimise expensive numerical stages without changing the underlying model;
6. validate intermediate and final outputs against physical and numerical expectations;
7. produce machine-readable results, figures, and tables for interpretation;
8. communicate the final result clearly, including limitations and uncertainty.

The repository therefore demonstrates not only astrophysical modelling, but also transferable skills in numerical analysis, data engineering, reproducibility, performance optimisation, and evidence-based interpretation.

# EoR TDE Recoils

Monte Carlo population-synthesis code developed for an MSc Astrophysics project on recoiling massive black-hole merger remnants and their tidal disruption events (TDEs) during the Epoch of Reionisation (EoR).

The project asks whether TDEs produced by gravitational-wave-recoiling black holes could provide a meaningful fraction of the hydrogen-ionising energy required to reionise the intergalactic medium (IGM). The calculation is intentionally optimistic: uncertain modelling choices are generally selected to favour external TDE production, so the resulting contribution is best interpreted as an upper bound.

A shorter, skills-focused summary for recruiters and data-analysis roles is available in [`PORTFOLIO.md`](PORTFOLIO.md).

## Headline result

For the current zero-delay, major-plus-minor merger calculation over $z=12\rightarrow6$, the model produces approximately

* $4\times10^{12}$ external TDEs;
* $3\times10^{65},\mathrm{erg}$ of hydrogen-ionising energy;
* compared with an EoR requirement of approximately $5\times10^{68},\mathrm{erg}$.

This gives

$$
f_{\mathrm{TDE}}\simeq 6\times10^{-4},
$$

or about **0.06 per cent** of the required ionising energy. Even under the deliberately favourable assumptions used here, the recoil-TDE channel is therefore energetically negligible for global hydrogen reionisation.

In this project, an **external TDE** is a disruption occurring after the recoiling remnant crosses the adopted central-region boundary. It does not necessarily imply that the black hole has escaped the host potential.

## Scientific model

The pipeline combines the following ingredients:

* a FIRE-2-calibrated high-redshift galaxy stellar mass function (GSMF);
* an observationally calibrated major-merger rate from Duan et al. (2025);
* a stellar-mass-ratio-dependent extension to minor mergers based on Rodriguez-Gomez et al. (2015);
* explicit Monte Carlo sampling of primary and secondary progenitor galaxies;
* black-hole/galaxy scaling relations and a non-spinning remnant-mass prescription;
* Morishita et al. (2024) galaxy sizes and composite NFW + Hernquist host potentials;
* a scan of $\mathcal{R}=V_{\mathrm{kick}}/V_{\mathrm{cent}}$ to identify kick ratios that maximise external TDE production;
* bound-cluster TDE rates with time-dependent depletion;
* a black-hole-mass-dependent ionising-energy prescription for each TDE;
* an independent photon-counting calculation of the hydrogen reionisation requirement.

The baseline model treats the galaxy merger and black-hole coalescence as occurring at the same simulation snapshot.

## Repository contents

| File                              | Purpose                                                                                                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `physics_relations.py`            | Shared physical relations, cosmology helpers, empirical scaling laws, GSMF interpolation, merger-rate relations, and TDE building blocks.                    |
| `gsmf_fitting.py`                 | Fits single-Schechter functions to the completeness-limited FIRE-2 GSMF data at $z=6$–12 and produces the GSMF diagnostic figures.                           |
| `merger_pair_sampling.py`         | Shared population builder for explicit major and minor galaxy pairs, descendant-mass-bin targets, black-hole assignment, cap sampling, and physical weights. |
| `ratio_scan_sample_gen.py`        | Generates the weighted galaxy-pair catalogues and deterministic physical target table used by the recoil-ratio scan.                                         |
| `ratio_scan_vcent.py`             | Scans $V_{\mathrm{kick}}/V_{\mathrm{cent}}$ for two central-region definitions and identifies the peak ratio for each descendant stellar-mass bin.           |
| `simulation.py`                   | Reuses the ratio-scan catalogues, applies the selected peak kick ratios, and computes recoil trajectories and central/external TDE counts.                   |
| `external_tde_ionizing_energy.py` | Converts weighted external TDE counts into the cumulative hydrogen-ionising energy budget and produces the energy figures/tables.                            |
| `igm_reionization_requirement.py` | Independently computes the photon-counting hydrogen-ionising requirement for the diffuse IGM over the EoR.                                                   |

Generated catalogues, figures, spreadsheets, and JSON outputs are not stored in the repository because they can be large and are reproducible from the source scripts.

## Pipeline workflow

The expensive galaxy-pair population is sampled once and reused downstream. This ensures that the recoil-ratio scan and final simulation operate on the same Monte Carlo events, descendant mass bins, and cap-rescaling weights.

```text
gsmf_fitting.py
        |
        v
ratio_scan_sample_gen.py
        |
        |-- uses merger_pair_sampling.py + physics_relations.py
        v
ratio_scan_vcent.py
        |
        |-- selects R_peak = V_kick / V_cent per descendant mass bin
        v
simulation.py
        |
        v
external_tde_ionizing_energy.py

igm_reionization_requirement.py   # independent EoR requirement calculation
```

## Current numerical setup

The current pipeline uses:

* EoR interval: $z=12$ to $z=6$;
* catalogue snapshots: $z=11.8,11.6,\ldots,6.2$ with $\Delta z=0.2$;
* 10 independent Monte Carlo runs with fixed seeds;
* explicit major ($\mu\geq0.25$) and minor ($\mu<0.25$) merger populations;
* one shared descendant stellar-mass-bin grid throughout the pipeline;
* a Monte Carlo cap of 100,000 sampled events per snapshot, descendant-mass bin, and merger class;
* a single physical weight $n_{\mathrm{phys}}/n_{\mathrm{samp}}$ attached to each sampled row when the physical target exceeds the cap;
* no galaxy-merger-to-BH-merger delay in the baseline calculation;
* a main external-region boundary of $R_{\mathrm{cent}}=0.05R_{\mathrm{eff}}$, with $R_{\mathrm{cent}}=R_{\mathrm{eff}}$ retained as a comparison case;
* a $1.3,M_\odot$, $1.3,R_\odot$ disrupted-star model in the current TDE calculation.

## Suggested run order

From the repository root:

```bash
python gsmf_fitting.py
python ratio_scan_sample_gen.py
python ratio_scan_vcent.py
python simulation.py
python external_tde_ionizing_energy.py
python igm_reionization_requirement.py
```

`merger_pair_sampling.py` and `physics_relations.py` are imported by the executable scripts and are not run separately.

The ratio-scan and simulation stages can be computationally expensive and generate large Parquet catalogues.

## Main generated outputs

Depending on which scripts are run, the pipeline produces files such as:

```text
figures/
ratio_scan_catalogue/
results_bin_targets/
simulation_results/

ratio_scan_vcent_rpeak__fcent0p05.json
ratio_scan_vcent_rpeak__fcent1p00.json
ratio_scan_vcent_tables__fcent0p05.xlsx
ratio_scan_vcent_tables__fcent1p00.xlsx
ratio_scan_vcent_figure_data.json

external_tde_ionizing_energy.json
external_tde_ionizing_energy_series.json
external_tde_ionizing_energy.xlsx

igm_reionization_requirement.json
igm_reionization_requirement.xlsx
```

The principal figures include the GSMF fits, the external-TDE ratio scan, the external ionising energy by descendant mass bin, and the total external ionising-energy history.

## Requirements

The scripts require Python 3.10 or later. Main third-party packages are:

```text
numpy
pandas
scipy
matplotlib
astropy
openpyxl
pyarrow
```

A minimal environment can be installed with:

```bash
pip install numpy pandas scipy matplotlib astropy openpyxl pyarrow
```

## Reproducibility notes

* Fixed random seeds are used for the independent Monte Carlo runs.
* Catalogue rows carry their physical cap-rescaling weight, avoiding repeated population rescaling in downstream stages.
* A population-model version string is stored in the generated catalogues so incompatible pipeline generations fail explicitly rather than being mixed silently.
* The final simulation reuses the ratio-scan population instead of drawing a second independent merger catalogue.
* BLAS/OpenMP thread counts are limited in the computational scripts, while independent snapshots can be parallelised at process level.
* Generated products are intentionally excluded from version control.

## Project status

This repository accompanies an MSc Astrophysics research project on the contribution of recoiling-black-hole TDEs to cosmic reionisation. The code is provided for transparency, reproducibility, and portfolio purposes; it is research software rather than a general-purpose astrophysical simulation package.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

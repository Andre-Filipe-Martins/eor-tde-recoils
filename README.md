# EoR TDE Recoils

Monte Carlo modelling for an MSc thesis project on recoiling massive black-hole remnants and their external tidal disruption events (TDEs) during the Epoch of Reionisation (EoR).

The project estimates whether TDEs produced by recoiling black-hole merger remnants can contribute significantly to the hydrogen-ionising energy budget required for cosmic reionisation. The calculation is deliberately optimistic: the recoil velocities are chosen from a ratio scan that maximises the number of external TDEs in each host-galaxy stellar-mass bin.

## Repository contents

This repository contains the Python scripts used for the population synthesis, ratio scan, TDE counting, and ionising-energy comparison.

| File | Purpose |
|---|---|
| `physics_relations.py` | Shared physical relations and empirical scaling laws used by the pipeline. |
| `gsmf_fitting.py` | Fits the FIRE-2 galaxy stellar mass function with a single-Schechter form. |
| `ratio_scan_sample_gen.py` | Generates the remnant catalogue used by the kick-ratio scan. |
| `ratio_scan_vcent.py` | Scans the kick-velocity ratio \(V_{\rm kick}/V_{\rm cent}\) and identifies the peak ratio per host-mass bin. |
| `simulation.py` | Runs the main Monte Carlo simulation using the peak kick ratios. |
| `external_tde_ionizing_energy.py` | Converts external TDE counts into a hydrogen-ionising energy budget. |
| `igm_reionization_requirement.py` | Computes the photon-counting ionising-energy requirement for reionising the IGM. |
| `gsmf_required_for_ftde_target.py` | Diagnostic calculation for the GSMF normalisation required to reach a target TDE contribution. |

Generated catalogues, figures, spreadsheets, and Parquet outputs are not included in the repository.

## Scientific workflow

The pipeline is organised as follows:

1. Fit the high-redshift galaxy stellar mass function (GSMF).
2. Generate black-hole merger remnant catalogues across the EoR redshift grid.
3. Scan \(V_{\rm kick}/V_{\rm cent}\) to find the kick ratio that maximises external TDE production.
4. Run the main Monte Carlo simulation using those peak ratios.
5. Convert external TDE counts into an ionising-energy budget.
6. Compute the ionising-energy requirement for hydrogen reionisation.
7. Compare the external-TDE contribution with the EoR requirement.

The code uses the term **external TDE** for TDEs occurring after the recoiling remnant crosses the adopted central-region boundary. This is distinct from true escape from the host potential.

## Suggested run order

```bash
python gsmf_fitting.py
python ratio_scan_sample_gen.py
python ratio_scan_vcent.py
python simulation.py
python external_tde_ionizing_energy.py
python igm_reionization_requirement.py
python gsmf_required_for_ftde_target.py
```

Some scripts can be computationally expensive and may generate large output files. The generated output directories are intentionally excluded from version control.

## Main generated outputs

Depending on which scripts are run, the pipeline can generate:

```text
figures/
ratio_scan_catalogue/
results_bin_targets/
simulation_results/
external_tde_ionizing_energy.json
external_tde_ionizing_energy.xlsx
igm_reionization_requirement.xlsx
gsmf_required_for_ftde_target.txt
```

These outputs are ignored by `.gitignore` because they are generated products rather than source code.

## Requirements

The scripts require Python 3.10 or later.

Main Python packages:

```text
numpy
pandas
scipy
matplotlib
astropy
openpyxl
pyarrow
```

A minimal installation can be created with:

```bash
pip install numpy pandas scipy matplotlib astropy openpyxl pyarrow
```

## Notes on reproducibility

The scripts use fixed random seeds where Monte Carlo sampling is involved. Several scripts also limit BLAS thread counts before importing NumPy to improve runtime stability and reproducibility across machines.

The large simulation catalogues are not stored in this repository. To reproduce them, run the scripts in the order listed above.

## Project status

This repository accompanies an MSc thesis project. The code is provided as research software for transparency and reproducibility. It is not intended as a general-purpose astrophysical simulation package.

## License

This project is released under the MIT License. See `LICENSE` for details.

# OASIS-2 Longitudinal Trajectory Analysis

This project investigates whether longitudinal measurements describe neurodegenerative change more faithfully than a single categorical diagnosis, and whether information available at baseline can identify subjects likely to follow a more adverse trajectory.

The pipeline discovers subject-level whole-brain-volume trajectories from repeated OASIS-2 observations, then trains a baseline-only model to predict the resulting trajectory group for previously unseen subjects. The evaluation split is performed at the subject level so visits from the same person cannot appear in both training and testing data.

## Research questions

### Can continuous trajectories describe people better than a categorical label?

Yes, generally. A longitudinal trajectory preserves information about starting level, direction, magnitude, and timing of change. A categorical label is useful as a compact summary, but compresses those details and can hide meaningful differences between subjects within the same group.

This implementation uses the sequence of normalized whole-brain-volume (`nWBV`) measurements across visits. K-Means summarizes these trajectories into two interpretable groups:

- **Cluster 0 - Healthy Stable:** higher-volume, comparatively stable trajectories.
- **Cluster 1 - Accelerated Atrophy:** lower-volume trajectories with greater longitudinal decline.

These are analytical groups rather than ground-truth diagnoses. The OASIS-2 data also contain cognitive measurements such as MMSE; a future extension should model cognition and brain volume jointly. The current pipeline clusters `nWBV` and does not claim that brain volume alone completely represents cognition.

### Can baseline information predict who declines faster?

This project tests that question at the subject level. The trajectory group is discovered from the longitudinal sequence for training subjects, while the forecasting model receives only baseline predictors: `Age`, sex, education, socioeconomic status, estimated total intracranial volume (`eTIV`), and atlas scaling factor (`ASF`).

Held-out subjects are never used to fit the trajectory clusters or baseline classifier. Their longitudinal sequences are used only to assign evaluation labels after the baseline-only prediction is made.

On the current fixed split, the pipeline produced approximately:

- **Held-out accuracy:** 0.67
- **ROC-AUC:** 0.756
- **Held-out subjects:** 30

This indicates useful predictive signal, but not perfect individual prediction. It should not be interpreted as a clinical diagnostic or prognostic tool.

## Method

1. Load and validate the longitudinal OASIS-2 table.
2. Split subjects into training and held-out partitions without sharing subject IDs.
3. Impute missing baseline covariates using within-subject values and training-derived medians.
4. Pivot repeated `nWBV` measurements into one trajectory per subject.
5. Fit a standardized two-cluster K-Means model using training subjects only.
6. Train a class-balanced Random Forest using baseline predictors only.
7. Assign held-out trajectories to the training-defined clusters and evaluate baseline predictions at the subject level.
8. Save confusion-matrix, ROC, feature-importance, and trajectory figures under `reports/figures/`.

## Data

The project uses the **OASIS-2: Longitudinal MRI Data in Older Adults** dataset, obtained from the official OASIS Brain Project website:

<https://sites.wustl.edu/oasisbrains/home/oasis-2/>

Please consult the OASIS-2 website for the dataset description, access conditions, documentation, and citation requirements. This repository does not redistribute the dataset.

## Installation

Create and activate a virtual environment, then install the project and its development tools:

```shell
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Running the pipeline

Place the OASIS-2 workbook at:

```text
data/oasis_longitudinal.xlsx
```

Then run:

```shell
python src/main.py
```

Generated figures are written to `reports/figures/`.

## Interpretation and limitations

- Trajectory groups are data-driven clusters, not externally validated disease categories.
- The current outcome is a two-group `nWBV` trajectory summary, not a continuous estimate of individual decline rate.
- OASIS-2 is observational; the model identifies associations and predictive patterns, not causal effects.
- Performance is based on one held-out subject split and should be confirmed with repeated subject-level resampling or external validation.
- Larger and more clinically diverse cohorts are needed before considering clinical use.

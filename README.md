# DILI-PLUS

Core research implementation accompanying **When model complexity offers limited gains: a prediction-time-aware EHR benchmark for a rare biochemical hepatic injury proxy**.

This repository contains the model and evaluation code, not a clinical prediction service. The outcome is a biochemical acute hepatic injury proxy, **not clinically adjudicated drug-induced liver injury**. Results are from internal validation; external validation is needed before clinical use.

## Research scope

The benchmark compares logistic regression, XGBoost, TextCNN, BiLSTM, a multimodal Transformer, and a time-aware multimodal Transformer under a shared information boundary. Prediction is fixed 24 hours before the index time; model inputs must precede prediction. Patient-grouped partitions separate training, model selection, calibration and testing.

The released implementation includes raw precision-recall evaluation, temperature calibration, fixed-budget evaluation, component ablations, earlier-cutoff evaluation, and the final 10,000-replicate patient-cluster bootstrap with multiplicity adjustment. Increasing model complexity did not produce consistent gains in the evaluated cohort. This is not a claim that simpler models universally outperform deep learning.

## Code map

| Component | Location |
|---|---|
| Cohort, proxy definition and prediction-time checks | `src/diliplus/data/` |
| Patient-grouped partitions | `src/diliplus/splits.py` |
| Model definitions and ablation registry | `src/diliplus/models/` |
| Training and loss | `src/diliplus/training/` |
| Temperature scaling | `src/diliplus/calibration.py` |
| Metrics and earlier-cutoff evaluation | `src/diliplus/evaluation/` |
| Final paired inference and Holm adjustment | `src/diliplus/evaluation/p0_statistical_refinement.py` |
| Configuration and synthetic contract tests | `configs/`, `tests/` |

`reporting/formal_assets.py` is retained because the final statistical analysis depends on its source-resolution contract. Publication graphics, historical wrappers and exploratory scripts are outside this core release.

## Installation

Use Python 3.11 and a virtual environment. Install a PyTorch build suitable for your platform before the remaining dependencies:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install -e .
```

The research environment used PyTorch `2.11.0.dev20251216+cu128`; this historical nightly is recorded for provenance, not assumed to remain downloadable. Other dependency versions are recorded in `requirements.txt`. Alternative PyTorch builds have not been established as numerically identical to the research environment.

## Check the implementation without clinical data

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/check_public_release.py
```

The tests construct small synthetic arrays, records and temporary databases. They check implementation contracts; they do not reproduce the paper's private-cohort results.

## Running with authorized data

The private institutional database, preprocessing mappings, fitted vocabularies, model weights and row-level predictions are **not distributed**. The source adapters refer to the study's database schema; a new dataset requires an explicitly reviewed adapter and compatible inputs. Do not run against a production database without authorization.

Set the database path in a local configuration derived from `configs/default.yaml`. It deliberately points to a nonexistent private-data location by default and requests read-only database access. Prepare institution-specific medication mappings locally before vocabulary construction.

```bash
python pipelines/01_build_dataset.py --config configs/local.yaml
python pipelines/02_train_models.py --config configs/local.yaml --run-id YOUR_RUN_ID
python pipelines/03_evaluate_models.py --config configs/local.yaml --run-id YOUR_RUN_ID --probability-mode raw
```

The primary analysis reports raw AUPRC; calibrated probabilities serve probability-quality and alert-budget analyses. The final statistical refinement is a separate, frozen-output step:

```bash
python pipelines/07_run_p0_statistical_refinement.py --config configs/local.yaml --workers 8
```

That step expects the original private run layout and its provenance manifest; it is not a turnkey command for a newly named run. The general training configuration retains the original 1,000-replicate evaluation setting, while the final P0 module explicitly uses 10,000 replicates. Do not conflate preliminary and final uncertainty estimates.

## Data and model availability

Only source code, configuration templates and synthetic tests are included in the current release. No patient-level data, trained parameters, private vocabularies, cohort memberships, prediction tables or case-level plots are supplied. Generated checkpoints can contain split information as well as fitted parameters and must remain private. See [PUBLICATION_POLICY.md](PUBLICATION_POLICY.md).

Public source access does not grant access to institutional patient data. This code alone cannot reproduce private-cohort numerical results. Any data request is subject to institutional permissions and applicable ethics and privacy requirements.

## Version provenance

The original experimental source baseline is commit `bc40825`. This streamlined release retains its core implementation, includes the subsequent local P0 statistical-analysis source used for the final manuscript, and removes non-core files from the current tree. A NumPy 1.x/2.x compatibility alias preserves the same trapezoidal integration rule. It is not a new training run. The existing Git history has **not** been rewritten; historical files remain accessible through older commits.

Use the full commit SHA of the reviewed checkout when recording a code version. No acceptance, publication DOI, or clinical-validation status is implied by this repository.

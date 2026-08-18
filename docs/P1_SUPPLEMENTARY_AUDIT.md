# P1 supplementary calibration, utility, subgroup, and observation-process audit

Date: 2026-08-18
Status: **PASS; post-hoc descriptive audit, not part of formal model selection or ranking**

## 1. Purpose and interpretation boundary

This P1 analysis adds the reporting elements that were reasonable in the external-review outlines without
changing the frozen Code-10 benchmark. It answers three limited questions:

1. What do the calibrated probabilities look like across the observed risk distribution?
2. How do retrospective net-benefit curves behave over a prespecified 0.5%--5.0% risk-threshold range?
3. How much of the prediction task can be recovered from observation intensity alone, and how heterogeneous
   are selected model results across recorded sex and observation-window strata?

These analyses do **not** establish clinical benefit, fairness, transportability, causality, or a new winner.
The six-model Code-10 results remain the formal benchmark. Recorded sex is an initial descriptive
heterogeneity audit; observation-window strata are healthcare-process strata, not fairness attributes.

## 2. Frozen inputs and cohort guards

- Formal run: `code10_formal_128d4h_seed0`.
- Cohort: 44,631 encounters, 44,611 patient clusters, 315 AHI-proxy events.
- Every frozen out-of-fold prediction is checked against `dataset_index`, `encounter_id`, `patient_id`, and
  the canonical label before analysis.
- The observation-process feature frame reuses the strict Code-08 Table 1 queries and one-to-one encounter
  join audits. It does not infer table interfaces from names or perform an unverified join.
- All post-hoc uncertainty intervals use 1,000 patient-cluster bootstrap replicates.

## 3. Observation-process baseline

The baseline is unweighted logistic regression trained and temperature-calibrated within the same five
patient-grouped training/selection/calibration/test roles used by Code-10. Its inputs are deliberately limited
to log observation duration, log medication/laboratory/diagnosis event counts, log modality event densities,
and laboratory/diagnosis availability indicators. It receives no clinical token identity, laboratory value,
demographic variable, or target measurement.

Pooled calibrated results:

| Metric | Estimate |
|---|---:|
| AUROC | 0.8434 (95% CI 0.8213--0.8649) |
| AUPRC | 0.0832 (95% CI 0.0634--0.1097) |
| AUPRC lift over prevalence | 11.8 |
| Brier score | 0.006790 |
| NLL | 0.03469 |
| Calibration slope | 0.986 |
| O:E ratio | 1.003 |
| Top-1% events captured | 70/315 (447 alerts; sensitivity 0.222) |

The baseline is therefore a meaningful measurement-process audit. It must not be described as a deployable
clinical model. Coefficients are descriptive only because duration, counts, densities, and availability are
structurally dependent and were not designed for causal interpretation.

## 4. Calibration and decision curves

`Figure S1A` uses ten equal-frequency risk groups and shows patient-cluster bootstrap intervals for the
observed event proportion under fixed group assignments. The highest-risk group illustrates residual
overprediction by the deep models: mean predicted risks range from 5.86% to 10.19%, while observed risks are
approximately 4.03%--4.23%. Logistic regression predicts 3.91% versus 4.73% observed; XGBoost predicts 5.04%
versus 4.53% observed. The plot is descriptive and does not replace continuous calibration statistics.

`Figure S1B` reuses the frozen Code-10 decision-curve outputs at 19 thresholds from 0.5% to 5.0%. Logistic
regression has the highest net benefit at 11 thresholds and XGBoost at 8. Mean net benefit per 100 encounters
over this grid is 0.256 for logistic regression, 0.253 for XGBoost, 0.220 for TextCNN, 0.202 for BiLSTM,
0.190 for the multimodal Transformer baseline, and 0.184 for TA-MMT. These are retrospective threshold
trade-offs under assumed utilities; no treatment effect, realised workflow benefit, or recommended threshold
is claimed.

## 5. Subgroup and process-stratum audit

Recorded-sex estimates use 19,402 female encounters/115 events and 24,048 male encounters/199 events;
1,181 encounters with unavailable sex are not assigned to either group. Intervals overlap substantially, so
the analysis does not support a claim of either fairness or disparity.

Observation-window tertiles are defined by the formal cohort distribution: short <=38.001 h, intermediate
38.001--98.745 h, and long >98.745 h. Each stratum has 14,877 encounters, with 31, 63, and 221 events,
respectively. The process-only AUPRC is 0.010, 0.032, and 0.107 across those strata. Logistic regression,
XGBoost, and TA-MMT also have materially lower within-stratum AUPRC in the short/intermediate windows than
in the long window. Because prevalence changes strongly across strata, Table S1 reports event counts and
AUPRC lift as well as raw AUPRC.

The correct interpretation is that prediction and apparent model performance are entangled with how long and
how intensively an encounter was observed. This supports the manuscript's measurement-process narrative, but
does not identify why a patient was measured more often.

## 6. Reproducible execution and Windows renderer isolation

Run from the repository root in the ML311 environment:

```powershell
python pipelines/06_build_p1_supplementary.py
```

The analysis stage uses ML311. On this workstation, importing/rendering Matplotlib after the ML analysis can
terminate the process with Windows native exception `0xC06D007F`, the same popup observed during earlier
figure runs. To preserve the analysis and avoid changing statistical code or saved results, the pipeline runs
analysis and rendering as isolated child processes; plotting/table/manifest generation uses the existing base
Anaconda Agg runtime. The final pipeline completed all five stages with `PASS`.

## 7. Outputs and paper mapping

- Aggregate manifest: `manifests/code12_p1_supplementary.json`.
- Metrics: `reports/runs/code12_p1_supplementary/metrics/`.
- Figure S1: `figures/Fig_S1_Calibration_Decision_Curves.pdf`.
- Figure S2: `figures/Fig_S2_Subgroup_Process_Audit.pdf`.
- Table S1: `reports/paper_assets/Table_S1_Subgroup_Process_Audit.tex` and `.csv`.
- Paper copies: `D:\PaperWorks\DILI-PLUS\Figures\Fig_S1_Calibration_Decision_Curves.pdf`,
  `D:\PaperWorks\DILI-PLUS\Figures\Fig_S2_Subgroup_Process_Audit.pdf`, and
  `D:\PaperWorks\DILI-PLUS\Tables\Table_S1_Subgroup_Process_Audit.tex`.
- Manuscript labels: `fig:supplementary_calibration_dca`, `fig:supplementary_subgroup_process`, and
  `tab:supplementary_subgroup_process`.

No new bibliography entry was introduced for P1, and `references.bib` was not edited. The paper compiled to
16 pages with no undefined reference or citation warning; pages 13--14 containing the new assets passed visual
inspection for clipping, overlap, and legibility. The final full repository suite passed **73/73** unit tests.

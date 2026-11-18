# Replicating the paper

Code, configuration and pooled results behind *Site-Level EHR Documentation Variation Impacts Model Inference and
Generalizability in Multi-Center Research*. The `das` package in `src/` holds the DAS method itself. This folder holds
the study pipeline that applied it to eICU and MIMIC-IV.

## Data

The analyses use the eICU Collaborative Research Database v2.0, MIMIC-IV v2.2 and MIMIC-IV-Note v2.2. All three are
available to credentialed users at https://physionet.org. No patient-level or hospital-level data are in this
repository. The scripts write stay-level files to the `rowlevel` folder named in `configs/external_v7.yaml`. Keep that
folder private.

## Setup

1. Convert the three datasets to parquet and set the `/path/to/` entries in `configs/external_v7.yaml` and `configs/external.yaml` (the two differ only in the term-list file they name).
2. Run every command from this folder with the run configuration set:

```bash
export EHRDOC_CONFIG=configs/external_v7.yaml EHRDOC_OUT=v7_2026-11-16
pip install duckdb pandas pyarrow pyyaml scikit-learn joblib xgboost statsmodels scipy
```

Results are written to `results_v7/` and `results_v7/v7_2026-11-16/`.

## Order of the steps

| Step | Script | Output used in the paper |
|---|---|---|
| Cohort and stay-level evidence | `01_build_flags.py`, `08_extract_history.py` (MIMIC-IV past history), `03_build_features.py`, `11_charlson_features.py` | cohort (Supplementary Figure S1) |
| Condition definitions | `68_eicu_hicl_list.py`, `80_dx_codes_first.py`, `76_build_flags_v7.py` with `configs/definitions_v7.yaml` | evidence flags (Supplementary Data) |
| Severity score and model inputs | `81_apache_from_inputs.py`, `09_apache_features.py` | APACHE IVa, official or rebuilt |
| ICU episode | `99_eicu_icu_episode.py` | outcome definitions |
| Training-site selection | `38_iecv.py split`, then `loop` (development, both samples), `pool` (screen and locked measure), `loop --half held_out`, `pool`, `mimic --measure mean` (both samples), all with `--spec v6 --out v7_2026-11-16` | Figures 4 and 5, Supplementary Tables S6, S8 to S12 |
| DAS against the membership model | `39_das_vs_membership.py v7_2026-11-16` | Supplementary Table S13 |
| Documentation variation and outcome models | `87_mimic_relatedness_v7.py`, `40_das_variation.py`, `43_partA_effects.py`, `51_partA_readmission.py`, `60_das_pair_quantiles.py`, `62_chance_beyond_noise.py`, `45_flow_counts.py`, `41_table1.py`, `42_venn_data.py` | Figures 1 and 2, Tables 1 to 3, Supplementary Tables S1 to S5 |
| Selection ceiling | `49_ceiling_iecv.py` (`--sample all`, `--sample with_medication`, `--pool-only`, `--half held_out`), `59_ceiling_mimic.py` | Figure 3, Supplementary Table S7 |
| Random-half check | `38_iecv.py loop/pool/mimic` with `--temporal --split-mode random`, for `EHRDOC_SPLIT_REP` 0 to 4 (output folders `random_half`, `random_half_r1` to `_r4`), then `93_random_half_combine.py` | Supplementary Table S14 |
| Size-matched sets | `92_membership_table.py --half held_out`, `58_size_matched_iecv.py run` (held_out, mimic), `58_size_matched_iecv.py pool` | sensitivity analysis |
| Other checks | `64_das_all_vs_modeled_stays.py`, `65_das_with_dropped_hospitals.py`, `110_discussion_numbers.py` | Supplementary Methods, Limitations |

Sensitivity analyses reuse `38_iecv.py` with options: `--k 5` or `--k 20` (set size), `--harmonized` (same predictors
as MIMIC-IV), `EHRDOC_APACHE_OFFICIAL=1` (official scores only), and `EHRDOC_VARIANT=origmeds|nobb|noloop|strict`
(medication lists, after rebuilding the flags with `68` and `76` under the same variable). Held-out results are pooled
over targets with at least 10 events for the outcome. The development screen pools all targets.

`check_das_against_paper.py` recomputes the DAS distribution with the `das` package from the stay-level flags. It
matches Supplementary Table S1 to the decimal.

The scripts keep options from earlier stages of the project. The paper uses only those listed above.

## Results

`results/` holds the pooled and summary results behind the paper's tables and figures, as CSV. Columns that list
hospital identifiers were removed, and no file has a row per hospital.

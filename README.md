# Final World Cup Modeling Pipeline
Authors
- Ahmed Alamin, Taha Ahmad, and Illir Bejleri


This directory is an isolated final-version pipeline. It contains the data cache, source code, model artifacts, report source, style file, and bibliography needed to rerun and submit the final project without depending on `papers/` or the older project scripts.

## Directory Structure

- `run_final_pipeline.py`: thin entry point.
- `src/pipeline.py`: main reproducible training/evaluation/simulation pipeline.
- `src/models/`: model-family organization notes for MLR, RF, NN, and Poisson variants.
- `src/data/`: feature-engineering notes.
- `src/tournament/`: tournament/draw-policy notes.
- `config/2026_groups.csv`: configured 2026 group field.
- `data/raw_matches.pkl`: raw cached matches used to regenerate all outputs.
- `data/international_results_2022_2026.csv`: supplemental post-cache international results used only for the final 2026 forecast.
- `outputs/`: generated features, metrics, model artifacts, figures, and 2026 forecasts.
- `report/`: LaTeX report, local `neurips_2025.sty`, local `bibliography.bib`, and compiled PDF.

## What Changed

- Trains supervised models on all available pre-tournament international matches, not only World Cup rows.
- Keeps World Cup rows higher-weighted so the model still targets tournament behavior.
- Compares several versions of MLR, RF, NN, and score-based Poisson/Dixon-Coles-style models.
- Uses walk-forward World Cup validation: train before every feasible World Cup checkpoint from 1938 through 2018 and validate on that next World Cup.
- Holds out 2022 as the final test tournament.
- Restores and reports the original 18 engineered feature families, including per-model feature usage counts.
- Explains how the expanded full models create 55 direct feature columns from the 18 original families.
- Explicitly zeroes draw probability for knockout matches, including the 2026 round of 32.
- Updates Elo, rolling form, final outcome models, and the final Poisson score layer with 2022-2026 supplemental data only for the 2026 winner forecast.
- Uses a classifier-calibrated score simulator for the primary 2026 forecast: the best held-out log-loss NN supplies W/D/L probabilities, while the Poisson score grid is reweighted to those probabilities before sampling scorelines.
- Tunes an ensemble using validation folds only.
- Simulates the 2026 World Cup from the configured group field in `config/2026_groups.csv`.

## Run

From this directory:

```bash
python3 run_final_pipeline.py
```

Outputs are written to `outputs/`.

## Main Outputs

- `features_all_matches.csv`: chronological supervised feature table for World Cup and international matches.
- `features_all_matches_2026_augmented.csv`: base feature table plus the 2022-2026 supplement, used only for the final 2026 forecast.
- `training_2026_augmented_summary.csv`: row counts and date ranges for the 2026-only augmented training table.
- `model_comparison.csv` / `model_comparison.md`: walk-forward CV and 2022 test metrics.
- `fold_metrics.csv`: per-model, per-World-Cup walk-forward accuracy/log-loss/draw-recall rows.
- `feature_glossary_18.csv`: definitions for the original 18 engineered feature families.
- `feature_blocks_55.csv`: how the 55-column expanded model is constructed from feature blocks.
- `feature_usage.csv` / `feature_usage_matrix.csv`: how many of the 18 feature families each model uses.
- `confusions.txt`: 2022 confusion matrices.
- `ensemble_weights.json`: validation-selected ensemble members and weights.
- `figures/`: report figures for dataset composition, feature usage, model metrics, fold metrics, and 2026 probabilities.
- `forecast_diagnostics.md`: knockout draw policy and Spain-vs-market diagnostic note.
- `team_2026_model_inputs.csv`: model-side team inputs used to diagnose 2026 forecast probabilities.
- `champion_probabilities_2026.csv`: Monte Carlo champion probabilities.
- `forecast_variant_summary.csv`: side-by-side top-team and Spain probabilities for Poisson-only, NN-hybrid, accuracy-NN-hybrid, and ensemble-hybrid forecast variants.
- `forecast_variants/`: full champion, stage, group-advancement, and deterministic-path outputs for each 2026 simulator variant.
- `team_stage_probabilities_2026.csv`: per-team probabilities for reaching each stage.
- `group_advancement_2026.csv`: group-stage advancement probabilities.
- `most_likely_2026_path.csv`: one deterministic projection from the best score model.
- `report_summary.md`: concise paper-ready findings.

## Source Notes

The raw historical match cache is bundled under `data/raw_matches.pkl`, originally built from `openfootball/worldcup` and `openfootball/internationals`.

The 2026-only supplement is bundled under `data/international_results_2022_2026.csv`, extracted from `martj42/international_results` on April 30, 2026. It covers matches from 2022-01-01 through 2026-03-31 after filtering out future tournament dates.

The 2026 group field is defined in `config/2026_groups.csv`. Team names are normalized to the naming used in the historical cache where needed.

The bundled historical cache currently ends with the 2022 World Cup and internationals through 2021. For a stronger production-grade 2026 forecast, refresh the raw cache with 2022-2026 internationals before rerunning.

# Final Pipeline Report Summary

## Methodological Upgrade

The final pipeline trains on every available pre-tournament supervised match: World Cup rows plus international rows. World Cup examples receive higher sample weight, while all rows receive time-decay weighting so recent matches matter more.

Validation remains leakage-resistant: hyperparameters are selected with every feasible World Cup checkpoint before 2022. Each checkpoint trains on matches before that tournament and validates on that next World Cup. The 2022 World Cup is the held-out final test.

## Best 2022 Model

Best by 2022 log loss: **NN-original-18** (NN) with test LL **0.9321** and accuracy **0.5156**.

Best by 2022 accuracy: **NN-expanded-no-underdog** (NN) with accuracy **0.6094** and test LL **0.9363**.

## Model Comparison

| Model                            | Family   |   #Feat |   Original18 Used | Best Config                                                                                                                                       |   Mean CV LL |   Std CV LL |   Mean CV Acc |   Test LL |   Test Acc |   Draw Recall |   CV-Test Gap |
|:---------------------------------|:---------|--------:|------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|------------:|--------------:|----------:|-----------:|--------------:|--------------:|
| NN-original-18                   | NN       |      18 |                18 | {"hidden1": 24, "hidden2": 12, "dropout": 0.3, "lr": 0.004, "weight_decay": 0.001}                                                                |       0.9657 |      0.0943 |        0.5133 |    0.9321 |     0.5156 |           0   |       -0.0336 |
| NN-full-regularized              | NN       |      55 |                18 | {"hidden1": 48, "hidden2": 16, "dropout": 0.5, "lr": 0.002, "weight_decay": 0.003}                                                                |       0.963  |      0.0991 |        0.4832 |    0.933  |     0.5312 |           0   |       -0.0301 |
| MLR-original-18                  | MLR      |      18 |                18 | {"C": 0.01}                                                                                                                                       |       1.0183 |      0.1296 |        0.4485 |    0.934  |     0.4844 |           0   |       -0.0843 |
| Validation-tuned family ensemble | Ensemble |      70 |                18 | {"members": ["Poisson-independent", "NN-full-regularized", "RF-full", "MLR-original-baseline-5"], "weights": [0.6000000000000001, 0.4, 0.0, 0.0]} |       0.936  |      0.0848 |        0.5553 |    0.9343 |     0.5469 |           0   |       -0.0017 |
| NN-strength                      | NN       |       8 |                 6 | {"hidden1": 24, "hidden2": 12, "dropout": 0.25, "lr": 0.005, "weight_decay": 0.001}                                                               |       0.9739 |      0.0994 |        0.4566 |    0.936  |     0.5    |           0   |       -0.0379 |
| NN-expanded-no-underdog          | NN       |      53 |                16 | {"hidden1": 32, "hidden2": 16, "dropout": 0.4, "lr": 0.003, "weight_decay": 0.002}                                                                |       0.9646 |      0.0866 |        0.5188 |    0.9363 |     0.6094 |           0.3 |       -0.0282 |
| NN-recent-no-elo                 | NN       |      28 |                10 | {"hidden1": 24, "hidden2": 12, "dropout": 0.35, "lr": 0.003, "weight_decay": 0.002}                                                               |       0.9906 |      0.0707 |        0.4712 |    0.9418 |     0.5    |           0   |       -0.0488 |
| RF-original-18                   | RF       |      18 |                18 | {"n_estimators": 350, "max_depth": 8, "min_samples_leaf": 10}                                                                                     |       0.9991 |      0.1155 |        0.4419 |    0.9475 |     0.5    |           0   |       -0.0516 |
| Poisson-independent              | Poisson  |      13 |                11 | {"alpha": 0.1, "draw_boost": 0.0}                                                                                                                 |       0.9526 |      0.1036 |        0.5387 |    0.9479 |     0.5312 |           0   |       -0.0047 |
| MLR-full                         | MLR      |      55 |                18 | {"C": 0.01}                                                                                                                                       |       1.0409 |      0.1607 |        0.4496 |    0.9519 |     0.5312 |           0.1 |       -0.089  |
| Poisson-draw-inflated            | Poisson  |      13 |                11 | {"alpha": 0.1, "draw_boost": 0.15}                                                                                                                |       0.9538 |      0.1012 |        0.5367 |    0.9527 |     0.5312 |           0   |       -0.0011 |
| MLR-recent-no-elo                | MLR      |      28 |                10 | {"C": 0.01}                                                                                                                                       |       1.0337 |      0.0941 |        0.4156 |    0.9546 |     0.4844 |           0.2 |       -0.0791 |
| MLR-original-matchup             | MLR      |       6 |                 6 | {"C": 0.01}                                                                                                                                       |       1.0344 |      0.1379 |        0.4261 |    0.9559 |     0.5312 |           0   |       -0.0785 |
| RF-full                          | RF       |      55 |                18 | {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 10}                                                                                    |       0.9926 |      0.1043 |        0.4507 |    0.9562 |     0.5625 |           0   |       -0.0364 |
| RF-recent-no-elo                 | RF       |      28 |                10 | {"n_estimators": 500, "max_depth": 9, "min_samples_leaf": 10}                                                                                     |       1.0073 |      0.092  |        0.4475 |    0.9602 |     0.4844 |           0.1 |       -0.0471 |
| MLR-strength                     | MLR      |       8 |                 6 | {"C": 0.01}                                                                                                                                       |       1.0244 |      0.1207 |        0.4297 |    0.9613 |     0.5312 |           0.1 |       -0.0631 |
| MLR-original-form                | MLR      |       8 |                 8 | {"C": 0.01}                                                                                                                                       |       1.0193 |      0.0679 |        0.4455 |    0.9637 |     0.4844 |           0   |       -0.0556 |
| MLR-original-baseline-5          | MLR      |       5 |                 6 | {"C": 0.01}                                                                                                                                       |       1.0002 |      0.0835 |        0.4442 |    0.9651 |     0.5469 |           0.1 |       -0.0351 |
| RF-strength                      | RF       |       8 |                 6 | {"n_estimators": 250, "max_depth": 5, "min_samples_leaf": 20}                                                                                     |       1.0115 |      0.1128 |        0.4027 |    0.9702 |     0.5312 |           0   |       -0.0413 |

## 2026 Forecast

The primary 2026 simulation is a classifier-calibrated score simulator: the best held-out log-loss NN supplies W/D/L probabilities, while the Poisson score layer is reweighted to those probabilities before sampling scorelines for group tiebreakers and knockout advancement. Model selection and 2022 testing use only the historical cache; the final 2026-only fit additionally uses the bundled 2022-2026 supplemental international results to update Elo, form, the outcome model, and the score layer.

Top champion probabilities:

| team        |   champion_probability |
|:------------|-----------------------:|
| Spain       |                 0.1486 |
| Argentina   |                 0.1256 |
| France      |                 0.0906 |
| Turkiye     |                 0.0654 |
| England     |                 0.0652 |
| Germany     |                 0.059  |
| Netherlands |                 0.0584 |
| Brazil      |                 0.0568 |
| Morocco     |                 0.0448 |
| Portugal    |                 0.0374 |
| Japan       |                 0.0272 |
| Colombia    |                 0.0246 |

Forecast variants:

| Forecast                                             | Top Team   |   Top Champion Probability |   Spain Champion Probability |
|:-----------------------------------------------------|:-----------|---------------------------:|-----------------------------:|
| Poisson-only score baseline                          | Spain      |                     0.251  |                       0.251  |
| NN-original-18 calibrated score simulator            | Spain      |                     0.1486 |                       0.1486 |
| NN-expanded-no-underdog calibrated score simulator   | Spain      |                     0.1476 |                       0.1476 |
| Validation-tuned ensemble calibrated score simulator | Spain      |                     0.1908 |                       0.1908 |

## Why Models Succeed Or Fail

- Poisson/Dixon-Coles-style models are useful score generators because football is low scoring, but direct outcome models can provide better W/D/L probabilities.
- MLR tends to be stable because the signal is mostly monotonic team strength and recent form; it can underfit nonlinear effects but avoids extreme probabilities.
- RF captures interactions but can overfit tournament idiosyncrasies when World Cup validation sets are small.
- NN models have the highest capacity and often improve CV on some folds, but they are fragile with only a few hundred World Cup validation matches unless heavily regularized.
- The ensemble is useful only when it is tuned on validation folds and includes genuinely different model families; test-set-selected ensembles should not be reported as final evidence.

## Bracket Caveat

The simulator uses the official 12 groups and top-two-plus-eight-best-thirds advancement rule. Because the full FIFA round-of-32 pairing table is awkward to maintain manually, the knockout bracket is seeded high-vs-low by simulated group performance and pre-tournament strength. This is transparent and reproducible, but it is an approximation of the official bracket path.
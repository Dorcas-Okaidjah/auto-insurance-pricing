# Results summary

Data: 677,983 motor policies, 358,341 policy-years, 26,373 claims. Train 508,418 / test 169,565 (random split). Individual claims capped at EUR 31,241 (the 99.5% quantile); the excess is spread back as a 1.347x load. Test-set metrics use capped claims x load, so a single very large claim cannot decide a segment's result.

## Model accuracy (test set)

| Metric | GLM (champion) | GBM (challenger) | Better |
|---|---|---|---|
| Poisson deviance, frequency | 0.4559 | 0.4498 | lower |
| Gamma deviance, severity | 1.1820 | 1.1818 | lower |
| Gini, frequency | 0.2913 | 0.3172 | higher |
| Gini, pure premium | 0.3196 | 0.3453 | higher |
| Predicted / actual losses (capped + load) | 0.9641 | 0.9658 | closer to 1 |
| Predicted / actual losses (raw) | 0.9458 | 0.9475 | closer to 1 |

Gini gain from GBM: **+0.0258** (95% bootstrap interval +0.0135 to +0.0371).

## Business case

Premium = expected loss / 70% target loss ratio. A customer switches insurer when the rival quotes at least 5% less. Losses are the test policies' own claims (capped + load).

| Scenario | Challenger wins | Challenger loss ratio | Incumbent loss ratio |
|---|---|---|---|
| We price with GBM, rival uses GLM | 45.4% | **73.8%** | 86.7% |
| We price with GLM, rival uses GBM | 37.1% | 94.1% | **73.1%** |
| We price with capped GBM (+/-15%), rival uses GLM | 45.4% | 66.9% | 86.7% |

Switching the existing book from GLM to GBM prices: 33.7% of customers move less than 10%; 27.0% move more than 25%. Capping year-one changes at +/-15% keeps 59.6% of the Gini gain.

## Actual vs expected by driver age (test set)

| Driver age | GLM A/E | GBM A/E |
|---|---|---|
| 18-24 | 1.132 | 1.124 |
| 25-39 | 0.976 | 0.987 |
| 40-59 | 1.023 | 1.014 |
| 60+ | 1.146 | 1.141 |

Charts: `lorenz_curve.png`, `lift_by_decile.png`, `double_lift.png`, `partial_dependence.png`, `feature_importance.png`, `dislocation.png`.

"""Pricing-model metrics an insurance pricing team actually uses.

* Deviance: statistical fit (lower is better).
* Gini on the ordered Lorenz curve: how well the model ranks risks, i.e. how
  well it separates good drivers from bad ones (higher is better).
* Lift / double-lift charts: does the predicted loss cost track actual loss
  cost across risk deciles, and when two models disagree, which one is right?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance


# np.trapezoid is numpy >= 2.0; np.trapz is the older name
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def poisson_deviance(df, freq_pred) -> float:
    return mean_poisson_deviance(df["ClaimNb"] / df["Exposure"], freq_pred,
                                 sample_weight=df["Exposure"])


def gamma_deviance(df, sev_pred) -> float:
    m = df["ClaimNb"] > 0
    y = df.loc[m, "ClaimAmountCapped"] / df.loc[m, "ClaimNb"]
    return mean_gamma_deviance(y, np.asarray(sev_pred)[m.values],
                               sample_weight=df.loc[m, "ClaimNb"])


def lorenz(pred, actual_loss, exposure):
    """Ordered Lorenz curve: policies sorted from safest to riskiest by the model."""
    order = np.argsort(pred)
    x = np.concatenate([[0], np.cumsum(np.asarray(exposure)[order])])
    y = np.concatenate([[0], np.cumsum(np.asarray(actual_loss)[order])])
    return x / x[-1], y / y[-1]


def gini(pred, actual_loss, exposure) -> float:
    x, y = lorenz(pred, actual_loss, exposure)
    return float(1 - 2 * _trapezoid(y, x))


def bootstrap_gini_diff(pred_a, pred_b, loss, expo, n=200, seed=0):
    """95% interval for Gini(b) - Gini(a) by resampling policies."""
    rng = np.random.default_rng(seed)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    loss, expo = np.asarray(loss), np.asarray(expo)
    k = len(loss)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        diffs.append(gini(pred_b[i], loss[i], expo[i]) - gini(pred_a[i], loss[i], expo[i]))
    return np.percentile(diffs, [2.5, 97.5])


def _weighted_bins(sort_key, exposure, n_bins):
    order = np.argsort(sort_key, kind="stable")
    cum = np.cumsum(np.asarray(exposure)[order]) / np.sum(exposure)
    bins = np.empty(len(order), dtype=int)
    bins[order] = np.minimum((cum * n_bins).astype(int), n_bins - 1) + 1
    return bins


def lift_table(df, pp_pred, n_bins=10) -> pd.DataFrame:
    """Equal-exposure risk deciles: predicted vs actual loss cost per policy-year."""
    t = pd.DataFrame({"bin": _weighted_bins(pp_pred, df["Exposure"], n_bins),
                      "expo": df["Exposure"].values,
                      "pred": np.asarray(pp_pred) * df["Exposure"].values,
                      "act": df["ClaimAmount"].values})
    g = t.groupby("bin").sum()
    return pd.DataFrame({"exposure": g["expo"],
                         "predicted_loss_cost": g["pred"] / g["expo"],
                         "actual_loss_cost": g["act"] / g["expo"]})


def double_lift(df, pp_a, pp_b, n_bins=10) -> pd.DataFrame:
    """Sort by the ratio of the two models' predictions.

    Bin 1 is where model B is cheapest relative to A; bin 10 where B is most
    expensive. If B is the better model, actual losses follow B's line.
    All three columns are indexed to the book average (1.00 = average).
    """
    pp_a, pp_b = np.asarray(pp_a), np.asarray(pp_b)
    e = df["Exposure"].values
    t = pd.DataFrame({"bin": _weighted_bins(pp_b / pp_a, e, n_bins), "e": e,
                      "a": pp_a * e, "b": pp_b * e, "act": df["ClaimAmount"].values})
    g = t.groupby("bin").sum()
    out = pd.DataFrame({c: (g[c] / g["e"]) / (t[c].sum() / t["e"].sum())
                        for c in ["a", "b", "act"]})
    out.columns = ["model_a", "model_b", "actual"]
    out["exposure"] = g["e"]
    return out


def actual_vs_expected(df, pp_pred, by: str) -> pd.DataFrame:
    t = pd.DataFrame({by: df[by].values, "e": df["Exposure"].values,
                      "pred": np.asarray(pp_pred) * df["Exposure"].values,
                      "act": df["ClaimAmount"].values})
    g = t.groupby(by, observed=True).sum()
    g["A/E"] = g["act"] / g["pred"]
    return g

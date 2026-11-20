"""Turn model accuracy into a business decision.

Three questions a pricing leader asks before a new model ships:

1. What is it worth?  Market simulation: an insurer that prices with the better
   model takes the customers its rival overcharges and leaves behind the ones
   its rival undercharges (adverse selection). We measure the loss ratio each
   side ends up with on held-out policies, using their real claims.
2. Who gets hurt?  Dislocation: how many existing customers would see a big
   price change if we switched models.
3. Can we ship it safely?  Rate capping: limit any one customer's change to
   +/- RATE_CAP in year one and measure how much of the gain survives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .metrics import gini


def premium(pp: np.ndarray) -> np.ndarray:
    """Technical premium for a full policy-year: expected loss / target loss ratio."""
    return np.asarray(pp) / C.TARGET_LOSS_RATIO


def market_simulation(df, pp_incumbent, pp_challenger, switch_threshold=0.05):
    """All customers start with the incumbent. Each one moves to the challenger
    if the challenger quotes at least `switch_threshold` cheaper.

    Premiums and losses are on the policies' actual exposure.
    """
    e = df["Exposure"].values
    loss = df["ClaimAmount"].values
    p_inc = premium(pp_incumbent) * e
    p_ch = premium(pp_challenger) * e
    moves = p_ch < p_inc * (1 - switch_threshold)

    def side(mask, prem):
        return {"policies": int(mask.sum()),
                "share": float(mask.mean()),
                "premium": float(prem[mask].sum()),
                "losses": float(loss[mask].sum()),
                "loss_ratio": float(loss[mask].sum() / prem[mask].sum())}

    return {"challenger": side(moves, p_ch), "incumbent": side(~moves, p_inc),
            "switch_threshold": switch_threshold}


def dislocation(pp_current, pp_new) -> pd.DataFrame:
    change = np.asarray(pp_new) / np.asarray(pp_current) - 1
    edges = [-np.inf] + C.DISLOCATION_BANDS + [np.inf]
    labels = ["< -25%", "-25% to -10%", "-10% to +10%", "+10% to +25%", "> +25%"]
    band = pd.cut(change, edges, labels=labels)
    t = pd.Series(band).value_counts(normalize=True).reindex(labels)
    return t.rename("share_of_policies").to_frame()


def capped_premium(pp_current, pp_new, cap=C.RATE_CAP):
    ratio = np.clip(np.asarray(pp_new) / np.asarray(pp_current), 1 - cap, 1 + cap)
    return np.asarray(pp_current) * ratio


def run(df, pp_glm, pp_gbm) -> dict:
    loss, e = df["ClaimAmount"].values, df["Exposure"].values
    pp_cap = capped_premium(pp_glm, pp_gbm)
    g_glm, g_gbm, g_cap = (gini(p, loss, e) for p in (pp_glm, pp_gbm, pp_cap))
    return {
        "gbm_vs_glm": market_simulation(df, pp_glm, pp_gbm),
        "glm_vs_gbm": market_simulation(df, pp_gbm, pp_glm),
        "dislocation": dislocation(pp_glm, pp_gbm),
        "gini": {"glm": g_glm, "gbm": g_gbm, "gbm_capped": g_cap,
                 "gain_kept_by_cap": (g_cap - g_glm) / (g_gbm - g_glm)
                 if g_gbm != g_glm else float("nan")},
        "capped_vs_glm": market_simulation(df, pp_glm, pp_cap),
    }

"""Explainability: what drives each model's price, in terms an underwriter can check."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

PDP_GRIDS = {
    "DrivAge": np.arange(18, 91, 2),
    "BonusMalus": np.arange(50, 151, 5),
    "VehAge": np.arange(0, 21, 1),
    "Density": np.unique(np.geomspace(2, 27000, 30).astype(int)),
}


def partial_dependence(model, df: pd.DataFrame, feature: str, grid, n=20000):
    """Average predicted claim frequency when every policy is given `feature`=v."""
    sample = df.sample(min(n, len(df)), random_state=C.RANDOM_STATE)
    out = []
    for v in grid:
        s = sample.copy()
        s[feature] = v
        out.append(model.predict_freq(s).mean())
    return np.array(out)


def gbm_importance(gbm_model) -> pd.DataFrame:
    rows = []
    for part, booster in [("frequency", gbm_model.freq), ("severity", gbm_model.sev)]:
        gain = booster.feature_importance("gain")
        rows.append(pd.Series(gain / gain.sum(), index=booster.feature_name(), name=part))
    return pd.concat(rows, axis=1).sort_values("frequency", ascending=False)


def glm_rating_table(glm_model, df: pd.DataFrame) -> pd.DataFrame:
    """Frequency relativities per rating factor level, the way rates are filed.

    Within each factor, the level with the most exposure is the base (1.00).
    """
    from .models import glm_frame

    pipe = glm_model.freq
    names = pipe[0].get_feature_names_out()
    coefs = pipe[-1].coef_
    X = glm_frame(df)
    rows = []
    for name, b in zip(names, coefs):
        if not name.startswith("cat__"):
            continue
        rest = name[len("cat__"):]
        factor = next(f for f in sorted(X.columns, key=len, reverse=True)
                      if rest.startswith(f + "_"))
        level = rest[len(factor) + 1:]
        rows.append((factor, level, b))
    t = pd.DataFrame(rows, columns=["factor", "level", "coef"])
    expo = df.assign(**{c: X[c] for c in X.columns})
    w = []
    for f, lvl in zip(t["factor"], t["level"]):
        w.append(expo.loc[expo[f].astype(str) == lvl, "Exposure"].sum())
    t["exposure"] = w
    base = t.loc[t.groupby("factor")["exposure"].idxmax(), ["factor", "coef"]]
    t = t.merge(base, on="factor", suffixes=("", "_base"))
    t["relativity"] = np.exp(t["coef"] - t["coef_base"]).round(3)
    t["exposure"] = t["exposure"].round(0).astype(int)
    return t[["factor", "level", "exposure", "relativity"]].sort_values(["factor", "level"])

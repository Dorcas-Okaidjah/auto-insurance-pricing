"""Frequency-severity pricing models: a GLM champion and a LightGBM challenger.

Pure premium (expected annual claim cost of one policy)
    = claim frequency (claims per policy-year)
    x claim severity (cost per claim)
    x large-loss load (recovers the part of big claims we capped)
    x off-balance factor (makes total predicted losses match the training book)

The GLM is how most insurers file rates today: transparent, one relativity
per rating factor. The GBM finds interactions and non-linear shapes the GLM
cannot see. The question this project answers is whether that extra
accuracy is worth money, and what it costs in customer disruption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

# --------------------------------------------------------------------------- #
# GLM feature engineering: the bands an actuary would file
# --------------------------------------------------------------------------- #
DRIV_AGE_BINS = [17, 20, 22, 25, 30, 40, 50, 60, 70, 120]
VEH_AGE_BINS = [-1, 0, 2, 5, 10, 15, 200]
GLM_CATS = ["DrivAgeBand", "VehAgeBand", "VehPowerBand"] + C.CAT_FEATURES
GLM_NUMS = ["logBonusMalus", "logDensity"]


def glm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["DrivAgeBand"] = pd.cut(df["DrivAge"], DRIV_AGE_BINS).astype(str)
    out["VehAgeBand"] = pd.cut(df["VehAge"], VEH_AGE_BINS).astype(str)
    out["VehPowerBand"] = df["VehPower"].clip(upper=12).astype(str)
    out["logBonusMalus"] = np.log(df["BonusMalus"].clip(upper=150))
    out["logDensity"] = np.log(df["Density"].clip(lower=1))
    for c in C.CAT_FEATURES:
        out[c] = df[c].astype(str)
    return out


def _glm_pipeline(estimator):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), GLM_CATS),
        ("num", StandardScaler(), GLM_NUMS),
    ])
    return make_pipeline(pre, estimator)


def gbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[C.FEATURES].copy()
    for c in C.CAT_FEATURES:
        X[c] = X[c].astype("category")
    return X


def _monotone_vector():
    return [1 if f in C.MONOTONE_INCREASING else 0 for f in C.FEATURES]


# --------------------------------------------------------------------------- #
class PricingModel:
    """One frequency model + one severity model -> pure premium."""

    def __init__(self, kind: str):
        assert kind in ("glm", "gbm")
        self.kind = kind
        self.freq = None
        self.sev = None
        self.large_loss_load = 1.0
        self.off_balance = 1.0
        self.n_rounds = {}

    # ---- frequency -------------------------------------------------------- #
    def _fit_freq(self, tr: pd.DataFrame):
        if self.kind == "glm":
            self.freq = _glm_pipeline(PoissonRegressor(alpha=1e-4, max_iter=1000))
            self.freq.fit(glm_frame(tr), tr["ClaimNb"] / tr["Exposure"],
                          poissonregressor__sample_weight=tr["Exposure"])
        else:
            import lightgbm as lgb
            fit, val = _inner_split(tr)
            params = dict(C.GBM_FREQ_PARAMS, monotone_constraints=_monotone_vector(),
                          seed=C.RANDOM_STATE)
            # log(exposure) as an offset: the model learns a rate per policy-year
            dtr = lgb.Dataset(gbm_frame(fit), fit["ClaimNb"],
                              init_score=np.log(fit["Exposure"]))
            dva = lgb.Dataset(gbm_frame(val), val["ClaimNb"],
                              init_score=np.log(val["Exposure"]), reference=dtr)
            self.freq = lgb.train(params, dtr, C.GBM_MAX_ROUNDS, valid_sets=[dva],
                                  callbacks=[lgb.early_stopping(C.GBM_EARLY_STOP,
                                                                verbose=False)])
            self.n_rounds["freq"] = self.freq.best_iteration

    def predict_freq(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claims per policy-year."""
        if self.kind == "glm":
            return self.freq.predict(glm_frame(df))
        return np.exp(self.freq.predict(gbm_frame(df), raw_score=True,
                                        num_iteration=self.freq.best_iteration))

    # ---- severity --------------------------------------------------------- #
    def _fit_sev(self, tr: pd.DataFrame):
        cl = tr[tr["ClaimNb"] > 0]
        y = cl["ClaimAmountCapped"] / cl["ClaimNb"]
        w = cl["ClaimNb"]
        if self.kind == "glm":
            self.sev = _glm_pipeline(GammaRegressor(alpha=1e-3, max_iter=1000))
            self.sev.fit(glm_frame(cl), y, gammaregressor__sample_weight=w)
        else:
            import lightgbm as lgb
            fit, val = _inner_split(cl)
            yf, yv = (d["ClaimAmountCapped"] / d["ClaimNb"] for d in (fit, val))
            params = dict(C.GBM_SEV_PARAMS, monotone_constraints=_monotone_vector(),
                          seed=C.RANDOM_STATE)
            dtr = lgb.Dataset(gbm_frame(fit), yf, weight=fit["ClaimNb"])
            dva = lgb.Dataset(gbm_frame(val), yv, weight=val["ClaimNb"], reference=dtr)
            self.sev = lgb.train(params, dtr, C.GBM_MAX_ROUNDS, valid_sets=[dva],
                                 callbacks=[lgb.early_stopping(C.GBM_EARLY_STOP,
                                                               verbose=False)])
            self.n_rounds["sev"] = self.sev.best_iteration

    def predict_sev(self, df: pd.DataFrame) -> np.ndarray:
        """Expected cost of one claim (capped), in EUR."""
        if self.kind == "glm":
            return self.sev.predict(glm_frame(df))
        return self.sev.predict(gbm_frame(df), num_iteration=self.sev.best_iteration)

    # ---- pure premium ----------------------------------------------------- #
    def fit(self, tr: pd.DataFrame, large_loss_load: float):
        self._fit_freq(tr)
        self._fit_sev(tr)
        self.large_loss_load = large_loss_load
        raw = self.predict_freq(tr) * self.predict_sev(tr) * large_loss_load
        actual = tr["ClaimAmount"].sum()          # uncapped: what was really paid
        self.off_balance = actual / (raw * tr["Exposure"]).sum()
        return self

    def predict_pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        """Expected claim cost for a full policy-year, in EUR."""
        return (self.predict_freq(df) * self.predict_sev(df)
                * self.large_loss_load * self.off_balance)


def _inner_split(df: pd.DataFrame, frac: float = 0.15):
    rng = np.random.default_rng(C.RANDOM_STATE + 1)
    m = rng.random(len(df)) < frac
    return df[~m], df[m]

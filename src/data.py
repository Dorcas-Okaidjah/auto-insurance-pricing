"""Download, clean and split the freMTPL2 motor insurance data.

Usage:
    python -m src.data --download          # fetch both tables (no login needed)
    python -m src.data --inspect           # print a data-quality summary
"""
from __future__ import annotations

import argparse
import io
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

FREQ_FILE = C.DATA_DIR / "freMTPL2freq.pkl"
SEV_FILE = C.DATA_DIR / "freMTPL2sev.pkl"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _from_casdatasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    import pyreadr  # reads R .rda files without needing R installed
    import tempfile

    out = []
    for name in ["freMTPL2freq", "freMTPL2sev"]:
        url = f"{C.CAS_BASE}/{name}.rda"
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            raw = r.read()
        with tempfile.NamedTemporaryFile(suffix=".rda") as tmp:
            tmp.write(raw)
            tmp.flush()
            out.append(next(iter(pyreadr.read_r(tmp.name).values())))
    return out[0], out[1]


def _from_openml() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.datasets import fetch_openml

    print("downloading from OpenML")
    freq = fetch_openml(data_id=C.OPENML_FREQ_ID, as_frame=True).frame
    sev = fetch_openml(data_id=C.OPENML_SEV_ID, as_frame=True).frame
    return freq, sev


def download(source: str = "auto") -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if source in ("auto", "cas"):
        try:
            freq, sev = _from_casdatasets()
        except Exception as e:  # network or format problem: try the mirror
            if source == "cas":
                raise
            print(f"CASdatasets download failed ({e}); trying OpenML")
            freq, sev = _from_openml()
    else:
        freq, sev = _from_openml()
    freq.to_pickle(FREQ_FILE)
    sev.to_pickle(SEV_FILE)
    print(f"saved {len(freq):,} policies and {len(sev):,} claims to {C.DATA_DIR}")


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
def clean(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (policies, claims): one row per policy with exposure, claim count
    and total claim cost, plus the claim rows that belong to kept policies.

    Rules (each one is printed so the README numbers are reproducible):
    * Claim counts are rebuilt from the claim-amount table, because the two
      published tables disagree for some policies. A count with no amount
      cannot be priced, and an amount with no count is invisible to a
      frequency model.
    * Exposure is capped at one policy-year.
    * Policies with more than MAX_CLAIMS claims are dropped as data errors.
    """
    freq = freq.copy()
    sev = sev.copy()
    freq["IDpol"] = freq["IDpol"].astype(float).astype(np.int64)
    sev["IDpol"] = sev["IDpol"].astype(float).astype(np.int64)

    per_pol = sev.groupby("IDpol")["ClaimAmount"].agg(n_claims="size", amount="sum")
    df = freq.drop(columns=["ClaimNb"]).merge(per_pol, how="left",
                                              left_on="IDpol", right_index=True)
    df["ClaimNb"] = df["n_claims"].fillna(0).astype(int)
    df["ClaimAmount"] = df["amount"].fillna(0.0)
    df = df.drop(columns=["n_claims", "amount"])

    orphan = (~sev["IDpol"].isin(freq["IDpol"])).sum()
    if orphan:
        print(f"  {orphan} claim rows have no matching policy and are dropped")

    n0 = len(df)
    df = df[df["ClaimNb"] <= C.MAX_CLAIMS]
    print(f"  dropped {n0 - len(df)} policies with > {C.MAX_CLAIMS} claims")
    df["Exposure"] = df["Exposure"].clip(upper=C.MAX_EXPOSURE)

    for col in C.CAT_FEATURES:
        df[col] = df[col].astype(str).str.strip("'").astype("category")
    for col in C.NUM_FEATURES:
        df[col] = pd.to_numeric(df[col])
    return df.reset_index(drop=True), sev[sev["IDpol"].isin(df["IDpol"])]


def large_loss_cap(sev: pd.DataFrame, train_ids,
                   quantile: float = C.LARGE_LOSS_QUANTILE) -> tuple[float, float]:
    """Choose a cap for individual claims and the load that recovers the excess.

    Large losses are rare and random; letting a few EUR 1M claims drive the
    severity model makes it chase noise. Standard practice is to cap them and
    spread the excess back as a flat load on every policy. Both numbers are
    estimated on training policies only, so nothing leaks from the test set.
    """
    train_claims = sev.loc[sev["IDpol"].isin(train_ids), "ClaimAmount"]
    cap = float(train_claims.quantile(quantile))
    load = float(train_claims.sum() / train_claims.clip(upper=cap).sum())
    return cap, load


def apply_cap(df: pd.DataFrame, sev: pd.DataFrame, cap: float) -> pd.DataFrame:
    per_pol = sev.assign(c=sev["ClaimAmount"].clip(upper=cap)).groupby("IDpol")["c"].sum()
    return df.assign(ClaimAmountCapped=df["IDpol"].map(per_pol).fillna(0.0))


def split(df: pd.DataFrame):
    rng = np.random.default_rng(C.RANDOM_STATE)
    is_test = rng.random(len(df)) < C.TEST_SIZE
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FREQ_FILE.exists():
        raise FileNotFoundError(
            "No data yet. Run:  python -m src.data --download")
    return clean(pd.read_pickle(FREQ_FILE), pd.read_pickle(SEV_FILE))


# --------------------------------------------------------------------------- #
# Synthetic data with the same schema (for tests and offline demos)
# --------------------------------------------------------------------------- #
def make_synthetic(n: int = 20000, seed: int = 0):
    """Fake portfolio with a known risk structure. Used by the smoke test only."""
    rng = np.random.default_rng(seed)
    regions = ["R11", "R24", "R82", "R93", "R53", "R52"]
    brands = ["B1", "B2", "B3", "B5", "B12"]
    df = pd.DataFrame({
        "IDpol": np.arange(1, n + 1),
        "Exposure": rng.uniform(0.05, 1.0, n).round(2),
        "VehPower": rng.integers(4, 15, n),
        "VehAge": rng.integers(0, 25, n),
        "DrivAge": rng.integers(18, 90, n),
        "BonusMalus": np.clip(rng.gamma(2, 15, n) + 50, 50, 230).astype(int),
        "VehBrand": rng.choice(brands, n),
        "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Area": rng.choice(list("ABCDEF"), n),
        "Density": np.exp(rng.normal(5.5, 1.8, n)).astype(int).clip(1, 27000),
        "Region": rng.choice(regions, n),
    })
    young = (df["DrivAge"] < 25).astype(float)
    lam = np.exp(-2.6 + 0.6 * young + 0.015 * (df["BonusMalus"] - 50)
                 + 0.08 * np.log1p(df["Density"]) - 0.01 * df["VehAge"])
    df["ClaimNb"] = rng.poisson(lam * df["Exposure"]).clip(0, C.MAX_CLAIMS)
    rows = []
    for pid, k in zip(df["IDpol"], df["ClaimNb"]):
        for _ in range(int(k)):
            rows.append((pid, float(rng.gamma(1.5, 1200) + 50)))
    sev = pd.DataFrame(rows, columns=["IDpol", "ClaimAmount"])
    return df, sev


# --------------------------------------------------------------------------- #
def inspect() -> None:
    df, sev = load_clean()
    print(f"\npolicies: {len(df):,}   claims: {int(df.ClaimNb.sum()):,}   "
          f"exposure: {df.Exposure.sum():,.0f} policy-years")
    print(f"claim frequency: {df.ClaimNb.sum() / df.Exposure.sum():.4f} per year")
    print(f"mean claim: EUR {sev.ClaimAmount.mean():,.0f}   "
          f"median: EUR {sev.ClaimAmount.median():,.0f}   "
          f"max: EUR {sev.ClaimAmount.max():,.0f}")
    print("\nclaim count distribution:")
    print(df.ClaimNb.value_counts().sort_index().to_string())
    print("\nnumeric features:")
    print(df[C.NUM_FEATURES + ["Exposure"]].describe().T.round(2).to_string())
    print("\nmissing values:", int(df[C.FEATURES].isna().sum().sum()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--source", default="auto", choices=["auto", "cas", "openml"])
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()
    if a.download:
        download(a.source)
    if a.inspect:
        inspect()

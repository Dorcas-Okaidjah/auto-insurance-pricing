"""Quote one policy with both models and explain the difference.

A small decision-support tool for an underwriter or product manager:

    python -m src.quote --driv-age 19 --bonus-malus 90 --veh-age 2 --region Ile-de-France
    python -m src.quote --driv-age 45 --bonus-malus 50 --density 50 --area A

Prints expected frequency, severity, pure premium and technical premium from the
GLM and the GBM, the proposed year-one price after rate capping, and how far
each rating factor moves the GBM price from the portfolio average.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .business_case import capped_premium, premium

DEFAULT = dict(VehPower=6, VehAge=6, DrivAge=44, BonusMalus=50, VehBrand="B12",
               VehGas="Regular", Area="C", Density=393, Region="Centre")


def load_models():
    path = C.MODEL_DIR / "pricing_models.joblib"
    if not path.exists():
        raise FileNotFoundError("No fitted models. Run:  python run_all.py")
    return joblib.load(path)


def quote(policy: dict, models=None) -> dict:
    models = models or load_models()
    row = pd.DataFrame([{**DEFAULT, **policy}])
    for c in C.CAT_FEATURES:
        row[c] = row[c].astype("category")
    out = {}
    for k, m in models.items():
        f, s = m.predict_freq(row)[0], m.predict_sev(row)[0]
        pp = m.predict_pure_premium(row)[0]
        out[k] = {"frequency": f, "severity": s, "pure_premium": pp,
                  "premium": float(premium(pp))}
    out["proposed_premium"] = float(premium(capped_premium(
        out["glm"]["pure_premium"], out["gbm"]["pure_premium"])))

    # One-factor-at-a-time attribution for the GBM: how much each of this
    # policy's values moves the price relative to the typical policy.
    base = models["gbm"].predict_pure_premium(_frame(DEFAULT))[0]
    drivers = {}
    for f in C.FEATURES:
        if policy.get(f) is None or policy[f] == DEFAULT[f]:
            continue
        p = models["gbm"].predict_pure_premium(_frame({**DEFAULT, f: policy[f]}))[0]
        drivers[f"{f}={policy[f]}"] = p / base - 1
    out["gbm_drivers"] = dict(sorted(drivers.items(), key=lambda kv: -abs(kv[1])))
    return out


def _frame(d):
    r = pd.DataFrame([d])
    for c in C.CAT_FEATURES:
        r[c] = r[c].astype("category")
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in DEFAULT.items():
        flag = "--" + "".join("-" + ch.lower() if ch.isupper() else ch
                              for ch in name).lstrip("-")
        p.add_argument(flag, dest=name, type=type(val), default=None)
    a = vars(p.parse_args())
    policy = {k: v for k, v in a.items() if v is not None}
    q = quote(policy)
    print(f"\nPolicy: {({**DEFAULT, **policy})}\n")
    print(f"{'':22}{'GLM':>12}{'GBM':>12}")
    for key, fmt in [("frequency", "{:.4f}"), ("severity", "EUR {:,.0f}"),
                     ("pure_premium", "EUR {:,.0f}"), ("premium", "EUR {:,.0f}")]:
        print(f"{key.replace('_', ' '):22}{fmt.format(q['glm'][key]):>12}"
              f"{fmt.format(q['gbm'][key]):>12}")
    print(f"\nProposed year-one premium (GBM, capped at +/-{C.RATE_CAP:.0%} vs GLM): "
          f"EUR {q['proposed_premium']:,.0f}")
    if q["gbm_drivers"]:
        print("\nWhat moves this price (GBM, vs a typical policy):")
        for k, v in q["gbm_drivers"].items():
            print(f"  {k:28} {v:+.0%}")


if __name__ == "__main__":
    main()

"""Run the whole project end to end.

    python run_all.py                  # real data (download first: python -m src.data --download)
    python run_all.py --sample 0.2     # 20% of policies, for a quick run
    python run_all.py --synthetic 20000   # fake data, no download (used by the smoke test)

Outputs land in reports/ (charts, tables, summary.md) and models/ (fitted models).
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

from src import business_case, explain, metrics, plots, report
from src import config as C
from src.data import (apply_cap, clean, large_loss_cap, load_clean,
                      make_synthetic, split)
from src.models import PricingModel


def main(sample: float = 1.0, synthetic: int = 0, out_dir=None):
    t0 = time.time()
    if out_dir is not None:
        C.REPORT_DIR = out_dir / "reports"
        C.MODEL_DIR = out_dir / "models"
        plots.C.REPORT_DIR = C.REPORT_DIR

    print("1/6 loading data")
    if synthetic:
        df, sev = clean(*make_synthetic(synthetic))
    else:
        df, sev = load_clean()
    if sample < 1:
        df = df.sample(frac=sample, random_state=C.RANDOM_STATE).reset_index(drop=True)
        sev = sev[sev["IDpol"].isin(df["IDpol"])]
    train, test = split(df)
    cap, load = large_loss_cap(sev, train["IDpol"])
    train, test = apply_cap(train, sev, cap), apply_cap(test, sev, cap)
    print(f"    train {len(train):,} / test {len(test):,} policies; "
          f"large-loss cap EUR {cap:,.0f}, load {load:.3f}")

    print("2/6 fitting GLM (champion)")
    glm = PricingModel("glm").fit(train, load)
    print("3/6 fitting GBM (challenger)")
    gbm = PricingModel("gbm").fit(train, load)

    print("4/6 evaluating on the test set")
    models = {"glm": glm, "gbm": gbm}
    freq = {k: m.predict_freq(test) for k, m in models.items()}
    sevp = {k: m.predict_sev(test) for k, m in models.items()}
    pp = {k: m.predict_pure_premium(test) for k, m in models.items()}
    # Evaluate on capped claims + the large-loss load. One EUR 1.4M claim in the
    # test set would otherwise decide every segment it lands in. Raw (uncapped)
    # totals are still reported as predicted_to_actual_raw.
    raw_loss = test["ClaimAmount"]
    test = test.assign(ClaimAmount=test["ClaimAmountCapped"] * load)
    loss, expo = test["ClaimAmount"], test["Exposure"]
    res = {
        "data": {"policies": len(df), "train": len(train), "test": len(test),
                 "claims": int(df["ClaimNb"].sum()),
                 "exposure_years": float(df["Exposure"].sum()),
                 "large_loss_cap": cap, "large_loss_load": load,
                 "synthetic": bool(synthetic), "sample": sample},
        "metrics": {},
        "gbm_rounds": gbm.n_rounds,
    }
    for k in models:
        res["metrics"][k] = {
            "poisson_deviance": metrics.poisson_deviance(test, freq[k]),
            "gamma_deviance": metrics.gamma_deviance(test, sevp[k]),
            "gini_frequency": metrics.gini(freq[k], test["ClaimNb"], expo),
            "gini_pure_premium": metrics.gini(pp[k], loss, expo),
            "predicted_to_actual": float((pp[k] * expo).sum() / loss.sum()),
            "predicted_to_actual_raw": float((pp[k] * expo).sum() / raw_loss.sum()),
        }
    lo, hi = metrics.bootstrap_gini_diff(pp["glm"], pp["gbm"], loss, expo)
    res["metrics"]["gini_gain_ci95"] = [float(lo), float(hi)]
    lifts = {k: metrics.lift_table(test, pp[k]) for k in models}
    dl = metrics.double_lift(test, pp["glm"], pp["gbm"])
    ae_age = {k: metrics.actual_vs_expected(
        test.assign(DrivAgeBand=np.select(
            [test.DrivAge < 25, test.DrivAge < 40, test.DrivAge < 60],
            ["18-24", "25-39", "40-59"], "60+")), pp[k], "DrivAgeBand")
        for k in models}

    print("5/6 explaining the models and building the business case")
    pdps = {}
    for feat, grid in explain.PDP_GRIDS.items():
        pdps[feat] = {"grid": grid,
                      **{k: explain.partial_dependence(m, train, feat, grid)
                         for k, m in models.items()}}
    imp = explain.gbm_importance(gbm)
    rating = explain.glm_rating_table(glm, train)
    biz = business_case.run(test, pp["glm"], pp["gbm"])

    print("6/6 writing charts, tables and the summary")
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plots.lorenz_curves(test, pp, {k: res["metrics"][k]["gini_pure_premium"]
                                   for k in models})
    plots.lift_charts(lifts)
    plots.double_lift_chart(dl)
    plots.pdp_grid(pdps)
    plots.importance_chart(imp)
    plots.dislocation_chart(biz["dislocation"], C.RATE_CAP)
    rating.to_csv(C.REPORT_DIR / "glm_rating_table.csv", index=False)
    imp.to_csv(C.REPORT_DIR / "gbm_feature_importance.csv")
    for k in models:
        lifts[k].to_csv(C.REPORT_DIR / f"lift_table_{k}.csv")
    dl.to_csv(C.REPORT_DIR / "double_lift.csv")
    res["business_case"] = {k: v for k, v in biz.items() if k != "dislocation"}
    res["business_case"]["dislocation"] = biz["dislocation"]["share_of_policies"].to_dict()
    res["seconds"] = round(time.time() - t0, 1)
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    report.write_summary(res, ae_age)
    joblib.dump(models, C.MODEL_DIR / "pricing_models.joblib")
    print(f"done in {res['seconds']}s -> {C.REPORT_DIR / 'summary.md'}")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=float, default=1.0)
    p.add_argument("--synthetic", type=int, default=0)
    a = p.parse_args()
    main(a.sample, a.synthetic)

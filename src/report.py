"""Write reports/summary.md: the results a pricing committee would read."""
from __future__ import annotations

from . import config as C


def _pct(x):
    return f"{x:.1%}"


def write_summary(res: dict, ae_age: dict) -> None:
    m, d, b = res["metrics"], res["data"], res["business_case"]
    g, gb = m["glm"], m["gbm"]
    lo, hi = m["gini_gain_ci95"]
    s1, s2, s3 = b["gbm_vs_glm"], b["glm_vs_gbm"], b["capped_vs_glm"]
    dis = b["dislocation"]
    big_move = dis["< -25%"] + dis["> +25%"]
    small_move = dis["-10% to +10%"]
    lines = []
    w = lines.append
    if d["synthetic"]:
        w("> **Synthetic data run** (smoke test). Numbers below are not meaningful.\n")
    w("# Results summary\n")
    w(f"Data: {d['policies']:,} motor policies, {d['exposure_years']:,.0f} policy-years, "
      f"{d['claims']:,} claims. Train {d['train']:,} / test {d['test']:,} (random split). "
      f"Individual claims capped at EUR {d['large_loss_cap']:,.0f} "
      f"(the {C.LARGE_LOSS_QUANTILE:.1%} quantile); the excess is spread back as a "
      f"{d['large_loss_load']:.3f}x load. Test-set metrics use capped claims x load, so "
      f"a single very large claim cannot decide a segment's result.\n")
    w("## Model accuracy (test set)\n")
    w("| Metric | GLM (champion) | GBM (challenger) | Better |")
    w("|---|---|---|---|")
    rows = [("Poisson deviance, frequency", "poisson_deviance", "lower"),
            ("Gamma deviance, severity", "gamma_deviance", "lower"),
            ("Gini, frequency", "gini_frequency", "higher"),
            ("Gini, pure premium", "gini_pure_premium", "higher"),
            ("Predicted / actual losses (capped + load)", "predicted_to_actual", "closer to 1"),
            ("Predicted / actual losses (raw)", "predicted_to_actual_raw", "closer to 1")]
    for label, key, better in rows:
        w(f"| {label} | {g[key]:.4f} | {gb[key]:.4f} | {better} |")
    w(f"\nGini gain from GBM: **{gb['gini_pure_premium'] - g['gini_pure_premium']:+.4f}** "
      f"(95% bootstrap interval {lo:+.4f} to {hi:+.4f}).\n")
    w("## Business case\n")
    w(f"Premium = expected loss / {C.TARGET_LOSS_RATIO:.0%} target loss ratio. "
      f"A customer switches insurer when the rival quotes at least "
      f"{s1['switch_threshold']:.0%} less. Losses are the test policies' own claims (capped + load).\n")
    w("| Scenario | Challenger wins | Challenger loss ratio | Incumbent loss ratio |")
    w("|---|---|---|---|")
    w(f"| We price with GBM, rival uses GLM | {_pct(s1['challenger']['share'])} | "
      f"**{_pct(s1['challenger']['loss_ratio'])}** | {_pct(s1['incumbent']['loss_ratio'])} |")
    w(f"| We price with GLM, rival uses GBM | {_pct(s2['challenger']['share'])} | "
      f"{_pct(s2['challenger']['loss_ratio'])} | **{_pct(s2['incumbent']['loss_ratio'])}** |")
    w(f"| We price with capped GBM (+/-{C.RATE_CAP:.0%}), rival uses GLM | "
      f"{_pct(s3['challenger']['share'])} | {_pct(s3['challenger']['loss_ratio'])} | "
      f"{_pct(s3['incumbent']['loss_ratio'])} |")
    w(f"\nSwitching the existing book from GLM to GBM prices: {_pct(small_move)} of "
      f"customers move less than 10%; {_pct(big_move)} move more than 25%. "
      f"Capping year-one changes at +/-{C.RATE_CAP:.0%} keeps "
      f"{_pct(b['gini']['gain_kept_by_cap'])} of the Gini gain.\n")
    w("## Actual vs expected by driver age (test set)\n")
    w("| Driver age | GLM A/E | GBM A/E |")
    w("|---|---|---|")
    for band in ae_age["glm"].index:
        w(f"| {band} | {ae_age['glm'].loc[band, 'A/E']:.3f} | "
          f"{ae_age['gbm'].loc[band, 'A/E']:.3f} |")
    w("\nCharts: `lorenz_curve.png`, `lift_by_decile.png`, `double_lift.png`, "
      "`partial_dependence.png`, `feature_importance.png`, `dislocation.png`.")
    (C.REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n")

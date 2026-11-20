"""Static charts for the README and report (matplotlib, PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .metrics import lorenz  # noqa: E402

COL = C.COLORS
NAMES = {"glm": "GLM (champion)", "gbm": "GBM (challenger)"}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=COL["actual"], pad=10)
    ax.set_xlabel(xlabel, color=COL["muted"])
    ax.set_ylabel(ylabel, color=COL["muted"])
    ax.grid(True, color=COL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(COL["grid"])
    ax.tick_params(colors=COL["muted"])


def _save(fig, name):
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.REPORT_DIR / name, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def lorenz_curves(test, preds: dict, ginis: dict):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for k, p in preds.items():
        x, y = lorenz(p, test["ClaimAmount"], test["Exposure"])
        ax.plot(x, y, color=COL[k], lw=2, label=f"{NAMES[k]}  Gini {ginis[k]:.3f}")
    ax.plot([0, 1], [0, 1], color=COL["muted"], lw=1, ls="--", label="Random pricing")
    _style(ax, "Risk ranking: ordered Lorenz curve (test set)",
           "Share of exposure, safest to riskiest", "Share of actual claim cost")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "lorenz_curve.png")


def lift_charts(lifts: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (k, t) in zip(axes, lifts.items()):
        ax.bar(t.index, t["actual_loss_cost"], color=COL["grid"], width=0.7,
               label="Actual")
        ax.plot(t.index, t["predicted_loss_cost"], color=COL[k], lw=2, marker="o",
                ms=6, label="Predicted")
        _style(ax, f"{NAMES[k]}: lift by risk decile", "Predicted risk decile",
               "Loss cost per policy-year (EUR)")
        ax.set_xticks(t.index)
        ax.legend(frameon=False, loc="upper left")
    _save(fig, "lift_by_decile.png")


def double_lift_chart(t):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(t.index, t["actual"], color=COL["actual"], lw=2, marker="o", ms=6,
            label="Actual")
    ax.plot(t.index, t["model_a"], color=COL["glm"], lw=2, marker="s", ms=6,
            label=NAMES["glm"])
    ax.plot(t.index, t["model_b"], color=COL["gbm"], lw=2, marker="^", ms=7,
            label=NAMES["gbm"])
    ax.axhline(1, color=COL["muted"], lw=0.8)
    _style(ax, "Double lift: where the models disagree, who is right?",
           "Decile of GBM / GLM price ratio  (1 = GBM much cheaper)",
           "Loss cost relative to book average")
    ax.set_xticks(t.index)
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "double_lift.png")


def pdp_grid(pdps: dict):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (feat, d) in zip(axes.ravel(), pdps.items()):
        for k in ("glm", "gbm"):
            ax.plot(d["grid"], d[k], color=COL[k], lw=2, label=NAMES[k])
        _style(ax, feat, feat, "Claims per policy-year")
        if feat == "Density":
            ax.set_xscale("log")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("How each model prices a rating factor (partial dependence)",
                 x=0.02, ha="left", fontsize=13)
    _save(fig, "partial_dependence.png")


def importance_chart(imp):
    t = imp["frequency"].sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(t.index, t.values, color=COL["gbm"], height=0.6)
    for y, v in enumerate(t.values):
        ax.text(v + 0.005, y, f"{v:.0%}", va="center", color=COL["muted"], fontsize=9)
    _style(ax, "GBM frequency model: share of total gain", "Share of gain", "")
    ax.grid(axis="y", visible=False)
    _save(fig, "feature_importance.png")


def dislocation_chart(t, cap):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index.astype(str), t["share_of_policies"], color=COL["gbm"], width=0.6)
    for i, v in enumerate(t["share_of_policies"]):
        ax.text(i, v + 0.008, f"{v:.0%}", ha="center", color=COL["actual"], fontsize=10)
    _style(ax, "Customer impact of switching GLM -> GBM (before capping)",
           "Change in premium", "Share of policies")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.text(0.99, 0.95, f"Year-one cap of +/-{cap:.0%} recommended",
            transform=ax.transAxes, ha="right", color=COL["muted"], fontsize=9)
    _save(fig, "dislocation.png")

"""End-to-end check on synthetic data: no download, about 30 seconds.

    python -m tests.smoke_test
"""
import json
import tempfile
from pathlib import Path

import numpy as np

import run_all
from src.business_case import capped_premium
from src.metrics import gini


def main():
    # unit checks
    rng = np.random.default_rng(0)
    loss = rng.gamma(1, 100, 5000)
    expo = np.ones(5000)
    assert abs(gini(loss, loss, expo)) > 0.3, "perfect ranking should have a high Gini"
    assert abs(gini(rng.random(5000), loss, expo)) < 0.05, "random ranking ~ 0 Gini"
    cp = capped_premium(np.array([100.0, 100.0]), np.array([200.0, 50.0]), cap=0.15)
    assert np.allclose(cp, [115.0, 85.0]), cp

    with tempfile.TemporaryDirectory() as tmp:
        res = run_all.main(synthetic=20000, out_dir=Path(tmp))
        rep = Path(tmp) / "reports"
        for f in ["summary.md", "metrics.json", "lorenz_curve.png", "double_lift.png",
                  "partial_dependence.png", "dislocation.png", "glm_rating_table.csv"]:
            assert (rep / f).exists(), f"missing {f}"
        assert (Path(tmp) / "models" / "pricing_models.joblib").exists()
        m = json.loads((rep / "metrics.json").read_text())["metrics"]
        for k in ("glm", "gbm"):
            # off-balance step means predicted losses ~ actual on new data too
            assert 0.7 < m[k]["predicted_to_actual"] < 1.3, m[k]
            assert m[k]["gini_frequency"] > 0.1, "model failed to learn known signal"
    print("\nSMOKE TEST PASSED")
    return res


if __name__ == "__main__":
    main()

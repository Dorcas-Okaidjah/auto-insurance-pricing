"""Central configuration. Change settings here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

# --- Data sources (both public, no login needed) ---------------------------
# freMTPL2: French motor third-party liability portfolio (~678k policies, one
# year of exposure) and the matching claim amounts. Published in the R package
# CASdatasets (Dutang & Charpentier, GPL-2+). We download at run time and never
# commit the data to the repo.
CAS_BASE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data"
OPENML_FREQ_ID = 41214   # fallback source: same data mirrored on OpenML
OPENML_SEV_ID = 41215

RANDOM_STATE = 42
TEST_SIZE = 0.25

# --- Cleaning rules (standard in the actuarial literature on this dataset) ---
MAX_EXPOSURE = 1.0        # a policy-year cannot exceed 1
MAX_CLAIMS = 4            # a handful of policies show 5-16 claims: data errors
LARGE_LOSS_QUANTILE = 0.995  # individual claims above this are capped and the
                             # excess is spread back as a flat load

# --- Features ----------------------------------------------------------------
CAT_FEATURES = ["VehBrand", "VehGas", "Area", "Region"]
NUM_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Regulators and underwriters expect a worse bonus-malus (claims history) score
# never to lower the price. We enforce that in the boosted models.
MONOTONE_INCREASING = ["BonusMalus"]

# --- LightGBM settings ---------------------------------------------------------
GBM_FREQ_PARAMS = dict(objective="poisson", learning_rate=0.05, num_leaves=31,
                       min_child_samples=500, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                       verbose=-1)
GBM_SEV_PARAMS = dict(objective="gamma", learning_rate=0.03, num_leaves=8,
                      min_child_samples=300, feature_fraction=0.8,
                      lambda_l2=5.0, verbose=-1)
GBM_MAX_ROUNDS = 3000
GBM_EARLY_STOP = 100

# --- Business case ---------------------------------------------------------
TARGET_LOSS_RATIO = 0.70   # premium = expected loss / target loss ratio
DISLOCATION_BANDS = [-0.25, -0.10, 0.10, 0.25]  # premium-change bands to report
RATE_CAP = 0.15            # max year-one premium change if the new model ships

# Chart colours (validated colour-blind-safe pair + neutral ink)
COLORS = {"glm": "#2a78d6", "gbm": "#eb6834", "actual": "#0b0b0b",
          "grid": "#e4e3df", "muted": "#52514e"}

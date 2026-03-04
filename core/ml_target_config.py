
import os

HORIZON_DAYS = 20

def _get_env_float(var, default):
	try:
		return float(os.environ.get(var, default))
	except Exception:
		return default

# Allow override via ML_UP_THRESHOLD, ML_DOWN_THRESHOLD
# 5% threshold matches Forward_Return_20d >= 5% target definition.
# Previous 8% was too aggressive, yielding ~20% positive class and AUC=0.553.
# At 5%: ~30% positive class gives the model more signal to learn from.
UP_THRESHOLD = _get_env_float("ML_UP_THRESHOLD", 0.05)
DOWN_THRESHOLD = _get_env_float("ML_DOWN_THRESHOLD", -0.02)

# Rank-based target configuration (used by V3.4+ training)
TARGET_MODE = os.environ.get("ML_TARGET_MODE", "rank")  # "rank" (recommended) or "absolute"
RANK_TOP_PCT = _get_env_float("ML_RANK_TOP_PCT", 0.20)   # Top 20% = winner
RANK_BOTTOM_PCT = _get_env_float("ML_RANK_BOTTOM_PCT", 0.40)  # Bottom 40% = loser

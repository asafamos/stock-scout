
import os

HORIZON_DAYS = 20

def _get_env_float(var, default):
	try:
		return float(os.environ.get(var, default))
	except Exception:
		return default

# Allow override via ML_UP_THRESHOLD, ML_DOWN_THRESHOLD
UP_THRESHOLD = _get_env_float("ML_UP_THRESHOLD", 0.08)
DOWN_THRESHOLD = _get_env_float("ML_DOWN_THRESHOLD", 0.00)

# V4 rank-based target configuration
TARGET_MODE = os.environ.get("ML_TARGET_MODE", "rank")  # "rank" or "absolute"
RANK_TOP_PCT = _get_env_float("ML_RANK_TOP_PCT", 0.20)   # Top 20% = winner
RANK_BOTTOM_PCT = _get_env_float("ML_RANK_BOTTOM_PCT", 0.40)  # Bottom 40% = loser

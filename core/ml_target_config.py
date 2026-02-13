
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

"""Configuration normalisation for the scan pipeline."""

import dataclasses
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _normalize_config(config: Any) -> Dict[str, Any]:
    """Normalize incoming config into a canonical dict.

    Supported inputs:
    - None: returns {}
    - dict: shallow-copied
    - objects with .to_dict(): use that
    - dataclasses: converted via dataclasses.asdict
    Otherwise: raise TypeError to avoid silent misconfiguration.
    Also applies selective environment overrides (e.g., METEOR_MODE, SMART_SCAN)
    after base normalization.
    """
    if config is None:
        normalized = {}
    elif isinstance(config, dict):
        normalized = dict(config)
    else:
        # Object types
        try:
            if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
                normalized = dict(config.to_dict())
            elif dataclasses.is_dataclass(config):
                normalized = dataclasses.asdict(config)
            else:
                raise TypeError(
                    f"Unsupported config type: {type(config).__name__}. "
                    "Provide dict or dataclass/Config."
                )
        except (AttributeError, TypeError):
            raise

    key_map = {
        "FUNDAMENTAL_ENABLED": "fundamental_enabled",
        "BETA_FILTER_ENABLED": "beta_filter_enabled",
        "BETA_MAX_ALLOWED": "beta_max_allowed",
        "BETA_TOP_K": "beta_top_k",
        "BETA_BENCHMARK": "beta_benchmark",
    }
    for old_key, new_key in key_map.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]

    # Enforce softened volume constraint for Tier 1 and downstream risk filters
    try:
        if "min_avg_volume" not in normalized or not isinstance(
            normalized.get("min_avg_volume"), (int, float)
        ):
            normalized["min_avg_volume"] = 100_000
        else:
            if float(normalized.get("min_avg_volume", 100_000)) > 100_000:
                normalized["min_avg_volume"] = 100_000
    except (TypeError, ValueError) as exc:
        logger.debug(f"min_avg_volume enforcement fallback: {exc}")
        normalized["min_avg_volume"] = 100_000

    # Environment overrides (post-normalization). Only apply to expected keys.
    try:
        def _env_bool(name: str) -> Optional[bool]:
            val = os.getenv(name)
            if val is None:
                return None
            s = str(val).strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
            return None

        # METEOR_MODE -> meteor_mode
        mm = _env_bool("METEOR_MODE")
        if mm is not None:
            normalized["meteor_mode"] = bool(mm)
        # SMART_SCAN -> smart_scan
        ss = _env_bool("SMART_SCAN")
        if ss is not None:
            normalized["smart_scan"] = bool(ss)
        # EXTERNAL_PRICE_VERIFY -> external_price_verify
        epv = _env_bool("EXTERNAL_PRICE_VERIFY")
        if epv is not None:
            normalized["external_price_verify"] = bool(epv)
    except (TypeError, ValueError, OSError) as exc:
        logger.debug(f"Environment override parsing skipped: {exc}")

    return normalized

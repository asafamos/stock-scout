"""
Configuration and constants for Stock Scout.
All configurable parameters in one place.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import os

# Import canonical weights so Config.weights stays in sync
from core.scoring_config import TECH_WEIGHTS as _CANONICAL_TECH_WEIGHTS

# Load .env early for Config defaults
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _clean_key(val: Optional[str]) -> Optional[str]:
    try:
        if val is None:
            return None
        return str(val).strip().strip('"').strip("'")
    except Exception:
        return val


def get_secret(key: str, default: Optional[str] = None, nested_sections: Optional[list[str]] = None) -> Optional[str]:
    """Unified secret loader without Streamlit dependency.

    Precedence: environment variables (with `.env` loaded) only.
    Returns `default` if nothing is found.
    """
    val = os.getenv(key)
    if val is not None:
        return _clean_key(val)
    return default

def _get_config_value(key: str, default: str) -> str:
    """Get config value using unified loader for backward compatibility."""
    val = get_secret(key, default)
    return str(val) if val is not None else default


@dataclass
class Config:
    """Main configuration for Stock Scout."""
    
    # Budget & Position Sizing
    budget_total: float = 5000.0
    min_position: float = 500.0
    max_position_pct: float = 15.0
    
    # Universe & Data
    universe_limit: int = int(_get_config_value('UNIVERSE_LIMIT', '3000'))
    lookback_days: int = int(_get_config_value('LOOKBACK_DAYS', '250'))
    smart_scan: bool = _get_config_value('SMART_SCAN', 'true').lower() in ('true', '1', 'yes')
    
    # Price & Volume Filters
    min_price: float = 3.0
    min_avg_volume: int = 500_000
    min_dollar_volume: int = 5_000_000
    min_market_cap: float = 100_000_000  # $100M
    
    # Technical Indicators
    ma_short: int = 20
    ma_long: int = 50
    # RSI Multi-Tier System (Nov 2025 - BALANCED UPDATE):
    # Tier A (Core): RSI 20-55 (oversold + neutral - ~62% win rate, better coverage)
    # Tier B (Spec): RSI 55-70 (slightly overbought zone)
    # Analysis: Previous strict RSI 25-40 resulted in 0 Core stocks
    rsi_core_bounds: tuple = (20, 55)  # Core = oversold + neutral (balanced)
    rsi_spec_bounds: tuple = (55, 70)  # Spec = higher momentum zone
    rsi_bounds: tuple = (20, 80)  # Overall acceptable range (for backwards compatibility)
    pullback_range: tuple = (0.85, 0.97)
    overext_soft: float = 0.20
    overext_hard: float = 0.30
    atr_price_hard: float = 0.08
    use_macd_adx: bool = True
    
    # Downside Protection Filters
    max_atr_pct: float = 6.0  # Reject extreme volatility (>6% ATR)
    min_rr_required: float = 1.5  # Minimum Risk/Reward ratio
    earnings_blackout_days: int = 7  # Skip stocks with earnings in next N days
    
    # Technical Weights — delegates to scoring_config.TECH_WEIGHTS (single source of truth)
    weights: Dict[str, float] = field(default_factory=lambda: dict(_CANONICAL_TECH_WEIGHTS))
    
    # Fundamentals
    fundamental_enabled: bool = True
    fundamental_weight: float = 0.15
    fundamental_top_k: int = 15
    surprise_bonus_on: bool = False
    
    # Risk Management
    earnings_check_topk: int = 12
    sector_cap_enabled: bool = True
    sector_cap_max: int = 3
    beta_filter_enabled: bool = True
    beta_benchmark: str = "SPY"
    beta_max_allowed: float = 2.0
    beta_top_k: int = 60
    
    # Sector Concentration Limits
    max_sector_allocation_pct: float = 35.0  # Max 35% in one sector
    max_positions_per_sector: int = 3  # Max 3 stocks per sector
    
    # External Verification
    external_price_verify: bool = True
    top_validate_k: int = 50  # Verify prices for ALL displayed stocks (was 12)
    
    # Results
    topn_results: int = int(_get_config_value('TOPN_RESULTS', '25'))
    topk_recommend: int = int(_get_config_value('TOPK_RECOMMEND', '10'))

    # Performance / Fast Mode
    perf_fast_mode: bool = False
    perf_multi_source_top_n: int = 8
    perf_alpha_enabled: bool = True
    perf_fund_timeout: int = 15
    perf_fund_timeout_fast: int = 6

    # Debug / Developer
    debug_mode: bool = os.getenv("STOCK_SCOUT_DEBUG", "false").lower() in ("true", "1", "yes")

    # Remote Autoscan
    use_remote_autoscan: bool = True
    remote_autoscan_repo: str = os.getenv("REMOTE_AUTOSCAN_REPO", "asafamos/stock-scout")
    remote_autoscan_branch: str = os.getenv("REMOTE_AUTOSCAN_BRANCH", "main")

    def to_dict(self) -> dict:
        """Convert config to flat dictionary for backward compatibility.

        Includes both snake_case (canonical) and UPPER_CASE (legacy) keys
        so callers using either convention continue to work.
        """
        d: Dict[str, Any] = {}
        for k in dir(self):
            if k.startswith("_") or callable(getattr(self, k)):
                continue
            d[k] = getattr(self, k)
            d[k.upper()] = getattr(self, k)
        return d

    def validate(self) -> None:
        """Validate config values at startup. Raises ValueError for invalid settings."""
        errors = []
        if self.budget_total <= 0:
            errors.append("budget_total must be positive")
        if self.min_position <= 0:
            errors.append("min_position must be positive")
        if not (0 < self.max_position_pct <= 100):
            errors.append("max_position_pct must be between 0 and 100")
        if self.universe_limit < 10:
            errors.append("universe_limit must be at least 10")
        if self.lookback_days < 50:
            errors.append("lookback_days must be at least 50")
        if self.min_price < 0:
            errors.append("min_price must be non-negative")
        if errors:
            raise ValueError(f"Config validation failed: {'; '.join(errors)}")


@dataclass
class APIKeys:
    """API keys configuration."""
    
    fmp: Optional[str] = None
    alpha_vantage: Optional[str] = None
    finnhub: Optional[str] = None
    polygon: Optional[str] = None
    tiingo: Optional[str] = None
    openai: Optional[str] = None
    marketstack: Optional[str] = None
    nasdaq: Optional[str] = None
    eodhd: Optional[str] = None
    simfin: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "APIKeys":
        """Load API keys via unified secret precedence."""
        def _get_key(key: str) -> Optional[str]:
            return get_secret(key, None)
        
        return cls(
            fmp=_get_key("FMP_API_KEY"),
            alpha_vantage=_get_key("ALPHA_VANTAGE_API_KEY"),
            finnhub=_get_key("FINNHUB_API_KEY"),
            polygon=_get_key("POLYGON_API_KEY"),
            tiingo=_get_key("TIINGO_API_KEY"),
            openai=_get_key("OPENAI_API_KEY"),
            marketstack=_get_key("MARKETSTACK_API_KEY"),
            nasdaq=_get_key("NASDAQ_API_KEY"),
            eodhd=_get_key("EODHD_API_KEY"),
            simfin=_get_key("SIMFIN_API_KEY"),
        )
    
    def has(self, provider: str) -> bool:
        """Check if a specific provider API key is available.

        Args:
            provider: Provider name matching an attribute (e.g. 'fmp', 'finnhub', 'openai').
        """
        val = getattr(self, provider, None)
        return val is not None and len(str(val)) > 0

    # Backward-compatible convenience methods
    def has_alpha_vantage(self) -> bool: return self.has("alpha_vantage")
    def has_finnhub(self) -> bool: return self.has("finnhub")
    def has_polygon(self) -> bool: return self.has("polygon")
    def has_tiingo(self) -> bool: return self.has("tiingo")
    def has_fmp(self) -> bool: return self.has("fmp")
    def has_openai(self) -> bool: return self.has("openai")
    def has_marketstack(self) -> bool: return self.has("marketstack")
    def has_nasdaq(self) -> bool: return self.has("nasdaq")
    def has_eodhd(self) -> bool: return self.has("eodhd")
    def has_simfin(self) -> bool: return self.has("simfin")


# Global config instance
_config: Optional[Config] = None
_api_keys: Optional[APIKeys] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def get_api_keys() -> APIKeys:
    """Get global API keys instance."""
    global _api_keys
    if _api_keys is None:
        _api_keys = APIKeys.from_env()
    return _api_keys

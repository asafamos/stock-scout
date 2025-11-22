"""
Configuration and constants for Stock Scout.
All configurable parameters in one place.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import streamlit as st

# Load .env early for Config defaults
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_config_value(key: str, default: str) -> str:
    """Get config value from Streamlit secrets (priority) or environment."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    # Fall back to environment variable
    return os.getenv(key, default)


@dataclass
class Config:
    """Main configuration for Stock Scout."""
    
    # Budget & Position Sizing
    budget_total: float = 5000.0
    min_position: float = 500.0
    max_position_pct: float = 15.0
    
    # Universe & Data
    universe_limit: int = int(_get_config_value('UNIVERSE_LIMIT', '50'))
    lookback_days: int = int(_get_config_value('LOOKBACK_DAYS', '90'))
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
    
    # Technical Weights (Optimized based on backtest analysis)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "ma": 0.20,
        "mom": 0.25,           # Reduced from 0.30
        "rsi": 0.20,           # INCREASED from 0.12 - RSI is highly predictive!
        "near_high_bell": 0.12,
        "vol": 0.02,           # REDUCED from 0.08 - volume surge doesn't predict well
        "overext": 0.08,
        "pullback": 0.05,
        "risk_reward": 0.05,   # Increased from 0.03
        "macd": 0.02,          # Increased from 0.01
        "adx": 0.01,
    })
    
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
    top_validate_k: int = 12
    
    # Results
    topn_results: int = int(_get_config_value('TOPN_RESULTS', '15'))
    topk_recommend: int = int(_get_config_value('TOPK_RECOMMEND', '5'))
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "budget_total": self.budget_total,
            "min_position": self.min_position,
            "max_position_pct": self.max_position_pct,
            "universe_limit": self.universe_limit,
            "lookback_days": self.lookback_days,
            "smart_scan": self.smart_scan,
            "min_price": self.min_price,
            "min_avg_volume": self.min_avg_volume,
            "min_dollar_volume": self.min_dollar_volume,
            "min_market_cap": self.min_market_cap,
            "sector_cap_max": self.sector_cap_max,
            "max_sector_allocation_pct": self.max_sector_allocation_pct,
            "max_positions_per_sector": self.max_positions_per_sector,
        }


@dataclass
class APIKeys:
    """API keys configuration."""
    
    alpha_vantage: Optional[str] = None
    finnhub: Optional[str] = None
    polygon: Optional[str] = None
    tiingo: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "APIKeys":
        """Load API keys from environment variables or Streamlit secrets."""
        def _get_key(key: str) -> Optional[str]:
            # Try Streamlit secrets first (supports nested sections)
            try:
                if "secrets" in dir(st):
                    sec = st.secrets
                    if isinstance(sec, dict) and key in sec:
                        return sec[key]
                    for section in ("api_keys", "keys", "secrets", "tokens"):
                        try:
                            container = sec.get(section) if hasattr(sec, 'get') else sec[section]
                            if isinstance(container, dict) and key in container:
                                return container[key]
                        except Exception:
                            continue
            except Exception:
                pass
            # Fall back to environment variable
            return os.getenv(key)
        
        return cls(
            alpha_vantage=_get_key("ALPHA_VANTAGE_API_KEY"),
            finnhub=_get_key("FINNHUB_API_KEY"),
            polygon=_get_key("POLYGON_API_KEY"),
            tiingo=_get_key("TIINGO_API_KEY"),
        )
    
    def has_alpha_vantage(self) -> bool:
        """Check if Alpha Vantage key is available."""
        return self.alpha_vantage is not None and len(self.alpha_vantage) > 0
    
    def has_finnhub(self) -> bool:
        """Check if Finnhub key is available."""
        return self.finnhub is not None and len(self.finnhub) > 0
    
    def has_polygon(self) -> bool:
        """Check if Polygon key is available."""
        return self.polygon is not None and len(self.polygon) > 0
    
    def has_tiingo(self) -> bool:
        """Check if Tiingo key is available."""
        return self.tiingo is not None and len(self.tiingo) > 0


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

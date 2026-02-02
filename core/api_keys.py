"""
Centralized API Key Management.
All API keys must be loaded from environment variables.
NO hardcoded keys anywhere in the codebase.

Usage:
    from core.api_keys import get_api_key, validate_keys, log_api_key_status
    
    # Get a required key (raises if missing)
    polygon_key = get_api_key("POLYGON_API_KEY", required=True)
    
    # Get an optional key (returns None if missing)
    alpha_key = get_api_key("ALPHA_VANTAGE_API_KEY")
    
    # Check multiple keys at once
    status = validate_keys(["POLYGON_API_KEY", "FINNHUB_API_KEY"])
    
    # Log status of all known keys (masked)
    log_api_key_status()
"""
import os
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIKeyStatus:
    """Status of a single API key."""
    name: str
    available: bool
    masked_value: str  # Show only last 4 chars


class APIKeyManager:
    """Centralized API key management with validation."""

    KNOWN_KEYS = [
        "POLYGON_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "FINNHUB_API_KEY",
        "TIINGO_API_KEY",
        "FMP_API_KEY",
        "MARKETSTACK_API_KEY",
        "NASDAQ_API_KEY",
        "EODHD_API_KEY",
        "SIMFIN_API_KEY",
        "OPENAI_API_KEY",
    ]

    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}

    def get_key(self, name: str, required: bool = False) -> Optional[str]:
        """
        Get API key from environment.
        
        Args:
            name: Name of the environment variable (e.g., "POLYGON_API_KEY")
            required: If True, raises EnvironmentError when key is missing
            
        Returns:
            The API key value, or None if not set and not required
            
        Raises:
            EnvironmentError: If required=True and key is not set
        """
        if name not in self._cache:
            self._cache[name] = os.environ.get(name)

        value = self._cache[name]

        if required and not value:
            raise EnvironmentError(
                f"Required API key '{name}' is not set. "
                f"Please set it in your environment or .env file."
            )

        return value

    def clear_cache(self) -> None:
        """Clear the key cache (useful for testing)."""
        self._cache.clear()

    def validate_required_keys(self, required: List[str]) -> Dict[str, bool]:
        """
        Check which required keys are available.
        
        Args:
            required: List of key names to check
            
        Returns:
            Dict mapping key name to availability (True/False)
        """
        return {key: bool(self.get_key(key)) for key in required}

    def get_status(self) -> List[APIKeyStatus]:
        """
        Get status of all known API keys (for logging/debugging).
        
        Returns:
            List of APIKeyStatus objects with masked values
        """
        statuses = []
        for key in self.KNOWN_KEYS:
            value = self.get_key(key)
            if value and len(value) > 4:
                masked = f"***{value[-4:]}"
            elif value:
                masked = "***"
            else:
                masked = "NOT SET"
            statuses.append(APIKeyStatus(key, bool(value), masked))
        return statuses

    def log_status(self) -> None:
        """Log API key availability (with masked values for security)."""
        logger.info("API Key Status:")
        for status in self.get_status():
            icon = "✓" if status.available else "✗"
            logger.info(f"  {icon} {status.name}: {status.masked_value}")

    def get_available_keys(self) -> List[str]:
        """
        Get list of available (set) API keys.
        
        Returns:
            List of key names that are set in the environment
        """
        return [key for key in self.KNOWN_KEYS if self.get_key(key)]

    def get_missing_keys(self) -> List[str]:
        """
        Get list of missing (not set) API keys.
        
        Returns:
            List of key names that are NOT set in the environment
        """
        return [key for key in self.KNOWN_KEYS if not self.get_key(key)]


# Global instance
_manager = APIKeyManager()


def get_api_key(name: str, required: bool = False) -> Optional[str]:
    """
    Get API key from environment.
    
    This is the primary function to use for API key access.
    
    Args:
        name: Name of the environment variable (e.g., "POLYGON_API_KEY")
        required: If True, raises EnvironmentError when key is missing
        
    Returns:
        The API key value, or None if not set and not required
        
    Raises:
        EnvironmentError: If required=True and key is not set
        
    Example:
        # Required key - will raise if not set
        polygon_key = get_api_key("POLYGON_API_KEY", required=True)
        
        # Optional key - returns None if not set
        alpha_key = get_api_key("ALPHA_VANTAGE_API_KEY")
    """
    return _manager.get_key(name, required)


def validate_keys(required: List[str]) -> Dict[str, bool]:
    """
    Validate multiple API keys at once.
    
    Args:
        required: List of key names to check
        
    Returns:
        Dict mapping key name to availability (True/False)
        
    Example:
        status = validate_keys(["POLYGON_API_KEY", "FINNHUB_API_KEY"])
        if not all(status.values()):
            missing = [k for k, v in status.items() if not v]
            print(f"Missing keys: {missing}")
    """
    return _manager.validate_required_keys(required)


def log_api_key_status() -> None:
    """
    Log status of all known API keys (with masked values).
    
    Useful for debugging and startup diagnostics.
    Values are masked for security (only last 4 chars shown).
    """
    _manager.log_status()


def get_available_keys() -> List[str]:
    """Get list of available (set) API key names."""
    return _manager.get_available_keys()


def get_missing_keys() -> List[str]:
    """Get list of missing (not set) API key names."""
    return _manager.get_missing_keys()


def clear_cache() -> None:
    """Clear the API key cache (useful for testing)."""
    _manager.clear_cache()

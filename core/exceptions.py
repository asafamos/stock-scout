"""
Stock Scout Custom Exceptions
=============================

Centralized exception hierarchy for proper error handling and debugging.
All exceptions inherit from StockScoutError for easy catching at boundaries.

Usage:
    from core.exceptions import DataFetchError, RateLimitError, DataValidationError
    
    try:
        data = fetch_from_provider(ticker)
    except RateLimitError as e:
        # Switch to fallback provider
        logger.warning(f"Rate limited: {e.provider}, switching to next")
    except DataFetchError as e:
        # Log and continue with cached data
        logger.error(f"Fetch failed: {e}")
"""

from typing import Optional, Any, Dict


class StockScoutError(Exception):
    """Base exception for all Stock Scout errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


# =============================================================================
# Data Fetch Errors
# =============================================================================

class DataFetchError(StockScoutError):
    """Error fetching data from an external provider."""
    
    def __init__(
        self,
        provider: str,
        ticker: Optional[str] = None,
        reason: str = "Unknown error",
        status_code: Optional[int] = None,
        url: Optional[str] = None
    ):
        self.provider = provider
        self.ticker = ticker
        self.reason = reason
        self.status_code = status_code
        self.url = url
        
        msg = f"Data fetch failed from {provider}"
        if ticker:
            msg += f" for {ticker}"
        msg += f": {reason}"
        if status_code:
            msg += f" (HTTP {status_code})"
        
        super().__init__(msg, {
            "provider": provider,
            "ticker": ticker,
            "status_code": status_code
        })


class RateLimitError(DataFetchError):
    """API rate limit exceeded (HTTP 429)."""
    
    def __init__(
        self,
        provider: str,
        ticker: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        self.retry_after = retry_after
        reason = "Rate limit exceeded"
        if retry_after:
            reason += f" (retry after {retry_after}s)"
        super().__init__(
            provider=provider,
            ticker=ticker,
            reason=reason,
            status_code=429
        )


class AuthenticationError(DataFetchError):
    """API authentication failed (HTTP 401/403)."""
    
    def __init__(self, provider: str, reason: str = "Invalid or missing API key"):
        super().__init__(
            provider=provider,
            reason=reason,
            status_code=401
        )


class ProviderUnavailableError(DataFetchError):
    """Provider is temporarily unavailable (HTTP 5xx or connection error)."""
    
    def __init__(self, provider: str, reason: str = "Service unavailable"):
        super().__init__(
            provider=provider,
            reason=reason,
            status_code=503
        )


# =============================================================================
# Data Validation Errors
# =============================================================================

class DataValidationError(StockScoutError):
    """Data failed validation checks."""
    
    def __init__(
        self,
        context: str,
        details: str,
        field: Optional[str] = None,
        value: Optional[Any] = None
    ):
        self.field = field
        self.value = value
        
        msg = f"Validation failed in {context}: {details}"
        if field:
            msg += f" (field={field}, value={value})"
        
        super().__init__(msg, {
            "context": context,
            "field": field,
            "value": str(value)[:100] if value is not None else None
        })


class InsufficientDataError(DataValidationError):
    """Not enough data points for calculation."""
    
    def __init__(self, ticker: str, required: int, actual: int, context: str = "calculation"):
        super().__init__(
            context=context,
            details=f"Need {required} rows, got {actual}",
            field="row_count",
            value=actual
        )
        self.ticker = ticker
        self.required = required
        self.actual = actual


class MissingColumnError(DataValidationError):
    """Required column missing from DataFrame."""
    
    def __init__(self, column: str, context: str = "DataFrame"):
        super().__init__(
            context=context,
            details=f"Missing required column",
            field="column",
            value=column
        )
        self.column = column


# =============================================================================
# Model Errors
# =============================================================================

class ModelError(StockScoutError):
    """Error in ML model operations."""
    
    def __init__(self, model_name: str, reason: str, recoverable: bool = True):
        self.model_name = model_name
        self.recoverable = recoverable
        
        msg = f"Model error ({model_name}): {reason}"
        super().__init__(msg, {
            "model": model_name,
            "recoverable": recoverable
        })


class ModelNotLoadedError(ModelError):
    """ML model not loaded or unavailable."""
    
    def __init__(self, model_name: str = "ML_20d"):
        super().__init__(
            model_name=model_name,
            reason="Model not loaded",
            recoverable=True
        )


class FeatureMismatchError(ModelError):
    """Input features don't match model expectations."""
    
    def __init__(self, model_name: str, expected: int, actual: int, missing: Optional[list] = None):
        self.expected = expected
        self.actual = actual
        self.missing = missing or []
        
        reason = f"Expected {expected} features, got {actual}"
        if missing:
            reason += f". Missing: {missing[:5]}"
        
        super().__init__(
            model_name=model_name,
            reason=reason,
            recoverable=True
        )


# =============================================================================
# Pipeline Errors
# =============================================================================

class PipelineError(StockScoutError):
    """Error in pipeline execution."""
    
    def __init__(self, stage: str, reason: str, recoverable: bool = False):
        self.stage = stage
        self.recoverable = recoverable
        
        msg = f"Pipeline failed at {stage}: {reason}"
        super().__init__(msg, {
            "stage": stage,
            "recoverable": recoverable
        })


class ScoringError(StockScoutError):
    """Error in score calculation."""
    
    def __init__(self, ticker: str, component: str, reason: str):
        self.ticker = ticker
        self.component = component
        
        msg = f"Scoring failed for {ticker} ({component}): {reason}"
        super().__init__(msg, {
            "ticker": ticker,
            "component": component
        })


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(StockScoutError):
    """Invalid or missing configuration."""
    
    def __init__(self, setting: str, reason: str):
        self.setting = setting
        msg = f"Configuration error for {setting}: {reason}"
        super().__init__(msg, {"setting": setting})


class MissingAPIKeyError(ConfigurationError):
    """Required API key not configured."""
    
    def __init__(self, provider: str):
        super().__init__(
            setting=f"{provider.upper()}_API_KEY",
            reason=f"API key for {provider} not found in environment or secrets"
        )
        self.provider = provider


# =============================================================================
# Utility Functions
# =============================================================================

def classify_http_error(status_code: int, provider: str, ticker: Optional[str] = None) -> DataFetchError:
    """Convert HTTP status code to appropriate exception."""
    if status_code == 429:
        return RateLimitError(provider=provider, ticker=ticker)
    elif status_code in (401, 403):
        return AuthenticationError(provider=provider)
    elif status_code >= 500:
        return ProviderUnavailableError(provider=provider)
    else:
        return DataFetchError(
            provider=provider,
            ticker=ticker,
            reason=f"HTTP error",
            status_code=status_code
        )


def is_recoverable(exc: Exception) -> bool:
    """Check if an exception is recoverable (should try fallback)."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, ProviderUnavailableError):
        return True
    if isinstance(exc, ModelError) and exc.recoverable:
        return True
    if isinstance(exc, DataFetchError):
        # Network errors are usually recoverable
        return exc.status_code is None or exc.status_code >= 500
    return False

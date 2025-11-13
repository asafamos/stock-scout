"""
Centralized logging configuration for Stock Scout.
"""
import logging
import sys
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(level: int = logging.INFO, name: str = "stock_scout") -> logging.Logger:
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    
    # Remove existing handlers
    _logger.handlers.clear()
    
    # Console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Format: [TIME] [LEVEL] [MODULE] Message
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    _logger.addHandler(handler)
    _logger.propagate = False
    
    return _logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Optional module name for logger
        
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None:
        _logger = setup_logging()
    
    if name:
        return _logger.getChild(name)
    
    return _logger


def mask_sensitive(text: str, show_last: int = 4) -> str:
    """
    Mask sensitive information (API keys, tokens, etc).
    
    Args:
        text: Text to mask
        show_last: Number of characters to show at end
        
    Returns:
        Masked text
        
    Examples:
        >>> mask_sensitive("ABCDEFGHIJK1234", 4)
        '***1234'
    """
    if not text or not isinstance(text, str):
        return "***"
    
    if len(text) <= show_last:
        return "***"
    
    return "***" + text[-show_last:]

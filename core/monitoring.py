"""
Production Monitoring System for Stock Scout

Provides:
- API health monitoring (rate limits, failures, latency)
- Model drift detection
- Scan quality alerts
- Performance degradation warnings
- Slack/Discord/Email notifications

Usage:
    from core.monitoring import Monitor, alert
    
    monitor = Monitor()
    monitor.check_api_health()
    
    # Or use decorator
    @alert(on_error=True, on_slow=5.0)
    def my_function():
        ...
"""
from __future__ import annotations
import logging
import time
import json
import functools
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
import threading
import atexit
import os
import traceback

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alerts."""
    # Thresholds
    api_failure_threshold: int = 3  # Consecutive failures before alert
    latency_threshold_ms: float = 5000  # 5 seconds
    model_drift_threshold: float = 0.10  # 10% prediction drift
    scan_quality_min_tickers: int = 20
    
    # Notification channels
    slack_webhook: Optional[str] = None
    discord_webhook: Optional[str] = None
    email_to: Optional[str] = None
    
    # Logging
    log_dir: str = "logs/monitoring"
    log_retention_days: int = 30


@dataclass
class APIHealthStatus:
    """Health status for a single API."""
    name: str
    is_healthy: bool
    last_check: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    avg_latency_ms: float = 0.0
    calls_today: int = 0
    rate_limit_remaining: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "is_healthy": self.is_healthy,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "avg_latency_ms": self.avg_latency_ms,
            "calls_today": self.calls_today,
            "rate_limit_remaining": self.rate_limit_remaining
        }


@dataclass
class ModelDriftStats:
    """Track model prediction drift."""
    baseline_mean: float = 0.5
    baseline_std: float = 0.2
    current_mean: float = 0.5
    current_std: float = 0.2
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    last_check: Optional[datetime] = None


@dataclass
class ScanQualityMetrics:
    """Quality metrics for a scan."""
    timestamp: datetime
    tickers_scanned: int
    tickers_passed_filters: int
    avg_probability: float
    max_probability: float
    sector_diversity: float  # Number of unique sectors
    has_warnings: bool = False
    warnings: List[str] = field(default_factory=list)


class AlertHistory:
    """Track alert history to prevent spam."""
    
    def __init__(self, cooldown_minutes: int = 30):
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.history: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def can_alert(self, alert_key: str) -> bool:
        """Check if we can send an alert (not in cooldown)."""
        with self._lock:
            last_sent = self.history.get(alert_key)
            if last_sent is None:
                return True
            return datetime.now() - last_sent > self.cooldown
    
    def record_alert(self, alert_key: str) -> None:
        """Record that an alert was sent."""
        with self._lock:
            self.history[alert_key] = datetime.now()


class Monitor:
    """
    Central monitoring system for Stock Scout.
    
    Tracks:
    - API health (rate limits, failures, latency)
    - Model performance (drift, predictions)
    - Scan quality (coverage, filters, diversity)
    """
    
    _instance: Optional["Monitor"] = None
    
    @classmethod
    def get_instance(cls) -> "Monitor":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """Initialize monitor."""
        self.config = config or AlertConfig()
        self.api_health: Dict[str, APIHealthStatus] = {}
        self.model_drift = ModelDriftStats()
        self.scan_history: List[ScanQualityMetrics] = []
        self.alert_history = AlertHistory()
        self._metrics_buffer: List[Dict] = []
        self._lock = threading.Lock()
        
        # Create log directory
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Register cleanup on exit
        atexit.register(self._flush_metrics)
        
        # Initialize API health tracking
        self._init_api_health()
    
    def _init_api_health(self) -> None:
        """Initialize health status for known APIs."""
        apis = [
            "alpha_vantage", "finnhub", "polygon", "tiingo",
            "fmp", "yfinance", "marketstack", "nasdaq"
        ]
        now = datetime.now()
        for api in apis:
            self.api_health[api] = APIHealthStatus(
                name=api,
                is_healthy=True,
                last_check=now
            )
    
    def record_api_call(
        self,
        api_name: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
        rate_limit_remaining: Optional[int] = None
    ) -> None:
        """Record an API call for monitoring."""
        with self._lock:
            if api_name not in self.api_health:
                self.api_health[api_name] = APIHealthStatus(
                    name=api_name,
                    is_healthy=True,
                    last_check=datetime.now()
                )
            
            status = self.api_health[api_name]
            status.last_check = datetime.now()
            status.calls_today += 1
            
            # Update latency (exponential moving average)
            alpha = 0.1
            status.avg_latency_ms = (
                alpha * latency_ms + (1 - alpha) * status.avg_latency_ms
            )
            
            if success:
                status.consecutive_failures = 0
                status.is_healthy = True
            else:
                status.consecutive_failures += 1
                status.last_error = error
                
                if status.consecutive_failures >= self.config.api_failure_threshold:
                    status.is_healthy = False
                    self._send_alert(
                        f"API_FAILURE:{api_name}",
                        f"API {api_name} has failed {status.consecutive_failures} times",
                        severity="high"
                    )
            
            if rate_limit_remaining is not None:
                status.rate_limit_remaining = rate_limit_remaining
                if rate_limit_remaining < 10:
                    self._send_alert(
                        f"RATE_LIMIT:{api_name}",
                        f"API {api_name} rate limit low: {rate_limit_remaining} remaining",
                        severity="medium"
                    )
            
            # Check latency
            if latency_ms > self.config.latency_threshold_ms:
                self._send_alert(
                    f"SLOW_API:{api_name}",
                    f"API {api_name} slow response: {latency_ms:.0f}ms",
                    severity="low"
                )
        
        # Log metric
        self._log_metric({
            "type": "api_call",
            "api": api_name,
            "success": success,
            "latency_ms": latency_ms,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def check_model_drift(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check for model prediction drift.
        
        Returns True if drift is detected.
        """
        current_mean = np.mean(predictions)
        current_std = np.std(predictions)
        
        # Compare to baseline
        mean_drift = abs(current_mean - self.model_drift.baseline_mean)
        std_drift = abs(current_std - self.model_drift.baseline_std)
        
        drift_magnitude = max(mean_drift, std_drift)
        
        self.model_drift.current_mean = current_mean
        self.model_drift.current_std = current_std
        self.model_drift.drift_magnitude = drift_magnitude
        self.model_drift.last_check = datetime.now()
        
        if drift_magnitude > self.config.model_drift_threshold:
            self.model_drift.drift_detected = True
            self._send_alert(
                "MODEL_DRIFT",
                f"Model drift detected: mean={current_mean:.3f} (baseline={self.model_drift.baseline_mean:.3f}), "
                f"drift={drift_magnitude:.3f}",
                severity="high"
            )
            return True
        
        self.model_drift.drift_detected = False
        return False
    
    def update_baseline(self, predictions: np.ndarray) -> None:
        """Update model baseline statistics."""
        self.model_drift.baseline_mean = np.mean(predictions)
        self.model_drift.baseline_std = np.std(predictions)
        logger.info(
            f"Updated model baseline: mean={self.model_drift.baseline_mean:.3f}, "
            f"std={self.model_drift.baseline_std:.3f}"
        )
    
    def record_scan(
        self,
        tickers_scanned: int,
        tickers_passed: int,
        probabilities: List[float],
        sectors: List[str]
    ) -> ScanQualityMetrics:
        """Record scan quality metrics."""
        metrics = ScanQualityMetrics(
            timestamp=datetime.now(),
            tickers_scanned=tickers_scanned,
            tickers_passed_filters=tickers_passed,
            avg_probability=np.mean(probabilities) if probabilities else 0,
            max_probability=max(probabilities) if probabilities else 0,
            sector_diversity=len(set(sectors))
        )
        
        # Check for warnings
        if tickers_passed < self.config.scan_quality_min_tickers:
            metrics.has_warnings = True
            metrics.warnings.append(
                f"Low filter pass rate: only {tickers_passed} tickers passed"
            )
        
        if metrics.avg_probability > 0.8:
            metrics.has_warnings = True
            metrics.warnings.append(
                f"Unusually high avg probability: {metrics.avg_probability:.2f}"
            )
        
        if metrics.sector_diversity < 3:
            metrics.has_warnings = True
            metrics.warnings.append(
                f"Low sector diversity: only {metrics.sector_diversity} sectors"
            )
        
        if metrics.has_warnings:
            self._send_alert(
                "SCAN_QUALITY",
                f"Scan quality warnings: {'; '.join(metrics.warnings)}",
                severity="medium"
            )
        
        with self._lock:
            self.scan_history.append(metrics)
            # Keep last 100 scans
            if len(self.scan_history) > 100:
                self.scan_history = self.scan_history[-100:]
        
        self._log_metric({
            "type": "scan",
            "tickers_scanned": tickers_scanned,
            "tickers_passed": tickers_passed,
            "avg_probability": metrics.avg_probability,
            "sector_diversity": metrics.sector_diversity,
            "timestamp": datetime.now().isoformat()
        })
        
        return metrics
    
    def _send_alert(
        self,
        alert_key: str,
        message: str,
        severity: str = "medium"
    ) -> None:
        """Send an alert via configured channels."""
        if not self.alert_history.can_alert(alert_key):
            logger.debug(f"Alert {alert_key} in cooldown, skipping")
            return
        
        self.alert_history.record_alert(alert_key)
        
        # Log locally
        log_level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR
        }.get(severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{severity.upper()}]: {message}")
        
        # Try Slack
        if self.config.slack_webhook:
            self._send_slack(message, severity)
        
        # Try Discord
        if self.config.discord_webhook:
            self._send_discord(message, severity)
        
        # Log to file
        self._log_alert(alert_key, message, severity)
    
    def _send_slack(self, message: str, severity: str) -> None:
        """Send alert to Slack."""
        try:
            import requests
            
            color = {"low": "#36a64f", "medium": "#ff9800", "high": "#d32f2f"}.get(severity, "#777")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Stock Scout Alert ({severity.upper()})",
                    "text": message,
                    "ts": int(time.time())
                }]
            }
            
            requests.post(
                self.config.slack_webhook,
                json=payload,
                timeout=5
            )
        except Exception as e:
            logger.debug(f"Failed to send Slack alert: {e}")
    
    def _send_discord(self, message: str, severity: str) -> None:
        """Send alert to Discord."""
        try:
            import requests
            
            color = {"low": 0x36a64f, "medium": 0xff9800, "high": 0xd32f2f}.get(severity, 0x777777)
            
            payload = {
                "embeds": [{
                    "title": f"Stock Scout Alert ({severity.upper()})",
                    "description": message,
                    "color": color
                }]
            }
            
            requests.post(
                self.config.discord_webhook,
                json=payload,
                timeout=5
            )
        except Exception as e:
            logger.debug(f"Failed to send Discord alert: {e}")
    
    def _log_alert(self, key: str, message: str, severity: str) -> None:
        """Log alert to file."""
        alert_file = Path(self.config.log_dir) / "alerts.jsonl"
        
        entry = {
            "key": key,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(alert_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log alert: {e}")
    
    def _log_metric(self, metric: Dict) -> None:
        """Buffer metric for logging."""
        with self._lock:
            self._metrics_buffer.append(metric)
            
            if len(self._metrics_buffer) >= 100:
                self._flush_metrics()
    
    def _flush_metrics(self) -> None:
        """Flush metrics buffer to file."""
        with self._lock:
            if not self._metrics_buffer:
                return
            
            date_str = datetime.now().strftime("%Y-%m-%d")
            metrics_file = Path(self.config.log_dir) / f"metrics_{date_str}.jsonl"
            
            try:
                with open(metrics_file, "a") as f:
                    for metric in self._metrics_buffer:
                        f.write(json.dumps(metric) + "\n")
                self._metrics_buffer = []
            except Exception as e:
                logger.debug(f"Failed to flush metrics: {e}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get overall health report."""
        unhealthy_apis = [
            api for api, status in self.api_health.items()
            if not status.is_healthy
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "apis": {
                name: status.to_dict()
                for name, status in self.api_health.items()
            },
            "unhealthy_apis": unhealthy_apis,
            "model_drift": {
                "detected": self.model_drift.drift_detected,
                "magnitude": self.model_drift.drift_magnitude,
                "current_mean": self.model_drift.current_mean
            },
            "recent_scans": len(self.scan_history),
            "overall_healthy": len(unhealthy_apis) == 0 and not self.model_drift.drift_detected
        }
    
    def print_status(self) -> None:
        """Print current monitoring status."""
        report = self.get_health_report()
        
        print("\n" + "="*60)
        print("STOCK SCOUT HEALTH REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Health: {'✅ HEALTHY' if report['overall_healthy'] else '❌ ISSUES DETECTED'}")
        
        print("\nAPI Status:")
        for name, status in report["apis"].items():
            health = "✅" if status["is_healthy"] else "❌"
            print(f"  {name}: {health} | Latency: {status['avg_latency_ms']:.0f}ms | Calls: {status['calls_today']}")
        
        print("\nModel Drift:")
        drift = report["model_drift"]
        drift_status = "⚠️ DRIFT DETECTED" if drift["detected"] else "✅ STABLE"
        print(f"  Status: {drift_status}")
        print(f"  Magnitude: {drift['magnitude']:.3f}")
        
        print("\nRecent Scans: " + str(report["recent_scans"]))


def alert(
    on_error: bool = True,
    on_slow: Optional[float] = None,
    api_name: Optional[str] = None
) -> Callable:
    """
    Decorator to add monitoring to a function.
    
    Args:
        on_error: Alert on exception
        on_slow: Alert if execution takes longer than this (seconds)
        api_name: Track as API call with this name
    
    Usage:
        @alert(on_error=True, on_slow=5.0, api_name="external_api")
        def fetch_data():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = Monitor.get_instance()
            start = time.time()
            error = None
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                
                if on_error:
                    monitor._send_alert(
                        f"ERROR:{func.__name__}",
                        f"Function {func.__name__} failed: {error}",
                        severity="high"
                    )
                raise
            finally:
                elapsed_ms = (time.time() - start) * 1000
                
                if api_name:
                    monitor.record_api_call(
                        api_name,
                        success=success,
                        latency_ms=elapsed_ms,
                        error=error
                    )
                
                if on_slow and elapsed_ms > on_slow * 1000:
                    monitor._send_alert(
                        f"SLOW:{func.__name__}",
                        f"Function {func.__name__} slow: {elapsed_ms:.0f}ms",
                        severity="low"
                    )
        
        return wrapper
    return decorator


class APICallTracker:
    """
    Context manager for tracking API calls.
    
    Usage:
        with APICallTracker("alpha_vantage") as tracker:
            response = requests.get(...)
            tracker.set_rate_limit(response.headers.get("X-RateLimit-Remaining"))
    """
    
    def __init__(self, api_name: str):
        self.api_name = api_name
        self.monitor = Monitor.get_instance()
        self.start_time: float = 0
        self.rate_limit: Optional[int] = None
        self.error: Optional[str] = None
        self.success = True
    
    def __enter__(self) -> "APICallTracker":
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        if exc_val:
            self.success = False
            self.error = str(exc_val)
        
        self.monitor.record_api_call(
            self.api_name,
            success=self.success,
            latency_ms=elapsed_ms,
            error=self.error,
            rate_limit_remaining=self.rate_limit
        )
        
        return False  # Don't suppress exceptions
    
    def set_rate_limit(self, remaining: Optional[Union[int, str]]) -> None:
        """Set rate limit remaining from response header."""
        if remaining is not None:
            try:
                self.rate_limit = int(remaining)
            except (ValueError, TypeError):
                pass
    
    def mark_failure(self, error: str) -> None:
        """Mark this call as failed."""
        self.success = False
        self.error = error


# Convenience functions
def get_monitor() -> Monitor:
    """Get the global monitor instance."""
    return Monitor.get_instance()


def log_api_call(api_name: str, success: bool, latency_ms: float, error: Optional[str] = None) -> None:
    """Log an API call to the monitor."""
    Monitor.get_instance().record_api_call(api_name, success, latency_ms, error)


def check_health() -> Dict[str, Any]:
    """Get current health status."""
    return Monitor.get_instance().get_health_report()


if __name__ == "__main__":
    # Demo
    monitor = Monitor()
    
    # Simulate some API calls
    monitor.record_api_call("alpha_vantage", True, 150.0)
    monitor.record_api_call("finnhub", True, 80.0)
    monitor.record_api_call("polygon", False, 5000.0, "Timeout")
    monitor.record_api_call("polygon", False, 5000.0, "Timeout")
    monitor.record_api_call("polygon", False, 5000.0, "Timeout")
    
    # Simulate a scan
    monitor.record_scan(
        tickers_scanned=500,
        tickers_passed=45,
        probabilities=[0.75, 0.72, 0.68, 0.65, 0.62],
        sectors=["Technology", "Healthcare", "Finance", "Consumer"]
    )
    
    # Check for drift
    fake_predictions = np.random.normal(0.5, 0.2, 100)
    monitor.update_baseline(fake_predictions)
    
    drifted_predictions = np.random.normal(0.7, 0.2, 100)  # Mean shifted
    monitor.check_model_drift(drifted_predictions)
    
    # Print status
    monitor.print_status()

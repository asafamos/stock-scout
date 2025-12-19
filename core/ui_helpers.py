"""
UI helpers for Stock Scout — 2026 modernization
------------------------------------------------
Clean separation of concerns: dynamic status, sources overview, progress tracking,
and performance profiling.
"""
import os
import time
from typing import Dict, Set, Optional, Tuple
import streamlit as st


class StatusManager:
    """Centralized status and progress management with performance instrumentation."""
    
    def __init__(self, stages: list[str]):
        """Initialize with predefined pipeline stages.
        
        Args:
            stages: List of stage names in order
        """
        self.stages = stages
        self.current_stage = 0
        self._progress = st.progress(0.0)
        self._status = st.empty()
        self._details = st.empty()
        
        # Timing instrumentation
        self._stage_times: Dict[str, float] = {}  # stage_name -> duration (seconds)
        self._stage_start_times: Dict[str, float] = {}  # stage_name -> start time (perf_counter)
        self._total_start = time.perf_counter()
        
    def advance(self, detail: str = "") -> None:
        """Advance to next stage with optional detail message."""
        # Record end time of previous stage if it was started
        if self.current_stage > 0:
            prev_stage_name = self.stages[self.current_stage - 1]
            if prev_stage_name in self._stage_start_times:
                elapsed = time.perf_counter() - self._stage_start_times[prev_stage_name]
                self._stage_times[prev_stage_name] = elapsed
        
        self.current_stage += 1
        progress = min(self.current_stage / len(self.stages), 1.0)
        self._progress.progress(progress)
        
        stage_name = self.stages[self.current_stage - 1] if self.current_stage <= len(self.stages) else "Complete"
        self._status.markdown(f"**Stage {self.current_stage}/{len(self.stages)}:** {stage_name}")
        
        # Record start time for new stage
        if self.current_stage <= len(self.stages):
            self._stage_start_times[stage_name] = time.perf_counter()
        
        if detail:
            self._details.caption(detail)
    
    def update_detail(self, message: str) -> None:
        """Update detail message without advancing stage."""
        self._details.caption(message)
    
    def complete(self, message: str = "✅ Pipeline complete") -> None:
        """Mark pipeline as complete, recording final stage timing."""
        # Record end time of last active stage
        if self.current_stage > 0:
            last_stage_name = self.stages[self.current_stage - 1] if self.current_stage <= len(self.stages) else None
            if last_stage_name and last_stage_name in self._stage_start_times:
                elapsed = time.perf_counter() - self._stage_start_times[last_stage_name]
                self._stage_times[last_stage_name] = elapsed
        
        self._progress.progress(1.0)
        self._status.success(message)
        self._details.empty()
    
    def render_timing_report(self) -> None:
        """Render performance report showing stage durations and percentages.
        
        Only displays if DEBUG_MODE environment variable is set or debug_mode session state is True.
        """
        debug_mode = os.getenv("DEBUG_MODE") or st.session_state.get("debug_mode", False)
        if not debug_mode:
            return
        
        if not self._stage_times:
            return
        
        with st.expander("⏱️ Performance Report", expanded=False):
            # Calculate total time
            total_time = sum(self._stage_times.values())
            
            # Build table data
            table_rows = []
            for stage_name in self.stages:
                if stage_name in self._stage_times:
                    duration = self._stage_times[stage_name]
                    percentage = (duration / total_time * 100) if total_time > 0 else 0
                    table_rows.append({
                        "Stage": stage_name,
                        "Duration (s)": f"{duration:.2f}",
                        "% of Total": f"{percentage:.1f}%"
                    })
            
            if table_rows:
                import pandas as pd
                df_timing = pd.DataFrame(table_rows)
                st.dataframe(df_timing, use_container_width=True)
                st.caption(f"⏱️ **Total pipeline time: {total_time:.2f}s**")


class SourcesOverview:
    """Dynamic data sources status overview."""
    
    # Provider metadata: roles and environment variable keys
    PROVIDERS = {
        "Yahoo": {"roles": {"price"}, "keys": []},
        "FMP": {"roles": {"fundamentals"}, "keys": ["FMP_API_KEY"]},
        "Alpha Vantage": {"roles": {"price", "fundamentals"}, "keys": ["ALPHA_VANTAGE_API_KEY"]},
        "Finnhub": {"roles": {"price", "fundamentals"}, "keys": ["FINNHUB_API_KEY"]},
        "Polygon": {"roles": {"price"}, "keys": ["POLYGON_API_KEY"]},
        "Tiingo": {"roles": {"price", "fundamentals"}, "keys": ["TIINGO_API_KEY"]},
        "SimFin": {"roles": {"fundamentals"}, "keys": ["SIMFIN_API_KEY"]},
        "EODHD": {"roles": {"fundamentals"}, "keys": ["EODHD_API_KEY"]},
        "Marketstack": {"roles": {"price"}, "keys": ["MARKETSTACK_API_KEY"]},
        "Nasdaq": {"roles": {"price"}, "keys": ["NASDAQ_API_KEY", "NASDAQ_DL_API_KEY"]},
        "OpenAI": {"roles": {"ml"}, "keys": ["OPENAI_API_KEY"]},
    }
    
    def __init__(self):
        """Initialize sources tracker."""
        self._usage: Dict[str, Set[str]] = {}
        self._placeholder = st.empty()
        
    def _has_key(self, provider: str) -> bool:
        """Check if provider has valid API key configured."""
        keys = self.PROVIDERS[provider]["keys"]
        if not keys:  # Yahoo, always available
            return True
        
        for key in keys:
            # Check env, st.secrets, and session state
            if os.getenv(key) or st.secrets.get(key):
                return True
        return False
    
    def _check_connectivity(self, provider: str) -> Dict[str, bool]:
        """Check which roles are active for provider.
        
        Returns:
            Dict with keys: "price", "fundamentals", "ml" → bool
        """
        has_key = self._has_key(provider)
        roles = self.PROVIDERS[provider]["roles"]
        
        # Special cases: some providers need additional checks
        if provider == "Yahoo":
            return {"price": True, "fundamentals": False, "ml": False}
        
        if not has_key:
            return {"price": False, "fundamentals": False, "ml": False}
        
        # Check session state for provider-specific status (set by validation functions)
        status_key = f"_{provider.lower().replace(' ', '_')}_ok"
        is_active = st.session_state.get(status_key, True)
        
        return {
            "price": is_active and "price" in roles,
            "fundamentals": is_active and "fundamentals" in roles,
            "ml": is_active and "ml" in roles,
        }
    
    def mark_usage(self, provider: str, category: str) -> None:
        """Record that provider was used for category.
        
        Args:
            provider: Provider name (e.g., "Alpha Vantage")
            category: Usage category ("price", "fundamentals", "ml")
        """
        if provider not in self._usage:
            self._usage[provider] = set()
        self._usage[provider].add(category)
    
    def render(self, show_legend: bool = True) -> None:
        """Render dynamic sources overview table.
        
        Args:
            show_legend: Whether to show status legend
        """
        # Status indicators
        dot_used = "🟢"  # Used this run
        dot_available = "🟡"  # Available but not yet used
        dot_missing = "⚫"  # Missing key or not connected
        
        # Build table rows
        rows = []
        used_count = 0
        
        for provider, meta in self.PROVIDERS.items():
            has_key = self._has_key(provider)
            connectivity = self._check_connectivity(provider)
            
            # Status dots for each role
            def get_dot(role: str) -> str:
                if provider in self._usage and role in self._usage[provider]:
                    return dot_used
                if connectivity.get(role, False):
                    return dot_available
                return dot_missing
            
            key_status = "✅" if has_key else "❌"
            price_dot = get_dot("price")
            fund_dot = get_dot("fundamentals")
            ml_dot = get_dot("ml")
            
            if provider in self._usage:
                used_count += 1
            
            rows.append(f"| {provider} | {key_status} | {price_dot} | {fund_dot} | {ml_dot} |")
        
        # Build markdown table
        table_md = "| Provider | Key | Price | Fundamentals | ML/AI |\n"
        table_md += "|----------|-----|-------|--------------|-------|\n"
        table_md += "\n".join(rows)
        
        if show_legend:
            table_md += f"\n\n**Legend:** {dot_used} Used ({used_count} providers) • {dot_available} Available • {dot_missing} Missing/Inactive"
        
        self._placeholder.markdown(table_md)
    
    def get_active_providers(self) -> Dict[str, list[str]]:
        """Get list of active providers by category.
        
        Returns:
            Dict with keys "price", "fundamentals", "ml" → list of provider names
        """
        active = {"price": [], "fundamentals": [], "ml": []}
        
        for provider in self.PROVIDERS:
            connectivity = self._check_connectivity(provider)
            for role, is_active in connectivity.items():
                if is_active:
                    active[role].append(provider)
        
        return active
    
    def check_critical_missing(self) -> Optional[str]:
        """Check if critical providers are missing.
        
        Returns:
            Warning message if critical providers missing, None otherwise
        """
        active = self.get_active_providers()
        
        # Need at least one fundamentals provider
        if not active["fundamentals"]:
            return "⚠️ **No fundamental data sources available.** Add at least one of: Alpha Vantage, Finnhub, FMP, Tiingo"
        
        return None


def get_pipeline_stages() -> list[str]:
    """Get ordered list of pipeline stages for progress tracking."""
    return [
        "Market Regime Detection",
        "Universe Building",
        "Historical Data Fetch",
        "Technical Indicators",
        "Beta Filter",
        "Advanced Filters",
        "Fundamentals Enrichment",
        "Risk Classification",
        "Price Verification",
        "Recommendations & Allocation"
    ]


def show_config_summary(config: dict) -> None:
    """Display compact configuration summary.
    
    Args:
        config: Configuration dictionary
    """
    st.caption(
        f"⚙️ **Config:** Universe={config.get('UNIVERSE_LIMIT', 'N/A')} • "
        f"Lookback={config.get('LOOKBACK_DAYS', 'N/A')}d • "
        f"Smart Scan={'✅' if config.get('SMART_SCAN') else '❌'}"
    )


def create_debug_expander(data: dict, title: str = "🔧 Debug Info") -> None:
    """Create collapsible debug section (only if debug mode enabled).
    
    Args:
        data: Debug data to display
        title: Expander title
    """
    if os.getenv("DEBUG_MODE") or st.session_state.get("debug_mode"):
        with st.expander(title, expanded=False):
            st.json(data)

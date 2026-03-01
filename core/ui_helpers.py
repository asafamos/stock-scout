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
        self._container = st.empty()

        # Timing instrumentation
        self._stage_times: Dict[str, float] = {}  # stage_name -> duration (seconds)
        self._stage_start_times: Dict[str, float] = {}  # stage_name -> start time (perf_counter)
        self._total_start = time.perf_counter()

        # Render initial empty state
        self._render_bar(0.0, "Initializing...", 0)

    def _render_bar(self, progress: float, stage_name: str, stage_num: int) -> None:
        """Render the custom progress bar matching recommendation card score bars."""
        pct = int(progress * 100)
        # Color gradient based on progress
        if pct >= 100:
            bar_gradient = "linear-gradient(90deg, #10b981, #34d399)"
            pct_color = "#10b981"
        elif pct >= 50:
            bar_gradient = "linear-gradient(90deg, #3b82f6, #60a5fa, #a78bfa)"
            pct_color = "#3b82f6"
        else:
            bar_gradient = "linear-gradient(90deg, #6366f1, #818cf8)"
            pct_color = "#6366f1"

        total = len(self.stages)
        self._container.markdown(f"""
        <div style="
            background: var(--ss-bg-card, #fff);
            border: 1px solid var(--ss-border, #e2e8f0);
            border-radius: var(--ss-radius-md, 12px);
            padding: 14px 18px;
            margin: 8px 0;
            box-shadow: var(--ss-shadow-sm, 0 1px 3px rgba(0,0,0,0.06));
            direction: ltr;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                <span style="font-size:0.82rem; font-weight:600; color:var(--ss-text-primary, #0f172a);">{stage_name}</span>
                <span style="font-size:0.82rem; font-weight:800; font-family:var(--ss-mono, monospace); color:{pct_color};">{pct}%</span>
            </div>
            <div style="
                height: 10px;
                background: var(--ss-bar-bg, #e2e8f0);
                border-radius: 5px;
                overflow: hidden;
            ">
                <div style="
                    width: {pct}%;
                    height: 100%;
                    background: {bar_gradient};
                    border-radius: 5px;
                    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                "></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px;">
                <span style="font-size:0.7rem; color:var(--ss-text-muted, #94a3b8);">Step {stage_num}/{total}</span>
                <span style="font-size:0.7rem; color:var(--ss-text-muted, #94a3b8);">{'🔄 Running...' if pct < 100 else '✅ Complete'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def advance(self, detail: str = "") -> None:
        """Advance to next stage with optional detail message."""
        total = len(self.stages)
        if self.current_stage >= total:
            return  # Already at or past final stage

        # Record end time of previous stage if it was started
        if 0 < self.current_stage <= total:
            prev_stage_name = self.stages[self.current_stage - 1]
            if prev_stage_name in self._stage_start_times:
                elapsed = time.perf_counter() - self._stage_start_times[prev_stage_name]
                self._stage_times[prev_stage_name] = elapsed

        self.current_stage += 1
        progress = min(self.current_stage / total, 1.0)

        stage_name = self.stages[self.current_stage - 1] if self.current_stage <= total else "Complete"
        self._render_bar(progress, stage_name, min(self.current_stage, total))

        # Record start time for new stage
        if self.current_stage <= len(self.stages):
            self._stage_start_times[stage_name] = time.perf_counter()

        # Detail caption removed — bar already shows stage name

    def update_detail(self, message: str) -> None:
        """Update detail message without advancing stage.

        Note: caption display disabled per user request (bar shows stage name).
        Kept for backward compatibility — callers can still call this safely.
        """
        pass  # Bar already shows stage name; no separate caption needed

    def complete(self, message: str = "Pipeline complete") -> None:
        """Mark pipeline as complete, recording final stage timing."""
        # Record end time of last active stage
        if self.current_stage > 0:
            last_stage_name = self.stages[self.current_stage - 1] if self.current_stage <= len(self.stages) else None
            if last_stage_name and last_stage_name in self._stage_start_times:
                elapsed = time.perf_counter() - self._stage_start_times[last_stage_name]
                self._stage_times[last_stage_name] = elapsed

        self._render_bar(1.0, message, len(self.stages))

    def set_progress(self, value: float, label: Optional[str] = None) -> None:
        """Explicitly set progress value (0.0-1.0) and optional label.
        Safe no-throw; updates internal progress bar and status text.
        """
        try:
            v = float(max(0.0, min(1.0, value)))
            stage_name = label or (self.stages[self.current_stage - 1] if self.current_stage > 0 else "Loading...")
            self._render_bar(v, stage_name, self.current_stage)
        except Exception:
            pass
    
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
                st.dataframe(df_timing, width='stretch')
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
            # Check env and safely check Streamlit secrets (bare mode friendly)
            if os.getenv(key):
                return True
            try:
                if hasattr(st, "secrets"):
                    val = st.secrets.get(key, None)
                    if val:
                        return True
            except Exception:
                # In bare mode or when secrets.toml is missing, st.secrets may raise
                pass
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

"""
Tests for UI helpers (status management, sources overview)
"""
import os
import time
import pytest
from unittest.mock import MagicMock, patch
from core.ui_helpers import StatusManager, SourcesOverview, get_pipeline_stages


class TestStatusManager:
    """Test centralized status and progress management."""
    
    def test_initialization(self):
        """Status manager initializes with stages."""
        stages = ["Stage 1", "Stage 2", "Stage 3"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty:
            
            mock_progress.return_value = MagicMock()
            mock_empty.return_value = MagicMock()
            
            manager = StatusManager(stages)
            
            assert manager.stages == stages
            assert manager.current_stage == 0
            mock_progress.assert_called_once_with(0.0)
    
    def test_advance_stages(self):
        """Advance through stages updates progress correctly."""
        stages = ["Stage 1", "Stage 2"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty:
            
            progress_bar = MagicMock()
            status_placeholder = MagicMock()
            details_placeholder = MagicMock()
            
            mock_progress.return_value = progress_bar
            mock_empty.side_effect = [status_placeholder, details_placeholder]
            
            manager = StatusManager(stages)
            
            # Advance to stage 1
            manager.advance("Processing first stage")
            assert manager.current_stage == 1
            progress_bar.progress.assert_called_with(0.5)  # 1/2
            
            # Advance to stage 2
            manager.advance("Processing second stage")
            assert manager.current_stage == 2
            progress_bar.progress.assert_called_with(1.0)  # 2/2
    
    def test_complete(self):
        """Complete marks pipeline as done."""
        stages = ["Stage 1"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty:
            
            progress_bar = MagicMock()
            status_placeholder = MagicMock()
            details_placeholder = MagicMock()
            
            mock_progress.return_value = progress_bar
            mock_empty.side_effect = [status_placeholder, details_placeholder]
            
            manager = StatusManager(stages)
            manager.complete("All done!")
            
            progress_bar.progress.assert_called_with(1.0)
            status_placeholder.success.assert_called_once()


class TestSourcesOverview:
    """Test dynamic sources status tracking."""
    
    def test_providers_metadata(self):
        """Providers have correct metadata structure."""
        overview = SourcesOverview()
        
        # Check required providers exist
        assert "Yahoo" in overview.PROVIDERS
        assert "Alpha Vantage" in overview.PROVIDERS
        assert "Finnhub" in overview.PROVIDERS
        
        # Verify structure
        for provider, meta in overview.PROVIDERS.items():
            assert "roles" in meta
            assert "keys" in meta
            assert isinstance(meta["roles"], set)
            assert isinstance(meta["keys"], list)
    
    def test_has_key_yahoo_always_available(self):
        """Yahoo always returns True (no key required)."""
        with patch('streamlit.empty'):
            overview = SourcesOverview()
            assert overview._has_key("Yahoo") is True
    
    def test_has_key_checks_env_and_secrets(self):
        """Has key checks both environment and st.secrets."""
        with patch('streamlit.empty'), \
             patch('streamlit.secrets') as mock_secrets:
            
            mock_secrets.get.return_value = None
            
            overview = SourcesOverview()
            
            # No key in env or secrets
            with patch.dict(os.environ, {}, clear=True):
                assert overview._has_key("Alpha Vantage") is False
            
            # Key in environment
            with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}):
                assert overview._has_key("Alpha Vantage") is True
            
            # Key in secrets
            mock_secrets.get.return_value = "secret_key"
            assert overview._has_key("Alpha Vantage") is True
    
    def test_mark_usage(self):
        """Mark usage tracks provider usage by category."""
        with patch('streamlit.empty'):
            overview = SourcesOverview()
            
            # Mark usage
            overview.mark_usage("Alpha Vantage", "fundamentals")
            overview.mark_usage("Yahoo", "price")
            overview.mark_usage("Alpha Vantage", "price")
            
            # Verify tracking
            assert "Alpha Vantage" in overview._usage
            assert "fundamentals" in overview._usage["Alpha Vantage"]
            assert "price" in overview._usage["Alpha Vantage"]
            
            assert "Yahoo" in overview._usage
            assert "price" in overview._usage["Yahoo"]
            assert len(overview._usage["Yahoo"]) == 1
    
    def test_get_active_providers(self):
        """Get active providers returns correct categorization."""
        with patch('streamlit.empty'), \
             patch('streamlit.secrets') as mock_secrets, \
             patch('streamlit.session_state', new_callable=dict):
            
            mock_secrets.get.return_value = None
            
            overview = SourcesOverview()
            
            # With no keys, only Yahoo should be active
            with patch.dict(os.environ, {}, clear=True):
                active = overview.get_active_providers()
                
                assert "Yahoo" in active["price"]
                assert len(active["fundamentals"]) == 0
                assert len(active["ml"]) == 0
            
            # With Alpha Vantage key
            with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test"}):
                active = overview.get_active_providers()
                
                assert "Yahoo" in active["price"]
                assert "Alpha Vantage" in active["price"]
                assert "Alpha Vantage" in active["fundamentals"]
    
    def test_check_critical_missing(self):
        """Critical missing check warns when no fundamentals sources."""
        with patch('streamlit.empty'), \
             patch('streamlit.secrets') as mock_secrets:
            
            mock_secrets.get.return_value = None
            
            overview = SourcesOverview()
            
            # No fundamental sources
            with patch.dict(os.environ, {}, clear=True):
                warning = overview.check_critical_missing()
                assert warning is not None
                assert "fundamental" in warning.lower()
            
            # With Alpha Vantage (has fundamentals)
            with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test"}):
                warning = overview.check_critical_missing()
                assert warning is None


class TestPipelineStages:
    """Test pipeline stage definitions."""
    
    def test_get_pipeline_stages(self):
        """Pipeline stages are properly ordered."""
        stages = get_pipeline_stages()
        
        # Should be a list of strings
        assert isinstance(stages, list)
        assert all(isinstance(s, str) for s in stages)
        
        # Should have expected stages
        assert "Universe Building" in stages
        assert "Historical Data Fetch" in stages
        assert "Technical Indicators" in stages
        assert "Fundamentals Enrichment" in stages
        assert "Recommendations & Allocation" in stages
        
        # Should be in logical order
        assert stages.index("Universe Building") < stages.index("Historical Data Fetch")
        assert stages.index("Historical Data Fetch") < stages.index("Technical Indicators")
        assert stages.index("Technical Indicators") < stages.index("Fundamentals Enrichment")


class TestIntegration:
    """Integration tests for UI helpers."""
    
    def test_status_and_sources_together(self):
        """Status manager and sources overview work together."""
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty:
            
            mock_progress.return_value = MagicMock()
            mock_empty.return_value = MagicMock()
            
            # Initialize both
            stages = get_pipeline_stages()
            status = StatusManager(stages)
            sources = SourcesOverview()
            
            # Simulate pipeline flow
            status.advance("Building universe")
            sources.mark_usage("Yahoo", "price")
            
            status.advance("Fetching fundamentals")
            sources.mark_usage("Alpha Vantage", "fundamentals")
            
            # Both should be tracking independently
            assert status.current_stage == 2
            assert len(sources._usage) == 2


class TestStatusManagerTiming:
    """Test performance instrumentation in StatusManager."""
    
    def test_timing_initialization(self):
        """Status manager initializes timing structures."""
        stages = ["Stage 1", "Stage 2"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty:
            
            mock_progress.return_value = MagicMock()
            mock_empty.return_value = MagicMock()
            
            manager = StatusManager(stages)
            
            assert manager._stage_times == {}
            assert manager._stage_start_times == {}
            assert manager._total_start is not None
    
    def test_timing_records_stage_duration(self):
        """Advancing stages records duration of previous stage."""
        stages = ["Stage 1", "Stage 2"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty:
            
            mock_progress.return_value = MagicMock()
            mock_empty.return_value = MagicMock()
            
            manager = StatusManager(stages)
            
            # Start stage 1
            manager.advance("Starting stage 1")
            time.sleep(0.01)  # Simulate work (10ms)
            
            # Advance to stage 2 - should record stage 1 duration
            manager.advance("Starting stage 2")
            
            assert "Stage 1" in manager._stage_times
            assert manager._stage_times["Stage 1"] >= 0.01
    
    def test_timing_on_complete(self):
        """Complete records duration of last stage."""
        stages = ["Stage 1"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty:
            
            mock_progress.return_value = MagicMock()
            mock_empty.return_value = MagicMock()
            
            manager = StatusManager(stages)
            manager.advance("Processing")
            time.sleep(0.01)
            manager.complete()
            
            # Last stage duration should be recorded
            assert "Stage 1" in manager._stage_times
            assert manager._stage_times["Stage 1"] >= 0.01
    
    def test_timing_report_requires_debug_mode(self):
        """Timing report only renders in debug mode."""
        stages = ["Stage 1"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty, \
             patch('os.getenv') as mock_getenv, \
             patch('streamlit.session_state', {}):
            
            mock_progress.return_value = MagicMock()
            mock_empty.return_value = MagicMock()
            mock_getenv.return_value = None
            
            manager = StatusManager(stages)
            manager.advance("Processing")
            manager.complete()
            
            # Capture what happens when render is called (should skip silently if not debug)
            with patch('streamlit.expander') as mock_expander:
                manager.render_timing_report()
                # In non-debug mode, expander shouldn't be called
                mock_expander.assert_not_called()
    
    def test_timing_report_with_debug_mode(self):
        """Timing report renders when debug mode enabled."""
        stages = ["Stage 1"]
        
        with patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty') as mock_empty, \
             patch('os.getenv') as mock_getenv, \
             patch('streamlit.session_state', {"debug_mode": True}), \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.caption') as mock_caption:
            
            mock_progress.return_value = MagicMock()
            mock_empty.return_value = MagicMock()
            mock_getenv.return_value = None
            mock_expander.return_value.__enter__ = MagicMock()
            mock_expander.return_value.__exit__ = MagicMock(return_value=False)
            
            manager = StatusManager(stages)
            manager.advance("Processing")
            time.sleep(0.01)
            manager.complete()
            
            # Render with debug mode enabled
            manager.render_timing_report()
            
            # Should create expander in debug mode
            mock_expander.assert_called_once()

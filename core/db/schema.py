"""DuckDB schema definitions for Stock Scout scan store.

All table definitions live here as the single source of truth.
Migrations are version-gated via the schema_version table.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Table: scans — one row per scan execution
# ---------------------------------------------------------------------------
CREATE_SCANS = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id          VARCHAR PRIMARY KEY,
    timestamp        TIMESTAMP NOT NULL,
    universe_name    VARCHAR,
    universe_size    INTEGER,
    market_regime    VARCHAR,
    regime_confidence DOUBLE,
    config_json      VARCHAR,          -- JSON string
    logic_version    VARCHAR,
    ml_model_version VARCHAR,
    total_scored     INTEGER,
    total_recommended INTEGER
);
"""

# ---------------------------------------------------------------------------
# Table: recommendations — one row per ticker per scan
# ---------------------------------------------------------------------------
CREATE_RECOMMENDATIONS = """
CREATE TABLE IF NOT EXISTS recommendations (
    id               VARCHAR PRIMARY KEY,   -- scan_id::ticker
    scan_id          VARCHAR NOT NULL,
    ticker           VARCHAR NOT NULL,
    scan_timestamp   TIMESTAMP NOT NULL,
    -- Scores
    final_score      DOUBLE,
    tech_score       DOUBLE,
    fundamental_score DOUBLE,
    ml_prob          DOUBLE,
    pattern_score    DOUBLE,
    big_winner_score DOUBLE,
    reliability_score DOUBLE,
    risk_meter       DOUBLE,
    -- Trade setup
    entry_price      DOUBLE,
    target_price     DOUBLE,
    stop_price       DOUBLE,
    rr_ratio         DOUBLE,
    -- Classification
    risk_class       VARCHAR,
    risk_label       VARCHAR,
    -- Context
    market_regime    VARCHAR,
    sector           VARCHAR,
    market_cap       DOUBLE,
    -- Key indicators
    rsi              DOUBLE,
    atr_pct          DOUBLE,
    volume_surge     DOUBLE,
    ma_alignment     DOUBLE,
    rs_vs_spy_20d    DOUBLE,
    -- Data quality
    fundamental_coverage_pct  DOUBLE,
    fundamental_sources_count INTEGER,
    data_quality     VARCHAR
);
"""

# ---------------------------------------------------------------------------
# Table: outcomes — forward returns tracked AFTER recommendation
# ---------------------------------------------------------------------------
CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    recommendation_id VARCHAR PRIMARY KEY,
    ticker           VARCHAR NOT NULL,
    entry_date       DATE NOT NULL,
    entry_price      DOUBLE NOT NULL,
    -- Forward returns (filled incrementally as trading days pass)
    return_5d        DOUBLE,
    return_10d       DOUBLE,
    return_20d       DOUBLE,
    return_40d       DOUBLE,
    -- Extremes during 20-day holding period
    max_price_20d    DOUBLE,
    min_price_20d    DOUBLE,
    max_drawdown_20d DOUBLE,
    max_upside_20d   DOUBLE,
    -- Did it hit targets?
    hit_target       BOOLEAN,
    hit_stop         BOOLEAN,
    days_to_target   INTEGER,
    days_to_stop     INTEGER,
    -- Benchmark comparison
    spy_return_20d   DOUBLE,
    excess_return_20d DOUBLE,
    -- Tracking status
    status           VARCHAR DEFAULT 'pending',  -- pending | partial | complete
    last_updated     TIMESTAMP
);
"""

# ---------------------------------------------------------------------------
# Table: schema_version — for forward-compatible migrations
# ---------------------------------------------------------------------------
CREATE_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

ALL_TABLES = [
    CREATE_SCHEMA_VERSION,
    CREATE_SCANS,
    CREATE_RECOMMENDATIONS,
    CREATE_OUTCOMES,
]

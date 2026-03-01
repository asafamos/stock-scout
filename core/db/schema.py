"""DuckDB schema definitions for Stock Scout scan store.

All table definitions live here as the single source of truth.
Migrations are version-gated via the schema_version table.
"""

from __future__ import annotations

SCHEMA_VERSION = 2

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

# ---------------------------------------------------------------------------
# Table: portfolio_positions — virtual portfolio tracking
# ---------------------------------------------------------------------------
CREATE_PORTFOLIO_POSITIONS = """
CREATE TABLE IF NOT EXISTS portfolio_positions (
    position_id      VARCHAR PRIMARY KEY,
    user_id          VARCHAR NOT NULL DEFAULT 'default',
    ticker           VARCHAR NOT NULL,
    -- Entry details (from recommendation card)
    entry_price      DOUBLE NOT NULL,
    target_price     DOUBLE,
    stop_price       DOUBLE,
    shares           INTEGER NOT NULL DEFAULT 100,
    entry_date       DATE NOT NULL,
    target_date      DATE,
    holding_days     INTEGER DEFAULT 20,
    -- Source recommendation link
    scan_id          VARCHAR,
    recommendation_id VARCHAR,
    -- Scores at entry (denormalized for self-contained reporting)
    final_score      DOUBLE,
    risk_class       VARCHAR,
    sector           VARCHAR,
    -- Live tracking
    current_price    DOUBLE,
    current_return_pct DOUBLE,
    max_price        DOUBLE,
    min_price        DOUBLE,
    -- Exit details
    exit_price       DOUBLE,
    exit_date        DATE,
    exit_reason      VARCHAR,
    realized_return_pct DOUBLE,
    -- Status
    status           VARCHAR NOT NULL DEFAULT 'open',
    prediction_correct BOOLEAN,
    -- Timestamps
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# ---------------------------------------------------------------------------
# Table: portfolio_snapshots — daily portfolio value for equity curve
# ---------------------------------------------------------------------------
CREATE_PORTFOLIO_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id               VARCHAR PRIMARY KEY,
    user_id          VARCHAR NOT NULL DEFAULT 'default',
    snapshot_date    DATE NOT NULL,
    total_positions  INTEGER,
    open_positions   INTEGER,
    total_invested   DOUBLE,
    current_value    DOUBLE,
    total_return_pct DOUBLE,
    win_count        INTEGER,
    loss_count       INTEGER,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

ALL_TABLES = [
    CREATE_SCHEMA_VERSION,
    CREATE_SCANS,
    CREATE_RECOMMENDATIONS,
    CREATE_OUTCOMES,
    CREATE_PORTFOLIO_POSITIONS,
    CREATE_PORTFOLIO_SNAPSHOTS,
]

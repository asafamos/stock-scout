-- ==========================================================================
-- Supabase (PostgreSQL) schema for Stock Scout portfolio tracking
--
-- Run this in the Supabase SQL Editor to set up portfolio persistence.
-- This replaces the local DuckDB portfolio_positions table so positions
-- survive Streamlit Cloud deployments.
-- ==========================================================================

-- Portfolio positions — one row per stock in the virtual portfolio
CREATE TABLE IF NOT EXISTS portfolio_positions (
    position_id          TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL DEFAULT 'default',
    ticker               TEXT NOT NULL,

    -- Entry details (from recommendation card)
    entry_price          DOUBLE PRECISION NOT NULL,
    target_price         DOUBLE PRECISION,
    stop_price           DOUBLE PRECISION,
    shares               INTEGER NOT NULL DEFAULT 100,
    entry_date           DATE NOT NULL,
    target_date          DATE,
    holding_days         INTEGER DEFAULT 20,

    -- Source recommendation link
    scan_id              TEXT,
    recommendation_id    TEXT,

    -- Scores at entry (denormalized for self-contained reporting)
    final_score          DOUBLE PRECISION,
    risk_class           TEXT,
    sector               TEXT,

    -- Live tracking
    current_price        DOUBLE PRECISION,
    current_return_pct   DOUBLE PRECISION,
    max_price            DOUBLE PRECISION,
    min_price            DOUBLE PRECISION,

    -- Exit details
    exit_price           DOUBLE PRECISION,
    exit_date            DATE,
    exit_reason          TEXT,
    realized_return_pct  DOUBLE PRECISION,

    -- Status
    status               TEXT NOT NULL DEFAULT 'open',
    prediction_correct   BOOLEAN,

    -- Timestamps
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_pp_user_status
    ON portfolio_positions (user_id, status);

CREATE INDEX IF NOT EXISTS idx_pp_ticker_status
    ON portfolio_positions (ticker, status);

CREATE INDEX IF NOT EXISTS idx_pp_exit_date
    ON portfolio_positions (exit_date)
    WHERE exit_date IS NOT NULL;

-- Portfolio snapshots — daily aggregate for equity curve
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id                   TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL DEFAULT 'default',
    snapshot_date        DATE NOT NULL,
    total_positions      INTEGER,
    open_positions       INTEGER,
    total_invested       DOUBLE PRECISION,
    current_value        DOUBLE PRECISION,
    total_return_pct     DOUBLE PRECISION,
    win_count            INTEGER,
    loss_count           INTEGER,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ps_user_date
    ON portfolio_snapshots (user_id, snapshot_date);

-- Row-Level Security (RLS) — enable for multi-user safety
-- Uncomment and adjust if using Supabase Auth:
-- ALTER TABLE portfolio_positions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Users see own positions" ON portfolio_positions
--     FOR ALL USING (user_id = auth.uid()::text);
-- CREATE POLICY "Users see own snapshots" ON portfolio_snapshots
--     FOR ALL USING (user_id = auth.uid()::text);

-- Auto-update updated_at on portfolio_positions changes
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON portfolio_positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();


-- ==========================================================================
-- Scan History — persists scan results across Streamlit deploys
-- ==========================================================================

-- Scan-level metadata — one row per scan execution
CREATE TABLE IF NOT EXISTS scan_history (
    scan_id              TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL DEFAULT 'default',
    timestamp            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    universe_name        TEXT,
    universe_size        INTEGER,
    market_regime        TEXT,
    regime_confidence    DOUBLE PRECISION,
    config_json          TEXT,
    logic_version        TEXT,
    ml_model_version     TEXT,
    total_scored         INTEGER,
    total_recommended    INTEGER,
    scan_type            TEXT DEFAULT 'manual',
    scan_duration_secs   DOUBLE PRECISION,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sh_user_ts
    ON scan_history (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sh_timestamp
    ON scan_history (timestamp DESC);

-- Scan recommendations — one row per ticker per scan
CREATE TABLE IF NOT EXISTS scan_recommendations (
    id                        TEXT PRIMARY KEY,
    scan_id                   TEXT NOT NULL REFERENCES scan_history(scan_id),
    user_id                   TEXT NOT NULL DEFAULT 'default',
    ticker                    TEXT NOT NULL,
    scan_timestamp            TIMESTAMPTZ NOT NULL,
    -- Scores
    final_score               DOUBLE PRECISION,
    tech_score                DOUBLE PRECISION,
    fundamental_score         DOUBLE PRECISION,
    ml_prob                   DOUBLE PRECISION,
    pattern_score             DOUBLE PRECISION,
    reliability_score         DOUBLE PRECISION,
    risk_meter                DOUBLE PRECISION,
    -- Trade setup
    entry_price               DOUBLE PRECISION,
    target_price              DOUBLE PRECISION,
    stop_price                DOUBLE PRECISION,
    rr_ratio                  DOUBLE PRECISION,
    holding_days              INTEGER,
    -- Classification
    risk_class                TEXT,
    risk_label                TEXT,
    -- Context
    market_regime             TEXT,
    sector                    TEXT,
    market_cap                DOUBLE PRECISION,
    -- Key indicators
    rsi                       DOUBLE PRECISION,
    atr_pct                   DOUBLE PRECISION,
    volume_surge              DOUBLE PRECISION,
    ma_alignment              DOUBLE PRECISION,
    rs_vs_spy_20d             DOUBLE PRECISION,
    -- Data quality
    fundamental_coverage_pct  DOUBLE PRECISION,
    fundamental_sources_count INTEGER,
    data_quality              TEXT,
    created_at                TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sr_scan_id
    ON scan_recommendations (scan_id);

CREATE INDEX IF NOT EXISTS idx_sr_ticker
    ON scan_recommendations (ticker);

CREATE INDEX IF NOT EXISTS idx_sr_user_ts
    ON scan_recommendations (user_id, scan_timestamp DESC);

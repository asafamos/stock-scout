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

-- ==========================================================================
-- Fix RLS policies for Stock Scout (single-user, anon key access)
--
-- Run this in the Supabase SQL Editor (https://supabase.com/dashboard)
-- This allows the anon key to read/write all tables.
-- ==========================================================================

-- 1) Ensure RLS is enabled (idempotent)
ALTER TABLE portfolio_positions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_snapshots  ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_history         ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_recommendations ENABLE ROW LEVEL SECURITY;

-- 2) Drop old policies if they exist (idempotent)
DROP POLICY IF EXISTS "anon_all_portfolio_positions"  ON portfolio_positions;
DROP POLICY IF EXISTS "anon_all_portfolio_snapshots"  ON portfolio_snapshots;
DROP POLICY IF EXISTS "anon_all_scan_history"         ON scan_history;
DROP POLICY IF EXISTS "anon_all_scan_recommendations" ON scan_recommendations;

-- 3) Create permissive policies for anon role (used by publishable key)
CREATE POLICY "anon_all_portfolio_positions"
    ON portfolio_positions FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);

CREATE POLICY "anon_all_portfolio_snapshots"
    ON portfolio_snapshots FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);

CREATE POLICY "anon_all_scan_history"
    ON scan_history FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);

CREATE POLICY "anon_all_scan_recommendations"
    ON scan_recommendations FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);

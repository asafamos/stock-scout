-- ============================================================
-- StockScout — Supabase storage cleanup + retention
-- ============================================================
-- Run via Supabase Studio (SQL Editor) when quota is exhausted —
-- the dashboard stays accessible per Supabase's restriction policy.
--
-- Context: 2026-05-17 hit quota (4.05GB / 1.1GB cap, 368% over).
-- Root cause: `scan_recommendations.full_row_json` stored the entire
-- pipeline row as a JSON blob (~5KB × 397 rows × 4 scans/day × many
-- days = multi-GB).
--
-- This script does 3 things:
--   1. Reports current sizes (read-only, run first)
--   2. Deletes scan_recommendations older than 30 days
--   3. NULLs the full_row_json column on remaining rows (kept for
--      schema compatibility but no longer fills storage)
--   4. (Optional) Creates a daily retention function + cron trigger
-- ============================================================

-- ── STEP 1: Diagnose (read-only) ─────────────────────────────
-- See what's eating space. Run this FIRST, before any DELETE.
SELECT
    'scan_recommendations' AS table_name,
    COUNT(*) AS rows,
    pg_size_pretty(pg_total_relation_size('scan_recommendations')) AS total_size,
    -- 2026-05-23: cast JSONB to text before COALESCE — Postgres tries
    -- to parse the '' default as JSON otherwise and errors out with
    -- "invalid input syntax for type json: input string ended unexpectedly"
    pg_size_pretty(SUM(LENGTH(COALESCE(full_row_json::text, '')))::bigint) AS blob_size,
    MIN(scan_timestamp) AS oldest,
    MAX(scan_timestamp) AS newest
FROM scan_recommendations
UNION ALL
SELECT
    'scan_history',
    COUNT(*),
    pg_size_pretty(pg_total_relation_size('scan_history')),
    NULL,
    MIN(scan_timestamp),
    MAX(scan_timestamp)
FROM scan_history
UNION ALL
SELECT
    'portfolio_positions',
    COUNT(*),
    pg_size_pretty(pg_total_relation_size('portfolio_positions')),
    NULL,
    MIN(timestamp),
    MAX(timestamp)
FROM portfolio_positions;

-- ── STEP 2: Drain the JSON blob from old rows ────────────────
-- Removes the heaviest column from rows older than 7 days.
-- We keep the structured scalars (score, ml_prob, etc.) for ML feedback.
UPDATE scan_recommendations
SET full_row_json = NULL
WHERE scan_timestamp < NOW() - INTERVAL '7 days'
  AND full_row_json IS NOT NULL;

-- ── STEP 3: Delete rows older than 30 days entirely ──────────
-- After this, the table only holds the last 30 days of scans.
-- Plenty for ML feedback + recent analytics.
DELETE FROM scan_recommendations
WHERE scan_timestamp < NOW() - INTERVAL '30 days';

DELETE FROM scan_history
WHERE scan_timestamp < NOW() - INTERVAL '30 days';

-- ── STEP 4: Reclaim space (forces VACUUM) ────────────────────
-- Postgres doesn't release deleted rows' disk until VACUUM runs.
-- This may take a few minutes but is what actually shrinks storage.
VACUUM FULL scan_recommendations;
VACUUM FULL scan_history;

-- ── STEP 5: (Optional) Schedule daily auto-retention ─────────
-- Requires pg_cron extension (enabled by default in Supabase Pro,
-- can be enabled on Free via Dashboard → Database → Extensions).
-- Comment out if you'd rather run cleanup manually.

-- CREATE OR REPLACE FUNCTION stockscout_daily_cleanup()
-- RETURNS void AS $$
-- BEGIN
--     UPDATE scan_recommendations
--     SET full_row_json = NULL
--     WHERE scan_timestamp < NOW() - INTERVAL '7 days'
--       AND full_row_json IS NOT NULL;
--
--     DELETE FROM scan_recommendations
--     WHERE scan_timestamp < NOW() - INTERVAL '30 days';
--
--     DELETE FROM scan_history
--     WHERE scan_timestamp < NOW() - INTERVAL '30 days';
-- END;
-- $$ LANGUAGE plpgsql;
--
-- SELECT cron.schedule(
--     'stockscout-daily-cleanup',
--     '0 3 * * *',   -- 03:00 UTC daily (off-market)
--     $$SELECT stockscout_daily_cleanup();$$
-- );

-- ── STEP 6: Verify (re-run STEP 1) ───────────────────────────
-- After the VACUUM completes, re-run the first query to confirm
-- the size dropped. Expected: < 200 MB total.

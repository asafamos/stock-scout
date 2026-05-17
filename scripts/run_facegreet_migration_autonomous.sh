#!/bin/bash
# ============================================================
# Fully-autonomous facegreet → R2 migration wrapper
# ============================================================
# Wraps run_facegreet_migration.sh --yes with:
#   - Logging to /tmp/facegreet-migration-YYYYMMDD.log
#   - Telegram notification at start, on success, on failure
#   - Network-online wait (in case Mac just woke up)
#
# Designed to be triggered by launchd on June 6, 2026 at noon IL.
# ============================================================

set +e  # don't exit on error — we want to send telegram on failures too

LOG_DIR="/tmp"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/facegreet-migration-${TS}.log"
PROJECT_DIR="/Users/asafamos/StockScout/stock-scout-2"

# Load Telegram creds from VPS env file (already on Mac for testing?)
# Fallback: read from .env.trading on Mac if it exists
TG_TOKEN=""
TG_CHAT=""
if [ -f "${PROJECT_DIR}/.streamlit/secrets.toml" ]; then
    TG_TOKEN=$(grep -E "^\s*TRADE_TELEGRAM_TOKEN" "${PROJECT_DIR}/.streamlit/secrets.toml" 2>/dev/null \
        | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
    TG_CHAT=$(grep -E "^\s*TRADE_TELEGRAM_CHAT_ID" "${PROJECT_DIR}/.streamlit/secrets.toml" 2>/dev/null \
        | head -1 | sed 's/.*=\s*"\(.*\)".*/\1/')
fi

notify() {
    local msg="$1"
    echo "[notify] ${msg}" >> "${LOG}"
    if [ -n "${TG_TOKEN}" ] && [ -n "${TG_CHAT}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            -d chat_id="${TG_CHAT}" \
            -d text="${msg}" \
            -d parse_mode="HTML" \
            > /dev/null 2>&1
    fi
}

cd "${PROJECT_DIR}" || {
    notify "🚨 facegreet migration FAILED: cannot cd to ${PROJECT_DIR}"
    exit 2
}

# Wait for network (Mac may have just woken; give it 60s)
for i in {1..12}; do
    if curl -s --max-time 5 https://api.cloudflare.com/client/v4/ > /dev/null 2>&1; then
        break
    fi
    sleep 5
done

notify "🚀 <b>facegreet → R2 migration starting</b>
Source: Supabase facegreet-videos (~734 files, ~3.5 GB)
Target: Cloudflare R2 facegreet-videos
Log: ${LOG}"

# Run the migration (interactive prompt bypassed via --yes)
bash scripts/run_facegreet_migration.sh --yes >> "${LOG}" 2>&1
RC=$?

if [ "${RC}" -eq 0 ]; then
    # Extract summary from log
    SUMMARY=$(grep -E "^\s*ok:|skipped_exists:|errors:|bytes moved:" "${LOG}" | tail -4 | sed 's/^.*\[INFO\]\s*//')
    notify "✅ <b>facegreet → R2 migration DONE</b>

${SUMMARY}

Next manual steps:
1. Spot-check ~5 files in R2 dashboard
2. Update facegreet app to read/write R2
3. DELETE FROM storage.objects WHERE bucket_id='facegreet-videos'
4. rm ${PROJECT_DIR}/.env.facegreet-migration

Full log: ${LOG}"
else
    LAST_ERR=$(grep -E "ERROR|CRITICAL|Traceback" "${LOG}" | tail -3)
    notify "🚨 <b>facegreet → R2 migration FAILED</b> (exit ${RC})

Last errors:
<pre>${LAST_ERR}</pre>

Full log: ${LOG}
Run manually:
  cd ${PROJECT_DIR}
  bash scripts/run_facegreet_migration.sh"
fi

exit "${RC}"

#!/bin/bash
# ============================================================
# StockScout Healthcheck + Weekly IB Key Reminder
#
# Runs every 15 min via systemd timer during market hours.
# Checks Docker Gateway, API port, monitor daemon.
# Sends weekly reminder to approve IB Key.
# ============================================================

TELEGRAM_TOKEN="${TRADE_TELEGRAM_TOKEN}"
CHAT_ID="${TRADE_TELEGRAM_CHAT_ID}"
IB_PORT=7496

# State dir for alert de-duplication. Files here act as "we already warned you
# about this recently" markers so we don't spam Telegram every 15 min while the
# same issue persists.
STATE_DIR="/var/tmp/stockscout-healthcheck"
mkdir -p "${STATE_DIR}" 2>/dev/null

send_alert() {
    local msg="$1"
    if [ -n "${TELEGRAM_TOKEN}" ] && [ -n "${CHAT_ID}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d chat_id="${CHAT_ID}" \
            -d text="${msg}" \
            -d parse_mode="HTML" \
            > /dev/null 2>&1
    fi
    echo "[ALERT] ${msg}"
}

# Send an alert only if we haven't already sent one of this kind in the last
# ${2:-7200} seconds (default 2h). Key must be a short filename-safe tag.
send_alert_dedup() {
    local key="$1"
    local msg="$2"
    local cooldown="${3:-7200}"
    local marker="${STATE_DIR}/alert_${key}"
    if [ -f "${marker}" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "${marker}" 2>/dev/null || echo 0) ))
        if [ "${age}" -lt "${cooldown}" ]; then
            echo "[SKIP-DEDUP] ${key} (last sent ${age}s ago < ${cooldown}s)"
            return 0
        fi
    fi
    send_alert "${msg}"
    touch "${marker}"
}

# Clear a dedup marker once the underlying issue recovers, so the next failure
# triggers a fresh alert instead of being silently suppressed.
clear_alert_dedup() {
    rm -f "${STATE_DIR}/alert_$1" 2>/dev/null
}

# Actual ib_insync handshake. Returns "OK" or "ERR:<name>" / "NOCONN".
# Use a RANDOM clientId per run (200-899): if a previous healthcheck left a
# zombie session on a fixed clientId, a fresh fixed id would get rejected
# with "client id already in use" → false handshake-failure alert.
#
# IMPORTANT: pass clientId as a positional arg to python, NOT via env + bash.
# Earlier version used `runuser --preserve-environment -- bash -c '...'`
# which carried HOME=/root into the stockscout shell; bash then tried to
# read /root/.bashrc (Permission denied) and EXITED before running Python,
# yielding empty output that looked like a handshake failure to the caller.
# That false alarm kept triggering the auto-heal → docker restart ibgateway
# → fresh 2FA push → which the user had JUST approved on their phone.
# Net effect: the healthcheck was destroying the session it was meant to
# protect. Passing clientId on argv bypasses the whole env-var path.
run_handshake_check() {
    local cid=$(( (RANDOM % 700) + 200 ))
    # NOTE: this service runs as User=stockscout (see systemd unit). Previous
    # version wrapped the python call with `runuser -u stockscout -- ...`
    # which FAILED silently because `runuser may not be used by non-root users`
    # — stderr "...may not be used by non-root users" → empty stdout → caller
    # saw empty result as "handshake failed" → auto-heal restarted the
    # container → killed the session the user had just approved.
    # Solution: drop the runuser wrapper entirely, we're already stockscout.
    /home/stockscout/stock-scout-2/.venv/bin/python -c "
import sys
from ib_insync import IB
ib = IB()
try:
    ib.connect('127.0.0.1', 7496, clientId=int(sys.argv[1]), timeout=10)
    if ib.isConnected():
        print('OK')
    else:
        print('NOCONN')
    ib.disconnect()
except Exception as e:
    print(f'ERR: {type(e).__name__}')
" "${cid}" 2>/dev/null
}

# Adaptive scope (changed 2026-05-05): outside market hours we still
# run a MINIMAL check so an overnight container death is caught before
# market open, instead of going undetected for 12-16h. The full deep
# check (handshake, drift, scan-outcomes) only runs during market
# hours to avoid noise alerts when IB is intentionally down for
# nightly maintenance.
DOW=$(date -u +%u)  # 1=Mon, 7=Sun
HOUR=$(date -u +%H)
MARKET_HOURS=0
if [ "$DOW" -le 5 ] && [ "$HOUR" -ge 13 ] && [ "$HOUR" -le 20 ]; then
    MARKET_HOURS=1
fi

# Weekly IB Key reminder — Sunday 15:00 UTC (before market opens Monday)
if [ "$DOW" -eq 7 ] && [ "$HOUR" -eq 15 ]; then
    send_alert "$(echo -e '\xF0\x9F\x94\x91') <b>Weekly IB Key Reminder</b>

IB Gateway needs re-authentication before market opens Monday.

1. Open: http://87.99.142.12:5800/vnc.html
2. Click Connect
3. Select IB Key in 2FA dialog
4. Approve on IBKR Mobile app

Takes 10 seconds!"
fi

# Off-hours: minimal check — Docker container alive + DEEP handshake.
#
# Updated 2026-05-18 after the Monday-morning outage where the Gateway
# stayed UP for 4 days as a container but its IBKR SESSION died over the
# Saturday-Sunday maintenance window, leaving us with a "container alive"
# false-positive 6 hours before market open. The old code only checked
# `docker ps` and skipped the handshake off-hours — so we'd discover the
# dead session at 16:00 IL on Monday (way too late). Now we run the same
# deep handshake + auto-restart logic off-hours, with longer dedup so we
# don't spam at 3am.
if [ "$MARKET_HOURS" -eq 0 ]; then
    OFF_HOURS_ISSUES=0

    # 1. Container existence (cheap, fast)
    if ! docker ps --format '{{.Names}}' | grep -q ibgateway; then
        send_alert_dedup "offhours_container_down" \
            "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') <b>OFF-HOURS</b> IB Gateway container DOWN — attempting restart" \
            14400  # 4h dedup so we don't spam overnight
        docker start ibgateway 2>/dev/null || docker restart ibgateway 2>/dev/null
        sleep 30
        if docker ps --format '{{.Names}}' | grep -q ibgateway; then
            send_alert "$(echo -e '\xe2\x9c\x85') Off-hours: ibgateway container restarted"
            clear_alert_dedup "offhours_container_down"
        else
            OFF_HOURS_ISSUES=1
            send_alert_dedup "offhours_container_restart_failed" \
                "$(echo -e '\xf0\x9f\x9a\xa8') OFF-HOURS CRITICAL: ibgateway restart FAILED" \
                3600
        fi
    fi

    # 2. DEEP handshake — session inside the container can be dead while
    # the container itself is alive (Saturday maintenance window).
    # On weekdays before market (DoW 1-5, HOUR 0-12 UTC = before US open)
    # and on Sunday afternoon onwards we definitely want this check to
    # auto-heal so Monday's market open finds an authenticated session.
    # On Saturday + Sunday morning we run it too but with longer dedup.
    if docker ps --format '{{.Names}}' | grep -q ibgateway; then
        OFFHOURS_API=$(run_handshake_check)
        if [ "$OFFHOURS_API" != "OK" ]; then
            # Dedup window: 1h on weekday-pre-market (we WANT to keep
            # nudging before open), 4h on weekend (less urgent).
            if [ "$DOW" -le 5 ]; then
                DEDUP_SEC=3600   # weekday pre-market: 1h cooldown
            else
                DEDUP_SEC=14400  # weekend: 4h cooldown
            fi

            # First attempt: auto-restart the container, give it 45s to
            # autologin + push 2FA. This usually works AFTER user has
            # approved a pending push on phone.
            echo "[OFF-HOURS AUTO-HEAL] handshake failed (${OFFHOURS_API}) — restarting ibgateway"
            send_alert_dedup "offhours_session_dead" \
                "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') <b>OFF-HOURS</b> IB session DEAD (handshake=${OFFHOURS_API})

Container is up but the IBKR session expired (weekly maintenance window).
Auto-restarting now — you should see a fresh 2FA push on IBKR Mobile.
Approve it to re-authenticate.

VNC if needed: http://87.99.142.12:5800/vnc.html" \
                ${DEDUP_SEC}
            docker restart ibgateway >/dev/null 2>&1
            sleep 45
            OFFHOURS_API2=$(run_handshake_check)
            if [ "$OFFHOURS_API2" = "OK" ]; then
                send_alert "$(echo -e '\xe2\x9c\x85') OFF-HOURS auto-heal SUCCESS: IB session restored"
                clear_alert_dedup "offhours_session_dead"
            else
                # Auto-heal couldn't recover (user hasn't approved push yet).
                # Send a LOUDER one-tap-recovery alert with full instructions.
                send_alert_dedup "offhours_2fa_needed" \
                    "$(echo -e '\xf0\x9f\x9a\xa8') <b>ACTION REQUIRED — IB Gateway needs 2FA</b>

handshake still failing after auto-restart (${OFFHOURS_API2}).

<b>To fix (10 sec on phone):</b>
1. Open IBKR Mobile app
2. Should see pending push: 'Approve sign-in'
3. Tap Approve with Face ID / IB Key

<b>If no push waiting:</b>
ssh root@87.99.142.12 'docker restart ibgateway'
(triggers fresh push)

<b>Manual via browser:</b>
http://87.99.142.12:5800/vnc.html" \
                    ${DEDUP_SEC}
                OFF_HOURS_ISSUES=$((OFF_HOURS_ISSUES + 1))
            fi
        else
            # Handshake OK off-hours → all clear; clear any stale alerts
            clear_alert_dedup "offhours_session_dead"
            clear_alert_dedup "offhours_2fa_needed"
        fi
    fi

    # 3. Monitor daemon
    if ! systemctl is-active --quiet stockscout-monitor; then
        send_alert_dedup "offhours_monitor_down" \
            "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') OFF-HOURS: monitor daemon DOWN — restarting" \
            7200
        sudo systemctl restart stockscout-monitor 2>/dev/null
    fi

    if [ "$OFF_HOURS_ISSUES" -eq 0 ]; then
        echo "Off-hours check OK"
    fi
    exit 0
fi

ISSUES=0

# 1. Check Docker IB Gateway container
if ! docker ps --format '{{.Names}}' | grep -q ibgateway; then
    send_alert_dedup "container_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') IB Gateway Docker container is DOWN — attempting restart"
    docker start ibgateway 2>/dev/null || docker restart ibgateway 2>/dev/null
    sleep 30
    if docker ps --format '{{.Names}}' | grep -q ibgateway; then
        send_alert "$(echo -e '\xe2\x9c\x85') IB Gateway container restarted"
        clear_alert_dedup "container_down"
    else
        send_alert_dedup "container_restart_failed" "$(echo -e '\xf0\x9f\x9a\xa8') CRITICAL: IB Gateway restart FAILED"
        ISSUES=$((ISSUES + 1))
    fi
fi

# 2. Check API port connectivity (TCP-level only)
if ! nc -z 127.0.0.1 ${IB_PORT} 2>/dev/null; then
    send_alert_dedup "port_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') IB Gateway port ${IB_PORT} not responding — may need 2FA approval

Open: http://87.99.142.12:5800/vnc.html"
    ISSUES=$((ISSUES + 1))
else
    clear_alert_dedup "port_down"
    # 2b. DEEP API CHECK — TCP open doesn't prove IB is authenticated.
    # The session can die while the container stays up (happens daily).
    # Try an actual ib_insync handshake; timeout fast (10s).
    API_CHECK=$(run_handshake_check)
    if [ "$API_CHECK" != "OK" ]; then
        # SELF-HEAL: session died but container is up. Most of the time a
        # `docker restart ibgateway` brings it back — the Gateway's autologin
        # then triggers a fresh 2FA push to IBKR Mobile, which can be approved
        # from the phone. Only alert if the restart fails to recover.
        echo "[AUTO-HEAL] handshake failed (${API_CHECK}) — restarting ibgateway container"
        docker restart ibgateway >/dev/null 2>&1
        # IB Gateway is slow to boot + autologin; give it enough time.
        sleep 45
        API_CHECK2=$(run_handshake_check)
        if [ "$API_CHECK2" = "OK" ]; then
            send_alert "$(echo -e '\xe2\x9c\x85') IB Gateway auto-recovered after handshake failure (was: ${API_CHECK})"
            clear_alert_dedup "handshake_failed"
        else
            send_alert_dedup "handshake_failed" "$(echo -e '\xf0\x9f\x9a\xa8') IB API handshake FAILED (${API_CHECK2}) — auto-restart did not recover.

Session needs IB Key re-approval. From phone: open IBKR Mobile and approve the pending push (container was just restarted, push should be waiting).

If no push arrived: ssh root@87.99.142.12 'docker restart ibgateway' to re-trigger autologin."
            ISSUES=$((ISSUES + 1))
        fi
    else
        clear_alert_dedup "handshake_failed"
    fi
fi

# 3. Check monitor daemon
if ! systemctl is-active --quiet stockscout-monitor; then
    send_alert_dedup "monitor_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Monitor daemon is DOWN — restarting"
    sudo systemctl restart stockscout-monitor
    sleep 5
    if systemctl is-active --quiet stockscout-monitor; then
        echo "Monitor daemon restarted"
        clear_alert_dedup "monitor_down"
    else
        send_alert_dedup "monitor_restart_failed" "$(echo -e '\xf0\x9f\x9a\xa8') Monitor daemon restart FAILED"
        ISSUES=$((ISSUES + 1))
    fi
else
    clear_alert_dedup "monitor_down"
    clear_alert_dedup "monitor_restart_failed"
fi

# 4. Check status bot daemon (handles /panic, /pnl, /status, /history)
if ! systemctl is-active --quiet stockscout-statusbot; then
    send_alert_dedup "statusbot_down" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Status bot is DOWN — restarting"
    sudo systemctl restart stockscout-statusbot
    sleep 3
    if systemctl is-active --quiet stockscout-statusbot; then
        echo "Status bot restarted"
        clear_alert_dedup "statusbot_down"
    else
        send_alert_dedup "statusbot_restart_failed" "$(echo -e '\xf0\x9f\x9a\xa8') Status bot restart FAILED — /panic unavailable"
        ISSUES=$((ISSUES + 1))
    fi
else
    clear_alert_dedup "statusbot_down"
    clear_alert_dedup "statusbot_restart_failed"
fi

# 5. Detect monitor stuck in restart loop (>3 restarts in last 5 min).
# Trim whitespace so the integer comparison doesn't choke on journalctl output.
MONITOR_RESTARTS=$(journalctl -u stockscout-monitor --since '5 min ago' --no-pager 2>/dev/null \
    | grep -c 'Started StockScout Position Monitor')
MONITOR_RESTARTS=${MONITOR_RESTARTS//[^0-9]/}
MONITOR_RESTARTS=${MONITOR_RESTARTS:-0}
if [ "$MONITOR_RESTARTS" -gt 3 ]; then
    send_alert_dedup "monitor_restart_loop" "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Monitor restarted ${MONITOR_RESTARTS}x in 5 min — check logs"
    ISSUES=$((ISSUES + 1))
fi

# 6. Orphan IB position drift — only report if we can prove the state.
# This service already runs as User=stockscout, so no runuser needed.
# If the IB query fails or returns no answer, SKIP the drift check entirely
# instead of false-alarming on every tracker ticker.
#
# 2026-06-01 FIX: tracker↔IB races during the monitor's reconcile cycle
# produced false-alarm DRIFT alerts. Specifically:
#   • OCA trail/target fires on IB → IB no longer holds the position.
#   • Monitor cycle hasn't yet run reconcile (CHECK_INTERVAL ≈ 5 min).
#   • Healthcheck timer fires in this window (every 15 min) → sees
#     "tracker has X, IB doesn't" → false DRIFT alert.
#   • A minute later the monitor reconciles + writes the trail_fired CLOSE
#     so the user sees DRIFT THEN immediate SELL — confusing.
# Symmetric race on the other side: monitor BUYs, fills in IB, healthcheck
# fires before the tracker write lands → false "IB has X but tracker doesn't".
#
# Fix: the IB query also returns RECENT FILLS (last GRACE_MIN minutes).
# If the orphan ticker has a recent fill in the matching direction, suppress
# the alert — it's a known reconcile race the monitor will close on its
# next cycle. The monitor-stale check (6b) is an independent safety net that
# catches a dead monitor, so suppressing here doesn't hide real failures.
#
# Env override: TRADE_DRIFT_RECENT_FILL_GRACE_MIN (default 30).
TRACKER="/home/stockscout/stock-scout-2/data/trades/open_positions.json"
if [ -f "${TRACKER}" ]; then
    # Random clientId (see run_handshake_check for rationale).
    DRIFT_CID=$(( (RANDOM % 700) + 200 ))
    export DRIFT_CID
    : "${TRADE_DRIFT_RECENT_FILL_GRACE_MIN:=30}"
    export TRADE_DRIFT_RECENT_FILL_GRACE_MIN
    IB_QUERY=$(bash -c 'cd /home/stockscout/stock-scout-2 && set -a && source .env.trading 2>/dev/null && set +a && /home/stockscout/stock-scout-2/.venv/bin/python -c "
import os
from datetime import datetime, timezone, timedelta
from ib_insync import IB
GRACE_MIN = int(os.environ.get(\"TRADE_DRIFT_RECENT_FILL_GRACE_MIN\", \"30\"))
try:
    ib = IB(); ib.connect(\"127.0.0.1\", 7496, clientId=int(os.environ[\"DRIFT_CID\"]), timeout=10)
    syms = sorted({p.contract.symbol for p in ib.positions() if p.position != 0})
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=GRACE_MIN)
    recent_buys, recent_sells = set(), set()
    try:
        for f in ib.fills():
            try:
                ex = f.execution
                if not getattr(ex, \"time\", None):
                    continue
                t = ex.time if ex.time.tzinfo else ex.time.replace(tzinfo=timezone.utc)
                if t < cutoff:
                    continue
                side = (ex.side or \"\").upper()
                if side in (\"BOT\", \"BUY\"):
                    recent_buys.add(f.contract.symbol)
                elif side in (\"SLD\", \"SELL\"):
                    recent_sells.add(f.contract.symbol)
            except Exception:
                continue
    except Exception:
        pass
    ib.disconnect()
    print(\"OK\", \" \".join(syms) + \"|\" + \" \".join(sorted(recent_buys)) + \"|\" + \" \".join(sorted(recent_sells)))
except Exception as e:
    print(\"ERR\", e)
"' 2>/dev/null)
    STATUS=$(echo "${IB_QUERY}" | awk '{print $1}')
    if [ "$STATUS" = "OK" ]; then
        BODY=$(echo "${IB_QUERY}" | cut -d' ' -f2-)
        IB_TICKERS=$(echo "${BODY}" | cut -d'|' -f1)
        RECENT_BUYS=$(echo "${BODY}" | cut -d'|' -f2)
        RECENT_SELLS=$(echo "${BODY}" | cut -d'|' -f3)
        TRACKER_TICKERS=$(python3 -c "
import json
try:
    with open('${TRACKER}') as f:
        syms = sorted({p['ticker'] for p in json.load(f)})
    print(' '.join(syms))
except Exception:
    print('')
" 2>/dev/null)
        for sym in ${IB_TICKERS}; do
            if [ -n "${sym}" ] && ! echo " ${TRACKER_TICKERS} " | grep -q " ${sym} "; then
                # Suppress: monitor BOUGHT this in the last GRACE_MIN min — tracker
                # write is in-flight. The monitor's own _drift_check will adopt or
                # write the OPEN on its next cycle.
                if echo " ${RECENT_BUYS} " | grep -q " ${sym} "; then
                    echo "DRIFT-SUPPRESS: IB has ${sym} (fresh BUY < ${TRADE_DRIFT_RECENT_FILL_GRACE_MIN}m, monitor will reconcile)"
                    continue
                fi
                send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') DRIFT: IB holds <b>${sym}</b> but tracker doesn't know"
                ISSUES=$((ISSUES + 1))
            fi
        done
        for sym in ${TRACKER_TICKERS}; do
            if [ -n "${sym}" ] && ! echo " ${IB_TICKERS} " | grep -q " ${sym} "; then
                # Suppress: OCA stop/target SOLD this in the last GRACE_MIN min —
                # monitor reconcile (CLOSE event with reason=trail_fired) is
                # pending on the next cycle. If the monitor is dead, check 6b
                # (snapshot freshness) fires independently.
                if echo " ${RECENT_SELLS} " | grep -q " ${sym} "; then
                    echo "DRIFT-SUPPRESS: tracker has ${sym} (fresh SELL < ${TRADE_DRIFT_RECENT_FILL_GRACE_MIN}m, monitor will close)"
                    continue
                fi
                send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') DRIFT: tracker holds <b>${sym}</b> but IB doesn't"
                ISSUES=$((ISSUES + 1))
            fi
        done
    else
        # Could not query IB (connection refused, no env vars, etc.)
        # Silent skip — this is NOT a drift, just an unreachable gateway.
        echo "Drift check skipped: IB query failed (${IB_QUERY})"
    fi
fi

# 6b. Portfolio snapshot freshness — alert if monitor daemon's per-cycle
# snapshot write has stalled. The monitor calls write_snapshot() at the end
# of each main loop iteration (~once per cycle = ~30-60s). If the file isn't
# mtime-bumped in >300s during ACTIVE market hours, the daemon is stuck.
#
# 2026-05-21 FIX: only fire from HOUR>=14 UTC. Reason: MARKET_HOURS in this
# script starts at HOUR>=13, but the actual US market open is at 13:30 UTC.
# Before then the monitor is in pre-market sleep and doesn't write snapshots,
# so a "stale" alert at 13:00-13:30 is a false positive. Same for first 30
# minutes after open (giving monitor time to do its first cycle).
SNAPSHOT="/home/stockscout/stock-scout-2/data/trades/portfolio_snapshot.json"
# 2026-05-29 FIX: was 300s, but the monitor's market-hours CHECK_INTERVAL
# is ALSO 300s — so a snapshot is written every ~305-330s (300s sleep +
# cycle exec time). With the threshold == the write interval, the age
# routinely crosses 300s in the few seconds before each NORMAL write, and
# the auto-recover then SIGKILLs a perfectly healthy monitor (observed
# firing repeatedly at ~318s on 2026-05-29). The stale threshold must be
# comfortably ABOVE the write interval. 660s = 2 full missed cycles + 60s
# buffer → only a genuinely stuck daemon (no write for 11 min) trips it,
# while normal 305-330s jitter does not.
SNAPSHOT_STALE_SEC=660  # 11 min — must exceed 2× monitor CHECK_INTERVAL (300s)
# 2026-05-26: gate this check on weekday too. Previously HOUR 14-20 only,
# which fired on Sat/Sun/holiday afternoons even though market is closed
# and (pre-fix monitor) the snapshot was legitimately stale. The Memorial
# Day weekend (Sat 23/5 → Tue 26/5 open) produced 3 false-positive STALE
# alerts on Sunday 21:45, 22:30, 23:15 IL because of this. DOW = 1-5
# means Mon-Fri (date +%u returns 1=Mon..7=Sun). Federal holidays still
# slip through (e.g., Memorial Day, Independence Day) — TODO add a small
# US-holiday allowlist. For now, weekend coverage alone eliminates ~95%
# of the false-positive rate. Pairs with the monitor-side fix that now
# touches the snapshot file every off-hours poll for liveness.
DOW=$(date -u +%u)
if [ "${HOUR}" -ge 14 ] && [ "${HOUR}" -le 20 ] && [ "${DOW}" -le 5 ]; then
    if [ -f "${SNAPSHOT}" ]; then
        SNAPSHOT_AGE=$(( $(date +%s) - $(stat -c %Y "${SNAPSHOT}" 2>/dev/null || echo 0) ))
        if [ "${SNAPSHOT_AGE}" -gt "${SNAPSHOT_STALE_SEC}" ]; then
            # 2026-05-26: auto-recover instead of just alerting and
            # waiting for the user to ssh in and pkill. The user got
            # 3 of these over Sunday night and had to wake up Tue
            # morning to manually kill the process — by which time
            # we'd already missed the (correctly-sleeping-but-looked-
            # stuck) Memorial-Day weekend. A SIGKILL + systemd auto-
            # restart costs ~10s of downtime and is reversible (the
            # tracker file is persisted so positions/OCAs aren't lost).
            # We still send the alert so the user sees what happened.
            echo "[snapshot-stale auto-recover] killing stuck monitor (age=${SNAPSHOT_AGE}s)"
            sudo pkill -KILL -u stockscout -f monitor_positions 2>/dev/null || true
            sleep 2
            sudo systemctl restart stockscout-monitor 2>/dev/null || true
            send_alert_dedup "snapshot_stale" \
                "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Portfolio snapshot was STALE — last write ${SNAPSHOT_AGE}s ago (>${SNAPSHOT_STALE_SEC}s). Monitor daemon was up but its main loop was stuck.

Auto-recovered: pkill -KILL + systemctl restart stockscout-monitor. Send <b>status</b> in ~30s to verify positions are tracked again.

Likely causes: stuck IB call, file lock, exception swallowed in cycle. If this repeats often, dig into journalctl -u stockscout-monitor around the times listed in the alert dedupe log." \
                1800  # 30-min dedup to avoid spam while you investigate
            ISSUES=$((ISSUES + 1))
        else
            clear_alert_dedup "snapshot_stale"
        fi
    else
        # Snapshot file missing — separate alert (different root cause)
        send_alert_dedup "snapshot_missing" \
            "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Portfolio snapshot file MISSING (${SNAPSHOT}). Monitor never reached write_snapshot since startup." \
            7200
        ISSUES=$((ISSUES + 1))
    fi
else
    # Outside active market window — snapshot age is expected to grow as
    # monitor only writes during run_check() (gated on is_market_open).
    # Clear any stale alert so we don't see a leftover from yesterday.
    clear_alert_dedup "snapshot_stale"
fi

# 7. Scan outcomes tracker health — alert if recording has stalled.
# The pending_scans.jsonl file should grow by ~50 rows per trading day.
# If no new entries in 2+ weekdays, something is wrong with the record cron.
PENDING="/home/stockscout/stock-scout-2/data/outcomes/pending_scans.jsonl"
if [ -f "${PENDING}" ]; then
    # Most-recent scan_date in the file
    LAST_SCAN=$(tail -100 "${PENDING}" 2>/dev/null | grep -oE '"scan_date": "[0-9-]+"' | tail -1 \
        | sed 's/.*"\([0-9-]*\)".*/\1/')
    if [ -n "${LAST_SCAN}" ]; then
        AGE_DAYS=$(( ($(date +%s) - $(date -d "${LAST_SCAN}" +%s 2>/dev/null || echo 0)) / 86400 ))
        if [ "${AGE_DAYS}" -gt 3 ]; then
            send_alert "$(echo -e '\xe2\x9a\xa0\xef\xb8\x8f') Scan outcomes tracker STALE — last record ${LAST_SCAN} (${AGE_DAYS} days ago). ML feedback loop broken."
            ISSUES=$((ISSUES + 1))
        fi
    fi
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "Healthcheck OK — all services running"
fi

exit 0

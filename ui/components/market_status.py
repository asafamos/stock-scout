"""Dynamic Market Status Indicator.

Renders a real-time market status bar using client-side JavaScript.
Updates every second without requiring Streamlit re-renders.
Handles pre-market, regular hours, after-hours, weekends, holidays, and early closes.
"""
import streamlit.components.v1 as components


def render_market_status(height: int = 52):
    """Render the dynamic market status indicator."""
    components.html(_MARKET_STATUS_HTML, height=height, scrolling=False)


_MARKET_STATUS_HTML = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent !important; }

  .ms-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 16px;
    border-radius: var(--ss-radius-md, 12px);
    background: var(--ss-bg-card, #ffffff);
    border: 1px solid var(--ss-border, #e2e8f0);
    font-family: var(--ss-font, 'Inter', system-ui, sans-serif);
    font-size: 0.84rem;
    color: var(--ss-text-secondary, #475569);
    direction: ltr;
  }

  .ms-left { display: flex; align-items: center; gap: 10px; }
  .ms-right { display: flex; align-items: center; gap: 14px; font-size: 0.8rem; }

  .ms-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .ms-dot.open {
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e88;
    animation: ms-pulse 1.8s ease-in-out infinite;
  }
  .ms-dot.pre, .ms-dot.after {
    background: #eab308;
    box-shadow: 0 0 4px #eab30866;
  }
  .ms-dot.closed {
    background: #94a3b8;
  }

  @keyframes ms-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(1.25); }
  }

  .ms-label {
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--ss-text-primary, #0f172a);
  }
  .ms-label.open { color: #16a34a; }
  .ms-label.pre  { color: #ca8a04; }
  .ms-label.after { color: #ca8a04; }
  .ms-label.closed { color: #94a3b8; }

  .ms-sep { color: var(--ss-text-muted, #94a3b8); }

  .ms-countdown {
    font-family: var(--ss-mono, 'JetBrains Mono', monospace);
    font-size: 0.78rem;
    color: var(--ss-text-muted, #94a3b8);
  }

  .ms-time {
    font-family: var(--ss-mono, 'JetBrains Mono', monospace);
    font-size: 0.78rem;
    color: var(--ss-text-secondary, #475569);
  }

  @media (prefers-color-scheme: dark) {
    .ms-bar {
      background: var(--ss-bg-card, #1e293b);
      border-color: var(--ss-border, #334155);
      color: var(--ss-text-secondary, #cbd5e1);
    }
    .ms-label { color: var(--ss-text-primary, #f1f5f9); }
    .ms-label.open { color: #4ade80; }
    .ms-label.pre, .ms-label.after { color: #fbbf24; }
    .ms-label.closed { color: #64748b; }
  }
</style>

<div class="ms-bar" id="ms-bar">
  <div class="ms-left">
    <div class="ms-dot" id="ms-dot"></div>
    <span class="ms-label" id="ms-label">—</span>
    <span class="ms-sep">·</span>
    <span class="ms-countdown" id="ms-countdown"></span>
  </div>
  <div class="ms-right">
    <span class="ms-time" id="ms-time"></span>
  </div>
</div>

<script>
(function() {
  // NYSE holidays 2026
  const HOLIDAYS = new Set([
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03',
    '2026-05-25','2026-07-03','2026-09-07','2026-11-26','2026-12-25'
  ]);
  const EARLY_CLOSE = new Set(['2026-11-27','2026-12-24']);

  const PRE_START   =  4 * 60;       // 04:00 ET
  const MKT_OPEN    =  9 * 60 + 30;  // 09:30 ET
  const MKT_CLOSE   = 16 * 60;       // 16:00 ET
  const EARLY_CLS   = 13 * 60;       // 13:00 ET
  const AFTER_END   = 20 * 60;       // 20:00 ET

  function toET(d) {
    const s = d.toLocaleString('en-US', { timeZone: 'America/New_York' });
    return new Date(s);
  }

  function fmtDate(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return y + '-' + m + '-' + day;
  }

  function isWeekend(d) { return d.getDay() === 0 || d.getDay() === 6; }
  function isHoliday(d) { return HOLIDAYS.has(fmtDate(d)); }
  function isEarlyClose(d) { return EARLY_CLOSE.has(fmtDate(d)); }
  function minsOfDay(d) { return d.getHours() * 60 + d.getMinutes(); }

  function nextBusinessDay(et) {
    const d = new Date(et);
    do {
      d.setDate(d.getDate() + 1);
    } while (isWeekend(d) || isHoliday(d));
    return d;
  }

  function fmtCountdown(totalSec) {
    if (totalSec <= 0) return '';
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    const parts = [];
    if (h > 0) parts.push(h + 'h');
    parts.push(String(m).padStart(2, '0') + 'm');
    parts.push(String(s).padStart(2, '0') + 's');
    return parts.join(' ');
  }

  function diffSeconds(fromET, targetMins, targetDate) {
    const tgt = new Date(targetDate || fromET);
    tgt.setHours(Math.floor(targetMins / 60), targetMins % 60, 0, 0);
    return Math.max(0, Math.round((tgt - fromET) / 1000));
  }

  function update() {
    const now = new Date();
    const et = toET(now);
    const mins = minsOfDay(et);
    const closed = isWeekend(et) || isHoliday(et);
    const early = isEarlyClose(et);
    const closeMin = early ? EARLY_CLS : MKT_CLOSE;

    let status, cls, countdownText;

    if (closed) {
      status = 'Market Closed';
      cls = 'closed';
      const nbd = nextBusinessDay(et);
      const dayNames = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
      const diff = Math.round((nbd.setHours(9,30,0,0) - et) / 1000);
      countdownText = 'Opens ' + dayNames[nbd.getDay()] + ' · ' + fmtCountdown(Math.max(0, diff));
    } else if (mins < PRE_START) {
      status = 'Market Closed';
      cls = 'closed';
      countdownText = 'Pre-market in ' + fmtCountdown(diffSeconds(et, PRE_START));
    } else if (mins < MKT_OPEN) {
      status = 'Pre-Market';
      cls = 'pre';
      countdownText = 'Opens in ' + fmtCountdown(diffSeconds(et, MKT_OPEN));
    } else if (mins < closeMin) {
      status = 'Market Open';
      cls = 'open';
      countdownText = 'Closes in ' + fmtCountdown(diffSeconds(et, closeMin));
      if (early) countdownText += ' (early close)';
    } else if (mins < AFTER_END) {
      status = 'After-Hours';
      cls = 'after';
      countdownText = 'Ends in ' + fmtCountdown(diffSeconds(et, AFTER_END));
    } else {
      status = 'Market Closed';
      cls = 'closed';
      const nbd = nextBusinessDay(et);
      const dayNames = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
      const nbdOpen = new Date(nbd); nbdOpen.setHours(9,30,0,0);
      const diff = Math.max(0, Math.round((nbdOpen - et) / 1000));
      countdownText = 'Opens ' + dayNames[nbd.getDay()] + ' · ' + fmtCountdown(diff);
    }

    const timeStr = et.toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: true, timeZone: 'America/New_York'
    }) + ' ET';

    document.getElementById('ms-dot').className = 'ms-dot ' + cls;
    const lbl = document.getElementById('ms-label');
    lbl.className = 'ms-label ' + cls;
    lbl.textContent = status;
    document.getElementById('ms-countdown').textContent = countdownText;
    document.getElementById('ms-time').textContent = timeStr;
  }

  update();
  setInterval(update, 1000);
})();
</script>
"""

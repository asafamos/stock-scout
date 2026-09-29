#!/bin/bash
# Idempotent: keep the IB Gateway API (7496) and its VNC (5800/5900) OFF the public internet.
# 2026-09-29: found all three published on 0.0.0.0 with the host firewall inactive; the
# Gateway trusts only localhost but the container relays remote connections as local, so
# anyone reaching 7496 could place orders on the live account (readOnlyApi=false).
# Loopback (monitor/healthcheck/pipeline on this host) is untouched; reach VNC via
# `ssh -L 5800:localhost:5800 root@<vps>` then http://localhost:5800/vnc.html
set -u
PORTS="7496 5800 5900"
EXT_IF=$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)
EXT_IF=${EXT_IF:-eth0}
for ipt in iptables ip6tables; do
  command -v "$ipt" >/dev/null || continue
  for p in $PORTS; do   # INPUT covers docker-proxy (IPv6 + userland path), non-loopback only
    $ipt -C INPUT -p tcp --dport "$p" ! -i lo -j DROP 2>/dev/null || $ipt -I INPUT -p tcp --dport "$p" ! -i lo -j DROP
  done
done
if iptables -S DOCKER-USER >/dev/null 2>&1; then   # DNAT'd IPv4 traffic to the container
  for p in $PORTS; do
    iptables -C DOCKER-USER -i "$EXT_IF" -p tcp -m conntrack --ctorigdstport "$p" --ctdir ORIGINAL -j DROP 2>/dev/null \
      || iptables -I DOCKER-USER -i "$EXT_IF" -p tcp -m conntrack --ctorigdstport "$p" --ctdir ORIGINAL -j DROP
  done
fi
echo "stockscout-firewall applied (ext_if=$EXT_IF, ports=$PORTS)"

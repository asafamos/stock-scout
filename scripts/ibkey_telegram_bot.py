"""IB Key 2FA via Telegram — semi-automated login for VPS.

When IB Gateway needs 2FA (weekly restart), this script:
1. Detects the 2FA dialog on the Gateway screen
2. Selects "IB Key" as the 2FA method
3. Reads the challenge number from the screen
4. Sends it to you via Telegram
5. You respond with the response code from IBKR Mobile
6. Script enters the code and completes login

Usage:
    python -m scripts.ibkey_telegram_bot

Runs as a daemon alongside IB Gateway on the VPS.
"""

import os
import re
import subprocess
import time
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Telegram config
TOKEN = os.getenv("TRADE_TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TRADE_TELEGRAM_CHAT_ID", "")

# If not in env, try secrets.toml
if not TOKEN:
    try:
        import toml
        secrets = toml.load(".streamlit/secrets.toml")
        TOKEN = secrets.get("TRADE_TELEGRAM_TOKEN", "")
        CHAT_ID = secrets.get("TRADE_TELEGRAM_CHAT_ID", "")
    except Exception:
        pass

DOCKER_CONTAINER = "ibgateway"
CHECK_INTERVAL = 30  # seconds between checks
RESPONSE_TIMEOUT = 300  # 5 min to respond


def send_telegram(msg: str) -> bool:
    """Send message via Telegram bot."""
    if not TOKEN or not CHAT_ID:
        logger.error("Telegram credentials not configured")
        return False
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        resp = requests.post(url, data={
            "chat_id": CHAT_ID,
            "text": msg,
            "parse_mode": "HTML",
        }, timeout=10)
        return resp.ok
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def get_telegram_updates(offset: int = 0) -> list:
    """Get new messages from Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
        resp = requests.get(url, params={
            "offset": offset,
            "timeout": 30,
        }, timeout=35)
        if resp.ok:
            return resp.json().get("result", [])
    except Exception as e:
        logger.error("Telegram getUpdates failed: %s", e)
    return []


def docker_exec(cmd: str) -> str:
    """Execute command inside Docker container."""
    try:
        result = subprocess.run(
            ["docker", "exec", DOCKER_CONTAINER, "bash", "-c", cmd],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"


def take_screenshot() -> str:
    """Take screenshot inside Docker and return path."""
    docker_exec("DISPLAY=:0 scrot /tmp/ibkey_screen.png")
    # Copy to host
    subprocess.run(
        ["docker", "cp", f"{DOCKER_CONTAINER}:/tmp/ibkey_screen.png", "/tmp/ibkey_screen.png"],
        capture_output=True, timeout=10,
    )
    return "/tmp/ibkey_screen.png"


def send_telegram_photo(photo_path: str, caption: str = "") -> bool:
    """Send photo via Telegram."""
    if not TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        with open(photo_path, "rb") as f:
            resp = requests.post(url, data={
                "chat_id": CHAT_ID,
                "caption": caption,
            }, files={"photo": f}, timeout=15)
        return resp.ok
    except Exception as e:
        logger.error("Telegram photo send failed: %s", e)
        return False


def check_gateway_status() -> str:
    """Check if Gateway is connected or needs 2FA."""
    logs = docker_exec("cat /tmp/ibg-jauto.log 2>/dev/null | tail -5")

    if "maintenance cycle" in logs:
        # Check if actually connected by looking for API port
        ports = docker_exec("cat /proc/net/tcp")
        if "0FA1" in ports:  # 4001 in hex
            return "connected"
        if "0FA0" in ports:  # 4000 in hex
            return "connected"
        return "maintenance_disconnected"

    if "two-factor" in logs.lower() or "2FA" in logs or "Second Factor" in logs:
        return "needs_2fa"

    if "login" in logs.lower():
        return "logging_in"

    return "unknown"


def handle_ibkey_2fa():
    """Handle IB Key 2FA flow via Telegram."""
    logger.info("2FA detected — starting IB Key flow")

    # Take screenshot to see what's on screen
    screenshot = take_screenshot()

    # Select IB Key in the 2FA dialog
    docker_exec("DISPLAY=:0 xdotool mousemove 655 388 click 1")
    time.sleep(1)
    docker_exec("DISPLAY=:0 xdotool mousemove 546 429 click 1")
    time.sleep(3)

    # Take screenshot of challenge
    screenshot = take_screenshot()

    # Send to user via Telegram
    send_telegram(
        "🔐 <b>IB Gateway needs authentication!</b>\n\n"
        "Open IBKR Mobile app → IB Key → Generate Response\n"
        "I'm sending the screen — look for the challenge number.\n"
        "Reply with the response code."
    )
    send_telegram_photo(screenshot, "IB Gateway 2FA screen")

    # Wait for response
    logger.info("Waiting for Telegram response (timeout: %ds)...", RESPONSE_TIMEOUT)

    # Get current update offset
    updates = get_telegram_updates()
    offset = updates[-1]["update_id"] + 1 if updates else 0

    start_time = time.time()
    while time.time() - start_time < RESPONSE_TIMEOUT:
        updates = get_telegram_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message", {})
            text = msg.get("text", "").strip()
            chat_id = str(msg.get("chat", {}).get("id", ""))

            if chat_id == CHAT_ID and text:
                # Check if it looks like a response code (digits only)
                code = re.sub(r"\D", "", text)
                if len(code) >= 6:
                    logger.info("Got response code: %s", code)
                    enter_response_code(code)
                    return True

        time.sleep(5)

    logger.warning("2FA response timeout — will retry on next cycle")
    send_telegram("⏰ 2FA timeout — Gateway will retry automatically.")
    return False


def enter_response_code(code: str):
    """Enter the IB Key response code into the Gateway."""
    # Type the code
    docker_exec(f"DISPLAY=:0 xdotool type '{code}'")
    time.sleep(1)
    # Press Enter or click OK
    docker_exec("DISPLAY=:0 xdotool key Return")
    time.sleep(5)

    # Check if login succeeded
    screenshot = take_screenshot()
    send_telegram_photo(screenshot, "After entering code")

    logger.info("Response code entered — checking connection...")
    time.sleep(10)

    status = check_gateway_status()
    if status == "connected":
        send_telegram("✅ IB Gateway connected successfully!")
        logger.info("Gateway connected!")
    else:
        send_telegram(f"⚠️ Gateway status: {status} — may need retry")
        logger.warning("Gateway not connected after code entry: %s", status)


def main():
    logger.info("IB Key Telegram Bot started")
    logger.info("Telegram: %s", "configured" if TOKEN else "NOT configured")

    send_telegram("🤖 IB Key bot started on VPS. I'll notify you when authentication is needed.")

    while True:
        try:
            status = check_gateway_status()

            if status == "connected":
                logger.debug("Gateway connected — all good")
            elif status == "needs_2fa":
                handle_ibkey_2fa()
            elif status == "maintenance_disconnected":
                logger.info("Gateway in maintenance but disconnected — checking...")
                # May need to wait for restart
            else:
                logger.debug("Gateway status: %s", status)

        except Exception as e:
            logger.error("Bot error: %s", e)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

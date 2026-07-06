import os
import time
import json
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ============================================================
# TELEGRAM / RENDER ENV SETTINGS
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

# Scan schedule. For 15m candles, scan once on clock boundaries:
# 9:00, 9:15, 9:30, 9:45, etc. CT.
# Small delay helps yfinance publish the just-closed candle. Set 0 if you want exact boundary.
SCAN_EVERY_MINUTES = int(os.environ.get("SCAN_EVERY_MINUTES", "15"))
SCAN_DELAY_SECONDS = int(os.environ.get("SCAN_DELAY_SECONDS", "30"))
SLEEP_BETWEEN_TICKERS_SEC = float(os.environ.get("SLEEP_BETWEEN_TICKERS_SEC", "0.35"))

INTERVAL = os.environ.get("KALMAN_INTERVAL", "15m")
PERIOD = os.environ.get("KALMAN_PERIOD", "60d")

# ============================================================
# SAME MAIN KALMAN DEFAULTS AS STREAMLIT TAB
# ============================================================
PARAMS = {
    "fast_gain": float(os.environ.get("KALMAN_FAST_GAIN", "0.34")),
    "slow_gain": float(os.environ.get("KALMAN_SLOW_GAIN", "0.055")),
    "polish_span": int(os.environ.get("KALMAN_POLISH_SPAN", "3")),
    "atr_window": int(os.environ.get("KALMAN_ATR_WINDOW", "14")),
    "rail_mult": float(os.environ.get("KALMAN_RAIL_MULT", "1.35")),
    "buffer_pct": float(os.environ.get("KALMAN_BUFFER_PCT", "0.0125")),  # 1.25%
    "confirm_bars": int(os.environ.get("KALMAN_CONFIRM_BARS", "3")),
    "min_hold_bars": int(os.environ.get("KALMAN_MIN_HOLD_BARS", "5")),
    "cooldown_bars": int(os.environ.get("KALMAN_COOLDOWN_BARS", "3")),
    "slope_confirm": os.environ.get("KALMAN_SLOPE_CONFIRM", "true").lower() == "true",
    "atr_safety": os.environ.get("KALMAN_ATR_SAFETY", "true").lower() == "true",
}

# Minimal fix: keep the original fast v12 engine, but load the exact Streamlit
# bundle first. Falls back to the old PER_TICKER_PARAMS_JSON env only if the
# bundle is unavailable. No optimizer is run in Render.
BUNDLE_FILE = os.environ.get(
    "STREAMLIT_KALMAN_BUNDLE_FILE",
    "/etc/secrets/streamlit_kalman_render_bundle.json",
).strip()

def _load_per_ticker_params():
    # 1) Exact Streamlit mirror bundle
    try:
        if BUNDLE_FILE and os.path.exists(BUNDLE_FILE):
            with open(BUNDLE_FILE, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            data = bundle.get("per_ticker_params", {}) if isinstance(bundle, dict) else {}
            if isinstance(data, dict) and data:
                clean = {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}
                print(f"✅ Loaded {len(clean)} ticker params from Streamlit bundle: {BUNDLE_FILE}")
                return clean
    except Exception as e:
        print(f"Streamlit bundle parse error: {e}")

    # 2) Old env fallback
    raw = os.environ.get("PER_TICKER_PARAMS_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        clean = {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}
        print(f"✅ Loaded {len(clean)} ticker params from PER_TICKER_PARAMS_JSON")
        return clean
    except Exception as e:
        print(f"PER_TICKER_PARAMS_JSON parse error: {e}")
        return {}

PER_TICKER_PARAMS = _load_per_ticker_params()

# Safer alert behavior: baseline first, then alert only on Render state transitions.
# This prevents false SELL messages from a historical last-bar recompute when the app was never baselined.
ALERTS_AFTER_BASELINE_ONLY = False  # v12: do not suppress valid first scheduled-bar alerts
SELL_CONFIRM_SCANS = int(os.environ.get("SELL_CONFIRM_SCANS", "1"))
pending_sell_counts = {}

# Optional live seed from your Streamlit visible Main Kalman open positions.
# Example: STREAMLIT_OPEN_TICKERS=PLTR,AAPL,NVDA
# This prevents Render from rewriting an already-open Streamlit trade to CASH on startup.
STREAMLIT_OPEN_TICKERS = {
    t.strip().upper() for t in os.environ.get("STREAMLIT_OPEN_TICKERS", "").split(",") if t.strip()
}

# CRITICAL SAFETY: when true, Render will NOT convert UNKNOWN tickers into CASH/LONG
# from a fresh historical recompute. It waits for /sync or STREAMLIT_OPEN_TICKERS first.
# This prevents the exact PLTR issue: old=UNKNOWN raw_state=CASH raw_alert=SELL.
REQUIRE_STREAMLIT_SYNC = False  # v12 exact-status mode never waits for manual sync
ledger_synced = False

# Persistent non-repaint lock, same idea as Streamlit's .pinehurst_main_kalman_signal_lock file.
seed_protected_until_bar = {}


WATCHLIST = [
    "AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR",
    "AMZN", "APEI", "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ",
    "BABA", "BBAI", "BE", "BR", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX",
    "CIFR", "CLSK", "CMG", "COIN", "CORT", "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP",
    "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", "FVRR", "GLXY", "GOOGL", "GTES", "HCC",
    "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", "IONQ", "IREN", "IRON",
    "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", "MTZ",
    "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN",
    "ORCL", "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX",
    "RDDT", "RDW", "RELX", "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX",
    "SCHW", "SEDG", "SG", "SHAK", "SHOP", "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM",
    "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", "UBER", "UFPT", "ULTA", "UNH", "UPST",
    "V", "VST", "WING", "WMT", "WULF", "XYZ"
]

STATE_FILE = os.environ.get("STATE_FILE", "kalman_render_state_v12.json")
SIGNAL_LOCK_FILE = os.environ.get("SIGNAL_LOCK_FILE", "kalman_render_signal_lock_v12.json")
UPDATE_OFFSET_FILE = os.environ.get("UPDATE_OFFSET_FILE", "kalman_render_update_offset_v12.json")
positions = {ticker: "UNKNOWN" for ticker in WATCHLIST}
last_alert_bar = {}
last_checked = {}
last_error = {}
scan_started_at = None
scan_finished_at = None
scan_in_progress = False
full_scans_completed = 0
rescan_requested = False
last_update_id = None
sync_wait_notice_sent = False
next_scheduled_scan_ct = None
if not REQUIRE_STREAMLIT_SYNC:
    ledger_synced = True

# ============================================================
# TELEGRAM HELPERS
# ============================================================
def _plain_from_html(text: str) -> str:
    """Very small fallback for Telegram if HTML parse fails."""
    return (text or "").replace("<b>", "").replace("</b>", "")


def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram disabled: BOT_TOKEN or CHAT_ID missing")
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        chunks = []
        txt = str(text or "")
        # Telegram max is 4096. Keep it lower to avoid HTML/cutoff issues.
        while len(txt) > 3500:
            cut = txt.rfind("\n", 0, 3500)
            if cut < 1000:
                cut = 3500
            chunks.append(txt[:cut])
            txt = txt[cut:].lstrip()
        chunks.append(txt)

        ok_all = True
        for ch in chunks:
            payload = {"chat_id": CHAT_ID, "text": ch, "parse_mode": "HTML", "disable_web_page_preview": True}
            r = requests.post(url, json=payload, timeout=8)
            if not r.ok:
                # Fallback without HTML parse mode so /status never silently fails.
                payload = {"chat_id": CHAT_ID, "text": _plain_from_html(ch), "disable_web_page_preview": True}
                r2 = requests.post(url, json=payload, timeout=8)
                if not r2.ok:
                    print(f"Telegram send failed: {r.status_code} {r.text} | fallback: {r2.status_code} {r2.text}")
                    ok_all = False
        return ok_all
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False

def load_state():
    global positions, last_alert_bar, seed_protected_until_bar, ledger_synced
    try:
        if os.path.exists(STATE_FILE):
            data = json.load(open(STATE_FILE, "r"))
            saved_pos = data.get("positions", {})
            saved_alerts = data.get("last_alert_bar", {})
            saved_checked = data.get("last_checked", {})
            saved_errors = data.get("last_error", {})
            saved_seed_bars = data.get("seed_protected_until_bar", {})
            saved_streamlit_open = data.get("streamlit_open_tickers", [])
            for t in WATCHLIST:
                if saved_pos.get(t) in ("LONG", "CASH", "UNKNOWN"):
                    positions[t] = saved_pos[t]
            if isinstance(saved_alerts, dict):
                last_alert_bar = saved_alerts
            if isinstance(saved_checked, dict):
                last_checked.update(saved_checked)
            if isinstance(saved_errors, dict):
                last_error.update(saved_errors)
            if isinstance(saved_seed_bars, dict):
                seed_protected_until_bar.update(saved_seed_bars)
            if isinstance(saved_streamlit_open, list):
                STREAMLIT_OPEN_TICKERS.update([str(x).upper() for x in saved_streamlit_open])
            ledger_synced = bool(data.get("ledger_synced", False)) or bool(STREAMLIT_OPEN_TICKERS) or (not REQUIRE_STREAMLIT_SYNC)
            # Seed known-open Streamlit positions at worker startup.
            # This is a ledger seed, not a fresh recompute. It avoids false startup sells.
            for t in STREAMLIT_OPEN_TICKERS:
                if t in positions and positions.get(t) in ("UNKNOWN", "CASH"):
                    positions[t] = "LONG"
            if STREAMLIT_OPEN_TICKERS:
                ledger_synced = True
    except Exception as e:
        print(f"State load error: {e}")


def save_state():
    try:
        json.dump({"positions": positions, "last_alert_bar": last_alert_bar, "last_checked": last_checked, "last_error": last_error, "full_scans_completed": full_scans_completed, "seed_protected_until_bar": seed_protected_until_bar, "streamlit_open_tickers": sorted(STREAMLIT_OPEN_TICKERS), "ledger_synced": ledger_synced}, open(STATE_FILE, "w"), indent=2)
    except Exception as e:
        print(f"State save error: {e}")


def ct_now():
    return datetime.now(ZoneInfo("America/Chicago"))

def fmt_ct_now():
    return ct_now().strftime("%Y-%m-%d %I:%M %p CT")

def next_quarter_scan_time(now=None):
    """Return next CT scan time on 15-minute clock boundary plus SCAN_DELAY_SECONDS."""
    now = now or ct_now()
    interval = max(1, int(SCAN_EVERY_MINUTES))
    base = now.replace(second=0, microsecond=0)
    minute_mod = base.minute % interval
    if minute_mod == 0 and now.second < SCAN_DELAY_SECONDS:
        boundary = base
    else:
        add_min = interval - minute_mod if minute_mod else interval
        boundary = base + timedelta(minutes=add_min)
    return boundary + timedelta(seconds=max(0, int(SCAN_DELAY_SECONDS)))

def fmt_ct_dt(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %I:%M:%S %p CT")

def sleep_until_scan_time(target_dt):
    """Sleep until target_dt while still answering Telegram commands every ~2 seconds."""
    global rescan_requested
    while True:
        if rescan_requested:
            return "manual_rescan"
        remaining = (target_dt - ct_now()).total_seconds()
        if remaining <= 0:
            return "scheduled"
        time.sleep(min(2.0, remaining))

def parse_ticker_csv(raw: str):
    raw = str(raw or "").upper().replace(";", ",").replace("\n", ",")
    return [x.strip() for x in raw.split(",") if x.strip()]

def sync_streamlit_open_ledger(open_tickers):
    """
    Hard-sync Render's position ledger to the Streamlit Main Kalman open list.
    This is needed because Render is a separate worker and cannot read Streamlit's session/trade-log state.
    """
    global STREAMLIT_OPEN_TICKERS, ledger_synced
    clean = {str(t).upper().strip() for t in open_tickers if str(t).upper().strip() in WATCHLIST}
    STREAMLIT_OPEN_TICKERS.clear()
    STREAMLIT_OPEN_TICKERS.update(clean)
    for t in WATCHLIST:
        positions[t] = "LONG" if t in clean else "CASH"
        pending_sell_counts.pop(t, None)
        seed_protected_until_bar.pop(t, None)
    ledger_synced = True
    globals()["sync_wait_notice_sent"] = False
    save_state()
    return sorted(clean)



def _load_update_offset():
    global last_update_id
    try:
        if os.path.exists(UPDATE_OFFSET_FILE):
            data = json.load(open(UPDATE_OFFSET_FILE, "r"))
            val = data.get("last_update_id")
            if isinstance(val, int):
                last_update_id = val
    except Exception as e:
        print(f"Could not load Telegram update offset: {e}")


def _save_update_offset():
    try:
        json.dump({"last_update_id": last_update_id}, open(UPDATE_OFFSET_FILE, "w"))
    except Exception as e:
        print(f"Could not save Telegram update offset: {e}")


def _parse_command(raw_text: str):
    """Supports /status, /status@BotName, mixed case, and arguments."""
    raw = (raw_text or "").strip()
    if not raw.startswith("/"):
        return "", "", raw
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower().split("@", 1)[0]
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg, raw
def handle_telegram_commands():
    global last_update_id, rescan_requested, STREAMLIT_OPEN_TICKERS
    if not BOT_TOKEN:
        return
    try:
        offset = None if last_update_id is None else str(last_update_id)
        params = {"timeout": 0}
        if offset is not None:
            params["offset"] = offset
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        resp = requests.get(url, params=params, timeout=8).json()
        if not resp.get("ok", True):
            print(f"Telegram getUpdates not ok: {resp}")
            return

        for res in resp.get("result", []):
            last_update_id = int(res["update_id"]) + 1
            _save_update_offset()

            msg_obj = res.get("message") or res.get("edited_message") or {}
            raw_text = msg_obj.get("text", "")
            cmd, arg, raw = _parse_command(raw_text)
            if not cmd:
                continue

            if cmd == "/status":
                longs = sorted([t for t, st in positions.items() if st == "LONG"])
                cash = sorted([t for t, st in positions.items() if st == "CASH"])
                unknown = sorted([t for t, st in positions.items() if st not in ("LONG", "CASH")])
                failed = sorted([t for t in WATCHLIST if last_error.get(t)])
                verified = len(longs) + len(cash)
                header_note = ""
                if scan_in_progress:
                    header_note = "\n⏳ <b>Scan in progress.</b> Some tickers may still be from the previous scan."
                if REQUIRE_STREAMLIT_SYNC and not ledger_synced:
                    header_note += "\n🛑 <b>Waiting for /sync.</b> Scanning is locked so UNKNOWN names are not turned into fake CASH."
                elif full_scans_completed == 0:
                    header_note += "\n⚠️ <b>No full baseline scan completed yet.</b> UNKNOWN/default tickers are not real CASH."

                msg = (
                    f"📋 <b>Main Kalman Status — {fmt_ct_now()}</b>\n"
                    f"Interval: <b>{INTERVAL}</b> | Period: <b>{PERIOD}</b>\n"
                    f"Verified: <b>{verified}/{len(WATCHLIST)}</b> | Full scans: <b>{full_scans_completed}</b>{header_note}\n"
                    f"Next scheduled scan: <b>{fmt_ct_dt(next_scheduled_scan_ct)}</b> | Scan cadence: <b>every {SCAN_EVERY_MINUTES}m</b>\n"
                    f"Ledger synced: <b>{ledger_synced}</b> | Require sync: <b>{REQUIRE_STREAMLIT_SYNC}</b> | Streamlit opens: <b>{len(STREAMLIT_OPEN_TICKERS)}</b>\n"
                    f"{len(longs)} LONG / {len(cash)} CASH / {len(unknown)} UNKNOWN / {len(failed)} DATA ERRORS\n\n"
                    + ("<b>LONG</b>\n" + "".join([f"🟢 {t}\n" for t in longs]) if longs else "<b>LONG</b>\nNone\n")
                    + "\n<b>CASH</b>\n"
                    + ("".join([f"⚪ {t}\n" for t in cash]) if cash else "None\n")
                )
                if unknown:
                    msg += "\n<b>UNKNOWN / NOT VERIFIED YET</b>\n" + "".join([f"❔ {t}\n" for t in unknown])
                if failed:
                    msg += "\n<b>DATA ERRORS</b>\n" + "".join([f"⚠️ {t}: {last_error.get(t)}\n" for t in failed[:60]])
                send_telegram(msg)

            elif cmd == "/ping":
                send_telegram(f"✅ <b>Bot is alive.</b> {fmt_ct_now()}\nLedger synced: <b>{ledger_synced}</b> | Require sync: <b>{REQUIRE_STREAMLIT_SYNC}</b>")

            elif cmd == "/rescan":
                rescan_requested = True
                send_telegram("🔄 <b>Rescan requested.</b> I will refresh the full watchlist on this worker loop.")

            elif cmd in ("/why", "/debug"):
                if arg:
                    send_telegram(debug_ticker(arg.split()[0].upper()))
                else:
                    send_telegram("Use /why TICKER, example: /why PLTR")

            elif cmd in ("/sync", "/sync_open", "/setopen"):
                syms = parse_ticker_csv(arg)
                synced = sync_streamlit_open_ledger(syms)
                send_telegram(
                    "✅ <b>Synced Render ledger to Streamlit open list.</b>\n"
                    f"LONG set: <b>{len(synced)}</b>\n"
                    + (", ".join(synced[:120]) if synced else "No tickers matched watchlist")
                    + (f"\n...and {len(synced)-120} more" if len(synced) > 120 else "")
                    + "\n\nNow use /status. From this point, Render tracks NEW transitions from that ledger."
                )

            elif cmd == "/seed":
                syms = parse_ticker_csv(arg)
                added = []
                for t in syms:
                    if t in WATCHLIST:
                        STREAMLIT_OPEN_TICKERS.add(t)
                        positions[t] = "LONG"
                        seed_protected_until_bar.pop(t, None)
                        added.append(t)
                save_state()
                send_telegram("✅ <b>Seeded Streamlit-open tickers as LONG:</b> " + (", ".join(sorted(added)) if added else "None matched watchlist"))

            elif cmd == "/unseed":
                syms = parse_ticker_csv(arg)
                removed = []
                for t in syms:
                    if t in STREAMLIT_OPEN_TICKERS:
                        STREAMLIT_OPEN_TICKERS.discard(t)
                        seed_protected_until_bar.pop(t, None)
                        removed.append(t)
                save_state()
                send_telegram("🧹 <b>Removed Streamlit seed tickers:</b> " + (", ".join(sorted(removed)) if removed else "None"))

            elif cmd == "/params":
                msg = "⚙️ <b>Main Kalman Params</b>\n" + "\n".join([f"{k}: <b>{v}</b>" for k, v in PARAMS.items()])
                msg += f"\nSTREAMLIT_OPEN_TICKERS: <b>{','.join(sorted(STREAMLIT_OPEN_TICKERS)) if STREAMLIT_OPEN_TICKERS else 'NONE'}</b>"
                msg += f"\nPER_TICKER_PARAMS loaded: <b>{len(PER_TICKER_PARAMS)}</b>"
                msg += f"\nREQUIRE_STREAMLIT_SYNC: <b>{REQUIRE_STREAMLIT_SYNC}</b> | ledger_synced: <b>{ledger_synced}</b>"
                msg += "\nCommands: /ping, /status, /why TICKER, /sync T1,T2,T3, /seed T1,T2, /unseed T1"
                send_telegram(msg)

            else:
                send_telegram("Commands: /ping, /status, /params, /why PLTR, /sync PLTR,AAPL,NVDA")

    except Exception as e:
        print(f"Telegram command error: {e}")
# ============================================================
# DATA
# ============================================================
def fetch_completed_bars(ticker: str):
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True, prepost=False, threads=False)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        close_col = "Close" if "Close" in df.columns else df.columns[-1]
        px = pd.Series(df[close_col]).dropna().astype(float)
        if len(px) < 80:
            return None

        # yfinance intraday bars are usually exchange-time/Eastern. Convert to CT for exact matching display.
        try:
            if px.index.tz is None:
                px.index = px.index.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
            px.index = px.index.tz_convert("America/Chicago").tz_localize(None)
        except Exception:
            pass

        # Drop latest candle only if still forming. For 15m, label is candle start.
        try:
            now_ct = pd.Timestamp.now(tz="America/Chicago").tz_localize(None)
            latest_start = pd.Timestamp(px.index[-1])
            interval_minutes = 15 if INTERVAL == "15m" else 5 if INTERVAL == "5m" else 30 if INTERVAL == "30m" else 60
            latest_close = latest_start + pd.Timedelta(minutes=interval_minutes)
            if latest_close > now_ct and len(px) > 2:
                px = px.iloc[:-1]
        except Exception:
            if len(px) > 2:
                px = px.iloc[:-1]

        return px.dropna()
    except Exception as e:
        print(f"{ticker} data error: {e}")
        return None

# ============================================================
# STREAMLIT MAIN KALMAN LOGIC — COPIED/PORTED WITHOUT STREAMLIT
# ============================================================
def institutional_adaptive_kalman_trend(prices, fast_gain=0.34, slow_gain=0.055, vol_window=20, polish_span=3):
    px = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if px.empty:
        return np.array([])

    ret = px.pct_change().abs()
    vol = ret.rolling(int(vol_window), min_periods=max(3, int(vol_window) // 3)).median().replace(0, np.nan)
    shock = (ret / (vol + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) / 3.0

    gains = (float(slow_gain) + (float(fast_gain) - float(slow_gain)) * shock).clip(
        min(float(slow_gain), float(fast_gain)),
        max(float(slow_gain), float(fast_gain))
    )

    out = np.zeros(len(px), dtype=float)
    out[0] = float(px.iloc[0])
    for i in range(1, len(px)):
        out[i] = out[i - 1] + float(gains.iloc[i]) * (float(px.iloc[i]) - out[i - 1])

    if int(polish_span) > 1:
        out = pd.Series(out, index=px.index).ewm(span=int(polish_span), adjust=False).mean().values
    return out


def institutional_trend_rail(prices, fast_gain=0.34, slow_gain=0.055, polish_span=3, atr_window=14, atr_mult=1.35):
    px = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if px.empty:
        return np.array([]), np.array([]), pd.Series(dtype=bool)

    center = pd.Series(
        institutional_adaptive_kalman_trend(
            px.values,
            fast_gain=fast_gain,
            slow_gain=slow_gain,
            vol_window=20,
            polish_span=polish_span,
        ),
        index=px.index,
    )

    atr = px.diff().abs().ewm(span=int(atr_window), adjust=False).mean()
    atr = atr.replace(0, np.nan).ffill().bfill()
    if atr.isna().all():
        atr = pd.Series(px.std() * 0.02 if len(px) > 2 else 1.0, index=px.index)
    atr = atr.fillna(float(px.iloc[-1]) * 0.015)

    slope = center.diff().ewm(span=5, adjust=False).mean().fillna(0)
    long_state = pd.Series(False, index=px.index)
    rail = pd.Series(index=px.index, dtype=float)

    state = True
    rail.iloc[0] = float(center.iloc[0] - float(atr.iloc[0]) * float(atr_mult))
    long_state.iloc[0] = state

    for i in range(1, len(px)):
        p = float(px.iloc[i])
        c = float(center.iloc[i])
        a = float(atr.iloc[i]) * float(atr_mult)
        sl = float(slope.iloc[i])
        prev_rail = float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else c

        if state:
            candidate = c - a
            if sl >= 0:
                candidate = max(candidate, prev_rail)
            if p < prev_rail:
                state = False
                candidate = c + a
        else:
            candidate = c + a
            if sl <= 0:
                candidate = min(candidate, prev_rail)
            if p > prev_rail:
                state = True
                candidate = c - a

        rail.iloc[i] = candidate
        long_state.iloc[i] = state

    rail = rail.ewm(span=2, adjust=False).mean()
    return rail.values, center.values, long_state


def params_for_ticker(ticker: str):
    p = dict(PARAMS)
    override = PER_TICKER_PARAMS.get(str(ticker).upper(), {})
    for k, v in override.items():
        if k in p:
            try:
                if isinstance(p[k], bool):
                    p[k] = bool(v)
                elif isinstance(p[k], int):
                    p[k] = int(v)
                else:
                    p[k] = float(v)
            except Exception:
                pass
    return p



def load_signal_lock():
    try:
        if os.path.exists(SIGNAL_LOCK_FILE):
            data = json.load(open(SIGNAL_LOCK_FILE, "r"))
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Signal lock load error: {e}")
    return {}


def save_signal_lock(store):
    try:
        json.dump(store, open(SIGNAL_LOCK_FILE, "w"), indent=2)
    except Exception as e:
        print(f"Signal lock save error: {e}")


def apply_render_signal_lock(ticker: str, sig: pd.Series):
    """
    Same purpose as Streamlit's _apply_signal_lock():
    already-seen bar timestamps are frozen and cannot be rewritten by a later 60d recompute.
    New completed bars are appended to the lock.
    """
    try:
        if sig is None or len(sig) == 0:
            return sig
        key = f"{str(ticker).upper()}|{INTERVAL}"
        store = load_signal_lock()
        locked = dict(store.get(key, {})) if isinstance(store.get(key), dict) else {}
        out = sig.copy()
        changed = False
        for dt in out.index:
            dtk = pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S")
            if dtk in locked:
                try:
                    out.loc[dt] = float(locked[dtk])
                except Exception:
                    pass
            else:
                try:
                    locked[dtk] = float(out.loc[dt])
                    changed = True
                except Exception:
                    pass
        if changed:
            if len(locked) > 6000:
                for old_key in sorted(locked.keys())[: len(locked) - 6000]:
                    locked.pop(old_key, None)
            store[key] = locked
            save_signal_lock(store)
        return out
    except Exception as e:
        print(f"Signal lock apply error {ticker}: {e}")
        return sig

def build_main_kalman_signal(px: pd.Series, ticker: str = ""):
    pms = params_for_ticker(ticker)
    rail, center, long_state = institutional_trend_rail(
        px,
        fast_gain=pms["fast_gain"],
        slow_gain=pms["slow_gain"],
        polish_span=pms["polish_span"],
        atr_window=pms["atr_window"],
        atr_mult=pms["rail_mult"],
    )

    rail_s = pd.Series(rail, index=px.index).ffill().bfill()
    trend_slope = rail_s.diff().ewm(span=5, adjust=False).mean().fillna(0)

    close_above = px > rail_s * (1.0 + pms["buffer_pct"])
    close_below = px < rail_s * (1.0 - pms["buffer_pct"])

    if pms["slope_confirm"]:
        entry_cond = close_above & (trend_slope >= 0)
        exit_cond = close_below & (trend_slope <= 0)
    else:
        entry_cond = close_above
        exit_cond = close_below

    if pms["atr_safety"]:
        atr_proxy = px.diff().abs().ewm(span=14, adjust=False).mean().replace(0, np.nan).ffill().bfill()
        safety_exit = px < (rail_s - 1.25 * atr_proxy)
        exit_cond = exit_cond | safety_exit.fillna(False)

    confirm_bars = int(pms["confirm_bars"])
    min_hold_bars = int(pms["min_hold_bars"])
    cooldown_bars = int(pms["cooldown_bars"])

    entry_ready = entry_cond.rolling(confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars).fillna(False)
    exit_ready = exit_cond.rolling(confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars).fillna(False)

    sig = pd.Series(0.0, index=px.index)
    in_pos = False
    bars_held = 0
    cooldown_left = 0

    for dt in px.index:
        if cooldown_left > 0:
            cooldown_left -= 1

        if not in_pos:
            if cooldown_left <= 0 and bool(entry_ready.loc[dt]):
                in_pos = True
                bars_held = 0
                sig.loc[dt] = 1.0
            else:
                sig.loc[dt] = 0.0
        else:
            bars_held += 1
            if bars_held >= min_hold_bars and bool(exit_ready.loc[dt]):
                in_pos = False
                cooldown_left = cooldown_bars
                bars_held = 0
                sig.loc[dt] = 0.0
            else:
                sig.loc[dt] = 1.0

    sig = sig.ffill().fillna(0.0).clip(0.0, 1.0)
    sig = apply_render_signal_lock(ticker, sig)
    return sig, rail_s



def run_streamlit_trade_log_status(px: pd.Series, sig: pd.Series, initial_capital: float = 10000.0):
    """Minimal port of Streamlit BacktestEngine.run_strategy trade-status accounting."""
    common_idx = px.index.intersection(sig.index)
    prices = pd.Series(px.loc[common_idx]).replace([np.inf, -np.inf], np.nan).dropna()
    signals = pd.Series(sig).reindex(prices.index).ffill().fillna(0.0).astype(float).clip(0.0, 1.0)
    if len(prices) == 0:
        return "CASH", None

    position = 0
    entry_price = None
    entry_date = None
    last_trade = None

    for dt in prices.index:
        price = float(prices.loc[dt])
        desired = float(signals.loc[dt])
        if position == 0 and desired > 0:
            position = 1
            entry_price = price
            entry_date = dt
        elif position == 1 and desired == 0:
            last_trade = {
                "Status": "Closed",
                "Entry Date": entry_date,
                "Exit Date": dt,
                "Buy Price": entry_price,
                "Sell Price": price,
                "PnL (%)": ((price - entry_price) / entry_price * 100.0) if entry_price else 0.0,
            }
            position = 0
            entry_price = None
            entry_date = None

    if position == 1:
        last_trade = {
            "Status": "Open",
            "Entry Date": entry_date,
            "Exit Date": "Open",
            "Buy Price": entry_price,
            "Sell Price": float(prices.iloc[-1]),
            "PnL (%)": ((float(prices.iloc[-1]) - entry_price) / entry_price * 100.0) if entry_price else 0.0,
        }
        return "LONG", last_trade
    return "CASH", last_trade

def latest_kalman_state(ticker: str):
    px = fetch_completed_bars(ticker)
    if px is None or len(px) < 80:
        return None

    sig, rail_s = build_main_kalman_signal(px, ticker)
    latest_sig = int(sig.iloc[-1])
    prev_sig = int(sig.iloc[-2]) if len(sig) >= 2 else latest_sig

    # Match Streamlit: status comes from the trade log/accounting, not only raw final signal.
    position, last_trade = run_streamlit_trade_log_status(px, sig)
    alert = "BUY" if latest_sig == 1 and prev_sig == 0 else "SELL" if latest_sig == 0 and prev_sig == 1 else "NO NEW ALERT"

    last_start = pd.Timestamp(px.index[-1])
    interval_minutes = 15 if INTERVAL == "15m" else 5 if INTERVAL == "5m" else 30 if INTERVAL == "30m" else 60
    candle_close = last_start + pd.Timedelta(minutes=interval_minutes)

    return {
        "ticker": ticker,
        "position": position,
        "alert": alert,
        "price": float(px.iloc[-1]),
        "rail": float(rail_s.iloc[-1]),
        "bar_start_ct": last_start,
        "candle_close_ct": candle_close,
        "last_trade": last_trade,
    }


def debug_ticker(ticker: str):
    """Telegram /why TICKER helper so we can see why Render differs from Streamlit."""
    ticker = str(ticker).upper().strip()
    try:
        px = fetch_completed_bars(ticker)
        if px is None or len(px) < 80:
            return f"⚠️ {ticker}: no enough completed bars from yfinance."
        pms = params_for_ticker(ticker)
        sig, rail_s = build_main_kalman_signal(px, ticker)
        trend_slope = rail_s.diff().ewm(span=5, adjust=False).mean().fillna(0)
        close_above = px > rail_s * (1.0 + pms["buffer_pct"])
        close_below = px < rail_s * (1.0 - pms["buffer_pct"])
        if pms["slope_confirm"]:
            entry_cond = close_above & (trend_slope >= 0)
            exit_cond = close_below & (trend_slope <= 0)
        else:
            entry_cond = close_above
            exit_cond = close_below
        if pms["atr_safety"]:
            atr_proxy = px.diff().abs().ewm(span=14, adjust=False).mean().replace(0, np.nan).ffill().bfill()
            safety_exit = px < (rail_s - 1.25 * atr_proxy)
            exit_cond = exit_cond | safety_exit.fillna(False)
        cb = int(pms["confirm_bars"])
        exit_ready = exit_cond.rolling(cb, min_periods=cb).sum().eq(cb).fillna(False)
        entry_ready = entry_cond.rolling(cb, min_periods=cb).sum().eq(cb).fillna(False)
        i = -1
        candle_close = pd.Timestamp(px.index[i]) + pd.Timedelta(minutes=15 if INTERVAL == "15m" else 5 if INTERVAL == "5m" else 30 if INTERVAL == "30m" else 60)
        lines = [
            f"<b>Render Kalman Debug — {ticker}</b>",
            f"Position now: {'LONG' if int(sig.iloc[i]) == 1 else 'CASH'} | Prev: {'LONG' if int(sig.iloc[i-1]) == 1 else 'CASH'}",
            f"Price: {float(px.iloc[i]):.2f} | Rail: {float(rail_s.iloc[i]):.2f}",
            f"Candle Close CT: {candle_close.strftime('%Y-%m-%d %I:%M %p CT')}",
            f"close_above: {bool(close_above.iloc[i])} | close_below: {bool(close_below.iloc[i])}",
            f"entry_ready: {bool(entry_ready.iloc[i])} | exit_ready: {bool(exit_ready.iloc[i])}",
            f"trend_slope: {float(trend_slope.iloc[i]):.5f}",
            f"params: buffer={pms['buffer_pct']}, confirm={pms['confirm_bars']}, hold={pms['min_hold_bars']}, cool={pms['cooldown_bars']}, rail_mult={pms['rail_mult']}",
            "Data fetch: auto_adjust=True, prepost=False, period=60d, interval=15m",
        ]
        if ticker in PER_TICKER_PARAMS:
            lines.append("Per-ticker override: YES")
        else:
            lines.append("Per-ticker override: NO — using defaults")
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ {ticker} debug error: {e}"

# ============================================================
# ENGINE LOOP
# ============================================================
def scan_once():
    """
    EXACT-STATUS MODE.

    Every scheduled 15-minute scan:
      1) fetch the same completed 60d/15m price series as Streamlit,
      2) rebuild the same Main Kalman signal history,
      3) apply the same non-repaint lock,
      4) derive current LONG/CASH from the reconstructed trade log,
      5) compare that exact current status with the previous scheduled scan.

    No separate Render ledger mode. No /sync. No UNKNOWN->ledger conversion rules.
    """
    global scan_started_at, scan_finished_at, scan_in_progress, full_scans_completed
    scan_in_progress = True
    scan_started_at = fmt_ct_now()

    for ticker in WATCHLIST:
        try:
            info = latest_kalman_state(ticker)
            if info is None:
                positions[ticker] = "UNKNOWN"
                last_error[ticker] = "not enough data / no yfinance data"
                last_checked[ticker] = fmt_ct_now()
                print(f"⚠️ {ticker}: not enough data")
                time.sleep(SLEEP_BETWEEN_TICKERS_SEC)
                continue

            old_state = positions.get(ticker, "UNKNOWN")
            new_state = info["position"]
            raw_alert = info.get("alert", "NO NEW ALERT")

            last_checked[ticker] = info["candle_close_ct"].strftime("%Y-%m-%d %I:%M %p CT")
            last_error.pop(ticker, None)
            positions[ticker] = new_state

            # Alert rules:
            # - Normal case: exact recomputed status changed since previous scheduled scan.
            # - First-ever scan: alert only when the latest completed bar itself contains
            #   the BUY/SELL transition. This catches AMD opening at 9:15 without creating
            #   a fake alert for an old historical position.
            transition_alert = "NO NEW ALERT"
            if old_state == "CASH" and new_state == "LONG":
                transition_alert = "BUY"
            elif old_state == "LONG" and new_state == "CASH":
                transition_alert = "SELL"
            elif old_state == "UNKNOWN" and raw_alert in ("BUY", "SELL"):
                # On baseline, trust only a transition on the latest completed candle.
                if (raw_alert == "BUY" and new_state == "LONG") or (raw_alert == "SELL" and new_state == "CASH"):
                    transition_alert = raw_alert

            bar_key = f"{ticker}|{transition_alert}|{info['candle_close_ct'].strftime('%Y-%m-%d %H:%M:%S')}"
            if transition_alert in ("BUY", "SELL") and last_alert_bar.get(ticker) != bar_key:
                last_alert_bar[ticker] = bar_key
                emoji = "🟢" if transition_alert == "BUY" else "🔴"
                msg = (
                    f"{emoji} <b>PINEHURST MAIN KALMAN {transition_alert}</b>\n"
                    f"Ticker: <b>{ticker}</b>\n"
                    f"Position: <b>{new_state}</b>\n"
                    f"Price: <b>{info['price']:.2f}</b>\n"
                    f"Rail: <b>{info['rail']:.2f}</b>\n"
                    f"Candle Close CT: <b>{info['candle_close_ct'].strftime('%Y-%m-%d %I:%M %p CT')}</b>\n"
                    f"Source: Exact Streamlit Main Kalman status recompute"
                )
                send_telegram(msg)
                print(f"📊 {ticker}: {old_state} -> {new_state} | {transition_alert} | {info['price']:.2f}")
            else:
                print(f"{ticker}: {new_state} | {info['price']:.2f} | previous={old_state} latest_bar_alert={raw_alert}")

            save_state()
            time.sleep(SLEEP_BETWEEN_TICKERS_SEC)
        except Exception as e:
            positions[ticker] = "UNKNOWN"
            last_error[ticker] = str(e)[:120]
            last_checked[ticker] = fmt_ct_now()
            print(f"{ticker} scan error: {e}")
            time.sleep(SLEEP_BETWEEN_TICKERS_SEC)

    scan_in_progress = False
    scan_finished_at = fmt_ct_now()
    full_scans_completed += 1
    save_state()


def telegram_command_loop():
    """Always-on Telegram polling so /status works while a scan is running."""
    print("💬 Telegram command thread active — /status available anytime")
    while True:
        try:
            handle_telegram_commands()
        except Exception as e:
            print(f"Telegram command loop error: {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    global_vars = globals()
    load_state()
    _load_update_offset()
    threading.Thread(target=telegram_command_loop, daemon=True, name="telegram-command-loop").start()
    print("🚀 Pinehurst Main Kalman FAST OLD ENGINE — Minimal Streamlit Param Fix")
    print(f"Interval={INTERVAL} Period={PERIOD} Watchlist={len(WATCHLIST)}")
    print(f"Scan schedule: every {SCAN_EVERY_MINUTES} minutes on CT clock boundary + {SCAN_DELAY_SECONDS}s delay")
    print("Params:", PARAMS)
    print("Streamlit seed opens:", sorted(STREAMLIT_OPEN_TICKERS))
    print(f"Exact-status mode: no manual sync required")

    next_scheduled_scan_ct = next_quarter_scan_time()
    globals()["next_scheduled_scan_ct"] = next_scheduled_scan_ct
    send_telegram(
        f"🚀 <b>Pinehurst Main Kalman FAST OLD ENGINE — Minimal Streamlit Param Fix</b>\n"
        f"I will scan once on each 15-minute CT boundary: 9:00, 9:15, 9:30, etc. Delay: {SCAN_DELAY_SECONDS}s.\n"
        f"Next scan: <b>{fmt_ct_dt(next_scheduled_scan_ct)}</b>. Use /status anytime."
    )

    while True:
        next_scheduled_scan_ct = next_quarter_scan_time()
        globals()["next_scheduled_scan_ct"] = next_scheduled_scan_ct
        print(f"⏳ Waiting until scheduled scan: {fmt_ct_dt(next_scheduled_scan_ct)}")
        reason = sleep_until_scan_time(next_scheduled_scan_ct)

        if reason == "manual_rescan":
            rescan_requested = False
            print("🔄 Manual /rescan requested. Running one full scan now.")

        scan_once()
        save_state()
        print("✅ Full scan complete. Next scan will be scheduled on the next 15-minute boundary.")

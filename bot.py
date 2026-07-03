import os
import time
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ============================================================
# TELEGRAM / RENDER ENV SETTINGS
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("CHAT_ID", "").strip()

# How often the full watchlist is scanned after one complete pass.
# Render background worker friendly. 300 = every 5 minutes after a pass.
SLEEP_AFTER_FULL_SCAN_SEC = int(os.environ.get("SLEEP_AFTER_FULL_SCAN_SEC", "60"))
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

STATE_FILE = os.environ.get("STATE_FILE", "kalman_render_state.json")
positions = {ticker: "CASH" for ticker in WATCHLIST}
last_alert_bar = {}
last_update_id = None

# ============================================================
# TELEGRAM HELPERS
# ============================================================
def send_telegram(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=8)
        return r.ok
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False


def load_state():
    global positions, last_alert_bar
    try:
        if os.path.exists(STATE_FILE):
            data = json.load(open(STATE_FILE, "r"))
            saved_pos = data.get("positions", {})
            saved_alerts = data.get("last_alert_bar", {})
            for t in WATCHLIST:
                if saved_pos.get(t) in ("LONG", "CASH"):
                    positions[t] = saved_pos[t]
            if isinstance(saved_alerts, dict):
                last_alert_bar = saved_alerts
    except Exception as e:
        print(f"State load error: {e}")


def save_state():
    try:
        json.dump({"positions": positions, "last_alert_bar": last_alert_bar}, open(STATE_FILE, "w"), indent=2)
    except Exception as e:
        print(f"State save error: {e}")


def handle_telegram_commands():
    global last_update_id
    if not BOT_TOKEN:
        return
    try:
        offset = "" if last_update_id is None else str(last_update_id)
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?timeout=0&offset={offset}"
        resp = requests.get(url, timeout=5).json()
        for res in resp.get("result", []):
            last_update_id = int(res["update_id"]) + 1
            text = res.get("message", {}).get("text", "").strip().lower()
            if text == "/status":
                longs = sorted([t for t, s in positions.items() if s == "LONG"])
                cash = sorted([t for t, s in positions.items() if s == "CASH"])
                msg = (
                    f"📋 <b>Main Kalman Status — {datetime.now().strftime('%Y-%m-%d %I:%M %p')}</b>\n"
                    f"Interval: <b>{INTERVAL}</b> | Period: <b>{PERIOD}</b>\n"
                    f"{len(longs)} LONG / {len(cash)} CASH\n\n"
                    + ("<b>LONG</b>\n" + "".join([f"🟢 {t}\n" for t in longs]) if longs else "<b>LONG</b>\nNone\n")
                    + "\n<b>CASH</b>\n"
                    + "".join([f"⚪ {t}\n" for t in cash])
                )
                send_telegram(msg)
            elif text == "/params":
                msg = "⚙️ <b>Main Kalman Params</b>\n" + "\n".join([f"{k}: <b>{v}</b>" for k, v in PARAMS.items()])
                send_telegram(msg)
    except Exception as e:
        print(f"Telegram command error: {e}")

# ============================================================
# DATA
# ============================================================
def fetch_completed_bars(ticker: str):
    try:
        df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, threads=False)
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


def build_main_kalman_signal(px: pd.Series):
    rail, center, long_state = institutional_trend_rail(
        px,
        fast_gain=PARAMS["fast_gain"],
        slow_gain=PARAMS["slow_gain"],
        polish_span=PARAMS["polish_span"],
        atr_window=PARAMS["atr_window"],
        atr_mult=PARAMS["rail_mult"],
    )

    rail_s = pd.Series(rail, index=px.index).ffill().bfill()
    trend_slope = rail_s.diff().ewm(span=5, adjust=False).mean().fillna(0)

    close_above = px > rail_s * (1.0 + PARAMS["buffer_pct"])
    close_below = px < rail_s * (1.0 - PARAMS["buffer_pct"])

    if PARAMS["slope_confirm"]:
        entry_cond = close_above & (trend_slope >= 0)
        exit_cond = close_below & (trend_slope <= 0)
    else:
        entry_cond = close_above
        exit_cond = close_below

    if PARAMS["atr_safety"]:
        atr_proxy = px.diff().abs().ewm(span=14, adjust=False).mean().replace(0, np.nan).ffill().bfill()
        safety_exit = px < (rail_s - 1.25 * atr_proxy)
        exit_cond = exit_cond | safety_exit.fillna(False)

    confirm_bars = int(PARAMS["confirm_bars"])
    min_hold_bars = int(PARAMS["min_hold_bars"])
    cooldown_bars = int(PARAMS["cooldown_bars"])

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
    return sig, rail_s


def latest_kalman_state(ticker: str):
    px = fetch_completed_bars(ticker)
    if px is None or len(px) < 80:
        return None

    sig, rail_s = build_main_kalman_signal(px)
    latest_sig = int(sig.iloc[-1])
    prev_sig = int(sig.iloc[-2]) if len(sig) >= 2 else latest_sig

    position = "LONG" if latest_sig == 1 else "CASH"
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
    }

# ============================================================
# ENGINE LOOP
# ============================================================
def scan_once():
    handle_telegram_commands()

    for ticker in WATCHLIST:
        try:
            info = latest_kalman_state(ticker)
            if info is None:
                print(f"⚠️ {ticker}: not enough data")
                time.sleep(SLEEP_BETWEEN_TICKERS_SEC)
                continue

            old_state = positions.get(ticker, "CASH")
            new_state = info["position"]
            positions[ticker] = new_state

            bar_key = f"{ticker}|{info['alert']}|{info['candle_close_ct'].strftime('%Y-%m-%d %H:%M:%S')}"

            if info["alert"] in ("BUY", "SELL") and last_alert_bar.get(ticker) != bar_key:
                last_alert_bar[ticker] = bar_key
                emoji = "🟢" if info["alert"] == "BUY" else "🔴"
                msg = (
                    f"{emoji} <b>PINEHURST MAIN KALMAN {info['alert']}</b>\n"
                    f"Ticker: <b>{ticker}</b>\n"
                    f"Position: <b>{new_state}</b>\n"
                    f"Price: <b>{info['price']:.2f}</b>\n"
                    f"Rail: <b>{info['rail']:.2f}</b>\n"
                    f"Candle Close CT: <b>{info['candle_close_ct'].strftime('%Y-%m-%d %I:%M %p CT')}</b>\n"
                    f"Source: Main Kalman Trend Rail logic"
                )
                send_telegram(msg)
                print(f"📊 {ticker}: {old_state} -> {new_state} | {info['alert']} | {info['price']:.2f}")
            else:
                print(f"{ticker}: {new_state} | {info['price']:.2f}")

            save_state()
            time.sleep(SLEEP_BETWEEN_TICKERS_SEC)
        except Exception as e:
            print(f"{ticker} scan error: {e}")
            time.sleep(SLEEP_BETWEEN_TICKERS_SEC)


if __name__ == "__main__":
    load_state()
    print("🚀 Pinehurst Main Kalman Render Engine Active")
    print(f"Interval={INTERVAL} Period={PERIOD} Watchlist={len(WATCHLIST)}")
    print("Params:", PARAMS)
    send_telegram("🚀 <b>Pinehurst Main Kalman Render Engine Active</b>")

    while True:
        scan_once()
        save_state()
        handle_telegram_commands()
        print(f"✅ Full scan complete. Sleeping {SLEEP_AFTER_FULL_SCAN_SEC}s")
        time.sleep(SLEEP_AFTER_FULL_SCAN_SEC)

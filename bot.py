import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
WATCHLIST = ["AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR", "AMZN", "APEI", "APLD", "APP", "APPF", "APPS", "ARM", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ", "BABA", "BBAI", "BE", "BKSY", "BR", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX", "CIFR", "CLSK", "CMG", "COIN", "CORT", "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP", "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FLY", "FSLR", "FVRR", "GLXY", "GOOGL", "GTES", "HCC", "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", "IONQ", "IRDM", "IREN", "IRON", "JKHY", "KKR", "LITE", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", "MTZ", "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN", "ORCL", "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "Q", "QBTS", "QCOM", "QNT", "QS", "QUBT", "RBLX", "RDDT", "RDW", "RELX", "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX", "SCHW", "SEDG", "SG", "SHAK", "SHOP", "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM", "T", "TDUP", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", "UBER", "UFPT", "ULTA", "UNH", "UPST", "V", "VST", "WING", "WMT", "WULF", "XYZ", "YSS"]

DEFAULT_PARAMS = {
    "buffer_pct": 0.0125,
    "confirm_bars": 3,
    "min_hold_bars": 5,
    "cooldown_bars": 3,
    "slope_confirm": True,
    "atr_safety": True,
    "fast_gain": 0.34,
    "slow_gain": 0.055,
    "rail_mult": 1.35,
    "polish_span": 3,
    "atr_window": 14,
}

# Locked profiles from the Streamlit optimizer. Paste each ticker's saved
# values here (from ~/.pinehurst_main_kalman_opt_params_V2_CLEAN.json) after
# you optimize it in Streamlit, then redeploy.
TICKER_PROFILES = {
    "CELH": {
        "buffer_pct": 0.02,
        "confirm_bars": 10,
        "min_hold_bars": 55,
        "cooldown_bars": 5,
        "slope_confirm": True,
        "atr_safety": True,
        "fast_gain": 0.34,
        "slow_gain": 0.055,
        "rail_mult": 1.35,
        "polish_span": 3,
        "atr_window": 14,
    },
    # "CELC": { ... },  # add more optimized tickers here
}

# UNKNOWN = not yet verified by a real scan. Never treated as CASH.
# This is the key safety property: we never fire a fake SELL alert just
# because startup hasn't finished checking a ticker yet.
positions = {ticker: "UNKNOWN" for ticker in WATCHLIST}
full_scans_completed = 0
last_update_id = 0
last_cycle_time_ct = None


def get_params_for_ticker(ticker):
    p = dict(DEFAULT_PARAMS)
    p.update(TICKER_PROFILES.get(str(ticker).upper(), {}))
    return p


# ==========================================
# 2. ADAPTIVE KALMAN MATH
# (ported exactly from the Streamlit app's institutional_adaptive_kalman_trend
#  and institutional_trend_rail - volatility-scaled gain blend + stateful
#  hysteresis rail that only flips when price breaks the PRIOR rail value)
# ==========================================
def institutional_adaptive_kalman_trend(prices, fast_gain=0.34, slow_gain=0.055, vol_window=20, polish_span=3):
    px = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if px.empty:
        return np.array([])
    ret = px.pct_change().abs()
    vol = ret.rolling(int(vol_window), min_periods=max(3, int(vol_window) // 3)).median().replace(0, np.nan)
    shock = (ret / (vol + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) / 3.0
    fast_gain = float(fast_gain)
    slow_gain = float(slow_gain)
    gains = (slow_gain + (fast_gain - slow_gain) * shock).clip(min(slow_gain, fast_gain), max(slow_gain, fast_gain))
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
        return np.array([]), np.array([]), pd.Series(dtype=float)

    center = pd.Series(
        institutional_adaptive_kalman_trend(
            px.values, fast_gain=float(fast_gain), slow_gain=float(slow_gain),
            vol_window=20, polish_span=int(polish_span)
        ),
        index=px.index
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
        p_ = float(px.iloc[i])
        c = float(center.iloc[i])
        a = float(atr.iloc[i]) * float(atr_mult)
        sl = float(slope.iloc[i])

        if state:
            candidate = c - a
            if sl >= 0:
                candidate = max(candidate, float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate)
            if p_ < (float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate):
                state = False
                candidate = c + a
        else:
            candidate = c + a
            if sl <= 0:
                candidate = min(candidate, float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate)
            if p_ > (float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate):
                state = True
                candidate = c - a

        rail.iloc[i] = candidate
        long_state.iloc[i] = state

    rail = rail.ewm(span=2, adjust=False).mean()
    return rail.values, center.values, long_state


# ==========================================
# 3. PATH-DEPENDENT TRADE-LOG STATE MACHINE
# (ported exactly from _build_main_kalman_trade_log_from_prices: slope
#  confirm, ATR safety exit, min-hold, cooldown, confirm-bar gating)
# ==========================================
def get_target_state(px, ticker):
    px = pd.Series(px).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 80:
        return None  # not enough data - caller keeps ticker as UNKNOWN

    p = get_params_for_ticker(ticker)

    rail, center, long_state = institutional_trend_rail(
        px,
        fast_gain=p["fast_gain"],
        slow_gain=p["slow_gain"],
        polish_span=p["polish_span"],
        atr_window=p["atr_window"],
        atr_mult=p["rail_mult"],
    )
    bt_trend = pd.Series(rail, index=px.index).ffill().bfill()
    trend_slope = bt_trend.diff().ewm(span=5, adjust=False).mean().fillna(0)

    close_above = px > bt_trend * (1.0 + p["buffer_pct"])
    close_below = px < bt_trend * (1.0 - p["buffer_pct"])

    if p["slope_confirm"]:
        entry_cond = close_above & (trend_slope >= 0)
        exit_cond = close_below & (trend_slope <= 0)
    else:
        entry_cond = close_above
        exit_cond = close_below

    if p["atr_safety"]:
        atr_proxy = px.diff().abs().ewm(span=14, adjust=False).mean().replace(0, np.nan).ffill().bfill()
        safety_exit = px < (bt_trend - 1.25 * atr_proxy)
        exit_cond = exit_cond | safety_exit.fillna(False)

    entry_ready = entry_cond.rolling(p["confirm_bars"], min_periods=p["confirm_bars"]).sum().eq(p["confirm_bars"]).fillna(False)
    exit_ready = exit_cond.rolling(p["confirm_bars"], min_periods=p["confirm_bars"]).sum().eq(p["confirm_bars"]).fillna(False)

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
        else:
            bars_held += 1
            if bars_held >= p["min_hold_bars"] and bool(exit_ready.loc[dt]):
                in_pos = False
                cooldown_left = p["cooldown_bars"]
                bars_held = 0

    return "LONG" if in_pos else "CASH"


def fetch_60d_15m(ticker):
    """Single-ticker fetch mirroring Streamlit's _main_monitor_fetch_15m:
    60 days of 15m bars, tz normalized to Chicago, forming candle dropped."""
    try:
        df = yf.download(ticker, period="60d", interval="15m", auto_adjust=True,
                          progress=False, prepost=False, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    close_col = "Close" if "Close" in df.columns else df.columns[-1]
    px = pd.Series(df[close_col]).dropna().astype(float)
    if len(px) < 80:
        return None
    try:
        if px.index.tz is None:
            px.index = px.index.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        px.index = px.index.tz_convert("America/Chicago").tz_localize(None)
    except Exception:
        pass
    try:
        now_ct = pd.Timestamp.now(tz="America/Chicago").tz_localize(None)
        latest_close = pd.Timestamp(px.index[-1]) + pd.Timedelta(minutes=15)
        if latest_close > now_ct and len(px) > 2:
            px = px.iloc[:-1]
    except Exception:
        if len(px) > 2:
            px = px.iloc[:-1]
    return px.dropna()


def send_alert(message):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=15)
    except Exception:
        pass


def is_market_open_now():
    """Regular session only: Mon-Fri, 8:30 AM - 3:00 PM Central (= 9:30-4:00 ET)."""
    now_ct = pd.Timestamp.now(tz="America/Chicago")
    if now_ct.weekday() >= 5:
        return False
    open_t = now_ct.replace(hour=8, minute=30, second=0, microsecond=0)
    close_t = now_ct.replace(hour=15, minute=0, second=0, microsecond=0)
    return open_t <= now_ct <= close_t


def send_eod_summary():
    today_str = pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d")
    long_list = sorted([t for t in WATCHLIST if positions.get(t) == "LONG"])
    cash_list = sorted([t for t in WATCHLIST if positions.get(t) == "CASH"])
    unknown_list = sorted([t for t in WATCHLIST if positions.get(t) == "UNKNOWN"])

    header = (f"📋 <b>End of Day Summary — {today_str}</b>\n"
              f"{len(long_list)} LONG / {len(cash_list)} CASH / {len(unknown_list)} UNKNOWN\n")
    body_lines = [f"🟢 {t}: LONG" for t in long_list] + [f"⚪ {t}: CASH" for t in cash_list]
    if unknown_list:
        body_lines += [f"❔ {t}: UNKNOWN" for t in unknown_list]

    chunk = header
    for line in body_lines:
        if len(chunk) + len(line) + 1 > 3900:
            send_alert(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        send_alert(chunk)


def send_status_report():
    """Same ticker list as the EOD summary, but labeled as an on-demand
    status check and includes scan/market-state context."""
    now_ct = pd.Timestamp.now(tz="America/Chicago")
    long_list = sorted([t for t in WATCHLIST if positions.get(t) == "LONG"])
    cash_list = sorted([t for t in WATCHLIST if positions.get(t) == "CASH"])
    unknown_list = sorted([t for t in WATCHLIST if positions.get(t) == "UNKNOWN"])

    market_state = "OPEN" if is_market_open_now() else "CLOSED"
    last_cycle_str = last_cycle_time_ct.strftime("%I:%M %p CT") if last_cycle_time_ct else "not yet run"

    header = (
        f"📋 <b>Status — {now_ct.strftime('%Y-%m-%d %I:%M %p CT')}</b>\n"
        f"Market: <b>{market_state}</b> | Baseline scans: <b>{full_scans_completed}</b> | "
        f"Last cycle: <b>{last_cycle_str}</b>\n"
        f"{len(long_list)} LONG / {len(cash_list)} CASH / {len(unknown_list)} UNKNOWN\n"
    )
    body_lines = [f"🟢 {t}: LONG" for t in long_list] + [f"⚪ {t}: CASH" for t in cash_list]
    if unknown_list:
        body_lines += [f"❔ {t}: UNKNOWN" for t in unknown_list]

    chunk = header
    for line in body_lines:
        if len(chunk) + len(line) + 1 > 3900:
            send_alert(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        send_alert(chunk)


def poll_telegram_commands():
    """Checks for new Telegram messages and responds to /status. Safe to
    call frequently - non-blocking (timeout=0) and never raises."""
    global last_update_id
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"offset": last_update_id + 1, "timeout": 0},
            timeout=10,
        )
        updates = resp.json().get("result", [])
    except Exception:
        return

    for upd in updates:
        last_update_id = max(last_update_id, upd.get("update_id", last_update_id))
        msg = upd.get("message", {}) or {}
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != str(CHAT_ID):
            continue
        text = str(msg.get("text", "")).strip().lower()
        if text in ("/status", "status"):
            send_status_report()


# ==========================================
# 4. BASELINE SCAN
# Runs a full pass over the whole watchlist, sets each ticker's real
# LONG/CASH state, but sends NO alerts. This prevents the classic false
# BUY/SELL that happens if you alert on transitions before you've ever
# established what the real starting state is.
# ==========================================
def run_baseline_scan():
    global full_scans_completed
    print("🚀 Running baseline scan (no alerts this pass)...")
    for idx, ticker in enumerate(WATCHLIST):
        px = fetch_60d_15m(ticker)
        state = get_target_state(px, ticker) if px is not None else None
        if state is None:
            positions[ticker] = "UNKNOWN"
            print(f"⚠️ {ticker}: not enough data, staying UNKNOWN")
        else:
            positions[ticker] = state
            print(f"  {ticker}: {state}")
        if idx % 15 == 0:
            poll_telegram_commands()  # keep /status responsive during the long first scan
        time.sleep(0.35)
    full_scans_completed += 1
    verified = sum(1 for v in positions.values() if v in ("LONG", "CASH"))
    print(f"✅ Baseline scan complete. {verified}/{len(WATCHLIST)} verified.")


# ==========================================
# 5. RUN LOOP
# Polls Telegram for commands every ~5 seconds (so /status answers
# immediately, any time) and runs the real ticker scan once per 15-minute
# clock boundary (aligned to :00/:15/:30/:45).
# ==========================================
cycle_count = 0
eod_summary_sent_date = None
last_scan_bucket = None  # (date, hour, quarter) of the last boundary we scanned

run_baseline_scan()
send_status_report()

while True:
    poll_telegram_commands()

    now_ct = pd.Timestamp.now(tz="America/Chicago")
    bucket = (now_ct.date(), now_ct.hour, now_ct.minute // 15)

    if bucket == last_scan_bucket:
        time.sleep(5)
        continue
    last_scan_bucket = bucket

    if not is_market_open_now():
        print(f"💤 {now_ct.strftime('%Y-%m-%d %I:%M %p CT')} - market closed, skipping this cycle")
        if now_ct.weekday() < 5 and now_ct.time() >= pd.Timestamp("15:00").time() and eod_summary_sent_date != now_ct.date():
            print(f"📋 Sending end-of-day summary for {now_ct.strftime('%Y-%m-%d')}...")
            send_eod_summary()
            eod_summary_sent_date = now_ct.date()
        time.sleep(5)
        continue

    cycle_count += 1
    changed = 0
    try:
        for idx, ticker in enumerate(WATCHLIST):
            px = fetch_60d_15m(ticker)
            if px is None:
                continue

            target_state = get_target_state(px, ticker)
            if target_state is None:
                continue

            current_state = positions.get(ticker, "UNKNOWN")

            # UNKNOWN tickers get set silently here (should be rare after
            # baseline, e.g. a ticker with no data at startup that now has
            # enough) - never alert off an UNKNOWN -> anything transition.
            if current_state == "UNKNOWN":
                positions[ticker] = target_state
                continue

            if current_state == "CASH" and target_state == "LONG":
                msg = (f"🟢 <b>{ticker} BUY</b>\n"
                       f"Price: ${round(px.iloc[-1], 2)}\n"
                       f"Date: {px.index[-1].strftime('%Y-%m-%d')}\n"
                       f"Time: {px.index[-1].strftime('%H:%M')} CT")
                send_alert(msg)
                positions[ticker] = "LONG"
                changed += 1

            elif current_state == "LONG" and target_state == "CASH":
                msg = (f"🔴 <b>{ticker} SELL</b>\n"
                       f"Price: ${round(px.iloc[-1], 2)}\n"
                       f"Date: {px.index[-1].strftime('%Y-%m-%d')}\n"
                       f"Time: {px.index[-1].strftime('%H:%M')} CT")
                send_alert(msg)
                positions[ticker] = "CASH"
                changed += 1

            if idx % 15 == 0:
                poll_telegram_commands()  # stay responsive during the ~1-2min scan pass

            time.sleep(0.35)

        last_cycle_time_ct = pd.Timestamp.now(tz="America/Chicago")
        print(f"🔄 Cycle {cycle_count} complete @ {last_cycle_time_ct.strftime('%I:%M %p CT')} "
              f"- {len(WATCHLIST)} tickers checked, {changed} status change(s)")
    except Exception as e:
        print(f"⚠️ Error: {e}")

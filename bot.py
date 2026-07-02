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

WATCHLIST = ["AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR", "AMZN", "APEI", "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ", "BABA", "BBAI", "BE", "BR", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX", "CIFR", "CLSK", "CMG", "COIN", "CORT", "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP", "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", "FVRR", "GLXY", "GOOGL", "GTES", "HCC", "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", "IONQ", "IREN", "IRON", "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", "MTZ", "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN", "ORCL", "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX", "RDDT", "RDW", "RELX", "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX", "SCHW", "SEDG", "SG", "SHAK", "SHOP", "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM", "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", "UBER", "UFPT", "ULTA", "UNH", "UPST", "V", "VST", "WING", "WMT", "WULF", "XYZ"]

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

# Locked profiles from the Streamlit optimizer.
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
}

positions = {ticker: "CASH" for ticker in WATCHLIST}

def get_params_for_ticker(ticker):
    p = dict(DEFAULT_PARAMS)
    p.update(TICKER_PROFILES.get(str(ticker).upper(), {}))
    return p

# ==========================================
# 2. ADAPTIVE KALMAN MATH
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
# ==========================================
def get_target_state(px, ticker):
    px = pd.Series(px).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 80:
        return "CASH"

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
    df = yf.download(ticker, period="60d", interval="15m", auto_adjust=True,
                      progress=False, prepost=False, threads=False)
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
                      json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    except Exception:
        pass

# ==========================================
# 4. INITIALIZATION & UTILITIES
# ==========================================
print("🚀 Pinehurst Engine Initialized. Loading baseline positions...")
for ticker in WATCHLIST:
    px = fetch_60d_15m(ticker)
    if px is None:
        print(f"⚠️ {ticker}: not enough data, defaulting to CASH")
        continue
    positions[ticker] = get_target_state(px, ticker)
print("✅ Initial state locked. Monitoring for absolute status changes...")

def is_market_open_now():
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

    header = f"📋 <b>Status Summary — {today_str}</b>\n{len(long_list)} LONG / {len(cash_list)} CASH\n"
    body_lines = [f"🟢 {t}: LONG" for t in long_list] + [f"⚪ {t}: CASH" for t in cash_list]

    chunk = header
    for line in body_lines:
        if len(chunk) + len(line) + 1 > 3900:
            send_alert(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        send_alert(chunk)

# ==========================================
# 5. RUN LOOP WITH ACTIVE LISTENER
# ==========================================
cycle_count = 0
eod_summary_sent_date = None
last_update_id = None

while True:
    now = datetime.now()
    sleep_time = (15 - (now.minute % 15)) * 60 - now.second
    end_sleep = time.time() + max(sleep_time, 1)

    # ACTIVE SLEEP: Listens for /status command while waiting for the next 15m mark
    while time.time() < end_sleep:
        if BOT_TOKEN:
            try:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?timeout=2"
                if last_update_id:
                    url += f"&offset={last_update_id}"
                
                resp = requests.get(url, timeout=5).json()
                for result in resp.get("result", []):
                    last_update_id = result["update_id"] + 1
                    text = result.get("message", {}).get("text", "").strip()
                    
                    if text == "/status":
                        print(f"📲 On-demand status requested @ {pd.Timestamp.now(tz='America/Chicago').strftime('%I:%M %p CT')}")
                        send_eod_summary()
            except Exception:
                pass
        time.sleep(2)

    if not is_market_open_now():
        now_ct = pd.Timestamp.now(tz="America/Chicago")
        print(f"💤 {now_ct.strftime('%Y-%m-%d %I:%M %p CT')} - market closed, skipping this cycle")

        if now_ct.weekday() < 5 and now_ct.time() >= pd.Timestamp("15:00").time() and eod_summary_sent_date != now_ct.date():
            print(f"📋 Sending end-of-day summary for {now_ct.strftime('%Y-%m-%d')}...")
            send_eod_summary()
            eod_summary_sent_date = now_ct.date()

        continue

    cycle_count += 1
    changed = 0
    try:
        for ticker in WATCHLIST:
            px = fetch_60d_15m(ticker)
            if px is None:
                continue

            target_state = get_target_state(px, ticker)
            current_state = positions[ticker]

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

            time.sleep(0.5)

        print(f"🔄 Cycle {cycle_count} complete @ {pd.Timestamp.now(tz='America/Chicago').strftime('%I:%M %p CT')} "
              f"- {len(WATCHLIST)} tickers checked, {changed} status change(s)")
    except Exception as e:
        print(f"⚠️ Error: {e}")

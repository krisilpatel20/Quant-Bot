import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

WATCHLIST = ["AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR", "AMZN", "APEI", "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ", "BABA", "BBAI", "BE", "BR", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX", "CIFR", "CLSK", "CMG", "COIN", "CORT", "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP", "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", "FVRR", "GLXY", "GOOGL", "GTES", "HCC", "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", "IONQ", "IREN", "IRON", "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", "MTZ", "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN", "ORCL", "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX", "RDDT", "RDW", "RELX", "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX", "SCHW", "SEDG", "SG", "SHAK", "SHOP", "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM", "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", "UBER", "UFPT", "ULTA", "UNH", "UPST", "V", "VST", "WING", "WMT", "WULF", "XYZ"]

last_signals = {}  # ticker -> "LONG" or "CASH", filled during initialization

# ==========================================
# MAIN KALMAN TRADE LOG PARAMETERS
# (ported 1:1 from the Streamlit app's _get_current_main_kalman_params
#  defaults, since this bot has no st.session_state to read sliders from)
# ==========================================
KALMAN_PARAMS = {
    "fast_gain": 0.34,
    "slow_gain": 0.055,
    "polish_span": 3,
    "atr_window": 14,
    "rail_mult": 1.35,      # atr_mult
    "buffer_pct": 0.0125,   # 1.25%
    "confirm_bars": 3,
    "min_hold_bars": 5,
    "cooldown_bars": 3,
    "slope_confirm": True,
    "atr_safety": True,
}


def institutional_adaptive_kalman_trend(prices, fast_gain=0.34, slow_gain=0.055, vol_window=20, polish_span=3):
    """
    Causal adaptive Kalman-style trend line.
    Ported exactly from the Streamlit "Main Kalman" tab so the Render bot's
    center line matches the visible chart/trade log.
    """
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
    """
    Directional trend rail. Ported exactly from the Streamlit "Main Kalman" tab
    (stateful hysteresis: the rail only flips support<->resistance when price
    breaks the PRIOR rail value, not from a static ema_fast > ema_slow check).
    """
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
        p = float(px.iloc[i])
        c = float(center.iloc[i])
        a = float(atr.iloc[i]) * float(atr_mult)
        sl = float(slope.iloc[i])

        if state:
            candidate = c - a
            if sl >= 0:
                candidate = max(candidate, float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate)
            if p < (float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate):
                state = False
                candidate = c + a
        else:
            candidate = c + a
            if sl <= 0:
                candidate = min(candidate, float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate)
            if p > (float(rail.iloc[i - 1]) if np.isfinite(rail.iloc[i - 1]) else candidate):
                state = True
                candidate = c - a

        rail.iloc[i] = candidate
        long_state.iloc[i] = state

    rail = rail.ewm(span=2, adjust=False).mean()
    return rail.values, center.values, long_state


def get_signal(px, params=KALMAN_PARAMS):
    """
    Reproduces _build_main_kalman_trade_log_from_prices' position state machine
    (entry/exit confirm bars, slope confirm, ATR safety exit, min-hold, cooldown)
    and returns the CURRENT position status: "LONG" or "CASH".

    This is deliberately NOT a bar-to-bar transition check. Comparing sig.iloc[-2]
    vs sig.iloc[-1] only catches a flip if the poll lands on the exact bar it
    happened - miss one poll (restart, rate limit, downtime) and that flip
    silently slides into the middle of the window and never fires. Returning
    the absolute current status instead lets the caller compare it against the
    last KNOWN status (persisted across polls) so a status change is always
    caught, no matter how many bars were missed in between.
    """
    px = pd.Series(px).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 80:
        return "CASH"

    rail, center, long_state = institutional_trend_rail(
        px,
        fast_gain=params["fast_gain"],
        slow_gain=params["slow_gain"],
        polish_span=params["polish_span"],
        atr_window=params["atr_window"],
        atr_mult=params["rail_mult"],
    )
    bt_trend = pd.Series(rail, index=px.index).ffill().bfill()
    trend_slope = bt_trend.diff().ewm(span=5, adjust=False).mean().fillna(0)

    buffer_pct = params["buffer_pct"]
    confirm_bars = params["confirm_bars"]
    min_hold_bars = params["min_hold_bars"]
    cooldown_bars = params["cooldown_bars"]

    close_above = px > bt_trend * (1.0 + buffer_pct)
    close_below = px < bt_trend * (1.0 - buffer_pct)

    if params["slope_confirm"]:
        entry_cond = close_above & (trend_slope >= 0)
        exit_cond = close_below & (trend_slope <= 0)
    else:
        entry_cond = close_above
        exit_cond = close_below

    if params["atr_safety"]:
        atr_proxy = px.diff().abs().ewm(span=14, adjust=False).mean().replace(0, np.nan).ffill().bfill()
        safety_exit = px < (bt_trend - 1.25 * atr_proxy)
        exit_cond = exit_cond | safety_exit.fillna(False)

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

    sig = sig.ffill().fillna(0).clip(0, 1)

    if sig.empty:
        return "CASH"

    return "LONG" if sig.iloc[-1] >= 1.0 else "CASH"


# ==========================================
# INITIALIZATION
# ==========================================
print("🚀 Initializing state...")
for i in range(0, len(WATCHLIST), 20):
    batch = WATCHLIST[i:i + 20]
    raw = yf.download(batch, period="1mo", interval="15m", group_by="ticker", threads=False)
    for ticker in batch:
        df = raw[ticker].dropna() if len(batch) > 1 else raw.dropna()
        df.index = df.index.tz_convert('America/Chicago')
        last_signals[ticker] = get_signal(df['Close'].tail(200))
print("✅ Ready. Monitoring for changes...")

# ==========================================
# RUN LOOP
# ==========================================
while True:
    now = datetime.now()
    sleep_time = (15 - (now.minute % 15)) * 60 - now.second
    time.sleep(max(sleep_time, 1))

    try:
        for i in range(0, len(WATCHLIST), 20):
            batch = WATCHLIST[i:i + 20]
            raw = yf.download(batch, period="1mo", interval="15m", group_by="ticker", threads=False)
            for ticker in batch:
                df = raw[ticker].dropna() if len(batch) > 1 else raw.dropna()
                df.index = df.index.tz_convert('America/Chicago')
                curr_status = get_signal(df['Close'].tail(200))
                prev_status = last_signals.get(ticker)

                # Only act on tickers whose status actually flipped LONG<->CASH.
                # Unchanged tickers (the vast majority every poll) are skipped
                # entirely - no message, no state churn.
                if prev_status is not None and curr_status != prev_status:
                    action = "BUY" if curr_status == "LONG" else "SELL"
                    msg = (f"{'🟢' if action=='BUY' else '🔴'} <b>{ticker} {action}</b>\n"
                           f"Price: ${round(df['Close'].iloc[-1],2)}\n"
                           f"Date: {df.index[-1].strftime('%Y-%m-%d')}\n"
                           f"Time: {df.index[-1].strftime('%H:%M')}")
                    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})

                last_signals[ticker] = curr_status
            time.sleep(10)
    except Exception as e:
        print(f"⚠️ Error: {e}")

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

PARAMS = {"buffer_pct": 0.015, "confirm_bars": 5, "min_hold_bars": 10, "cooldown_bars": 5, "slope_confirm": True, "atr_safety": True, "fast_gain": 0.34, "slow_gain": 0.055, "rail_mult": 1.35, "polish_span": 3, "atr_window": 14}
positions = {ticker: "CASH" for ticker in WATCHLIST}

# ==========================================
# 2. FIXED CORE MATH
# ==========================================
def fetch_60d_15m(ticker):
    df = yf.download(ticker, period="60d", interval="15m", progress=False)
    if df.empty: return None
    px = df['Close'].values.flatten() if 'Close' in df else df.iloc[:, 0].values.flatten()
    return px if len(px) >= 80 else None

def institutional_adaptive_kalman_trend(px):
    px = pd.Series(px).ffill().bfill().values
    ret = np.abs(np.diff(px, prepend=px[0]) / (px + 1e-12))
    vol = pd.Series(ret).rolling(20, min_periods=3).median().fillna(0.01).values
    shock = np.clip(ret / (vol + 1e-12), 0, 3) / 3.0
    gains = np.clip(PARAMS["slow_gain"] + (PARAMS["fast_gain"] - PARAMS["slow_gain"]) * shock, PARAMS["slow_gain"], PARAMS["fast_gain"])
    out = np.zeros_like(px)
    out[0] = px[0]
    for i in range(1, len(px)):
        out[i] = out[i-1] + gains[i] * (px[i] - out[i-1])
    return pd.Series(out).ewm(span=PARAMS["polish_span"]).mean().values

def get_target_state(px):
    center = institutional_adaptive_kalman_trend(px)
    atr = pd.Series(px).diff().abs().ewm(span=PARAMS["atr_window"]).mean().fillna(px[-1]*0.01).values
    rail = center - (atr * PARAMS["rail_mult"])
    close_above = px > (rail * (1.0 + PARAMS["buffer_pct"]))
    close_below = px < (rail * (1.0 - PARAMS["buffer_pct"]))
    entry_ready = pd.Series(close_above).rolling(PARAMS["confirm_bars"]).sum() == PARAMS["confirm_bars"]
    exit_ready = pd.Series(close_below).rolling(PARAMS["confirm_bars"]).sum() == PARAMS["confirm_bars"]
    state = "CASH"
    for i in range(len(px)):
        if state == "CASH" and entry_ready.iloc[i]: state = "LONG"
        elif state == "LONG" and exit_ready.iloc[i]: state = "CASH"
    return state

# ==========================================
# 3. RUNNER WITH HIGH-FREQUENCY TELEGRAM LISTENER
# ==========================================
print("🚀 Pinehurst Engine Active.")
last_update_id = None

while True:
    for ticker in WATCHLIST:
        # A. Check Telegram command every single loop
        if BOT_TOKEN:
            try:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?timeout=0"
                if last_update_id: url += f"&offset={last_update_id}"
                resp = requests.get(url, timeout=2).json()
                for result in resp.get("result", []):
                    last_update_id = result["update_id"] + 1
                    if result.get("message", {}).get("text", "").strip() == "/status":
                        longs = [t for t, state in positions.items() if state == "LONG"]
                        msg = f"📋 <b>Report</b>\nLongs: {', '.join(longs) if longs else 'None'}"
                        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                                      json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
            except: pass

        # B. Scan the Ticker
        px = fetch_60d_15m(ticker)
        if px is not None:
            target = get_target_state(px)
            if positions[ticker] != target:
                print(f"📊 {ticker}: {positions[ticker]} -> {target}")
                positions[ticker] = target
        
        # C. Small pause to keep connection healthy
        time.sleep(0.5)

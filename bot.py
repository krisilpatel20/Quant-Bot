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

# Baseline institutional parameters
PARAMS = {
    "buffer_pct": 0.015,
    "confirm_bars": 5,
    "min_hold_bars": 10,
    "cooldown_bars": 5,
    "slope_confirm": True,
    "atr_safety": True,
    "fast_gain": 0.34,
    "slow_gain": 0.055,
    "rail_mult": 1.35,
    "polish_span": 3,
    "atr_window": 14,
}

positions = {ticker: "CASH" for ticker in WATCHLIST}

# ==========================================
# 2. CORE MATH (NO EXTERNAL FILES)
# ==========================================
def institutional_adaptive_kalman_trend(prices):
    px = pd.Series(prices).astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    ret = px.pct_change().abs()
    vol = ret.rolling(20, min_periods=3).median().replace(0, np.nan)
    shock = (ret / (vol + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 3) / 3.0
    gains = (PARAMS["slow_gain"] + (PARAMS["fast_gain"] - PARAMS["slow_gain"]) * shock).clip(PARAMS["slow_gain"], PARAMS["fast_gain"])
    out = np.zeros(len(px), dtype=float)
    out[0] = float(px.iloc[0])
    for i in range(1, len(px)):
        out[i] = out[i - 1] + float(gains.iloc[i]) * (float(px.iloc[i]) - out[i - 1])
    return pd.Series(out, index=px.index).ewm(span=PARAMS["polish_span"], adjust=False).mean().values

def fetch_60d_15m(ticker):
    df = yf.download(ticker, period="60d", interval="15m", auto_adjust=True, progress=False, threads=False)
    if df.empty: return None
    close_col = "Close" if "Close" in df.columns else df.columns[-1]
    px = pd.Series(df[close_col]).dropna().astype(float)
    return px if len(px) >= 80 else None

def get_target_state(px):
    rail, center, _ = institutional_trend_rail(px)
    bt_trend = pd.Series(rail, index=px.index)
    close_above = px > bt_trend * (1.0 + PARAMS["buffer_pct"])
    close_below = px < bt_trend * (1.0 - PARAMS["buffer_pct"])
    
    entry_ready = close_above.rolling(PARAMS["confirm_bars"]).sum().eq(PARAMS["confirm_bars"])
    exit_ready = close_below.rolling(PARAMS["confirm_bars"]).sum().eq(PARAMS["confirm_bars"])
    
    in_pos = False
    for i in range(len(px)):
        if not in_pos and entry_ready.iloc[i]: in_pos = True
        elif in_pos and exit_ready.iloc[i]: in_pos = False
    return "LONG" if in_pos else "CASH"

# Simplified Rail Logic
def institutional_trend_rail(px):
    center = institutional_adaptive_kalman_trend(px.values)
    atr = px.diff().abs().ewm(span=PARAMS["atr_window"]).mean().fillna(px.iloc[-1]*0.01)
    rail = center - (atr * PARAMS["rail_mult"])
    return rail, center, None

# ==========================================
# 3. RUNNER
# ==========================================
print("🚀 Pinehurst Engine Freshly Initialized.")
while True:
    for ticker in WATCHLIST:
        px = fetch_60d_15m(ticker)
        if px is None: continue
        
        target = get_target_state(px)
        if positions[ticker] != target:
            print(f"📊 {ticker} changed to {target}")
            positions[ticker] = target
        time.sleep(1)
    time.sleep(300) # Sleep between full market scans

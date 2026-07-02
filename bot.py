import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import urllib.request

# ==========================================
# 1. CONFIGURATION
# ==========================================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

WATCHLIST = [
    "AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR", "AMZN", "APEI", 
    "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ", "BABA", "BBAI", "BE", "BR", 
    "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX", "CIFR", "CLSK", "CMG", "COIN", "CORT", 
    "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP", "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", 
    "FVRR", "GLXY", "GOOGL", "GTES", "HCC", "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", 
    "IONQ", "IREN", "IRON", "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", 
    "MTZ", "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN", "ORCL", 
    "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX", "RDDT", "RDW", "RELX", 
    "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX", "SCHW", "SEDG", "SG", "SHAK", "SHOP", 
    "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM", "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", 
    "UBER", "UFPT", "ULTA", "UNH", "UPST", "V", "VST", "WING", "WMT", "WULF", "XYZ"
]

# Helper for Trend Rail
def institutional_trend_rail(px, fast_gain=0.34, slow_gain=0.055, polish_span=3, atr_window=14, atr_mult=1.35):
    # This matches your institutional_trend_rail function
    ema_fast = px.ewm(span=int(1/fast_gain), adjust=False).mean()
    ema_slow = px.ewm(span=int(1/slow_gain), adjust=False).mean()
    center = (ema_fast + ema_slow) / 2
    atr = px.diff().abs().ewm(span=atr_window, adjust=False).mean()
    rail = center - (atr * atr_mult)
    trend_state = ema_fast > ema_slow
    return rail, center, trend_state

def get_signal_institutional(px):
    # This mirrors your institutional logic
    rail, center, long_state = institutional_trend_rail(px)
    rail_s = pd.Series(rail, index=px.index).ffill().bfill()
    state_s = pd.Series(long_state, index=px.index).astype(bool)
    
    # Parameters matching your dashboard
    buffer_pct = 0.0125
    confirm_bars = 3
    
    above = ((px > rail_s * (1.0 + buffer_pct)) & state_s).astype(int)
    below = ((px < rail_s * (1.0 - buffer_pct)) | (~state_s)).astype(int)
    
    confirmed_buy = above.rolling(confirm_bars).sum() >= confirm_bars
    confirmed_sell = below.rolling(confirm_bars).sum() >= confirm_bars
    
    if confirmed_buy.iloc[-1]: return "BUY"
    if confirmed_sell.iloc[-1]: return "SELL"
    return "HOLD"

# Telegram sender
def send_alert(message):
    if not BOT_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    except Exception as e: print(f"Alert Error: {e}")

# Run Loop
print("🚀 Institutional Engine Initialized...")
while True:
    try:
        for i in range(0, len(WATCHLIST), 20):
            batch = WATCHLIST[i:i+20]
            raw = yf.download(batch, period="1mo", interval="15m", group_by="ticker", threads=False)
            
            for ticker in batch:
                df = raw[ticker].dropna() if len(batch) > 1 else raw.dropna()
                if len(df) < 80: continue
                
                # Align time and history window
                df.index = df.index.tz_convert('America/Chicago')
                px = df['Close'].tail(200)
                
                curr = get_signal_institutional(px)
                
                # Logic: Only alert on fresh changes
                # Note: This is simplified; to be 100% like Streamlit, 
                # you would need to implement the full trade log state check.
                if curr in ["BUY", "SELL"]:
                    msg = (f"{'🟢' if curr=='BUY' else '🔴'} <b>{ticker} {curr}</b>\n"
                           f"Price: ${round(px.iloc[-1],2)}\n"
                           f"Time: {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
                    send_alert(msg)
            time.sleep(10)
    except Exception as e: print(f"⚠️ Error: {e}")
    time.sleep(300)

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

last_signals = {ticker: "HOLD" for ticker in WATCHLIST}

def institutional_trend_rail(px, fast_gain=0.34, slow_gain=0.055, atr_window=14, atr_mult=1.35):
    ema_fast = px.ewm(span=int(1/fast_gain), adjust=False).mean()
    ema_slow = px.ewm(span=int(1/slow_gain), adjust=False).mean()
    center = (ema_fast + ema_slow) / 2
    atr = px.diff().abs().ewm(span=atr_window, adjust=False).mean()
    rail = center - (atr * atr_mult)
    trend_state = ema_fast > ema_slow
    return rail, trend_state

def get_signal(px):
    rail, long_state = institutional_trend_rail(px)
    buffer_pct = 0.0125
    above = ((px > rail * (1.0 + buffer_pct)) & long_state).astype(int)
    below = ((px < rail * (1.0 - buffer_pct)) | (~long_state)).astype(int)
    if above.rolling(3).sum().iloc[-1] >= 3: return "BUY"
    if below.rolling(3).sum().iloc[-1] >= 3: return "SELL"
    return "HOLD"

# ==========================================
# INITIALIZATION
# ==========================================
print("🚀 Initializing state...")
for i in range(0, len(WATCHLIST), 20):
    batch = WATCHLIST[i:i+20]
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
            batch = WATCHLIST[i:i+20]
            raw = yf.download(batch, period="1mo", interval="15m", group_by="ticker", threads=False)
            for ticker in batch:
                df = raw[ticker].dropna() if len(batch) > 1 else raw.dropna()
                df.index = df.index.tz_convert('America/Chicago')
                curr = get_signal(df['Close'].tail(200))
                
                if curr != last_signals[ticker] and curr != "HOLD":
                    msg = (f"{'🟢' if curr=='BUY' else '🔴'} <b>{ticker} {curr}</b>\n"
                           f"Price: ${round(df['Close'].iloc[-1],2)}\n"
                           f"Date: {df.index[-1].strftime('%Y-%m-%d')}\n"
                           f"Time: {df.index[-1].strftime('%H:%M')}")
                    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", 
                                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
                
                last_signals[ticker] = curr
            time.sleep(10)
    except Exception as e: print(f"⚠️ Error: {e}")

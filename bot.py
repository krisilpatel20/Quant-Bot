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

WATCHLIST = [
    "AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR", "AMZN", "APEI", 
    "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ", "BABA", "BBAI", "BE", "BR", 
    "BRK.B", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX", "CIFR", "CLSK", "CMG", "COIN", "CORT", 
    "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP", "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", 
    "FVRR", "GLXY", "GOOGL", "GTES", "HCC", "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", 
    "IONQ", "IREN", "IRON", "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", 
    "MTZ", "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN", "ORCL", 
    "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX", "RDDT", "RDW", "RELX", 
    "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX", "SCHW", "SEDG", "SG", "SHAK", "SHOP", 
    "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM", "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", 
    "UBER", "UFPT", "ULTA", "UNH", "UPST", "V", "VST", "WING", "WMT", "WULF", "XYZ"
]

last_signals = {ticker: None for ticker in WATCHLIST}

def send_alert(message):
    if not BOT_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    except Exception as e: print(f"Alert Error: {e}")

def wait_for_next_15m():
    now = datetime.now()
    next_minute = (now.minute // 15 + 1) * 15
    if next_minute >= 60: next_time = now.replace(hour=now.hour + 1, minute=0, second=5, microsecond=0)
    else: next_time = now.replace(minute=next_minute, second=5, microsecond=0)
    time.sleep((next_time - now).total_seconds())

class KalmanFilterTrend:
    def __init__(self, Q=1e-3, R=5e-2): self.Q, self.R = Q, R
    def filter(self, data):
        n = len(data); mean = np.zeros(n); cov = np.zeros(n)
        mean[0] = data.iloc[0]; cov[0] = 1.0
        for t in range(1, n):
            pred_m = mean[t-1]; pred_c = cov[t-1] + self.Q
            K = pred_c / (pred_c + self.R)
            mean[t] = pred_m + K * (data.iloc[t] - pred_m)
            cov[t] = (1 - K) * pred_c
        return pd.Series(mean, index=data.index)

def calculate_kalman_15m_signal(px):
    if len(px) < 20: return "HOLD"
    kf = KalmanFilterTrend()
    centerline = kf.filter(px)
    atr = px.diff().abs().ewm(span=14, adjust=False).mean()
    rail = centerline - (atr * 1.1)
    slope = centerline.diff().ewm(span=3, adjust=False).mean() >= 0
    above = ((px > rail * 1.01) & slope).astype(int).rolling(3).sum() >= 3
    below = ((px < rail * 0.99) & (~slope)).astype(int).rolling(3).sum() >= 3
    if above.iloc[-1]: return "BUY"
    elif below.iloc[-1]: return "SELL"
    return "HOLD"

print("🚀 Quant Engine Initialized...")
while True:
    wait_for_next_15m()
    try:
        for i in range(0, len(WATCHLIST), 20):
            batch = WATCHLIST[i:i+20]
            raw = yf.download(batch, period="5d", interval="15m", group_by="ticker", threads=False)
            for ticker in batch:
                df = raw[ticker].dropna() if len(batch) > 1 else raw.dropna()
                if len(df) < 20: continue
                curr = calculate_kalman_15m_signal(df['Close'].astype(float))
                if last_signals[ticker] and curr != last_signals[ticker] and curr in ["BUY", "SELL"]:
                    send_alert(f"{'🟢' if curr=='BUY' else '🔴'} <b>{ticker} {curr}</b>\nPrice: ${round(df['Close'].iloc[-1],2)}\nTime: {df.index[-1].strftime('%H:%M')}")
                last_signals[ticker] = curr
            time.sleep(10)
    except Exception as e: print(f"⚠️ Error: {e}")

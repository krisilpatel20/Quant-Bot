import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 1. SECURE CONFIGURATION
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

# State Memory - initialized to None to avoid initial spam
last_signals = {ticker: None for ticker in WATCHLIST}

def send_alert(message):
    if not BOT_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"})
    except Exception as e:
        print(f"Alert Error: {e}")

class KalmanFilterTrend:
    def __init__(self, process_noise=1e-3, measurement_noise=5e-2):
        self.Q, self.R = process_noise, measurement_noise
    def filter(self, data):
        data_vals = data.values
        n = len(data_vals)
        state_mean = np.zeros(n)
        state_cov = np.zeros(n)
        state_mean[0] = data_vals[0]
        state_cov[0] = 1.0
        for t in range(1, n):
            pred_mean = state_mean[t-1]
            pred_cov = state_cov[t-1] + self.Q
            K = pred_cov / (pred_cov + self.R)
            state_mean[t] = pred_mean + K * (data_vals[t] - pred_mean)
            state_cov[t] = (1 - K) * pred_cov
        return pd.Series(state_mean, index=data.index)

def calculate_kalman_15m_signal(px):
    if len(px) < 20: return "HOLD"
    kf = KalmanFilterTrend()
    centerline = kf.filter(px)
    atr = px.diff().abs().ewm(span=14, adjust=False).mean()
    # Rail logic adjusted for sensitivity
    rail_s = pd.Series(centerline - (atr * 1.1), index=px.index).ffill().bfill()
    trend_slope = centerline.diff().ewm(span=3, adjust=False).mean()
    state_s = trend_slope >= 0 
    # Confirmation logic
    above = ((px > rail_s * 1.01) & state_s).astype(int).rolling(3).sum() >= 3
    below = ((px < rail_s * 0.99) & (~state_s)).astype(int).rolling(3).sum() >= 3
    if above.iloc[-1]: return "BUY"
    elif below.iloc[-1]: return "SELL"
    else: return "HOLD"

print("🚀 Quant Engine Initialized...")
while True:
    try:
        raw_data = yf.download(WATCHLIST, period="5d", interval="15m", group_by="ticker", threads=True)
        for ticker in WATCHLIST:
            df = raw_data[ticker].dropna() if len(WATCHLIST) > 1 else raw_data.dropna()
            if df.empty or len(df) < 20: continue
            
            # Run signal logic
            current_state = calculate_kalman_15m_signal(df['Close'].astype(float))
            
            # Send alert only on change
            if last_signals[ticker] is not None and current_state != last_signals[ticker] and current_state in ["BUY", "SELL"]:
                msg = f"{'🟢' if current_state == 'BUY' else '🔴'} <b>{ticker} Signal: {current_state}</b>\nPrice: ${round(df['Close'].iloc[-1], 2)}\nTime: {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
                send_alert(msg)
            
            last_signals[ticker] = current_state
            
    except Exception as e: print(f"⚠️ Error: {e}")
    time.sleep(900)

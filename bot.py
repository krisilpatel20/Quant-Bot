import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ADD THIS LINE RIGHT HERE
print("--- SCRIPT IS STARTING UP NOW ---")

# ==========================================
# 1. SECURE CONFIGURATION
# ==========================================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("🚨 ERROR: Missing Environment Variables! Check Render settings.")

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

# ==========================================
# 2. STATE MEMORY
# ==========================================
last_signals = {}

def send_alert(message):
    """Sends a formatted message to Telegram with a strict speed limit."""
    if not BOT_TOKEN or not CHAT_ID:
        return
        
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"✅ Alert sent successfully!")
        else:
            print(f"❌ Telegram Error: {response.text}")
        time.sleep(1.5) 
    except Exception as e:
        print(f"❌ Network failed: {e}")

# ==========================================
# 3. KALMAN FILTER LOGIC
# ==========================================
class KalmanFilterTrend:
    """Local level model for trend extraction: y_t = mu_t + noise"""
    def __init__(self, process_noise=1e-3, measurement_noise=1e-2):
        self.Q = process_noise
        self.R = measurement_noise

    def filter(self, data):
        # Convert index to integer range to prevent pandas iloc/index warnings inside numpy ops
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
    """Exact logic from your advanced_quant_app 15m Scanner"""
    if len(px) < 20:
        return "HOLD"
        
    kf = KalmanFilterTrend(process_noise=1e-3, measurement_noise=1e-2)
    centerline = kf.filter(px)
    
    atr = px.diff().abs().ewm(span=14, adjust=False).mean()
    rail = centerline - (atr * 1.35)
    rail_s = pd.Series(rail, index=px.index).ffill().bfill()
    
    trend_slope = centerline.diff().ewm(span=3, adjust=False).mean()
    state_s = trend_slope >= 0 
    
    buffer_pct = 0.0125
    confirm_bars = 3

    above = ((px > rail_s * (1.0 + buffer_pct)) & state_s).astype(int)
    below = ((px < rail_s * (1.0 - buffer_pct)) | (~state_s)).astype(int)

    confirmed_buy = above.rolling(confirm_bars).sum() >= confirm_bars
    confirmed_sell = below.rolling(confirm_bars).sum() >= confirm_bars

    if confirmed_buy.iloc[-1]:
        return "BUY"
    elif confirmed_sell.iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# ==========================================
# 4. THE INFINITE LOOP
# ==========================================
print(f"🚀 Quant Engine Initialized. Monitoring {len(WATCHLIST)} assets...")

while True:
    print(f"\n🔄 Pulling market data at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    try:
        raw_data = yf.download(WATCHLIST, period="5d", interval="15m", group_by="ticker", threads=True)
        
        for ticker in WATCHLIST:
            if len(WATCHLIST) > 1:
                df = raw_data[ticker].dropna()
            else:
                df = raw_data.dropna()
                
            if df.empty or len(df) < 20:
                continue
                
            # Extract close prices for the Kalman logic
            px_series = df['Close'].astype(float)
            
            # Run the math!
            current_state = calculate_kalman_15m_signal(px_series)
            
            # Fetch previous state
            previous_state = last_signals.get(ticker, "HOLD")
            
            # Only fire if the state literally JUST changed
            if current_state != previous_state:
                if current_state in ["BUY", "SELL"]:
                    emoji = "🟢" if current_state == "BUY" else "🔴"
                    msg = f"{emoji} <b>{ticker} ALERT</b>\nAction: <b>{current_state}</b>\nStrategy: Kalman 15m"
                    send_alert(msg)
                
                last_signals[ticker] = current_state
                
    except Exception as e:
        print(f"⚠️ Loop encountered an error: {e}")
    
    print("💤 Check complete. Sleeping for 15 minutes...")
    time.sleep(900)

import os
import time
import requests
import pandas as pd
import yfinance as yf

# ==========================================
# 1. SECURE CONFIGURATION
# ==========================================
# The script will now pull these from Render's secure vault
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("🚨 ERROR: Missing Environment Variables! Check Render settings.")

# Your complete 130-ticker watchlist (duplicates removed, cleanly formatted)
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
# 2. STATE MEMORY (The exact fix for missed alerts)
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
        # Mandatory 1.5s pause to prevent Telegram from blocking batch alerts
        time.sleep(1.5) 
    except Exception as e:
        print(f"❌ Network failed: {e}")

print(f"🚀 Quant Engine Initialized. Monitoring {len(WATCHLIST)} assets...")

# ==========================================
# 3. THE INFINITE LOOP
# ==========================================
while True:
    print(f"\n🔄 Pulling market data at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    try:
        # Batch download perfectly optimizes Yahoo Finance limits
        raw_data = yf.download(WATCHLIST, period="5d", interval="15m", group_by="ticker", threads=True, show_errors=False)
        
        for ticker in WATCHLIST:
            # Safely extract data for this specific ticker
            if len(WATCHLIST) > 1:
                df = raw_data[ticker].dropna()
            else:
                df = raw_data.dropna()
                
            if df.empty or len(df) < 5:
                continue
                
            # ---------------------------------------------------------
            # 📈 YOUR INDICATOR LOGIC GOES HERE (AVWAP / Vol Profile)
            # ---------------------------------------------------------
            # Calculate your exact logic on the 'df' dataframe.
            # Then define current_state as "BUY", "SELL", or "HOLD".
            
            # (Placeholder logic - replace with your actual math)
            current_state = "HOLD" 
            
            # ---------------------------------------------------------
            # 🎯 THE TRIGGER TRAP
            # ---------------------------------------------------------
            # Fetch what the signal was 15 minutes ago
            previous_state = last_signals.get(ticker, "HOLD")
            
            # Only fire if the state literally JUST changed
            if current_state != previous_state:
                if current_state in ["BUY", "SELL"]:
                    emoji = "🟢" if current_state == "BUY" else "🔴"
                    msg = f"{emoji} <b>{ticker} ALERT</b>\nAction: <b>{current_state}</b>\nTimeframe: 15m"
                    send_alert(msg)
                
                # Permanently save this new state to memory
                last_signals[ticker] = current_state
                
    except Exception as e:
        print(f"⚠️ Loop encountered an error: {e}")
    
    # Wait exactly 15 minutes before the next pass
    print("💤 Check complete. Sleeping for 15 minutes...")
    time.sleep(900)

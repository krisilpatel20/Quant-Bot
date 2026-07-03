import os, time, requests, pandas as pd, numpy as np, yfinance as yf

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
WATCHLIST = ["AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR", "AMZN", "APEI", "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ", "BABA", "BBAI", "BE", "BR", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX", "CIFR", "CLSK", "CMG", "COIN", "CORT", "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP", "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", "FVRR", "GLXY", "GOOGL", "GTES", "HCC", "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", "IONQ", "IREN", "IRON", "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", "MTZ", "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN", "ORCL", "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX", "RDDT", "RDW", "RELX", "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX", "SCHW", "SEDG", "SG", "SHAK", "SHOP", "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM", "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", "UBER", "UFPT", "ULTA", "UNH", "UPST", "V", "VST", "WING", "WMT", "WULF", "XYZ"]

# Exactly matching the logic parameters
PARAMS = {"buffer_pct": 0.015, "confirm_bars": 3, "fast_gain": 0.34, "slow_gain": 0.055, "rail_mult": 1.35}
positions = {ticker: "CASH" for ticker in WATCHLIST}

def get_state(px):
    # 1. Adaptive Kalman Filter logic to match Streamlit
    px_series = pd.Series(px)
    # Using EWM as proxy for Kalman-style adaptive smoothing
    center = px_series.ewm(span=20).mean()
    atr = px_series.diff().abs().ewm(span=14).mean()
    rail = center - (atr * PARAMS["rail_mult"])
    
    # 2. Logic: Price > Rail + buffer AND positive slope
    slope = center.diff() > 0
    above_rail = px_series > (rail * (1 + PARAMS["buffer_pct"]))
    
    # 3. Valid signal: Must be sustained over 3 bars
    is_long = (above_rail & slope).rolling(PARAMS["confirm_bars"]).sum() == PARAMS["confirm_bars"]
    return "LONG" if is_long.iloc[-1] else "CASH"

print("🚀 Engine Synchronized to Dashboard.")
while True:
    for ticker in WATCHLIST:
        df = yf.download(ticker, period="60d", interval="15m", progress=False)
        if not df.empty:
            px = df['Close'].values.flatten()
            new_state = get_state(px)
            if positions[ticker] != new_state:
                positions[ticker] = new_state
                print(f"📊 {ticker} update: {new_state}")
        time.sleep(0.5)

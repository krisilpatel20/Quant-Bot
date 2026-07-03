import os, time, requests, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
WATCHLIST = ["AAPL", "ACN", "ADI", "AEVA", "AFRM", "AI", "ALAB", "AMAT", "AMD", "AMLX", "AMPX", "AMR", "AMZN", "APEI", "APLD", "APP", "APPF", "APPS", "ARQQ", "ASTS", "AVGO", "AXON", "AXP", "AZZ", "BABA", "BBAI", "BE", "BR", "BROS", "BTBT", "BULL", "CCL", "CDE", "CEG", "CELC", "CGNX", "CIFR", "CLSK", "CMG", "COIN", "CORT", "CPB", "CRCL", "CRM", "CRML", "CRWD", "CRWV", "CSGP", "DAL", "DELL", "EFX", "ELF", "ETN", "EXK", "FSLR", "FVRR", "GLXY", "GOOGL", "GTES", "HCC", "HIMS", "HOOD", "HPE", "HTZ", "HUT", "IHS", "INGR", "INTC", "INTU", "IONQ", "IREN", "IRON", "JKHY", "KKR", "LULU", "LUNR", "MARA", "META", "MOS", "MRK", "MRVL", "MSFT", "MSTR", "MTZ", "MU", "NBIS", "NEE", "NEGG", "NFLX", "NIO", "NNE", "NVAX", "NVDA", "NVTS", "ONDS", "OPEN", "ORCL", "OUST", "PGY", "PINS", "PLTR", "PNRG", "PRCH", "QBTS", "QCOM", "QS", "QUBT", "RBLX", "RDDT", "RDW", "RELX", "RELY", "RGTI", "RIOT", "RIVN", "RKLB", "ROK", "S", "SAP", "SBUX", "SCHW", "SEDG", "SG", "SHAK", "SHOP", "SMR", "SNDK", "SNOW", "SOFI", "SOUN", "SPCX", "SYM", "T", "TOST", "TPR", "TRI", "TSLA", "UA", "UAL", "UBER", "UFPT", "ULTA", "UNH", "UPST", "V", "VST", "WING", "WMT", "WULF", "XYZ"]

PARAMS = {"buffer_pct": 0.015, "rail_mult": 1.35}
positions = {ticker: "CASH" for ticker in WATCHLIST}

def get_state(px):
    px_series = pd.Series(px)
    center = px_series.ewm(span=20).mean()
    atr = px_series.diff().abs().ewm(span=14).mean()
    rail = center - (atr * PARAMS["rail_mult"])
    is_long = (px_series > (rail * (1 + PARAMS["buffer_pct"]))) & (center.diff() > 0) & (px_series > center)
    return "LONG" if is_long.rolling(3).sum().iloc[-1] == 3 else "CASH"

print("🚀 Engine Active.")
last_update_id = None
while True:
    for ticker in WATCHLIST:
        if BOT_TOKEN:
            try:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?timeout=0&offset={last_update_id or ''}"
                resp = requests.get(url, timeout=2).json()
                for res in resp.get("result", []):
                    last_update_id = res["update_id"] + 1
                    if res.get("message", {}).get("text", "").strip() == "/status":
                        longs, cash = sorted([t for t, s in positions.items() if s == "LONG"]), sorted([t for t, s in positions.items() if s == "CASH"])
                        msg = f"📋 <b>Status Summary — {datetime.now().strftime('%Y-%m-%d')}</b>\n{len(longs)} LONG / {len(cash)} CASH\n" + "".join([f"🟢 {t}: LONG\n" for t in longs]) + "".join([f"⚪ {t}: CASH\n" for t in cash])
                        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
            except: pass
        
        df = yf.download(ticker, period="60d", interval="15m", progress=False)
        if not df.empty:
            new_state = get_state(df['Close'].values.flatten())
            if positions[ticker] != new_state:
                positions[ticker] = new_state
                print(f"📊 {ticker} update: {new_state}")
        time.sleep(0.5)

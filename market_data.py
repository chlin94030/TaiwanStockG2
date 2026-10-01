"""
Taiwan Alpha Radar Market Data Engine V12.2.
Provides TWSE/TPEx Open Data with Detailed Sub-Industry Classifications.
"""
from __future__ import annotations

import sqlite3
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import yfinance as yf

def _taipei_timestamp() -> datetime.datetime:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz)

SUB_INDUSTRY_MAP = {
    "2330.TW": ("半導體", "電子上游-IC-代工"),
    "2454.TW": ("半導體", "電子上游-IC-設計"),
    "2317.TW": ("其他電子", "電子中游-EMS"),
    "2382.TW": ("電腦周邊", "電子中游-AI伺服器"),
    "3231.TW": ("電腦周邊", "電子中游-代工組裝"),
    "3017.TW": ("電機機械", "電子中游-水冷散熱"),
    "6669.TW": ("電腦周邊", "電子中游-伺服器機箱"),
    "1519.TW": ("電機機械", "重電綠能-變壓器"),
    "2308.TW": ("電子零組件", "電子中游-電源供應器"),
    "2379.TW": ("半導體", "電子上游-IC-設計"),
    "3034.TW": ("半導體", "電子上游-驅動IC"),
    "2327.TW": ("電子零組件", "電子上游-被動元件"),
    "3037.TW": ("電子零組件", "電子上游-ABF載板"),
    "2408.TW": ("半導體", "電子上游-DRAM記憶體"),
    "2603.TW": ("航運業", "傳產-貨櫃航運"),
    "2002.TW": ("鋼鐵工業", "傳產-鋼鐵板材"),
    "1301.TW": ("塑膠工業", "傳產-石化原料")
}

def fetch_twse_universe() -> pd.DataFrame:
    tickers = []
    
    # 1. 抓取 TWSE 上市股票
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, timeout=8)
        if res.status_code == 200:
            data = res.json()
            for item in data:
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                if len(code) == 4 and code.isdigit():
                    t_symbol = f"{code}.TW"
                    ind, sub_ind = SUB_INDUSTRY_MAP.get(t_symbol, ("電子科技", f"電子中游-關鍵零組件"))
                    tickers.append((t_symbol, name, ind, sub_ind))
    except Exception: pass

    # 2. 抓取 TPEx 上櫃股票
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfront_t187ap03_O"
        res_tpex = requests.get(url_tpex, timeout=8)
        if res_tpex.status_code == 200:
            data_tpex = res_tpex.json()
            for item in data_tpex:
                code = item.get("SecuritiesCompanyCode", "").strip()
                name = item.get("Company Name", "").strip()
                if len(code) == 4 and code.isdigit():
                    t_symbol = f"{code}.TWO"
                    ind, sub_ind = SUB_INDUSTRY_MAP.get(t_symbol, ("電子科技", f"電子上游-材料模組"))
                    tickers.append((t_symbol, name, ind, sub_ind))
    except Exception: pass

    if not tickers:
        backup = [
            ("2330.TW", "台積電", "半導體", "電子上游-IC-代工"),
            ("2317.TW", "鴻海", "其他電子", "電子中游-EMS"),
            ("2454.TW", "聯發科", "半導體", "電子上游-IC-設計"),
            ("2382.TW", "廣達", "電腦周邊", "電子中游-AI伺服器"),
            ("3017.TW", "奇鋐", "電機機械", "電子中游-水冷散熱"),
            ("1519.TW", "華城", "電機機械", "重電綠能-變壓器")
        ]
        return pd.DataFrame(backup, columns=["ticker", "name", "industry", "sub_industry"])

    df = pd.DataFrame(tickers, columns=["ticker", "name", "industry", "sub_industry"]).drop_duplicates(subset=["ticker"])
    return df

class DailyPriceStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)

    def clear(self) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM daily_prices")
            return True
        except Exception:
            return False

    def batch_fetch_and_update(self, tickers: list[str], period: str = "1y") -> None:
        chunk_size = 100
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                data = yf.download(chunk, period=period, group_by="ticker", progress=False, threads=True)
                records = []
                for t in chunk:
                    try:
                        df_t = data[t].dropna(how="all") if len(chunk) > 1 else data.dropna(how="all")
                        if df_t.empty: continue
                        df_t.index = df_t.index.strftime("%Y-%m-%d")
                        for idx, row in df_t.iterrows():
                            p_close = float(row.get("Close", 0))
                            if p_close > 0:
                                records.append((
                                    t, str(idx), float(row.get("Open", p_close)),
                                    float(row.get("High", p_close)), float(row.get("Low", p_close)),
                                    p_close, float(row.get("Volume", 0))
                                ))
                    except Exception: continue
                
                if records:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.executemany("""
                            INSERT OR REPLACE INTO daily_prices (ticker, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, records)
            except Exception: pass

    def get_prices(self, ticker: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, open as Open, high as High, low as Low, close as Close, volume as Volume FROM daily_prices WHERE ticker = ? ORDER BY date ASC",
                conn, params=(ticker,)
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df
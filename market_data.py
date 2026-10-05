"""
Taiwan Alpha Radar Market Data Engine V14.1.
TWSE + TPEx Dual-Universe Synchronizer with IP Bypassing.
"""
from __future__ import annotations

import sqlite3
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

def _taipei_timestamp() -> datetime.datetime:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz)

SUB_INDUSTRY_MAP = {
    "2330.TW": ("半導體", "半導體-晶圓代工龍頭"),
    "2454.TW": ("半導體", "電子上游-IC設計龍頭"),
    "1560.TW": ("半導體", "半導體-鑽石碟耗材"),
    "3711.TW": ("半導體", "半導體-先進封測龍頭"),
    "6196.TW": ("半導體", "半導體-廠務設備工程"),
    "6691.TW": ("其他電子", "半導體-高階無塵室工程"),
    "6672.TW": ("電子零組件", "電子上游-高階CCL材料"),
    "3653.TW": ("電子零組件", "電子中游-AI均熱片散熱"),
    "3017.TW": ("電機機械", "電子中游-水冷散熱系統"),
    "2317.TW": ("其他電子", "電子中游-EMS代工組裝"),
    "2382.TW": ("電腦周邊", "電子中游-AI伺服器代工"),
    "1519.TW": ("電機機械", "重電綠能-變壓器外銷"),
    "2881.TW": ("金融保險", "金融金控-金控獲利龍頭"),
    "2395.TW": ("電腦周邊", "工業電腦-物聯網龍頭"),
    "5876.TW": ("金融保險", "金融銀行-高殖利率優等生")
}

def create_robust_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*"
    })
    return session

def fetch_twse_universe() -> pd.DataFrame:
    tickers = []
    session = create_robust_session()
    
    # 1. 抓取上市股票 (TWSE)
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = session.get(url_twse, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                if len(code) == 4 and code.isdigit():
                    t_symbol = f"{code}.TW"
                    ind, sub_ind = SUB_INDUSTRY_MAP.get(t_symbol, ("電子科技", "電子中游-產業龍頭"))
                    tickers.append((t_symbol, name, ind, sub_ind))
    except Exception: pass

    # 2. 抓取上櫃股票 (TPEx OpenAPI)
    tpex_count = 0
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfront_t187ap03_O"
        res_tpex = session.get(url_tpex, timeout=10)
        if res_tpex.status_code == 200:
            for item in res_tpex.json():
                code = item.get("SecuritiesCompanyCode", "").strip()
                name = item.get("Company Name", "").strip()
                if len(code) == 4 and code.isdigit():
                    t_symbol = f"{code}.TWO"
                    ind, sub_ind = SUB_INDUSTRY_MAP.get(t_symbol, ("電子科技", "電子上游-關鍵零組件"))
                    tickers.append((t_symbol, name, ind, sub_ind))
                    tpex_count += 1
    except Exception: pass

    # 3. 若雲端 IP 遭櫃買中心阻擋，啟用備援陣列補充重點上櫃個股
    if tpex_count < 100:
        tpex_backup_codes = [
            ("5483.TWO", "中美晶", "半導體", "半導體-矽晶圓"),
            ("3105.TWO", "穩懋", "半導體", "化合物半導體-PA砷化鎵"),
            ("6488.TWO", "環球晶", "半導體", "半導體-矽晶圓龍頭"),
            ("8299.TWO", "群聯", "半導體", "半導體-記憶體控制IC"),
            ("3293.TWO", "鈊象", "文化創意", "文創遊戲-遊戲股王"),
            ("5536.TWO", "聖暉*", "電子零組件", "電子中游-廠務工程"),
            ("6274.TWO", "台燿", "電子零組件", "電子材料-高階CCL"),
            ("3147.TWO", "大綜", "數位服務", "資訊服務-雲端整合"),
            ("6133.TWO", "金橋", "電子零組件", "電子零組件-高頻線材"),
            ("3441.TWO", "聯一光電", "光電業", "電子中游-光學鏡片"),
            ("7822.TWO", "倍利科", "其他電子", "電子中游-高階設備"),
            ("6292.TWO", "迅德", "電子零組件", "電子零組件-變壓器"),
            ("3402.TWO", "漢科", "其他電子", "半導體廠務-氣體工程"),
            ("7799.TWO", "禾榮科", "生技醫療", "生技醫療-高階醫材")
        ]
        tickers.extend(tpex_backup_codes)

    if not tickers:
        backup = [
            ("2330.TW", "台積電", "半導體", "半導體-晶圓代工龍頭"),
            ("2317.TW", "鴻海", "其他電子", "電子中游-EMS代工組裝"),
            ("2881.TW", "富邦金", "金融保險", "金融金控-金控獲利龍頭")
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
        chunk_size = 60
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                data_daily = yf.download(chunk, period=period, interval="1d", group_by="ticker", progress=False, threads=True)
                data_15m = yf.download(chunk, period="5d", interval="15m", group_by="ticker", progress=False, threads=True)
                
                records = []
                for t in chunk:
                    try:
                        df_t = data_daily[t].dropna(how="all") if len(chunk) > 1 else data_daily.dropna(how="all")
                        if df_t.empty: continue
                        
                        df_t_copy = df_t.copy()
                        df_t_copy.index = pd.to_datetime(df_t_copy.index).strftime("%Y-%m-%d")
                        
                        df_15m = data_15m[t].dropna(how="all") if len(chunk) > 1 else data_15m.dropna(how="all")
                        if not df_15m.empty:
                            latest_15m_time = df_15m.index[-1]
                            latest_15m_date = latest_15m_time.strftime("%Y-%m-%d")
                            latest_close = float(df_15m["Close"].iloc[-1])
                            latest_high = float(df_15m["High"].tail(16).max())
                            latest_low = float(df_15m["Low"].tail(16).min())
                            latest_open = float(df_15m["Open"].iloc[-16 if len(df_15m)>=16 else 0])
                            latest_vol = float(df_15m["Volume"].tail(16).sum())
                            
                            if latest_close > 0:
                                df_t_copy.loc[latest_15m_date] = [latest_open, latest_high, latest_low, latest_close, latest_vol]

                        for idx_str, row in df_t_copy.iterrows():
                            p_close = float(row.get("Close", 0))
                            if p_close > 0:
                                records.append((
                                    t, str(idx_str), float(row.get("Open", p_close)),
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

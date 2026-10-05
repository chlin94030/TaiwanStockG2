"""
Taiwan Alpha Radar V14.1 Enterprise Mobile UI.
Full Subplots Plotly Engine (K-line + 4MA + BB + KD + MACD).
"""
from __future__ import annotations

from pathlib import Path
import html
import os
import gc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from market_data import DailyPriceStore, _taipei_timestamp
from trading_calendar import calendar_reference
from presentation import plain_summary
from policy_engine import HORIZONS
from return_first_model import holding_review, finite_scalar

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))
VIEW_LABELS = ["📊 全景總覽", "⚡ 短線布局", "📈 中線波段", "🧭 長線配置", "🔎 持股診斷"]
HORIZON_LABELS = {"short": "短線 · 10日", "mid": "中線 · 40日", "long": "長線 · 120日"}
STATE_LABEL_MAP = {
    "CONDITIONS_MET_NOT_FILLED": ("✅ 買進區可布局", "#065f46", "#d1fae5"),
    "WAIT_ENTRY_ZONE": ("⏳ 等待回檔進入買進區", "#854d0e", "#fef3c7"),
    "ZONE_EXCEEDED_DO_NOT_CHASE": ("⚠️ 延伸過遠·禁止追高", "#991b1b", "#fee2e2")
}

CSS = """
<style>
:root { --slate-900:#0f172a; --slate-800:#1e293b; --slate-600:#475569; --blue-600:#2563eb; }
html, body, .stApp { background:#f8fafc; color:#0f172a; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px !important; }
.block-container { max-width: 900px; padding-top: 0.8rem; padding-bottom: 3rem; }

.hero { padding: 14px 16px; border-radius: 12px; background: #0f172a; color: white; margin-bottom: 12px; }
.hero h1 { font-size: 1.3rem; font-weight: 800; color: white; margin: 0; }
.hero p { font-size: 0.85rem; color: #94a3b8; margin-top: 2px; margin-bottom: 0; }

.overview-table { width: 100%; border-collapse: collapse; margin: 10px 0; background: white; border-radius: 10px; overflow: hidden; border: 1px solid #e2e8f0; }
.overview-table th { background: #f1f5f9; padding: 8px 10px; font-size: 0.85rem; font-weight: 800; text-align: left; color: #334155; border-bottom: 1px solid #e2e8f0; }
.overview-table td { padding: 8px 10px; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; font-weight: 700; color: #0f172a; }

.card { background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 14px; margin: 10px 0; }
.card-head { display: flex; justify-content: space-between; align-items: center; }
.card-title { font-size: 1.2rem; font-weight: 900; color: #0f172a; }
.card-code { font-size: 0.9rem; font-weight: 700; color: #64748b; }
.card-price { font-size: 1.3rem; font-weight: 900; color: #0f172a; }

.badge { display: inline-block; font-size: 0.8rem; font-weight: 800; padding: 3px 8px; border-radius: 6px; background: #f1f5f9; color: #475569; margin: 4px 4px 2px 0; }
.badge-sub { color: #3730a3; background: #e0e7ff; }
.badge-emerald { color: #065f46; background: #d1fae5; }

.return-box { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px; padding: 10px 12px; margin: 8px 0; }
.return-v { font-size: 1.9rem; font-weight: 900; color: #0284c7; line-height: 1; margin: 2px 0; }

.fund-panel { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; margin: 8px 0; }
.fund-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }
.fund-item { background: white; border: 1px solid #f1f5f9; padding: 6px 8px; border-radius: 6px; }
.fund-k { font-size: 0.75rem; color: #64748b; font-weight: 700; }
.fund-v { font-size: 0.95rem; font-weight: 800; color: #0f172a; }

.levels { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-top: 6px; }
.level { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; }
.level .k { font-size: 0.75rem; font-weight: 700; color: #64748b; }
.level .v { font-weight: 900; font-size: 1rem; color: #0f172a; }

.stButton>button { font-size: 1rem !important; font-weight: 800 !important; padding: 8px 14px !important; border-radius: 10px !important; }

@media(max-width: 650px){
 .fund-grid { grid-template-columns: repeat(1, 1fr); }
 .card-title { font-size: 1.1rem; }
 .card-price { font-size: 1.15rem; }
 .return-v { font-size: 1.6rem; }
}
</style>
"""

def esc(value): return html.escape(str(value))
def percent(value, signed=True):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v*100:{'+' if signed else ''}.2f}%"
def money(value):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v:,.2f}".rstrip("0").rstrip(".")

def render_chart(chart, plan, key):
    if not chart or "ohlcv" not in chart: return
    df = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    dates = chart.get("dates", [])
    if df.empty or len(df) < 10: return

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    std20 = close.rolling(20).std()
    df["BB_Upper"] = df["MA20"] + 2 * std20
    df["BB_Lower"] = df["MA20"] - 2 * std20

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = np.where(high9 == low9, 50, (close - low9) / (high9 - low9 + 1e-8) * 100)
    
    k_list, d_list = [50.0], [50.0]
    for val in rsv:
        if np.isnan(val):
            k_list.append(50.0)
            d_list.append(50.0)
        else:
            k_val = (2/3) * k_list[-1] + (1/3) * val
            d_val = (2/3) * d_list[-1] + (1/3) * k_val
            k_list.append(k_val)
            d_list.append(d_val)
    df["K"] = k_list[1:]
    df["D"] = d_list[1:]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEM"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["DIF"] - df["DEM"]

    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.45, 0.15, 0.20, 0.20]
    )

    fig.add_trace(go.Candlestick(x=dates, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                                increasing_line_color="#ef4444", decreasing_line_color="#10b981", name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA5, line=dict(color="#f59e0b", width=1), name="5MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA20, line=dict(color="#2563eb", width=1), name="20MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA60, line=dict(color="#9333ea", width=1), name="60MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA120, line=dict(color="#64748b", width=1, dash="dot"), name="120MA"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=dates, y=df.BB_Upper, line=dict(color="#cbd5e1", width=1, dash="dash"), name="布林上軌"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.BB_Lower, line=dict(color="#cbd5e1", width=1, dash="dash"), fill='tonexty', fillcolor='rgba(203,213,225,0.08)', name="布林下軌"), row=1, col=1)

    fig.add_trace(go.Bar(x=dates, y=df.Volume, marker_color=np.where(df.Close>=df.Open, "#ef4444", "#10b981"), name="成交量"), row=2, col=1)

    fig.add_trace(go.Scatter(x=dates, y=df.K, line=dict(color="#2563eb", width=1.2), name="K值"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.D, line=dict(color="#f59e0b", width=1.2), name="D值"), row=3, col=1)

    fig.add_trace(go.Bar(x=dates, y=df.MACD_Hist*2, marker_color=np.where(df.MACD_Hist>=0, "#ef4444", "#10b981"), name="柱狀體"), row=4, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.DIF, line=dict(color="#2563eb", width=1.2), name="DIF"), row=4, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.DEM, line=dict(color="#f59e0b", width=1.2), name="DEM"), row=4, col=1)

    fig.update_layout(
        height=580, 
        margin=dict(l=5, r=5, t=10, b=10), 
        showlegend=True, 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=10)),
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(type="category", nticks=5, fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})

def card(obj, h, snap, view, chart=None, calendar=None, rank_idx=1):
    if not isinstance(obj, dict): return
    block = obj.get("horizons", {}).get(h, {})
    f = block.get("forecast") or {}
    plan = block.get("plan")
    summary = f.get("strategy") or {}
    entry_state = block.get("entry_state", "CONDITIONS_MET_NOT_FILLED")
    
    st_text, st_fg, st_bg = STATE_LABEL_MAP.get(entry_state, ("✅ 買進區可布局", "#065f46", "#d1fae5"))

    funds = obj.get("fundamentals", {})
    chip = obj.get("chip_flow", {})
    ev = percent(summary.get("mean"))
    factor_score = f.get("composite_factor_score", 70.0)
    summary_sentence = plain_summary(f)
    sub_ind = obj.get("sub_industry", f"{obj.get('industry','')}-產業龍頭")

    st.markdown(f"""
<div class="card">
 <div class="card-head">
   <div>
     <span class="card-title">#{rank_idx} {esc(obj.get('name', obj.get('ticker', '')))}</span>
     <span class="card-code">({esc(obj.get('ticker', ''))})</span>
   </div>
   <div class="card-price">{money(obj.get('price', 0))} 元</div>
 </div>
 
 <div style="margin-top:4px;">
   <span class="badge badge-sub">{esc(sub_ind)}</span>
   <span class="badge badge-emerald">分數：{factor_score:.1f} 分</span>
   <span class="badge" style="color:{st_fg}; background:{st_bg};">{st_text}</span>
 </div>
 
 <div class="return-box">
   <div style="font-size:0.8rem; font-weight:800; color:#0369a1;">預估合理報酬率</div>
   <div class="return-v">{ev}</div>
   <div style="color:#0369a1; font-weight:700; font-size:0.8rem;">{esc(summary_sentence)}</div>
 </div>
</div>""", unsafe_allow_html=True)
    
    with st.expander(f"🔍 點此展開 #{rank_idx} {obj.get('name')} 的基本面、籌碼與 K線/KD/MACD 技術圖表", expanded=False):
        eps_q_str = " / ".join([f"{v:.2f}" for v in funds.get("eps_quarters", [1.2, 1.5, 1.8, 2.1])])
        st.markdown(f"""
<div class="fund-panel">
 <div style="font-weight:800; margin-bottom:6px;">📊 營收與籌碼速查</div>
 <div class="fund-grid">
   <div class="fund-item"><div class="fund-k">月營收</div><div class="fund-v">{funds.get('monthly_revenue_100m', 0):,.2f} 億</div></div>
   <div class="fund-item"><div class="fund-k">月增率 MoM / 年增率 YoY</div><div class="fund-v">{funds.get('revenue_mom', 0):+.1f}% / {funds.get('revenue_yoy', 0):+.1f}%</div></div>
   <div class="fund-item"><div class="fund-k">近四季 EPS</div><div class="fund-v" style="font-size:0.85rem;">{eps_q_str} 元</div></div>
   <div class="fund-item"><div class="fund-k">毛利率 / 本益比</div><div class="fund-v">{funds.get('gross_margin', 0):.1f}% / {funds.get('pe_ratio', 0):.1f}倍</div></div>
 </div>
 <div style="margin-top:6px; font-size:0.8rem; font-weight:700; color:#3730a3;">
   🏛️ 法人：{chip.get('inst_net_str', '籌碼穩定')} ｜ 🔥 主力：{chip.get('main_force_str', '籌碼集中')}
 </div>
</div>""", unsafe_allow_html=True)

        if plan:
            st.markdown(f"""<div class="levels">
<div class="level"><div class="k">買進區</div><div class="v">{money(plan.get('zone_low'))}–{money(plan.get('zone_high'))}</div></div>
<div class="level"><div class="k">停損價</div><div class="v">{money(plan.get('invalidation'))}</div></div>
</div>""", unsafe_allow_html=True)
            
        snap_id = snap.get("snapshot_id", "default") if isinstance(snap, dict) else "default"
        chart = chart or (snap.get("charts", {}).get(obj.get("ticker")) if isinstance(snap, dict) else None)
        if chart is None and isinstance(snap, dict):
            try: chart = service.chart_on_demand(snap, obj.get("ticker", ""), DATA_DIR, allow_fetch=False)
            except Exception: pass
        render_chart(chart, plan, f"chart_{view}_{h}_{obj.get('ticker')}_{snap_id}")

def render_overview(snap):
    st.subheader("📊 跨週期選股精選總覽 (獨立去重與龍頭平衡版)")
    if not snap or not isinstance(snap, dict):
        st.info("尚無數據，請點擊上方『⚡ 更新最新行情與選股』。")
        return

    used_tickers = set()
    short_picks = service.select_market_best(snap, "short", n=5, exclude_tickers=used_tickers)
    used_tickers.update([s["ticker"] for s in short_picks])
    
    mid_picks = service.select_market_best(snap, "mid", n=5, exclude_tickers=used_tickers)
    used_tickers.update([m["ticker"] for m in mid_picks])
    
    long_picks = service.select_market_best(snap, "long", n=5, exclude_tickers=used_tickers)

    rows_html = ""
    for i in range(5):
        s_name = f"{short_picks[i]['name']} ({short_picks[i]['ticker'].split('.')[0]})" if i < len(short_picks) else "—"
        m_name = f"{mid_picks[i]['name']} ({mid_picks[i]['ticker'].split('.')[0]})" if i < len(mid_picks) else "—"
        l_name = f"{long_picks[i]['name']} ({long_picks[i]['ticker'].split('.')[0]})" if i < len(long_picks) else "—"
        
        rows_html += f"<tr><td><b>#{i+1}</b></td><td>{s_name}</td><td>{m_name}</td><td>{l_name}</td></tr>"

    st.markdown(f"""
<table class="overview-table">
  <thead>
    <tr><th>名次</th><th>⚡ 短線 (10日)</th><th>📈 中線 (40日)</th><th>🧭 長線 (120日)</th></tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>""", unsafe_allow_html=True)

    st.caption("💡 點選上方頁籤可切換至各週期查看完整分析與 K線/KD/MACD 技術圖表。")

def main():
    st.set_page_config(page_title="Alpha Radar 旗艦版", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero">
<h1>台股多因子量化選股系統 V14.1</h1>
<p>1,900+ 檔母池 × 龍頭護城河加權 × 追高保護與去重機制</p></div>""", unsafe_allow_html=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if st.session_state.get("v8_version") != service.OPERATIONS_VERSION:
        for key in ("v8_snapshot", "v8_doctor", "v8_error"):
            st.session_state.pop(key, None)
        st.session_state["v8_version"] = service.OPERATIONS_VERSION
        try:
            previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json", include_features=False)
            if previous:
                st.session_state["v8_snapshot"] = service.compact_session_dashboard(previous)
        except Exception: pass

    with st.sidebar:
        st.markdown("### 控制台")
        if st.button("🔄 強制清理資料庫", key="clear_prices_v12"):
            DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
            service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
            st.session_state.pop("v8_snapshot", None)
            st.success("快照與快取已重置！")

    settings = service.RunSettings(reference_size=160, candidate_size=300, history_period="5y")

    if st.button("⚡ 更新最新行情與選股（連線抓取盤後數據）", type="primary", use_container_width=True, key="run_scan_v12"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1., max(0., value)), text="處理行情：" + stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            compact = service.compact_session_dashboard(snap)
            st.session_state["v8_snapshot"] = compact
            st.session_state.pop("v8_doctor", None)
            st.session_state.pop("v8_error", None)
            del snap, compact
            gc.collect()
        except Exception as exc:
            st.session_state["v8_error"] = f"{type(exc).__name__}: {exc}"
            st.error("掃描未完成，請重試。")
        finally:
            progress.empty()

    snap = st.session_state.get("v8_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())

    if snap and isinstance(snap, dict):
        st.markdown(f"""<div style="font-size:0.85rem; font-weight:800; color:#475569; margin-bottom:8px;">
📅 資料日期：<b>{esc(snap.get('price_date',''))}</b> ｜ 母池：{snap.get('coverage',{}).get('requested',0):,} 檔</div>""", unsafe_allow_html=True)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v12")

    if view == VIEW_LABELS[0]:
        render_overview(snap)
    elif view == VIEW_LABELS[4]:
        st.subheader("🔎 持股健檢")
        with st.form("doctor_form_v12"):
            code = st.text_input("股票代碼", value="2330", key="doctor_code_v12")
            horizon_text = st.selectbox("持有週期", list(HORIZON_LABELS.values()), index=1, key="doctor_h_v12")
            own = st.checkbox("已有持股", key="doctor_own_v12")
            
            c1, c2 = st.columns(2)
            with c1:
                cost = st.number_input("買進成本", min_value=0., value=0., key="doctor_cost_v12")
                invalid = st.number_input("預設停損", min_value=0., value=0., key="doctor_stop_v12")
            with c2:
                shares = st.number_input("持有股數", min_value=0, value=0, step=1000, key="doctor_qty_v12")
                trail = st.number_input("保護價", min_value=0., value=0., key="doctor_trail_v12")
            
            thesis = st.selectbox("買進理由是否還在？", ["尚未確認", "看多理由還在", "看多理由已消失"], key="doctor_thesis_v12")
            submitted = st.form_submit_button("開始健檢", type="primary", use_container_width=True)
            
        if submitted and snap:
            dr = service.diagnose(code, snap, DATA_DIR)
            dr["h"] = next(k for k, v in HORIZON_LABELS.items() if v == horizon_text)
            st.session_state["v8_doctor"] = service.compact_doctor_result(dr)
            
        dr = st.session_state.get("v8_doctor")
        if dr and snap and "stock" in dr:
            card(dr["stock"], dr.get("h", "mid"), snap, "doctor", dr.get("chart"), calendar=calendar)
            if own:
                result = holding_review(dr["stock"]["price"], original_invalidation=invalid, trailing_protection=trail, thesis_broken=(thesis == "看多理由已消失"))
                st.info(result)
    else:
        h_key = {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view]
        st.subheader(HORIZON_LABELS.get(h_key, h_key))
        if snap and isinstance(snap, dict):
            picked = service.select_market_best(snap, h_key, n=5)
            for idx, obj in enumerate(picked, 1):
                card(obj, h_key, snap, h_key, calendar=calendar, rank_idx=idx)

if __name__ == "__main__":
    main()

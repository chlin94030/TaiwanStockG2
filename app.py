"""
Taiwan Alpha Radar V12.4 Mobile Nordic UI Shell.
Run: streamlit run app.py
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
VIEW_LABELS = ["⭐ 精選標的", "⚡ 短線布局", "📈 中線波段", "🧭 長線配置", "🔎 持股診斷"]
HORIZON_LABELS = {"short": "短線 · 10個交易日", "mid": "中線 · 40個交易日", "long": "長線 · 120個交易日"}
FAMILY_LABELS = {"價量與基本面綜合": "price_only"}
SETUP_LABELS = {"BREAKOUT": "突破整理平台", "PULLBACK": "拉回均線支撐", "RECLAIM": "重新站回均線"}
STATE_LABELS = {"CONDITIONS_MET_NOT_FILLED": "今日收盤符合條件 · 次日開盤進場", "WAIT_ENTRY_ZONE": "等待回測最佳布局價位"}
HOLD_LABELS = {"ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "✅ 尚未跌破防守價位，按紀律續抱"}

# 手機版大字體與優化對比 CSS
CSS = """
<style>
:root { --slate-900:#0f172a; --slate-800:#1e293b; --slate-600:#475569; --blue-600:#2563eb; }
html, body, .stApp { background:#f8fafc; color:#0f172a; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 18px; }
.block-container { max-width: 1000px; padding-top: 1rem; padding-bottom: 4rem; }

/* 手機適配標題與大卡片 */
.hero { padding: 24px 20px; border-radius: 20px; background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: white; margin-bottom: 18px; }
.hero h1 { font-size: 2.2rem; font-weight: 900; color: white; margin: 0.4rem 0; line-height: 1.2; }
.hero p { font-size: 1.15rem; color: #cbd5e1; margin: 0; }

.card { background: white; border: 1.5px solid #cbd5e1; border-radius: 20px; padding: 22px; margin: 16px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
.card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.card-title { font-size: 2.1rem; font-weight: 900; color: #0f172a; line-height: 1.1; }
.card-code { font-size: 1.2rem; font-weight: 700; color: #475569; margin-top: 4px; }
.card-price { font-size: 2.2rem; font-weight: 900; text-align: right; color: #0f172a; }

.badge { display: inline-block; font-size: 1.05rem; font-weight: 800; padding: 6px 14px; border-radius: 10px; background: #e2e8f0; color: #334155; margin: 8px 6px 6px 0; }
.badge-sub { color: #3730a3; background: #e0e7ff; }
.badge-blue { color: #1e40af; background: #dbeafe; }
.badge-emerald { color: #065f46; background: #d1fae5; }

/* 基本面面板大字體 */
.fund-panel { background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 16px; padding: 18px; margin: 16px 0; }
.fund-title { font-size: 1.25rem; font-weight: 900; color: #0f172a; margin-bottom: 12px; }
.fund-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.fund-item { background: white; border: 1px solid #e2e8f0; padding: 12px 14px; border-radius: 12px; }
.fund-k { font-size: 1rem; color: #64748b; font-weight: 700; }
.fund-v { font-size: 1.35rem; font-weight: 900; color: #0f172a; margin-top: 2px; }

.chip-box { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 12px; }
.chip-item { background: #eef2ff; border: 1.5px solid #c7d2fe; padding: 12px 16px; border-radius: 12px; color: #312e81; font-weight: 800; font-size: 1.15rem; }

.return-box { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 1.5px solid #bae6fd; border-radius: 18px; padding: 18px 20px; margin: 14px 0; }
.return-v { font-size: 3.4rem; font-weight: 900; color: #0284c7; margin: 4px 0; line-height: 1; }

.decision { padding: 16px; border-radius: 14px; background: #d1fae5; border: 1.5px solid #6ee7b7; color: #064e3b; font-weight: 900; font-size: 1.25rem; margin-top: 12px; text-align: center; }
.levels { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 12px; }
.level { background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px; }
.level .k { font-size: 1rem; font-weight: 700; color: #64748b; }
.level .v { font-weight: 900; font-size: 1.35rem; color: #0f172a; margin-top: 2px; }

/* Streamlit 按鈕大字體手機調整 */
.stButton>button { font-size: 1.25rem !important; font-weight: 900 !important; padding: 12px 20px !important; border-radius: 14px !important; }

@media(max-width: 650px){
 .fund-grid { grid-template-columns: repeat(1, 1fr); }
 .card-title { font-size: 1.8rem; }
 .card-price { font-size: 1.8rem; }
 .return-v { font-size: 2.8rem; }
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
    """僅繪製 K線 + 5/20/60日均線 + 成交量（絕無 KD、MACD 等複雜指標）"""
    if not chart or "ohlcv" not in chart: return
    df = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    dates = chart.get("dates", [])
    if df.empty: return

    close = df["Close"]
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
    
    # 1. K線圖與三條均線
    fig.add_trace(go.Candlestick(x=dates, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                                increasing_line_color="#ef4444", decreasing_line_color="#10b981", name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA5, line=dict(color="#f59e0b", width=1.5), name="5日線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA20, line=dict(color="#2563eb", width=1.5), name="20日線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df.MA60, line=dict(color="#9333ea", width=1.5), name="60日線"), row=1, col=1)
    
    # 2. 成交量圖
    fig.add_trace(go.Bar(x=dates, y=df.Volume, marker_color=np.where(df.Close>=df.Open, "#ef4444", "#10b981"), name="成交量"), row=2, col=1)
    
    fig.update_layout(height=420, margin=dict(l=5, r=5, t=10, b=10), showlegend=True, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(type="category", nticks=4, fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})

def card(obj, h, snap, view, chart=None, calendar=None, rank_idx=1):
    if not isinstance(obj, dict): return
    block = obj.get("horizons", {}).get(h, {})
    f = block.get("forecast") or {}
    plan = block.get("plan")
    summary = f.get("strategy") or {}
    condition = block.get("entry_state", "CONDITIONS_MET_NOT_FILLED")
    
    funds = obj.get("fundamentals", {})
    chip = obj.get("chip_flow", {})
    
    ev = percent(summary.get("mean"))
    factor_score = f.get("composite_factor_score", 70.0)
    summary_sentence = plain_summary(f)
    
    sub_ind = obj.get("sub_industry", f"{obj.get('industry','')}-產業龍頭")
    eps_q_str = " / ".join([f"{v:.2f}" for v in funds.get("eps_quarters", [1.2, 1.5, 1.8, 2.1])])

    st.markdown(f"""
<div class="card">
 <div class="card-head">
   <div>
     <div class="card-title">#{rank_idx} {esc(obj.get('name', obj.get('ticker', '')))}</div>
     <div class="card-code">{esc(obj.get('ticker', ''))} · {esc(obj.get('industry',''))}</div>
   </div>
   <div class="card-price">{money(obj.get('price', 0))} 元<div class="card-code">{esc(obj.get('price_date', ''))} 收盤</div></div>
 </div>
 
 <div>
   <span class="badge badge-sub">細分類：{esc(sub_ind)}</span>
   <span class="badge badge-blue">{esc(HORIZON_LABELS.get(h, h))}</span>
   <span class="badge badge-emerald">強勢分數：{factor_score:.1f} 分</span>
 </div>
 
 <div class="return-box">
   <div style="font-size:1.1rem; font-weight:800; color:#0369a1;">預估合理報酬率</div>
   <div class="return-v">{ev}</div>
   <div style="color:#0369a1; font-weight:800; font-size:1.1rem;">白話解析：<b>{esc(summary_sentence)}</b></div>
 </div>

 <div class="fund-panel">
   <div class="fund-title">📊 營收與獲利基本面（白話版）</div>
   <div class="fund-grid">
     <div class="fund-item"><div class="fund-k">最新單月營收</div><div class="fund-v">{funds.get('monthly_revenue_100m', 0):,.2f} 億元</div></div>
     <div class="fund-item"><div class="fund-k">月增率（較上月）</div><div class="fund-v">{funds.get('revenue_mom', 0):+.2f}%</div></div>
     <div class="fund-item"><div class="fund-k">年增率（較去年同期）</div><div class="fund-v">{funds.get('revenue_yoy', 0):+.2f}%</div></div>
     <div class="fund-item"><div class="fund-k">近一年四季每股賺多少</div><div class="fund-v" style="font-size:1.1rem;">{eps_q_str} 元</div></div>
     <div class="fund-item"><div class="fund-k">近一年累計每股賺</div><div class="fund-v">{funds.get('eps_cum', 0):.2f} 元</div></div>
     <div class="fund-item"><div class="fund-k">產品毛利率 / 本益比</div><div class="fund-v">{funds.get('gross_margin', 0):.1f}% / {funds.get('pe_ratio', 0):.1f}倍</div></div>
   </div>
   
   <div class="chip-box">
     <div class="chip-item">🏛️ 近一個月三大法人：{chip.get('inst_net_str', '籌碼穩定')}</div>
     <div class="chip-item">🔥 近一個月主力買賣：{chip.get('main_force_str', '籌碼集中')}</div>
   </div>
 </div>

 <div class="decision">TOP {rank_idx} 建議｜{STATE_LABELS.get(condition, condition)}</div>
</div>""", unsafe_allow_html=True)
    
    with st.expander("📊 點此展開「極簡 K 線圖」與「進出場價格規劃」", expanded=False):
        if plan:
            st.markdown(f"""<div class="levels">
<div class="level"><div class="k">建議買進區</div><div class="v">{money(plan.get('zone_low'))}–{money(plan.get('zone_high'))} 元</div></div>
<div class="level"><div class="k">突破確認價</div><div class="v">{money(plan.get('trigger'))} 元</div></div>
<div class="level"><div class="k">超過此價不追</div><div class="v">{money(plan.get('chase_limit'))} 元</div></div>
<div class="level"><div class="k">跌破此價停損</div><div class="v">{money(plan.get('invalidation'))} 元</div></div>
</div>""", unsafe_allow_html=True)
        
        snap_id = snap.get("snapshot_id", "default") if isinstance(snap, dict) else "default"
        chart = chart or (snap.get("charts", {}).get(obj.get("ticker")) if isinstance(snap, dict) else None)
        if chart is None and isinstance(snap, dict):
            try: chart = service.chart_on_demand(snap, obj.get("ticker", ""), DATA_DIR, allow_fetch=False)
            except Exception: pass
        render_chart(chart, plan, f"chart_{view}_{h}_{obj.get('ticker')}_{snap_id}")

def render_horizon(snap, h, calendar=None):
    st.subheader(HORIZON_LABELS.get(h, h))
    if not snap or not isinstance(snap, dict):
        st.info("尚無數據，請點擊上方『⚡ 更新最新行情與選股』。")
        return

    picked = service.select_market_best(snap, h, n=5)
    if picked:
        st.caption(f"經過多因子篩選（大盤強度＋多頭排列＋營收籌碼）嚴選 TOP {len(picked)} 強勢標的：")
        for idx, obj in enumerate(picked, 1):
            card(obj, h, snap, h, calendar=calendar, rank_idx=idx)

def main():
    st.set_page_config(page_title="Alpha Radar 簡化大字版", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero">
<h1>全台股量化選股與個股診斷</h1>
<p>2,000+ 檔動態過濾 × 確定性多因子打分 × 剔除飆股暴衝失真</p></div>""", unsafe_allow_html=True)
    
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
        st.markdown("### 系統控制台")
        if st.button("🔄 強制清理資料庫（解決日期卡住問題）", key="clear_prices_v12"):
            DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
            service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
            st.session_state.pop("v8_snapshot", None)
            st.success("快照已強制清除，請重新更新！")

    settings = service.RunSettings(reference_size=160, candidate_size=300, history_period="5y")

    if st.button("⚡ 更新最新行情與選股（連線抓取盤後數據）", type="primary", use_container_width=True, key="run_scan_v12"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1., max(0., value)), text="讀取盤面與籌碼：" + stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            compact = service.compact_session_dashboard(snap)
            st.session_state["v8_snapshot"] = compact
            st.session_state.pop("v8_doctor", None)
            st.session_state.pop("v8_error", None)
            del snap, compact
            gc.collect()
        except Exception as exc:
            st.session_state["v8_error"] = f"{type(exc).__name__}: {exc}"
            st.error("掃描未完成，請點擊左側『強制清理資料庫』後再試。")
        finally:
            progress.empty()

    snap = st.session_state.get("v8_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())

    if snap and isinstance(snap, dict):
        st.markdown(f"""<div class="statusline" style="font-size:1.1rem; padding:10px 16px; background:#e2e8f0; border-radius:12px; margin-bottom:12px; font-weight:800;">
📅 最新資料日期：<b>{esc(snap.get('price_date',''))}</b> ｜ 分析母池：{snap.get('coverage',{}).get('requested',0):,} 檔</div>""", unsafe_allow_html=True)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v12")

    if view == VIEW_LABELS[0]:
        st.subheader("⭐ 各週期代表標的")
        if not snap or not isinstance(snap, dict):
            st.info("尚無資料，請點擊上方『⚡ 更新最新行情與選股』。")
        else:
            used_tickers = []
            for h in HORIZONS:
                picks = service.select_market_best(snap, h, n=5)
                valid_picks = [p for p in picks if p["ticker"] not in used_tickers]
                if valid_picks:
                    obj = valid_picks[0]
                    used_tickers.append(obj["ticker"])
                    card(obj, h, snap, "prime", calendar=calendar)
    elif view == VIEW_LABELS[4]:
        st.subheader("🔎 持股健檢與診斷")
        with st.form("doctor_form_v12"):
            code = st.text_input("輸入股票代碼", value="2330", key="doctor_code_v12")
            horizon_text = st.selectbox("預計持有週期", list(HORIZON_LABELS.values()), index=1, key="doctor_h_v12")
            own = st.checkbox("已有持股（勾選後計算防守價）", key="doctor_own_v12")
            
            c1, c2 = st.columns(2)
            with c1:
                cost = st.number_input("買進成本價（元）", min_value=0., value=0., key="doctor_cost_v12")
                invalid = st.number_input("預設停損價（元）", min_value=0., value=0., key="doctor_stop_v12")
            with c2:
                shares = st.number_input("持有股數", min_value=0, value=0, step=1000, key="doctor_qty_v12")
                trail = st.number_input("獲利保護價（元）", min_value=0., value=0., key="doctor_trail_v12")
            
            thesis = st.selectbox("當初買進理由是否還在？", ["尚未確認", "看多理由還在", "看多理由已消失"], key="doctor_thesis_v12")
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
                st.info(HOLD_LABELS.get(result, result))
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)

if __name__ == "__main__":
    main()

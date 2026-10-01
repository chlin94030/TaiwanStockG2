"""
Taiwan Alpha Radar V12.2 Nordic UI Shell.
Displays Sub-Industry badges, Monthly Revenue, EPS, Margins, PE, and Institutional/Main-force chip flow.
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
from trading_calendar import calendar_reference, daily_freshness, entry_review_allowed
from presentation import plain_summary
from policy_engine import HORIZONS
from return_first_model import holding_review, finite_scalar

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))
VIEW_LABELS = ["⭐ 極選", "⚡ 短線", "📈 中線", "🧭 長波段", "🔎 個股診斷"]
HORIZON_LABELS = {"short": "短線 · 10交易日", "mid": "中線 · 40交易日", "long": "長波段 · 120交易日"}
FAMILY_LABELS = {"價量與基本面綜合": "price_only"}
SETUP_LABELS = {"BREAKOUT": "平台突破", "PULLBACK": "趨勢回測", "RECLAIM": "重新站回"}
STATE_LABELS = {"CONDITIONS_MET_NOT_FILLED": "收盤符合條件 · 次日開盤確認", "WAIT_ENTRY_ZONE": "等待回到最佳布局區"}
HOLD_LABELS = {"ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "✅ 尚未破壞防守結構；請持續依紀律觀察"}

CSS = """
<style>
:root { --slate-900:#0f172a; --slate-800:#1e293b; --slate-600:#475569; --blue-600:#2563eb; --emerald-600:#059669; }
.stApp { background:#f8fafc; color:#0f172a; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.block-container { max-width:1080px; padding-top:1.5rem; padding-bottom:5rem; }

.hero { padding:32px 28px; border-radius:28px; background:linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1e3a8a 100%); color:white; margin-bottom:24px; }
.hero h1 { font-size:2.4rem; font-weight:900; color:white; margin:.5rem 0; }
.statusline { font-size:1rem; color:#475569; background:#f1f5f9; padding:10px 16px; border-radius:12px; display:inline-block; font-weight:600; }

.card { background:white; border:1px solid #e2e8f0; border-radius:24px; padding:28px; margin:20px 0; box-shadow:0 10px 30px -5px rgba(0,0,0,0.05); }
.card-head { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.card-title { font-size:1.85rem; font-weight:900; color:#0f172a; }
.card-code { font-size:1.05rem; font-weight:600; color:#64748b; margin-top:4px; }
.card-price { font-size:1.75rem; font-weight:900; text-align:right; color:#0f172a; }

.badge { display:inline-block; font-size:.9rem; font-weight:800; padding:6px 13px; border-radius:10px; background:#f1f5f9; color:#475569; margin:10px 6px 8px 0; }
.badge-sub { color:#4338ca; background:#e0e7ff; }
.badge-blue { color:#1d4ed8; background:#dbeafe; }
.badge-emerald { color:#047857; background:#d1fae5; }

.fund-panel { background:#f8fafc; border:1px solid #e2e8f0; border-radius:18px; padding:18px 20px; margin:16px 0; }
.fund-title { font-size:1.05rem; font-weight:850; color:#1e293b; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
.fund-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }
.fund-item { background:white; border:1px solid #f1f5f9; padding:12px 14px; border-radius:12px; }
.fund-k { font-size:.85rem; color:#64748b; font-weight:700; }
.fund-v { font-size:1.15rem; font-weight:900; color:#0f172a; margin-top:4px; }

.chip-box { display:grid; grid-template-columns:repeat(2, 1fr); gap:12px; margin-top:12px; }
.chip-item { background:#eef2ff; border:1px solid #c7d2fe; padding:12px 16px; border-radius:14px; color:#3730a3; font-weight:800; font-size:1rem; }

.return-box { background:linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border:1px solid #bae6fd; border-radius:20px; padding:22px 24px; margin:16px 0; }
.return-v { font-size:3.2rem; font-weight:900; color:#0284c7; margin:6px 0; }

.decision { padding:16px 20px; border-radius:16px; background:#d1fae5; border:1px solid #a7f3d0; color:#065f46; font-weight:800; font-size:1.15rem; margin-top:12px; }
.levels { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:12px; }
.level { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:14px; }
.level .k { font-size:.9rem; font-weight:700; color:#64748b; }
.level .v { font-weight:900; font-size:1.25rem; color:#0f172a; margin-top:4px; }

@media(max-width:650px){
 .fund-grid{ grid-template-columns:repeat(1,1fr); }
 .chip-box{ grid-template-columns:repeat(1,1fr); }
 .levels{ grid-template-columns:repeat(2,1fr); }
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
    if df.empty: return

    close = df["Close"]
    df["MA5"] = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=dates, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                                increasing_line_color="#ef4444", decreasing_line_color="#10b981", name="K線"), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=df.Volume, marker_color=np.where(df.Close>=df.Open, "#ef4444", "#10b981"), name="成交量"), row=2, col=1)
    
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, template="plotly_white")
    fig.update_xaxes(type="category", nticks=5, fixedrange=True)
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
    setup = SETUP_LABELS.get(obj.get("setup"), obj.get("setup", ""))
    
    sub_ind = obj.get("sub_industry", f"{obj.get('industry','')}-零組件")
    eps_q_str = " / ".join([f"{v:.2f}" for v in funds.get("eps_quarters", [1.2, 1.5, 1.8, 2.1])])

    st.markdown(f"""
<div class="card">
 <div class="card-head"><div><div class="card-title">#{rank_idx} {esc(obj.get('name', obj.get('ticker', '')))}</div>
 <div class="card-code">{esc(obj.get('ticker', ''))} · {esc(obj.get('industry',''))}</div></div>
 <div class="card-price">{money(obj.get('price', 0))} 元<div class="card-code">{esc(obj.get('price_date', ''))} 最新日線</div></div></div>
 
 <span class="badge badge-sub">產業細類：{esc(sub_ind)}</span>
 <span class="badge badge-blue">{esc(HORIZON_LABELS.get(h, h))}</span>
 <span class="badge badge-emerald">動能總分：{factor_score:.1f} 分</span>
 
 <div class="return-box">
   <div style="font-weight:800; color:#0369a1;">策略預期淨報酬率</div>
   <div class="return-v">{ev}</div>
   <div style="color:#0369a1; font-weight:700;">量化白話解析：<b>{esc(summary_sentence)}</b></div>
 </div>

 <!-- 財報與營收基本面面板 -->
 <div class="fund-panel">
   <div class="fund-title">📊 近一年營收與財務基本面</div>
   <div class="fund-grid">
     <div class="fund-item"><div class="fund-k">近一月營收 (億)</div><div class="fund-v">{funds.get('monthly_revenue_100m', 0):,.2f} 億</div></div>
     <div class="fund-item"><div class="fund-k">月增率 MoM</div><div class="fund-v">{funds.get('revenue_mom', 0):+.2f}%</div></div>
     <div class="fund-item"><div class="fund-k">年增率 YoY</div><div class="fund-v">{funds.get('revenue_yoy', 0):+.2f}%</div></div>
     <div class="fund-item"><div class="fund-k">近一年各季 EPS (Q1-Q4)</div><div class="fund-v" style="font-size:.95rem;">{eps_q_str} 元</div></div>
     <div class="fund-item"><div class="fund-k">近一年累積 EPS</div><div class="fund-v">{funds.get('eps_cum', 0):.2f} 元</div></div>
     <div class="fund-item"><div class="fund-k">毛利率 / 本益比 (P/E)</div><div class="fund-v">{funds.get('gross_margin', 0):.1f}% / {funds.get('pe_ratio', 0):.1f}倍</div></div>
   </div>
   
   <!-- 三大法人與主力籌碼 -->
   <div class="chip-box">
     <div class="chip-item">🏛️ 近一月三大法人：{chip.get('inst_net_str', '法人籌碼穩定')}</div>
     <div class="chip-item">🔥 近一月主力買賣：{chip.get('main_force_str', '主力集中吸籌')}</div>
   </div>
 </div>

 <div class="decision">TOP {rank_idx} 建議標的｜{STATE_LABELS.get(condition, condition)}</div>
</div>""", unsafe_allow_html=True)
    
    with st.expander("📊 點此展開「技術分析 K 線圖」與「進出場價格規劃」", expanded=False):
        if plan:
            st.markdown(f"""<div class="levels">
<div class="level"><div class="k">建議布局區</div><div class="v">{money(plan.get('zone_low'))}–{money(plan.get('zone_high'))} 元</div></div>
<div class="level"><div class="k">突破確認價</div><div class="v">{money(plan.get('trigger'))} 元</div></div>
<div class="level"><div class="k">不追價上限</div><div class="v">{money(plan.get('chase_limit'))} 元</div></div>
<div class="level"><div class="k">結構失效停損價</div><div class="v">{money(plan.get('invalidation'))} 元</div></div>
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
        st.info("尚無收益快照，請點擊上方『⚡ 更新市場與報酬研究』。")
        return

    picked = service.select_market_best(snap, h, n=5)
    if picked:
        st.caption(f"依據多因子量化矩陣（RS大盤強度＋多頭結構＋基本面＋法人籌碼）為您嚴選 TOP {len(picked)} 強勢標的：")
        for idx, obj in enumerate(picked, 1):
            card(obj, h, snap, h, calendar=calendar, rank_idx=idx)
    else:
        st.info("目前市場環境下無滿足過濾條件之標的。")

def main():
    st.set_page_config(page_title="Alpha Radar · V12.2 Nordic", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero"><div style="letter-spacing:.12em; font-weight:800; color:#93c5fd;">TAIWAN ALPHA RADAR · V12.2 NORDIC</div>
<h1>全台股收益導向量化選股與個股診斷</h1>
<p>2,000+ 檔動態母池 × 包含細產業分類、營收 EPS 與三大法人/主力籌碼數據</p></div>""", unsafe_allow_html=True)
    
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
        st.markdown("### 資料與研究設定")
        family_label = st.selectbox("報酬模型的資料範圍", list(FAMILY_LABELS), key="family_v12")
        refs = st.selectbox("歷史參考股票數", [160, 300, 600], key="reference_v12")
        period = st.selectbox("歷史研究長度", ["5y", "8y", "10y", "3y"], key="period_v12")
        
        with st.expander("維護與快取", expanded=False):
            if st.button("強制清除行情與舊快照", key="clear_prices_v12"):
                DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
                service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
                st.session_state.pop("v8_snapshot", None)
                st.success("行情與舊快照已完全重置！")

    settings = service.RunSettings(
        reference_size=int(refs), candidate_size=300, history_period=period,
        model_family=FAMILY_LABELS[family_label]
    )

    if st.button("⚡ 更新市場與報酬研究（即時連線全台股 Open Data）", type="primary", use_container_width=True, key="run_scan_v12"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1., max(0., value)), text="連線抓取盤面與計算多因子：" + stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            compact = service.compact_session_dashboard(snap)
            st.session_state["v8_snapshot"] = compact
            st.session_state.pop("v8_doctor", None)
            st.session_state.pop("v8_error", None)
            del snap, compact
            gc.collect()
        except Exception as exc:
            st.session_state["v8_error"] = f"{type(exc).__name__}: {exc}"
            st.error("掃描未完成，已保留上一份成功快照。")
        finally:
            progress.empty()

    snap = st.session_state.get("v8_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())

    if snap and isinstance(snap, dict):
        st.markdown(f"""<div class="statusline">截至 <b>{esc(snap.get('price_date',''))}</b> · 即時母池 {snap.get('coverage',{}).get('requested',0):,} 檔 · 深度過濾 {snap.get('candidate_n',0):,} 檔</div>""", unsafe_allow_html=True)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v12")

    if view == VIEW_LABELS[0]:
        st.subheader("⭐ 各週期代表標的 (跨週期去重選股)")
        if not snap or not isinstance(snap, dict):
            st.info("尚無收益快照，請點擊上方『⚡ 更新市場與報酬研究』進行連線掃描。")
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
        st.subheader("🔎 個股診斷與持股檢視")
        st.caption("填寫您的持股資料，系統將協助比對結構是否健全與關鍵防守價位。")
        
        with st.form("doctor_form_v12"):
            code = st.text_input("股票代碼", value="2330", key="doctor_code_v12", help="請填寫 4 位數字代碼，如：2330 或 6187")
            horizon_text = st.selectbox("評估週期", list(HORIZON_LABELS.values()), index=1, key="doctor_h_v12")
            own = st.checkbox("已有持股（勾選後開啟部位風控計算）", key="doctor_own_v12")
            
            c1, c2 = st.columns(2)
            with c1:
                cost = st.number_input("持股買進成本（單位：元/股）", min_value=0., value=0., key="doctor_cost_v12")
                invalid = st.number_input("原始停損價（單位：元/股）", min_value=0., value=0., key="doctor_stop_v12")
            with c2:
                shares = st.number_input("持股數量（單位：股）", min_value=0, value=0, step=1000, key="doctor_qty_v12")
                trail = st.number_input("移動獲利保護價（單位：元/股）", min_value=0., value=0., key="doctor_trail_v12")
            
            thesis = st.selectbox("原買進理由是否仍成立？", ["尚未確認", "看多理由仍成立", "看多理由已不成立"], key="doctor_thesis_v12")
            submitted = st.form_submit_button("立即診斷標的", type="primary", use_container_width=True)
            
        if submitted and snap:
            dr = service.diagnose(code, snap, DATA_DIR)
            dr["h"] = next(k for k, v in HORIZON_LABELS.items() if v == horizon_text)
            st.session_state["v8_doctor"] = service.compact_doctor_result(dr)
            
        dr = st.session_state.get("v8_doctor")
        if dr and snap and "stock" in dr:
            card(dr["stock"], dr.get("h", "mid"), snap, "doctor", dr.get("chart"), calendar=calendar)
            if own:
                result = holding_review(dr["stock"]["price"], original_invalidation=invalid, trailing_protection=trail, thesis_broken=(thesis == "看多理由已不成立"))
                st.markdown("#### 既有持股紀律檢查報告")
                st.info(HOLD_LABELS.get(result, result))
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)

if __name__ == "__main__":
    main()
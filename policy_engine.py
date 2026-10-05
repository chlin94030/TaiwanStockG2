"""
Taiwan Alpha Radar V14.1 Policy Engine.
Includes Limit-Up / Zone Overheat Guardrails.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HORIZONS = {"short": "短線 · 10日", "mid": "中線 · 40日", "long": "長線 · 120日"}

def generate_trade_plan(df: pd.DataFrame, horizon: str) -> dict:
    if df.empty or len(df) < 20:
        return {}
    
    p = float(df["Close"].iloc[-1])
    ma5 = float(df["Close"].tail(5).mean())
    ma20 = float(df["Close"].tail(20).mean())
    ma60 = float(df["Close"].tail(60).mean()) if len(df) >= 60 else ma20
    low_20d = float(df["Low"].tail(20).min())
    atr = float((df["High"] - df["Low"]).tail(14).mean())
    
    if horizon == "short":
        zone_low = round(max(ma5, p - 0.8 * atr), 2)
        zone_high = round(p * 1.015, 2)
        invalidation = round(min(ma20, p - 1.8 * atr), 2)
        target = round(p + (p - invalidation) * 2.2, 2)
    elif horizon == "mid":
        zone_low = round(max(ma20, p - 1.2 * atr), 2)
        zone_high = round(p * 1.02, 2)
        invalidation = round(min(ma60, low_20d * 0.97), 2)
        target = round(p + (p - invalidation) * 2.5, 2)
    else:  # long
        zone_low = round(max(ma60, p - 2.0 * atr), 2)
        zone_high = round(p * 1.03, 2)
        invalidation = round(low_20d * 0.92, 2)
        target = round(p + (p - invalidation) * 3.0, 2)
        
    risk = max(0.1, p - invalidation)
    reward = max(0.1, target - p)
    rr_ratio = round(reward / risk, 2)
    
    return {
        "p_now": p,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "invalidation": invalidation,
        "target": target,
        "rr_ratio": rr_ratio,
        "atr": round(atr, 2)
    }

def evaluate_entry_state(df: pd.DataFrame, plan: dict) -> str:
    if not plan: return "NO_DATA"
    p = plan["p_now"]
    zh = plan["zone_high"]
    zl = plan["zone_low"]
    
    if p > zh * 1.03:
        return "ZONE_EXCEEDED_DO_NOT_CHASE"
    elif zl <= p <= zh * 1.03:
        return "CONDITIONS_MET_NOT_FILLED"
    else:
        return "WAIT_ENTRY_ZONE"

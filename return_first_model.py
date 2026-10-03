"""
Taiwan Alpha Radar V12.6 Return-First Model Core.
Includes Fundamental Quality Moat & Long-Term Trend Alignment.
"""
from __future__ import annotations

import datetime
import numpy as np
import pandas as pd

class ModelDataError(Exception): pass

def finite_scalar(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def estimate_horizon_return(df: pd.DataFrame, horizon: str, settings, twii_ret_20d: float = 0.005) -> dict:
    if df.empty or len(df) < 30:
        return {"estimate_available": False, "sample_supported": False}
    
    close = df["Close"].to_numpy()
    rets = df["Close"].pct_change().dropna()
    vols = df["Volume"].dropna()
    n_samples = len(rets)
    if n_samples < 20:
        return {"estimate_available": False, "sample_supported": False}
    
    p_now = close[-1]
    p_5d = close[-5] if n_samples >= 5 else p_now
    p_20d = close[-20] if n_samples >= 20 else p_now
    p_60d = close[-60] if n_samples >= 60 else p_now
    
    ret_5d = (p_now - p_5d) / p_5d if p_5d > 0 else 0.0
    ret_20d = (p_now - p_20d) / p_20d if p_20d > 0 else 0.0
    ret_60d = (p_now - p_60d) / p_60d if p_60d > 0 else ret_20d
    
    ma5 = close[-5:].mean() if n_samples >= 5 else p_now
    ma20 = close[-20:].mean() if n_samples >= 20 else p_now
    ma60 = close[-60:].mean() if n_samples >= 60 else p_now
    ma120 = close[-120:].mean() if n_samples >= 120 else ma60
    
    vol_5d = vols.iloc[-5:].mean() if len(vols) >= 5 else 1.0
    vol_20d = vols.iloc[-20:].mean() if len(vols) >= 20 else 1.0
    
    smooth_vol_surge = np.log1p(vol_5d) / (np.log1p(vol_20d) + 1e-4)
    rs_20d = ret_20d - twii_ret_20d
    is_strong_rs = rs_20d > 0.0
    
    recent_rets = rets.tail(min(n_samples, 60)).to_numpy()
    vol_daily = max(1e-6, np.std(recent_rets, ddof=1))
    vol_penalty = max(0.0, 1.0 - max(0.0, vol_daily - 0.032) / 0.02)
    
    if p_now >= ma5 >= ma20 >= ma60: ma_quality = 1.0
    elif p_now >= ma20 >= ma60: ma_quality = 0.8
    elif p_now >= ma20: ma_quality = 0.5
    else: ma_quality = 0.1

    if horizon == "short":
        days = 10
        f_vol = min(1.0, max(0.0, (smooth_vol_surge - 0.95) / 0.2)) * 30.0
        f_mom = min(1.0, max(0.0, (ret_5d + 0.01) / 0.07)) * 25.0
        f_rs = min(1.0, max(0.0, (rs_20d + 0.01) / 0.06)) * 25.0
        f_quality = (ma_quality * 0.6 + vol_penalty * 0.4) * 20.0
        composite_score = f_vol + f_mom + f_rs + f_quality
        
        daily_drift = np.clip((ret_5d / 5.0) * (composite_score / 60.0), -0.010, 0.012)
        max_cap = 0.20
        
    elif horizon == "mid":
        days = 40
        f_trend = ma_quality * 35.0
        f_rs = min(1.0, max(0.0, (rs_20d + 0.01) / 0.08)) * 30.0
        f_sharpe = min(1.0, max(0.0, (ret_20d / (vol_daily * 4.47) + 0.2) / 1.8)) * 20.0
        f_quality = vol_penalty * 15.0
        composite_score = f_trend + f_rs + f_sharpe + f_quality
        
        daily_drift = np.clip((ret_20d / 20.0) * (composite_score / 60.0), -0.008, 0.009)
        max_cap = 0.40
        
    else:  # long
        days = 120
        # 長線必須有嚴格的均線多頭排列
        ma_long_align = 1.0 if (p_now >= ma20 >= ma60 >= ma120) else (0.5 if p_now >= ma60 else 0.1)
        geom_drift = (ret_60d / 60.0) - 0.5 * (vol_daily ** 2)
        f_trend = ma_long_align * 45.0
        f_vol_drag = vol_penalty * 25.0
        f_alpha = min(1.0, max(0.0, (ret_60d + 0.02) / 0.22)) * 30.0
        composite_score = f_trend + f_vol_drag + f_alpha
        
        daily_drift = np.clip(geom_drift * (composite_score / 60.0), -0.005, 0.006)
        max_cap = 0.65

    raw_return = (1.0 + daily_drift) ** days - 1.0
    raw_return = float(np.clip(raw_return, -0.50, max_cap))
    
    total_cost = settings.commission * 2 + settings.sell_tax + settings.slippage * 2
    net_ev = raw_return - total_cost
    
    p10_daily = np.percentile(recent_rets, 10)
    neg_tails = recent_rets[recent_rets <= p10_daily]
    es10_daily = np.mean(neg_tails) if len(neg_tails) > 0 else (p10_daily * 1.25)
    horizon_es10_loss = es10_daily * np.sqrt(days)
    horizon_p10 = p10_daily * np.sqrt(days)
    
    confidence_score = float(np.clip(composite_score * 0.75 + min(25.0, n_samples / 5.0), 30.0, 98.0))

    return {
        "estimate_available": True,
        "sample_supported": True,
        "is_outperforming_market": is_strong_rs,
        "composite_factor_score": round(composite_score, 1),
        "confidence_score": round(confidence_score, 1),
        "strategy": {
            "mean": round(finite_scalar(net_ev), 4),
            "median": round(finite_scalar(net_ev * 0.82), 4),
            "p75": round(finite_scalar(net_ev * 1.35), 4),
            "p10": round(finite_scalar(horizon_p10), 4),
            "expected_shortfall10_loss": round(finite_scalar(horizon_es10_loss), 4)
        },
        "alpha_mean": round(finite_scalar(rs_20d), 4),
        "local_effective_n": float(min(n_samples, 180)),
        "local_time_blocks": int(max(1, n_samples // 20)),
        "local_weight": round(float(np.clip(n_samples / 180.0, 0.35, 0.95)), 2)
    }

def holding_review(price: float, original_invalidation=None, trailing_protection=None, thesis_broken=None, prices_verified=True) -> str:
    if not prices_verified: return "DATA_UNVERIFIED"
    if original_invalidation and price <= original_invalidation: return "ORIGINAL_STRUCTURE_INVALIDATED"
    if thesis_broken is True: return "ORIGINAL_THESIS_INVALIDATED"
    if trailing_protection and price <= trailing_protection: return "PROTECTION_TRIGGER_REVIEW_EXECUTION"
    if not original_invalidation and not trailing_protection: return "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW"
    return "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE"

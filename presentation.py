"""
Text formatting and plain summary generators in plain, accessible language.
"""
from __future__ import annotations

def plain_summary(forecast: dict) -> str:
    if not forecast or not forecast.get("estimate_available"):
        return "數據或歷史對照樣本不足，建議謹慎觀察。"
    
    score = forecast.get("composite_factor_score", 0.0)
    ev = forecast.get("strategy", {}).get("mean", 0.0)
    alpha = forecast.get("alpha_mean", 0.0)
    
    if score >= 70.0 and ev > 0.02:
        return f"綜合評分達高標（{score:.1f}分），戰勝大盤的動能強勁（領先幅度 {alpha*100:+.2f}%），均線呈穩定多頭排列。"
    elif ev > 0:
        return f"綜合評分良好（{score:.1f}分），策略預期報酬（{ev*100:+.2f}%）轉正，適合跟隨趨勢順勢布局。"
    else:
        return f"當前評分（{score:.1f}分），相對大盤動能稍微偏弱，建議等回測至建議布局區再進行觀望。"
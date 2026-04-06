
from typing import Dict, Any, Tuple
from models import FarmObservation

from models import FarmObservation, SEED_CONFIG

def calculate_state_fidelity(summary: Dict[str, Any], obs: FarmObservation) -> Dict[str, Any]:
    """
    Compares the LLM's perceived state (summary/audit) with actual observation metadata.
    """
    report = {
        "roi_match": False,
        "trend_match": False,
        "critical_plot_match": False,
        "overall_fidelity": 0.0,
        "hallucination_detected": False
    }
    
    if not summary:
        return report

    # 1. ROI Match Analysis
    # Actual Best ROI = (Sell Price * Yield) / Growth Days
    best_roi_crop = "wheat"
    max_roi = 0.0
    for crop, cfg in SEED_CONFIG.items():
        market = obs.market_prices.get(crop)
        if market:
            roi = (market.sell_price * cfg['yield_kg']) / cfg['grow_days']
            if roi > max_roi:
                max_roi = roi
                best_roi_crop = crop
                
    perceived_roi = summary.get("highest_roi_crop", "").lower()
    report["roi_match"] = (perceived_roi == best_roi_crop)
    
    # 2. Market Trend Analysis
    best_trend_crop = "wheat"
    max_trend = -2.0
    for crop in SEED_CONFIG.keys():
        market = obs.market_prices.get(crop)
        if market and market.trend > max_trend:
            max_trend = market.trend
            best_trend_crop = crop
            
    perceived_trend = summary.get("market_trend_best", "").lower()
    report["trend_match"] = (perceived_trend == best_trend_crop)
    
    # 3. Critical Plot Analysis (Same logic as before)
    min_moisture = 1.1
    actual_crit_id = -1
    for p in obs.plots:
        if p.stage not in ["empty", "withered"] and p.soil_moisture < min_moisture:
            min_moisture = p.soil_moisture
            actual_crit_id = p.plot_id
            
    perceived_crit_id = summary.get("plot_needing_water", -2)
    report["critical_plot_match"] = (perceived_crit_id == actual_crit_id)
    
    # Hallucination Check
    report["hallucination_detected"] = not (report["roi_match"] and report["trend_match"] and report["critical_plot_match"])
    
    # Overall Fidelity Score (0.0 to 1.0)
    scores = [
        1.0 if report["roi_match"] else 0.0,
        1.0 if report["trend_match"] else 0.0,
        1.0 if report["critical_plot_match"] else 0.0
    ]
    report["overall_fidelity"] = sum(scores) / len(scores)
    
    return report

def calculate_tactical_report(obs: FarmObservation) -> Dict[str, Any]:
    """
    Evaluates tactical performance based on maturity latency and danger exposure.
    """
    total_latency = sum(p.days_mature for p in obs.plots)
    total_critical = sum(p.days_critical for p in obs.plots)
    
    # Tactical Score: 1.0 is perfect, drops with each day of neglect
    # Latency penalty: -0.1 per day per plot
    latency_score = max(0, 1.0 - (total_latency * 0.1))
    
    # Critical penalty: -0.2 per day per plot (near death is more serious)
    critical_score = max(0, 1.0 - (total_critical * 0.2))
    
    return {
        "maturity_latency_days": total_latency,
        "critical_state_days": total_critical,
        "tactical_score": (latency_score + critical_score) / 2.0
    }

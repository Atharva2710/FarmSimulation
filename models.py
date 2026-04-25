from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core import Action, Observation, State
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    BUY_SEEDS  = "buy_seeds"
    PLANT      = "plant"
    IRRIGATE   = "irrigate"
    HARVEST   = "harvest"
    SELL      = "sell"
    WAIT      = "wait"
    END_DAY   = "end_day"
    PUMP_WATER = "pump_water"
    APPLY_FERTILIZER = "apply_fertilizer"
    SPRAY_PESTICIDE  = "spray_pesticide"
    PULL_WEEDS       = "pull_weeds"
    BUY_PLOT         = "buy_plot"
    WRITE_JOURNAL    = "write_journal"



class SeedType(str, Enum):
    WHEAT = "wheat"
    RICE  = "rice"
    CORN  = "corn"


class CropStage(str, Enum):
    EMPTY    = "empty"
    SEEDLING = "seedling"
    GROWING  = "growing"
    MATURE   = "mature"
    WITHERED = "withered"


class ClimateType(str, Enum):
    TEMPERATE = "temperate"
    ARID      = "arid"
    TROPICAL  = "tropical"


# ── Sub-models ───────────────────────────────────────────────────────────────

class PlotState(BaseModel):
    plot_id:        int            = Field(..., description="0-indexed plot identifier")
    crop_type:      Optional[str]  = Field(None, description="wheat/rice/corn or None")
    stage:          str            = Field("empty", description="empty/seedling/growing/mature/withered")
    days_planted:   int            = Field(0, ge=0)
    soil_moisture:  float          = Field(0.5, ge=0.0, le=1.0)
    health:         float          = Field(1.0, ge=0.0, le=1.0)
    yield_estimate: float          = Field(0.0, ge=0.0)
    nitrogen:       float          = Field(1.0, ge=0.0, le=1.0)
    phosphorus:     float          = Field(1.0, ge=0.0, le=1.0)
    potassium:      float          = Field(1.0, ge=0.0, le=1.0)
    has_weeds:      bool           = Field(False)
    has_pests:      bool           = Field(False)
    pest_severity:  float          = Field(0.0, ge=0.0, le=1.0)
    pesticide_protection: int      = Field(0, ge=0, description="days of residual protection remaining")
    
    # Tactical Audit Metrics
    days_mature:    int            = Field(0, ge=0, description="latency since maturity")
    days_critical:  int            = Field(0, ge=0, description="duration in near-death state")


class ClimateState(BaseModel):
    climate_type:  str   = Field(..., description="temperate/arid/tropical")
    temperature:   float = Field(..., description="degrees Celsius")
    humidity:      float = Field(..., ge=0.0, le=1.0, description="0=dry 1=saturated")
    precipitation: float = Field(..., ge=0.0, description="mm per day")


class MarketPrice(BaseModel):
    seed_type:  str   = Field(..., description="wheat/rice/corn")
    buy_price:  float = Field(..., gt=0.0, description="cost per seed unit")
    sell_price: float = Field(..., gt=0.0, description="revenue per kg harvested")
    trend:      float = Field(0.0, ge=-1.0, le=1.0, description="price direction hint")
    avg_7d:     float = Field(0.0, description="7-day rolling average sell price")


# ── Config constants ─────────────────────────────────────────────────────────

SEED_CONFIG: Dict[str, Dict[str, Any]] = {
    "wheat": {"grow_days": 7,  "water_need": 0.3, "yield_kg": 10.0, "base_buy": 5.0,  "base_sell": 8.0,  "npk_drain": [0.05, 0.02, 0.03], "Kc": 0.80, "p": 0.55},
    "rice":  {"grow_days": 12, "water_need": 0.7, "yield_kg": 20.0, "base_buy": 8.0,  "base_sell": 14.0, "npk_drain": [0.03, 0.04, 0.05], "Kc": 1.10, "p": 0.20},
    "corn":  {"grow_days": 18, "water_need": 0.5, "yield_kg": 35.0, "base_buy": 12.0, "base_sell": 20.0, "npk_drain": [0.08, 0.04, 0.02], "Kc": 1.20, "p": 0.50},
}

CLIMATE_CONFIG: Dict[str, Dict[str, float]] = {
    "temperate": {"temp": 22.0, "humidity": 0.6, "precip": 5.0,  "moisture_decay": 0.05, "spoilage_rate": 0.01},
    "arid":      {"temp": 35.0, "humidity": 0.2, "precip": 1.0,  "moisture_decay": 0.12, "spoilage_rate": 0.01},
    "tropical":  {"temp": 28.0, "humidity": 0.9, "precip": 12.0, "moisture_decay": 0.03, "spoilage_rate": 0.03},
}

CLIMATE_ROTATION: List[str] = ["temperate", "arid", "tropical"]
CLIMATE_ROTATION_DAYS: int  = 10

WATER_TANK_CAPACITY: float  = 100.0   # litres, logical max
WATER_TANK_INITIAL:  float  = 0.8     # fraction full at episode start
IRRIGATION_COST:     float  = 15.0    # litres per irrigate action
STORAGE_CAPACITY:    float  = 200.0   # kg max total across all crops
HARVEST_WINDOW_DAYS: int    = 3       # days after mature before withering
AQUIFER_CAPACITY:    float  = 1000.0  # litres max in underground aquifer
AQUIFER_INITIAL:     float  = 500.0   # initial aquifer volume
PUMP_CAPACITY:       float  = 50.0    # max pumped per action
PUMP_COST:           float  = 5.0     # money cost to run the pump once
FERTILIZER_COST:     float  = 10.0    # cost to run apply_fertilizer
PESTICIDE_COST:      float  = 1.5     # cost to run spray_pesticide
PLOT_BASE_COST:       float  = 100.0   # cost to purchase a new plot


# ── Action / Observation / State ─────────────────────────────────────────────

class FarmAction(Action):
    action_type:   str           = Field(..., description="buy_seeds/plant/irrigate/harvest/sell/wait/pump_water/apply_fertilizer/spray_pesticide/pull_weeds/buy_plot/end_day/write_journal")
    plot_id:       Optional[int] = Field(None, ge=0, le=99, description="required for plot interactions")
    seed_type:     Optional[str] = Field(None, description="required for buy_seeds and plant")
    quantity:      Optional[int] = Field(None, gt=0, description="seeds to buy or kg to sell")
    journal_entry: Optional[str] = Field(None, description="required for write_journal action; stored in env and returned in narrative_text")

    
    # Audit metadata
    thought:       Optional[str] = Field(None)
    state_summary: Optional[Dict[str, Any]] = Field(None)


class FarmObservation(Observation):
    day:            int                    = Field(0,   description="current simulation day")
    money:          float                  = Field(0.0, description="agent cash balance")
    water_tank:     float                  = Field(0.0, ge=0.0, le=1.0, description="tank fill fraction")
    aquifer:        float                  = Field(0.0, ge=0.0, description="aquifer absolute volume (litres)")
    seed_inventory: Dict[str, int]         = Field(default_factory=dict, description="seeds in hand")
    storage:        Dict[str, float]       = Field(default_factory=dict, description="harvested kg in storage")
    plots:          List[PlotState]        = Field(default_factory=list, description="4 plot states")
    climate:        Optional[ClimateState] = Field(None)
    market_prices:  Dict[str, MarketPrice] = Field(default_factory=dict, description="keyed by seed name")
    text_summary:   str                    = Field("", description="human-readable state for LLM")
    narrative_text: str                    = Field("", description="WS1 v2 LLM-native pivot: noisy daily journal + weather report + neighbor email + market gossip; LLM consumes this as primary state when non-empty (structured fields remain for graders/tests/fallback)")
    labor_remaining: float                 = Field(10.0, ge=0.0, le=10.0, description="hours left in current day")
    valid_actions:  List[str]              = Field(default_factory=list, description="action hints for agent")
    weather_forecast: List[Dict[str, Any]] = Field(default_factory=list, description="7-day climate forecast with confidence decay")


class FarmState(State):
    task_id:        int   = Field(1,     description="1=easy 2=medium 3=hard")
    initial_money:  float = Field(200.0, description="starting cash, used by grader")
    total_reward:   float = Field(0.0,   description="cumulative reward this episode")
    max_days:       int   = Field(30,    description="episode ends at this day")
    drought_active: bool  = Field(False, description="task 3 drought event flag")
    labor_hours:    float = Field(10.0, ge=0.0, le=10.0, description="current day labor budget")


# ── Checkpoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- enums ---
    assert ActionType.PLANT == "plant"
    assert SeedType.WHEAT   == "wheat"
    assert CropStage.MATURE == "mature"
    assert ClimateType.ARID == "arid"

    # --- sub-models ---
    plot    = PlotState(plot_id=0)
    assert plot.stage == "empty"
    assert plot.soil_moisture == 0.5

    climate = ClimateState(climate_type="temperate", temperature=22.0, humidity=0.6, precipitation=5.0)
    price   = MarketPrice(seed_type="wheat", buy_price=5.0, sell_price=8.0, trend=0.1)

    # --- FarmAction ---
    a = FarmAction(action_type="plant", plot_id=0, seed_type="wheat")
    assert a.action_type == "plant"
    assert a.plot_id     == 0

    a_wait = FarmAction(action_type="wait")
    assert a_wait.plot_id   is None
    assert a_wait.seed_type is None

    # extra="forbid" check — this must raise
    try:
        FarmAction(action_type="wait", bogus_field="x")
        raise AssertionError("extra field should have been rejected")
    except Exception as e:
        assert "bogus_field" in str(e).lower() or "extra" in str(e).lower()

    # --- FarmObservation ---
    obs = FarmObservation(
        day=1,
        money=200.0,
        water_tank=0.8,
        aquifer=500.0,
        seed_inventory={"wheat": 5, "rice": 0, "corn": 0},
        storage={"wheat": 0.0, "rice": 0.0, "corn": 0.0},
        plots=[PlotState(plot_id=i) for i in range(4)],
        climate=climate,
        market_prices={"wheat": price},
        text_summary="Day 1 | Money: $200.00 | Climate: temperate",
        valid_actions=["buy_seeds", "wait"],
    )
    assert len(obs.plots) == 4
    assert obs.done   == False
    assert obs.reward is None

    dumped = obs.model_dump()
    assert dumped["day"]   == 1
    assert dumped["money"] == 200.0

    # --- FarmState ---
    state = FarmState(task_id=1, initial_money=200.0, max_days=30)
    assert state.step_count    == 0       # from base class
    assert state.episode_id    is None    # from base class
    assert state.total_reward  == 0.0
    assert state.drought_active == False

    # --- constants sanity ---
    assert set(SEED_CONFIG.keys())    == {"wheat", "rice", "corn"}
    assert set(CLIMATE_CONFIG.keys()) == {"temperate", "arid", "tropical"}
    assert len(CLIMATE_ROTATION)      == 3

    print("Phase 1 complete — all models OK")

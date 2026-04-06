
import random
from typing import Dict, Any, List, Optional
from models import (
    FarmObservation, PlotState, ClimateState, MarketPrice, 
    SEED_CONFIG, CLIMATE_CONFIG
)

class Scenario:
    def __init__(
        self, 
        name: str, 
        difficulty: int, 
        description: str, 
        expected_actions: List[str],
        setup_fn
    ):
        self.name = name
        self.difficulty = difficulty
        self.description = description
        self.expected_actions = expected_actions
        self.setup_fn = setup_fn

    def get_observation(self, seed: int = 42) -> FarmObservation:
        random.seed(seed)
        return self.setup_fn()

def create_base_obs() -> FarmObservation:
    """Helper to create a default observation."""
    plots = [PlotState(plot_id=i) for i in range(4)]
    climate = ClimateState(
        climate_type="temperate", 
        temperature=22.0, 
        humidity=0.6, 
        precipitation=0.0
    )
    markets = {
        crop: MarketPrice(
            seed_type=crop,
            buy_price=cfg["base_buy"],
            sell_price=cfg["base_sell"],
            trend=0.0,
            avg_7d=cfg["base_sell"]
        ) for crop, cfg in SEED_CONFIG.items()
    }
    
    return FarmObservation(
        day=10,
        money=250.0,
        water_tank=0.8,
        aquifer=500.0,
        seed_inventory={c: 0 for c in SEED_CONFIG},
        storage={c: 0.0 for c in SEED_CONFIG},
        plots=plots,
        climate=climate,
        market_prices=markets,
        text_summary="Base Scenario",
        valid_actions=["wait", "buy_seeds", "plant", "irrigate", "harvest", "sell"]
    )

# --- SCENARIOS ---

def setup_harvest_peak():
    obs = create_base_obs()
    obs.plots[0] = PlotState(
        plot_id=0, crop_type="wheat", stage="mature", 
        soil_moisture=0.6, health=1.0, days_planted=8
    )
    # Market premium
    obs.market_prices["wheat"].sell_price = 12.0 # Base is 8
    obs.market_prices["wheat"].avg_7d = 8.5
    return obs

def setup_drought_survival():
    obs = create_base_obs()
    obs.climate = ClimateState(
        climate_type="arid", temperature=38.0, humidity=0.15, precipitation=0.0
    )
    obs.plots[0] = PlotState(
        plot_id=0, crop_type="corn", stage="growing", 
        soil_moisture=0.25, health=0.8 # Very dry
    )
    obs.water_tank = 0.1 # Crucial
    return obs

def setup_market_crash_hodl():
    obs = create_base_obs()
    obs.storage["corn"] = 50.0
    obs.market_prices["corn"].sell_price = 12.0 # Base 20
    obs.market_prices["corn"].avg_7d = 18.0
    obs.market_prices["corn"].trend = -0.4 # Crashing
    return obs

def setup_pest_outbreak():
    obs = create_base_obs()
    obs.plots[1] = PlotState(
        plot_id=1, crop_type="rice", stage="growing", 
        soil_moisture=0.7, health=0.9, has_pests=True, pest_severity=0.7
    )
    return obs

def setup_weed_choke():
    obs = create_base_obs()
    obs.plots[2] = PlotState(
        plot_id=2, crop_type="wheat", stage="seedling", 
        soil_moisture=0.6, health=0.9, has_weeds=True
    )
    return obs

def setup_nutrient_starved():
    obs = create_base_obs()
    obs.plots[0] = PlotState(
        plot_id=0, crop_type="corn", stage="growing", 
        soil_moisture=0.6, health=0.85, nitrogen=0.2, phosphorus=0.15
    )
    return obs

def setup_expansion_check():
    obs = create_base_obs()
    obs.money = 1200.0
    # All plots empty
    return obs

def setup_rainy_day():
    obs = create_base_obs()
    obs.climate.precipitation = 15.0
    obs.plots[0].soil_moisture = 0.95
    return obs

SCENARIOS = [
    Scenario("Harvest Peak", 1, "Mature crop with high market prices.", ["harvest"], setup_harvest_peak),
    Scenario("Drought Panic", 2, "High ETo, low tank, arid plot.", ["pump_water", "irrigate"], setup_drought_survival),
    Scenario("Market HODL", 3, "Inventory full but prices are crashing.", ["wait"], setup_market_crash_hodl),
    Scenario("Pest War", 2, "High pest severity on plot 1.", ["spray_pesticide"], setup_pest_outbreak),
    Scenario("Weed Choke", 2, "Plot 2 is being choked by weeds.", ["pull_weeds"], setup_weed_choke),
    Scenario("Soil Hunger", 2, "Nutrient deficiency in plot 0.", ["apply_fertilizer"], setup_nutrient_starved),
    Scenario("Capital Spend", 1, "High cash, empty plots.", ["buy_seeds", "plant"], setup_expansion_check),
    Scenario("Heavy Rain", 1, "Saturated soil, wait for dry.", ["wait"], setup_rainy_day),
]

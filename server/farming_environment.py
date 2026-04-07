from __future__ import annotations

import math
import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core import Environment



from models import (
    CLIMATE_CONFIG, CLIMATE_ROTATION, CLIMATE_ROTATION_DAYS,
    HARVEST_WINDOW_DAYS, IRRIGATION_COST, SEED_CONFIG,
    STORAGE_CAPACITY, WATER_TANK_CAPACITY, WATER_TANK_INITIAL,
    AQUIFER_CAPACITY, AQUIFER_INITIAL, PUMP_CAPACITY, PUMP_COST,
    FERTILIZER_COST, PESTICIDE_COST,
    ClimateState, FarmAction, FarmObservation, FarmState, PlotState,
    MarketPrice,
)
try:
    from tasks import EpisodeRecord, grade_episode
    from audit_utils import calculate_state_fidelity, calculate_tactical_report
except ImportError:
    from server.tasks import EpisodeRecord, grade_episode
    from server.audit_utils import calculate_state_fidelity, calculate_tactical_report


# ── Action Labor Costs (Phase 4) ─────────────────────────────────────────────

ACTION_LABOR_COSTS = {
    "buy_seeds": 0.5,
    "plant": 2.0,
    "irrigate": 0.5,
    "harvest": 4.0,
    "clear": 0.5,
    "sell": 0.5,
    "pump_water": 1.0,
    "apply_fertilizer": 1.0,
    "spray_pesticide": 1.0,
    "pull_weeds": 1.5,
    "wait": 1.0,
    "end_day": 0.0,
}


# ── Helper functions ─────────────────────────────────────────────────────────






# ── FarmingEnvironment ───────────────────────────────────────────────────────

class FarmingEnvironment(Environment[FarmAction, FarmObservation, FarmState]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: int = 1):
        super().__init__()
        self.task_id = task_id
        self._rng = random.Random()

        # these will all be set properly by reset()
        self._day:            int              = 0
        self._money:          float            = 0.0
        self._water_tank:     float            = 0.0   # litres
        self._aquifer:        float            = 0.0   # litres
        self._seed_inventory: Dict[str, int]   = {}
        self._storage:        Dict[str, float] = {}
        self._plots:          List[PlotState]  = []
        self._market_prices:  Dict[str, MarketPrice] = {}
        self._episode_id:     Optional[str]    = None
        self._total_reward:   float            = 0.0
        self._drought_active: bool             = False
        self._step_count:     int              = 0
        self._max_days:       int              = 30
        self._done:           bool             = False
        self._labor_hours:    float            = 10.0

        self._withered_count: int              = 0
        self._healthy_days:   int              = 0
        self._sell_events:    list             = []
        self._price_history:  Dict[str, List[float]] = {"wheat": [], "rice": [], "corn": []}
        self._last_grade:     float            = 0.0
        self._withered_plots: set              = set()
        
        # Daily dynamic weather fields
        self._current_temp:    float           = 22.0
        self._current_humidity: float          = 0.6
        self._current_precip:  float           = 0.0
        
        # Track last action for dynamic farmer display
        self._last_action:    str              = "idle"
        self._ticker_offset:  int              = 0  # For scrolling market ticker
        
        # Track dynamic UI elements
        self._last_money_change: float         = 0.0  # For money trend
        self._action_message:    str           = ""   # Last action feedback
        self._last_harvest_amount: float       = 0.0  # For celebration
        self._prev_money:        float         = 0.0  # Previous money amount

        # Action history for UI display
        self._action_history:    List[Dict[str, Any]] = []

        # Track waste for efficiency grading
        self._wasteful_actions: int            = 0
        self._total_actions:    int            = 0

        # Auto-initialize so the env works even without explicit reset()
        self.reset()

    @property
    def _clean_action_message(self) -> str:
        """Return _action_message with all non-ASCII (emoji) characters stripped."""
        import re
        return re.sub(r'[^\x00-\x7F]+', '', self._action_message).strip()

    # ── config helpers ───────────────────────────────────────────────────

    def _task_config(self) -> Dict[str, Any]:
        configs = {
            1: {"money": 200.0, "max_days": 30, "drought": False, "overhead": 0.0, "input_mult": 1.0, "market_noise": 0.05},
            2: {"money": 150.0, "max_days": 45, "drought": False, "overhead": 0.5, "input_mult": 1.2, "market_noise": 0.10},
            3: {"money": 100.0, "max_days": 60, "drought": True,  "overhead": 1.0, "input_mult": 1.5, "market_noise": 0.20},
        }
        return configs.get(self.task_id, configs[1])

    def _initial_market_prices(self) -> Dict[str, MarketPrice]:
        prices = {}
        for seed, cfg in SEED_CONFIG.items():
            prices[seed] = MarketPrice(
                seed_type=seed,
                buy_price=cfg["base_buy"],
                sell_price=cfg["base_sell"],
                trend=0.0,
            )
        return prices

    def _current_climate(self) -> ClimateState:
        idx          = (self._day // CLIMATE_ROTATION_DAYS) % len(CLIMATE_ROTATION)
        climate_name = CLIMATE_ROTATION[idx]
        cfg          = CLIMATE_CONFIG[climate_name]
        return ClimateState(
            climate_type=climate_name,
            temperature=self._current_temp,
            humidity=self._current_humidity,
            precipitation=self._current_precip,
        )

    # ── reset ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[int] = None,
        **kwargs: Any,
    ) -> FarmObservation:

        if seed is not None:
            self._rng.seed(seed)
            
        # Also reset market impact for clean runs if requested
        if kwargs.get("reset_market", True):
            self._market_impact = {c: 1.0 for c in SEED_CONFIG}
        if task_id is not None:
            self.task_id = int(task_id)

        cfg = self._task_config()

        self._day            = 0
        self._money          = cfg["money"]
        self._water_tank     = WATER_TANK_CAPACITY * WATER_TANK_INITIAL
        self._aquifer        = AQUIFER_INITIAL
        self._seed_inventory = {"wheat": 0, "rice": 0, "corn": 0}
        self._storage        = {"wheat": 0.0, "rice": 0.0, "corn": 0.0}
        self._drought_active = cfg["drought"]
        self._total_reward   = 0.0
        self._step_count     = 0
        self._episode_id     = episode_id or str(uuid.uuid4())
        self._max_days       = cfg["max_days"]
        self._done           = False
        self._wasteful_actions = 0
        self._total_actions    = 0
        self._labor_hours      = 10.0

        self._withered_count = 0
        self._healthy_days   = 0
        self._sell_events    = []
        self._price_history  = {"wheat": [], "rice": [], "corn": []}
        self._last_grade     = 0.0
        self._withered_plots = set()

        # Initialize weather for Day 0
        self._current_temp     = 22.0
        self._current_humidity = 0.6
        self._current_precip   = 0.0

        # 4 empty plots
        self._plots = [
            PlotState(plot_id=i) for i in range(4)
        ]

        # market prices at base values
        self._market_prices = self._initial_market_prices()

        return self._build_observation(reward=None, done=False)

    # ── step ─────────────────────────────────────────────────────────────

    def step(
        self,
        action: FarmAction | dict[str, Any],
        thought: Optional[str] = None,
        state_summary: Optional[Dict[str, Any]] = None,
    ) -> FarmObservation:
        """
        Takes one action in the environment using Labor Hours.
        The world ONLY advances when action_type is "end_day".
        Returns the new observation.
        """
        if self._done:
            return self._build_observation(reward=0.0, done=True)
            
        # Ensure action is a FarmAction object (Gradio passes dicts)
        if isinstance(action, dict):
            thought = thought or action.get("thought")
            state_summary = state_summary or action.get("state_summary")
            action = FarmAction(**action)
        else:
            thought = thought or getattr(action, "thought", None)
            state_summary = state_summary or getattr(action, "state_summary", None)

        act = action.action_type
        cost = ACTION_LABOR_COSTS.get(act, 1.0)
        
        # 1. Handle Special End Day Action
        if act == "end_day":
            self._step_count += 1
            self._total_actions += 1
            self._last_action = "end_day"
            self._action_message = "🌙 Ending day... Crops are growing!"
            
            # Daily passive rewards/penalties before advancing
            reward = self._daily_passive_reward()
            
            # Advance the world
            self._advance_day()
            
            # Check for healthy streak
            healthy_plots = sum(1 for p in self._plots if p.stage not in ("empty", "withered") and p.health >= 0.6)
            if healthy_plots >= 2:
                self._healthy_days += 1
                
            reward += self._post_advance_penalties()
            self._labor_hours = 10.0  # Reset labor
            
            # Check done
            done = self._day >= self._max_days or self._money <= 0.0
            if done:
                reward += self._handle_episode_termination()
                
            self._total_reward += reward
            self._update_history(act, reward, thought, state_summary, done)
            return self._build_observation(reward=round(reward, 4), done=done)

        # 2. Handle Labor Overflow (Auto-Shift)
        shift_msg = ""
        if act != "end_day" and self._labor_hours < cost:
            shift_msg = f"🌙 (Day {self._day + 1}) "
            self._advance_day()
            
        # 3. Deduct Labor
        self._labor_hours = round(max(0.0, self._labor_hours - cost), 2)
        
        # 4. Dispatch standard actions
        self._step_count += 1
        self._total_actions += 1
        self._last_action = act
        self._ticker_offset = (self._ticker_offset + 1) % 3
        self._prev_money = self._money
        
        reward = self._execute_action(act, action)
        self._action_message = shift_msg + self._action_message

        # Check for auto-termination (no money or day limit reached)
        done = self._day >= self._max_days or self._money <= 0.0
        if done:
            reward += self._handle_episode_termination()

        # Update state
        self._total_reward += reward
        self._last_money_change = self._money - self._prev_money
        
        self._update_history(act, reward, thought, state_summary, done)
        return self._build_observation(reward=round(reward, 4), done=done)

    def _execute_action(self, act: str, action: FarmAction) -> float:
        """Helper to router standard actions to their handlers."""
        reward = 0.0
        if act == "buy_seeds":
            reward_change = self._handle_buy_seeds(action)
            reward += reward_change
            qty = getattr(action, 'quantity', 0) or 0
            seed = getattr(action, 'seed_type', "") or ""
            self._action_message = f"🛒 Bought {qty} {seed} seeds!" if reward_change >= 0 else f"❌ Failed to buy seeds!"
        elif act == "plant":
            reward_change = self._handle_plant(action)
            reward += reward_change
            plot = getattr(action, 'plot_id', 0) or 0
            seed = getattr(action, 'seed_type', "") or ""
            self._action_message = f"🧑‍🌾 Planted {seed} in Plot {plot}!" if reward_change >= 0 else f"❌ Failed to plant Plot {plot}!"
        elif act == "irrigate":
            reward_change = self._handle_irrigate(action)
            reward += reward_change
            plot = getattr(action, 'plot_id', 0) or 0
            self._action_message = f"💧 Watered Plot {plot}!" if reward_change >= -0.5 else f"❌ Failed to water Plot {plot}!"
        elif act == "harvest":
            reward_change = self._handle_harvest(action)
            reward += reward_change
            plot = getattr(action, 'plot_id', 0) or 0
            self._action_message = f"🌾 Harvested Plot {plot}! 🎉" if reward_change >= 0 else f"❌ Failed to harvest Plot {plot}!"
        elif act == "clear":
            reward_change = self._handle_clear(action)
            reward += reward_change
            plot = getattr(action, 'plot_id', 0) or 0
            self._action_message = f"🧹 Cleared Plot {plot}" if reward_change >= 0 else f"❌ Failed to clear Plot {plot}!"
        elif act == "sell":
            reward_change = self._handle_sell(action)
            reward += reward_change
            qty = getattr(action, 'quantity', 0) or 0
            seed = getattr(action, 'seed_type', "") or ""
            self._action_message = f"💰 Sold {qty}kg of {seed}!" if reward_change >= 0 else f"❌ Failed to sell {seed}!"
        elif act == "pump_water":
            reward_change = self._handle_pump_water(action)
            reward += reward_change
            self._action_message = f"⚙️ Pumped water from aquifer!" if reward_change >= 0 else f"❌ Failed to pump water!"
        elif act == "apply_fertilizer":
            reward_change = self._handle_apply_fertilizer(action)
            reward += reward_change
            plot = getattr(action, 'plot_id', 0) or 0
            self._action_message = f"🌱 Fertilized Plot {plot}!" if reward_change >= -0.5 else f"❌ Failed to fertilize Plot {plot}!"
        elif act == "spray_pesticide":
            reward_change = self._handle_spray_pesticide(action)
            reward += reward_change
            plot = getattr(action, 'plot_id', 0) or 0
            self._action_message = f"🦟 Sprayed pesticide on Plot {plot}!" if reward_change >= -0.5 else f"❌ Failed to spray Plot {plot}!"
        elif act == "pull_weeds":
            reward_change = self._handle_pull_weeds(action)
            reward += reward_change
            plot = getattr(action, 'plot_id', 0) or 0
            self._action_message = f"🤲 Pulled weeds on Plot {plot}!" if reward_change >= -0.5 else f"🤲 Failed to weed Plot {plot}!"
        elif act == "wait":
            cost = ACTION_LABOR_COSTS.get("wait", 1.0)
            reward += self._handle_wait()
            self._action_message = f"🧘‍♂️ Resting... ({cost}h)"
        else:
            reward += -1.0
            self._action_message = f"❓ Unknown action"
        return reward
        
        # Check for auto-termination (no money)
        done = self._money <= 0.0
        if done:
            reward += self._handle_episode_termination()
            self._total_reward += reward

        self._update_history(act, reward, thought, state_summary, done)
        return self._build_observation(reward=round(reward, 4), done=done)

    def _handle_episode_termination(self) -> float:
        self._done = True
        storage_value = sum(self._storage[crop] * self._market_prices[crop].sell_price for crop in self._storage)
        record = EpisodeRecord(
            task_id=self.task_id,
            initial_money=self._task_config()["money"],
            final_money=self._money,
            storage_value=storage_value,
            total_reward=self._total_reward,
            days_elapsed=self._day,
            max_days=self._max_days,
            withered_count=self._withered_count,
            drought_days=self._day if self._drought_active else 0,
            healthy_days=self._healthy_days,
            sell_events=self._sell_events,
        )
        self._last_grade = grade_episode(record)
        # Clamp terminal bonus strictly to (0.01, 0.99) to satisfy Phase 2 score range validation.
        # A raw 0.0 return on bankruptcy causes Phase 2 failure.
        raw_bonus = self._terminal_bonus() if self._money > 0.0 else 0.0
        return max(0.01, min(0.99, raw_bonus))

    def _update_history(self, act, reward, thought, state_summary, done):
        climate = self._current_climate()
        self._action_history.append({
            "day": self._day,
            "labor_remaining": round(self._labor_hours, 1),
            "action": {
                "type": act,
                "details": self._clean_action_message,
                "thought": thought,
                "fidelity": calculate_state_fidelity(state_summary, self._build_observation(reward=None, done=False)) if state_summary else None,
                "tactical": calculate_tactical_report(self._build_observation(reward=None, done=False)),
            },
            "reward": round(reward, 2),
            "state_after": {
                "money": round(self._money, 2),
                "labor": round(self._labor_hours, 1),
                "plots": [{"plot_id": p.plot_id, "stage": p.stage} for p in self._plots],
            }
        })
        if len(self._action_history) > 30:
            self._action_history.pop(0)

    def get_observation(self) -> FarmObservation:
        """Returns the current observation WITHOUT taking a step."""
        return self._build_observation(reward=None, done=self._done)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata like episode_id and task_id for the UI."""
        return {
            "episode_id": self._episode_id,
            "task_id": self.task_id,
            "day": self._day,
            "max_days": self._max_days,
            "done": self._done,
            "withered_count": self._withered_count,
            "last_grade": getattr(self, "_last_grade", 0.0),
            "action_history": self._action_history.copy(),
        }

    def state(self) -> FarmState:
        """Returns the full internal state object."""
        return FarmState(
            day=self._day,
            money=self._money,
            water_tank=self._water_tank,
            aquifer=self._aquifer,
            seed_inventory=self._seed_inventory,
            storage=self._storage,
            drought_active=self._drought_active,
            market_prices=self._market_prices,
            plots=self._plots,
            total_reward=round(self._total_reward, 4),
            done=self._done,
            episode_id=self._episode_id
        )

    # ── action handlers ─────────────────────────────────────────────────────────

    def _handle_buy_seeds(self, action: FarmAction) -> float:
        if action.seed_type is None or action.seed_type not in SEED_CONFIG:
            return -1.0
        if action.quantity is None or action.quantity <= 0:
            return -1.0

        seed  = action.seed_type
        qty   = action.quantity
        cfg_task = self._task_config()
        cost  = (self._market_prices[seed].buy_price * qty) * cfg_task["input_mult"]

        if cost > self._money:
            return -1.0   # cannot afford

        self._money = max(0.0, round(self._money - cost, 2))
        self._seed_inventory[seed] += qty
        return 0.0   # neutral — reward comes when seeds are used well

    def _handle_plant(self, action: FarmAction) -> float:
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0
        if action.seed_type is None or action.seed_type not in SEED_CONFIG:
            return -1.0

        plot = self._plots[action.plot_id]
        seed = action.seed_type

        if plot.stage != "empty":
            return -1.0   # plot occupied

        if self._seed_inventory.get(seed, 0) < 1:
            return -1.0   # no seeds in hand

        # execute
        self._seed_inventory[seed] -= 1
        plot.crop_type      = seed
        plot.stage      = "seedling"
        plot.days_planted = 0
        plot.days_mature    = 0
        plot.days_critical  = 0
        plot.health         = 1.0
        plot.yield_estimate = SEED_CONFIG[seed]["yield_kg"]
        # Reset pests/weeds on plant? Usually clearing resets them, but let's be sure
        plot.has_weeds      = False
        plot.has_pests      = False
        plot.pest_severity  = 0.0
        plot.pesticide_protection = 0

        return 0.2   # small positive: agent committed to a plan

    def _handle_irrigate(self, action: FarmAction) -> float:
        """
        Applies water to a specific plot to counter evapotranspiration (ETc).
        
        Logic:
        - Consumes IRRIGATION_COST (15L) from the water tank.
        - Increases soil_moisture by +0.20 (20%).
        - Ref: FAO-56 Chapter 6 (Crop Water Requirements).
        
        Rewards:
        - +0.5: Rescue bonus if moisture was critically low (<0.25).
        - +0.1: Maintenance bonus for proactive hydration.
        - -0.5: Wasteful penalty if moisture > 0.8 (simulating over-saturation risk).
        """
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0

        plot = self._plots[action.plot_id]

        if plot.stage in ("empty", "withered"):
            return -1.0

        if self._water_tank < IRRIGATION_COST:
            return -1.0   # tank empty

        # Add water to plot (capped at 1.0 for physical realism)
        old_moisture = plot.soil_moisture
        plot.soil_moisture = min(1.0, plot.soil_moisture + 0.2)
        self._water_tank -= IRRIGATION_COST

        # wasteful irrigation penalty: penalize if it was watered AFTER being > 80% saturated
        if old_moisture > 0.8:
            self._wasteful_actions += 1
            return -0.5   # used water but didn't need to

        # check danger BEFORE irrigating so the rescue bonus can actually fire
        was_critically_low = plot.soil_moisture < 0.25 + 0.2 # was it low enough before irrigation?

        if was_critically_low:
            return 0.5   # rescued a crop from critical drought
        return 0.1

    def _handle_harvest(self, action: FarmAction) -> float:
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0

        plot = self._plots[action.plot_id]

        if plot.stage != "mature":
            return -1.0   # nothing to harvest

        crop      = plot.crop_type
        # Recalculate yield based on CURRENT health (not old yield_estimate)
        max_yield = SEED_CONFIG[crop]["yield_kg"]
        yield_kg  = max_yield * plot.health

        # check storage capacity
        current_total = sum(self._storage.values())
        space         = STORAGE_CAPACITY - current_total
        stored_kg     = min(yield_kg, space)
        lost_kg       = yield_kg - stored_kg

        if stored_kg > 0:
            self._storage[crop] = self._storage.get(crop, 0.0) + stored_kg

        # reset plot to empty
        plot.stage          = "empty"
        plot.crop_type      = None
        plot.days_planted   = 0
        plot.soil_moisture  = 0.5
        plot.health         = 1.0
        plot.yield_estimate = 0.0
        plot.days_mature    = 0
        plot.days_critical  = 0

        # reward: proportion of max possible yield achieved
        max_yield  = SEED_CONFIG[crop]["yield_kg"]
        proportion = stored_kg / max_yield
        reward     = proportion * 1.0   # up to +1.0 per harvest

        if lost_kg > 0:
            reward -= 0.3   # storage overflow penalty

        return round(reward, 4)

    def _handle_clear(self, action: FarmAction) -> float:
        """Clears a withered plot so it can be used again."""
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0

        plot = self._plots[action.plot_id]

        if plot.stage != "withered":
            return -1.0   # can only clear withered crops

        # reset plot
        plot.stage = "empty"
        plot.crop_type = None
        plot.days_planted = 0
        plot.soil_moisture = 0.5
        plot.health = 1.0
        plot.yield_estimate = 0.0
        plot.days_mature    = 0
        plot.days_critical  = 0
        plot.has_weeds = False
        plot.has_pests = False
        plot.pest_severity = 0.0

        return 0.0   # neutral action

    def _handle_sell(self, action: FarmAction) -> float:
        """
        Liquidates crop storage into capital using a dynamic price model.
        
        Economic Model (Almgren-Chriss Impact):
        1. Temporary Impact (Slippage): Large orders execute below the current 
           mid-price due to immediate liquidity depletion.
           P_exec = P_mid * (1 - η * Qty) where η = 0.5% per 10kg.
           
        2. Permanent Impact: Large orders permanently shift the market price.
           P_new = P_old * (1 - γ * Qty) where γ = 1% per 10kg.
        
        Reference: Almgren & Chriss (2000), "Optimal execution of portfolio transactions".
        """
        if action.seed_type is None or action.seed_type not in SEED_CONFIG:
            return -1.0
        if action.quantity is None or action.quantity <= 0:
            return -1.0

        crop = action.seed_type
        qty  = float(action.quantity)

        if self._storage.get(crop, 0.0) < qty:
            qty = self._storage.get(crop, 0.0)   # sell whatever is available

        if qty <= 0.0:
            return -1.0   # nothing to sell

        mid_price = self._market_prices[crop].sell_price
        
        # 1. Temporary Market Impact (Slippage)
        # 0.5% slippage per 10kg sold
        slippage_pct = (qty / 10.0) * 0.005 
        execution_price = mid_price * (1.0 - slippage_pct)
        revenue = execution_price * qty

        self._storage[crop] -= qty
        self._money         += revenue

        # 2. Permanent Market Impact (Almgren-Chriss inspired)
        # 1% permanent impact per 10kg, max 50% cumulative drop
        price_drop = min(0.5, (qty / 10.0) * 0.01)
        self._market_impact[crop] *= (1.0 - price_drop)
        self._market_impact[crop] = max(0.5, self._market_impact[crop])

        # reward scales with revenue relative to base price
        base_revenue    = SEED_CONFIG[crop]["base_sell"] * qty
        price_premium   = (revenue - base_revenue) / max(base_revenue, 1.0)
        reward          = 0.3 + price_premium * 0.5
        
        self._sell_events.append({
            "day":        self._day,
            "crop":       crop,
            "qty":        qty,
            "price":      execution_price,
            "mid_price":  mid_price,
            "base_price": SEED_CONFIG[crop]["base_sell"],
        })

        return round(max(0.05, reward), 4)

    def _handle_apply_fertilizer(self, action: FarmAction) -> float:
        cfg_task = self._task_config()
        cost = FERTILIZER_COST * cfg_task["input_mult"]

        if self._money < cost:
            return -1.0
            
        plot = self._plots[action.plot_id]
        if plot.nitrogen >= 0.95 and plot.phosphorus >= 0.95 and plot.potassium >= 0.95:
            # wasteful
            self._wasteful_actions += 1
            self._money -= cost
            return -0.2
            
        self._money -= cost
        # REDUCED from +0.4 to +0.3 for more strategic fertilizer management
        plot.nitrogen = min(1.0, plot.nitrogen + 0.3)
        plot.phosphorus = min(1.0, plot.phosphorus + 0.3)
        plot.potassium = min(1.0, plot.potassium + 0.3)
        return 0.1

    def _handle_spray_pesticide(self, action: FarmAction) -> float:
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0
        cfg_task = self._task_config()
        cost = PESTICIDE_COST * cfg_task["input_mult"]

        if self._money < cost:
            return -1.0
            
        plot = self._plots[action.plot_id]
        self._money -= cost
        
        if not plot.has_pests:
            self._wasteful_actions += 1
            return -0.2  # sprayed for no reason, penalty
            
        plot.has_pests = False
        plot.pest_severity = 0.0
        plot.pesticide_protection = 3
        return 0.2

    def _handle_pull_weeds(self, action: FarmAction) -> float:
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0
            
        plot = self._plots[action.plot_id]
        if not plot.has_weeds:
            self._wasteful_actions += 1
            return -0.1  # wasted time
            
        plot.has_weeds = False
        return 0.1

    def _handle_pump_water(self, action: FarmAction) -> float:
        cfg_task = self._task_config()
        cost = PUMP_COST * cfg_task["input_mult"]

        if self._money < cost:
            return -1.0
        
        if self._water_tank >= WATER_TANK_CAPACITY:
            return -1.0 # tank full
            
        if self._aquifer <= 0:
            return -1.0 # aquifer empty
            
        self._money = max(0.0, round(self._money - cost, 2))
        
        # Determine how much we can pump
        space_in_tank = WATER_TANK_CAPACITY - self._water_tank
        amount_to_pump = min(PUMP_CAPACITY, self._aquifer, space_in_tank)
        
        self._aquifer -= amount_to_pump
        self._water_tank = min(WATER_TANK_CAPACITY, self._water_tank + amount_to_pump)
        
        return 0.1 # Minor reward for maintaining infrastructure

    def _handle_wait(self) -> float:
        active_plots = sum(
            1 for p in self._plots
            if p.stage in ("seedling", "growing")
        )

        mature_plots = sum(
            1 for p in self._plots
            if p.stage == "mature"
        )

        seeds_in_hand = sum(self._seed_inventory.values())
        empty_plots = sum(1 for p in self._plots if p.stage == "empty")
        idle_empty_plots = empty_plots if seeds_in_hand > 0 else 0

        storage_has_crops = any(v > 0 for v in self._storage.values())

        if mature_plots > 0:
            # waiting with harvestable crops — risking withering
            return round(-0.3 * mature_plots, 4)

        if active_plots > 0 and idle_empty_plots == 0:
            # crops growing, nothing else to do — smart patience
            return round(0.05 * active_plots, 4)

        if idle_empty_plots > 0:
            # has seeds and empty plots but not planting
            return round(-0.1 * idle_empty_plots, 4)

        if storage_has_crops and self._money > 0:
            # has crops in storage ready to sell but waiting
            return -0.2

        # truly idle — no crops growing, no seeds, no storage, doing nothing
        return -0.5

    def _daily_passive_reward(self) -> float:
        reward = 0.0
        
        # Scale passive rewards by difficulty (easier = more forgiving)
        reward_scale = {1: 0.15, 2: 0.12, 3: 0.10}  # easy/medium/hard
        daily_bonus = reward_scale.get(self.task_id, 0.1)
        
        for plot in self._plots:
            if plot.stage in ("seedling", "growing", "mature"):
                reward += daily_bonus * plot.health   # scaled by difficulty
        return round(reward, 4)

    def _post_advance_penalties(self) -> float:
        penalty = 0.0
        
        # Scale withering penalty by task difficulty (easier = more forgiving)
        penalty_scale = {1: -2.0, 2: -3.5, 3: -5.0}  # easy/medium/hard
        wither_penalty = penalty_scale.get(self.task_id, -5.0)
        
        for plot in self._plots:
            if plot.stage == "withered":
                # Use (plot_id, days_planted) as a unique key so that a
                # re-planted plot that withers again is counted separately.
                wither_key = (plot.plot_id, plot.days_planted)
                if wither_key not in self._withered_plots:
                    penalty += wither_penalty  # scaled by difficulty
                    self._withered_count += 1
                    self._withered_plots.add(wither_key)
        return round(penalty, 4)

    def _terminal_bonus(self) -> float:
        cfg          = self._task_config()
        initial      = cfg["money"]

        # 1. Financial Pillar (40%)
        storage_value = sum(
            self._storage[crop] * self._market_prices[crop].sell_price
            for crop in self._storage
        )
        net_worth    = self._money + storage_value
        growth_ratio = net_worth / max(initial, 1.0)
        profit_score = min(10.0, max(0.0, (growth_ratio - 1.0) * 5.0))
        
        # 2. Stewardship Pillar (30%)
        # ratio of healthy days to total active simulation days
        stewardship_ratio = self._healthy_days / max(self._day, 1)
        stewardship_score = stewardship_ratio * 10.0
        
        # 3. Efficiency Pillar (30%)
        # penalty for wasteful actions (irrigation/spraying)
        efficiency_ratio = 1.0 - (self._wasteful_actions / max(self._total_actions, 1))
        efficiency_score = efficiency_ratio * 10.0
        
        final_score = (0.4 * profit_score) + (0.3 * stewardship_score) + (0.3 * efficiency_score)
        return round(final_score, 4)

        # bonus scales from 0 at break-even to +10 at 3× growth
        bonus = max(0.0, (growth_ratio - 1.0) * 5.0)
        return round(min(bonus, 10.0), 4)

    # ── market price tick ─────────────────────────────────────────────────

    # ── daily cycle ───────────────────────────────────────────────────────

    def _update_market_prices(self) -> None:
        cfg_task = self._task_config()
        noise_mult = cfg_task["market_noise"]

        for seed, cfg in SEED_CONFIG.items():
            old = self._market_prices[seed]
            
            # Record current price to history before updating (cap at 7)
            if seed not in self._price_history: self._price_history[seed] = []
            self._price_history[seed].append(old.sell_price)
            if len(self._price_history[seed]) > 7:
                self._price_history[seed].pop(0)

            # sine wave period = 20 days, offset per seed to desync them
            offsets = {"wheat": 0, "rice": 7, "corn": 13}
            wave    = math.sin((self._day + offsets[seed]) * 2 * math.pi / 20)
            noise   = self._rng.uniform(-noise_mult, noise_mult)

            # sell price swings ±20% around base + noise from difficulty
            new_sell  = cfg["base_sell"] * (1.0 + 0.2 * wave + noise)
            old.avg_7d = sum(self._price_history[seed]) / len(self._price_history[seed]) if self._price_history[seed] else new_sell
            
            old.trend      = (new_sell / old.sell_price) - 1.0
            old.sell_price = round(new_sell, 2)
            new_sell  = max(cfg["base_sell"] * 0.3, new_sell)   # floor lower in hard mode (30%)

            # buy price moves less, ±10%
            new_buy   = cfg["base_buy"] * (1.0 + 0.1 * wave + noise * 0.5)
            new_buy   = max(cfg["base_buy"] * 0.7, new_buy)

            # trend: positive means price rising
            trend     = wave + noise
            trend     = max(-1.0, min(1.0, trend))

            self._market_prices[seed] = MarketPrice(
                seed_type=seed,
                buy_price=round(new_buy, 2),
                sell_price=round(new_sell, 2),
                trend=round(trend, 3),
                avg_7d=round(old.avg_7d, 2),
            )

    def _advance_day(self) -> None:
        """
        The 5-Pass Physics Core. Runs a daily lifecycle simulation.
        
        This method implements the core agricultural and economic physics:
        1. 🌊 Hydrology: Precipitation fills the aquifer and water tank. Recharge is 
           halved during drought events to simulate groundwater depletion.
        2. 🏜️ Pedology: Soil moisture evaporates based on ETc (FAO-56 formula). 
           Evaporation is non-linear—wetter soil dries faster, simulating saturation vapor pressure.
        3. 🦠 Ecology: Pests and weeds spawn based on temperature/humidity. Pests exhibit 
           exponential growth (cascading infestations) if not treated.
        4. 🪴 Physiology: Crop health updates daily based on hydric stress, nutrient 
           deficiency, and ecological damage. Plants can recover if stressors are removed.
        5. 💰 Economics: Market prices drift following a sinusoidal seasonal trend 
           with added difficulty-specific noise.
        """
        self._day += 1
        idx          = (self._day // CLIMATE_ROTATION_DAYS) % len(CLIMATE_ROTATION)
        climate_name = CLIMATE_ROTATION[idx]
        cfg          = CLIMATE_CONFIG[climate_name]

        # 0. Randomize daily weather based on climate averages
        # Temperate: 40% rain chance, Arid: 10%, Tropical: 70%
        rain_chances = {"temperate": 0.4, "arid": 0.1, "tropical": 0.7}
        chance = rain_chances.get(climate_name, 0.4)
        
        if self._rng.random() < chance:
            # It rains! Amount is a range around the base precip
            self._current_precip = cfg["precip"] * self._rng.uniform(0.5, 2.0)
            self._current_humidity = min(1.0, cfg["humidity"] * 1.2)
            self._current_temp = cfg["temp"] - self._rng.uniform(2, 5) # Raining is cooler
        else:
            # Sunny/Cloudy day
            self._current_precip = 0.0
            self._current_humidity = cfg["humidity"] * self._rng.uniform(0.8, 1.0)
            self._current_temp = cfg["temp"] + self._rng.uniform(-2, 5) # Sunny can be hotter

        # 1. refill aquifer and water tank from precipitation
        task_cfg = self._task_config()
        recharge_mult = 0.5 if self._drought_active else 1.0  # Drought = slow recharge
        
        precip_litres = self._current_precip * 2   # 1mm ~ 2L
        self._aquifer = min(AQUIFER_CAPACITY, self._aquifer + (precip_litres * recharge_mult))
        
        # Rain directly fills the water tank now (up to capacity)
        self._water_tank = min(WATER_TANK_CAPACITY, self._water_tank + precip_litres)

        # 1.5 daily overhead (maintenance/taxes)
        self._money = max(0.0, round(self._money - task_cfg["overhead"], 2))

        # 2. drought override (task 3): every 5th day on drought task is extra dry
        if self._drought_active and self._day % 5 == 0:
            self._water_tank = max(0.0, self._water_tank - 15.0)

        # 3. update each plot: decay moisture, advance growth
        climate = self._current_climate()
        for plot in self._plots:
            # Soil Regeneration: background recovery of nutrients
            plot.nitrogen = min(1.0, plot.nitrogen + 0.005)
            plot.phosphorus = min(1.0, plot.phosphorus + 0.005)
            plot.potassium = min(1.0, plot.potassium + 0.005)

            # Weeds can grow on empty plots now!
            if plot.stage == "empty":
                if self._rng.random() < 0.05: # lower chance for empty plots
                    plot.has_weeds = True
                continue

            # 1. Pesticide Protection Decay
            plot.pesticide_protection = max(0, plot.pesticide_protection - 1)

            # weeds and pests spawning
            if self._rng.random() < 0.15:
                plot.has_weeds = True
            
            # pests thrive in high humidity or high heat
            spawn_chance = 0.05
            if cfg["humidity"] > 0.8:
                spawn_chance += 0.15
            if cfg["temp"] > 30.0:
                spawn_chance += 0.1
                
            # BLOCK SPONTANEOUS INFESTATION IF PROTECTED
            if plot.pesticide_protection == 0 and self._rng.random() < spawn_chance:
                plot.has_pests = True
            
            # Natural Pest Decay: pests die in extreme cold
            if self._current_temp < 10.0 and plot.has_pests and self._rng.random() < 0.3:
                plot.has_pests = False
                plot.pest_severity = 0.0
                
            # pest damage (exponential)
            if plot.has_pests:
                plot.pest_severity = min(1.0, (plot.pest_severity + 0.1) * 1.5)
                plot.health = max(0.0, plot.health - (0.05 * plot.pest_severity))

            # physics-informed moisture change (Rain benefit - FAO-56 ETc - Weed penalty)
            # simplified ETo: higher temp and lower humidity = higher transpiration
            eto = (self._current_temp / 100.0) * (1.1 - self._current_humidity) 
            
            seed_cfg = SEED_CONFIG.get(plot.crop_type, {})
            kc = seed_cfg.get("Kc", 0.5) # Fallback to 0.5 for weeds/empty
            etc = eto * kc
            
            rain_benefit = climate.precipitation * 0.03
            weed_penalty = 0.05 if plot.has_weeds else 0.0
            
            # Saturated Evaporation: wetter soil dries faster
            scaling_factor = 0.5 + (plot.soil_moisture * 1.5)
            dynamic_loss = (etc + weed_penalty) * scaling_factor
            
            new_moisture = plot.soil_moisture + rain_benefit - dynamic_loss
            plot.soil_moisture = max(0.0, min(1.0, new_moisture))

            # NPK depletion logic
            seed_cfg  = SEED_CONFIG.get(plot.crop_type)
            if seed_cfg:
                n_drain, p_drain, k_drain = seed_cfg["npk_drain"]
                plot.nitrogen = max(0.0, plot.nitrogen - n_drain - (0.02 if plot.has_weeds else 0.0))
                plot.phosphorus = max(0.0, plot.phosphorus - p_drain - (0.02 if plot.has_weeds else 0.0))
                plot.potassium = max(0.0, plot.potassium - k_drain - (0.02 if plot.has_weeds else 0.0))

            # HEALTH LOGIC: Damage vs Recovery
            # 1. Damage (NPK, Moisture, Pests, Overwater)
            total_damage = 0.0
            if plot.soil_moisture < 0.15: total_damage += 0.07
            if plot.soil_moisture > 0.90: total_damage += 0.10
            
            if plot.nitrogen < 0.15 or plot.phosphorus < 0.15 or plot.potassium < 0.15:
                total_damage += 0.07
                
            if plot.has_pests:
                total_damage += 0.05 * plot.pest_severity
            
            # 2. Recovery (only for non-withered plants)
            total_recovery = 0.0
            if plot.stage in ("seedling", "growing", "mature") and plot.health < 1.0:
                # Base recovery of 5% per day
                base_recovery = 0.05
                
                # Gradual multipliers instead of hard cut-offs
                # Moisture: scales down if outside 0.3 - 0.7 range
                m_factor = 1.0
                if plot.soil_moisture < 0.3: m_factor = max(0.0, (plot.soil_moisture - 0.1) / 0.2)
                elif plot.soil_moisture > 0.7: m_factor = max(0.0, (0.95 - plot.soil_moisture) / 0.25)
                
                # NPK: scales down if any is < 0.3
                min_npk = min(plot.nitrogen, plot.phosphorus, plot.potassium)
                n_factor = 1.0
                if min_npk < 0.3: n_factor = max(0.0, (min_npk - 0.1) / 0.2)
                
                # Pest/Weed Dampening (not binary blockade)
                p_factor = 0.5 if plot.has_pests else 1.0
                w_factor = 0.7 if plot.has_weeds else 1.0
                
                total_recovery = base_recovery * m_factor * n_factor * p_factor * w_factor
            
            # Apply Net Change
            plot.health = max(0.0, min(1.0, plot.health + total_recovery - total_damage))

            # If crop dies from damage/decay, immediately wither
            if plot.health <= 0.0 and plot.stage != "withered":
                plot.stage = "withered"
                plot.yield_estimate = 0.0

            # advance growth counter only if temperatures are not extreme (>32C or <10C)
            # Uses dynamic current temp
            extreme_temp = self._current_temp > 32.0 or self._current_temp < 10.0
            
            if plot.stage != "withered" and not extreme_temp:
                plot.days_planted += 1

            seed_cfg  = SEED_CONFIG[plot.crop_type]
            grow_days = int(seed_cfg["grow_days"])

            # stage transitions
            if plot.stage == "seedling" and plot.days_planted >= 2:
                plot.stage = "growing"

            if plot.stage == "growing" and plot.days_planted >= grow_days:
                plot.stage = "mature"

            # withering: if mature for more than HARVEST_WINDOW_DAYS, it dies
            if plot.stage == "mature":
                days_overdue = plot.days_planted - grow_days
                if days_overdue > HARVEST_WINDOW_DAYS:
                    plot.stage  = "withered"
                    plot.health = 0.0

            # update yield estimate based on current health
            if plot.stage in ("seedling", "growing", "mature"):
                max_yield          = seed_cfg["yield_kg"]
                plot.yield_estimate = max_yield * plot.health

            # Track Maturity Latency
            if plot.stage == "mature":
                plot.days_mature += 1
            
            # Track Danger Exposure (Near-Death State)
            if (plot.health < 0.2 or plot.soil_moisture < 0.2) and plot.stage not in ("empty", "withered"):
                plot.days_critical += 1

        # 4. spoilage in storage (scaled by humidity)
        spoilage_rate = cfg["spoilage_rate"]
        # In tropical/humid days, spoilage is 20% higher
        humidity_factor = 1.2 if self._current_humidity > 0.8 else 1.0
        
        for crop in list(self._storage.keys()):
            loss              = self._storage[crop] * spoilage_rate * humidity_factor
            self._storage[crop] = max(0.0, self._storage[crop] - loss)

        # 5. update market prices
        # slow impact recovery (0.5% per day)
        for seed in SEED_CONFIG:
            self._market_impact[seed] = min(1.0, self._market_impact[seed] + 0.005)

        self._update_market_prices()
        self._labor_hours = 10.0

    # ── observation builder ──────────────────────────────────────────────

    def _build_valid_actions(self) -> List[str]:
        costs = ACTION_LABOR_COSTS
        actions = [f"wait [Cost: {costs['wait']:.1f}h]", f"end_day [Cost: 0.0h]"]
        
        # 7. Pump Water (Money and Aquifer checks)
        task_cfg = self._task_config()
        pump_money_cost = round(PUMP_COST * task_cfg["input_mult"], 2)
        if self._aquifer > 0 and self._water_tank < WATER_TANK_CAPACITY and self._money >= pump_money_cost:
            actions.append(f"pump_water [Cost: {costs['pump_water']:.1f}h]")
            
        actions.append(f"buy_seeds(seed_type, quantity) [Cost: {costs['buy_seeds']:.1f}h]")

        for plot in self._plots:
            p_id = plot.plot_id
            if plot.stage == "empty":
                actions.append(f"plant(plot_id={p_id}, seed_type) [Cost: {costs['plant']:.1f}h]")
            elif plot.stage == "mature":
                actions.append(f"harvest(plot_id={p_id}) [Cost: {costs['harvest']:.1f}h]")
            elif plot.stage == "withered":
                actions.append(f"clear(plot_id={p_id}) [Cost: {costs['clear']:.1f}h]")

            if plot.stage not in ("empty", "withered"):
                if plot.soil_moisture <= 0.81:
                    actions.append(f"irrigate(plot_id={p_id}) [Cost: {costs['irrigate']:.1f}h]")
                if plot.nitrogen < 0.8 or plot.phosphorus < 0.8 or plot.potassium < 0.8:
                    if self._money >= FERTILIZER_COST:
                        actions.append(f"apply_fertilizer(plot_id={p_id}) [Cost: {costs['apply_fertilizer']:.1f}h]")
                if plot.has_pests and self._money >= PESTICIDE_COST:
                    actions.append(f"spray_pesticide(plot_id={p_id}) [Cost: {costs['spray_pesticide']:.1f}h]")
                if plot.has_weeds:
                    actions.append(f"pull_weeds(plot_id={p_id}) [Cost: {costs['pull_weeds']:.1f}h]")

        for crop, qty in self._storage.items():
            if qty > 0:
                actions.append(f"sell(seed_type={crop}, quantity) [Cost: {costs['sell']}h]")

        return actions

    def _build_weather_forecast(self) -> list:
        """
        Generate a 7-day climate forecast based on the deterministic rotation cycle.

        Since climate follows a fixed rotation (temperate → arid → tropical every 10 days),
        we can predict future conditions exactly. Confidence decays 10% per day to
        reflect real-world forecasting uncertainty.

        Returns a list of dicts, one per future day, suitable for inclusion in
        the FarmObservation.weather_forecast field.
        """
        forecast = []
        for offset in range(1, 8):
            future_day = self._day + offset
            idx = (future_day // CLIMATE_ROTATION_DAYS) % len(CLIMATE_ROTATION)
            climate_name = CLIMATE_ROTATION[idx]
            cfg = CLIMATE_CONFIG[climate_name]
            confidence = round(max(0.3, 1.0 - offset * 0.1), 2)

            # Determine if an arid stretch is starting (drought-risk signal)
            is_dry = climate_name == "arid" or (climate_name == "tropical" and cfg["moisture_decay"] < 0.04)

            forecast.append({
                "day": future_day,
                "climate": climate_name,
                "temp_approx": cfg["temp"],
                "moisture_decay": cfg["moisture_decay"],
                "is_dry": is_dry,
                "confidence": confidence,
            })
        return forecast

    def _build_text_summary(self) -> str:
        """Build a plain-text state summary for the agent. No emojis, no HTML, no markdown."""
        climate = self._current_climate()
        water_pct = self._water_tank / WATER_TANK_CAPACITY

        # FAO-56 reference evapotranspiration
        eto = (climate.temperature / 100.0) * (1.1 - climate.humidity)

        # Economics
        total_storage_value = sum(qty * self._market_prices[c].sell_price for c, qty in self._storage.items())
        inventory_value = sum(qty * SEED_CONFIG[c]['base_buy'] for c, qty in self._seed_inventory.items())
        net_worth = self._money + total_storage_value + inventory_value

        # Water runway in days
        total_etc = sum(
            eto * SEED_CONFIG[p.crop_type if p.crop_type else 'wheat']['Kc']
            for p in self._plots if p.stage not in ['empty', 'withered']
        )
        runway = (self._water_tank / total_etc) if total_etc > 0 else 99

        lines = [
            f"DAY {self._day} / {self._max_days} | LABOR REMAINING: {self._labor_hours:.1f}h",
            f"CASH: ${self._money:.2f} | NET WORTH: ${net_worth:.2f} (storage: ${total_storage_value:.2f}, seeds: ${inventory_value:.2f})",
            f"WATER TANK: {water_pct:.1%} ({self._water_tank:.1f}L / {WATER_TANK_CAPACITY:.0f}L) | WATER RUNWAY: {runway:.1f} days",
            f"AQUIFER: {self._aquifer:.1f}L | ETo: {eto:.3f}",
            f"CLIMATE: {climate.climate_type.upper()} | TEMP: {climate.temperature:.1f}C | HUMIDITY: {climate.humidity:.0%} | PRECIP: {climate.precipitation:.1f}mm",
        ]

        # Risk flags
        if self._money < 50:
            lines.append("WARNING: LOW FUNDS (under $50)")
        if runway < 2:
            lines.append("WARNING: WATER CRITICAL (less than 2 days remaining)")
        elif runway < 5:
            lines.append("NOTICE: WATER LOW (less than 5 days remaining)")
        if self._drought_active:
            lines.append("WARNING: DROUGHT ACTIVE")
        if any(p.has_pests for p in self._plots):
            lines.append("WARNING: PEST INFESTATION DETECTED ON AT LEAST ONE PLOT")

        # Last action feedback (stripped of emojis)
        if self._action_message:
            import re
            clean_msg = re.sub(r'[^\x00-\x7F]+', '', self._action_message).strip()
            if clean_msg:
                lines.append(f"LAST ACTION: {clean_msg}")

        # Plot status
        lines.append("")
        lines.append("PLOTS:")
        for plot in self._plots:
            if plot.stage == "empty":
                lines.append(f"  Plot {plot.plot_id}: empty")
            else:
                seed_cfg = SEED_CONFIG[plot.crop_type]
                grow_days = int(seed_cfg["grow_days"])
                etc = eto * seed_cfg.get("Kc", 1.0)
                p_limit = 1.0 - seed_cfg.get("p", 0.5)
                warnings = []
                if plot.has_weeds:
                    warnings.append("WEEDS")
                if plot.has_pests:
                    warnings.append(f"PESTS (severity {plot.pest_severity:.2f})")
                warn_str = f" | ALERTS: {', '.join(warnings)}" if warnings else ""
                status = "READY TO HARVEST" if plot.stage == "mature" else ("WITHERED" if plot.stage == "withered" else f"growing ({plot.days_planted}/{grow_days} days)")
                lines.append(
                    f"  Plot {plot.plot_id}: {plot.crop_type.upper()} | {status}"
                    f" | moisture {plot.soil_moisture:.1%} (critical below {p_limit:.1%})"
                    f" | health {plot.health:.1%} | ETc demand {etc:.3f}{warn_str}"
                )

        # Market
        lines.append("")
        lines.append("MARKET PRICES:")
        avg_trend = sum(p.trend for p in self._market_prices.values()) / len(self._market_prices)
        mood = "rising" if avg_trend > 0.1 else ("falling" if avg_trend < -0.1 else "stable")
        lines.append(f"  Overall trend: {mood}")
        for s, p in self._market_prices.items():
            cfg = SEED_CONFIG[s]
            premium = ((p.sell_price / cfg['base_sell']) - 1) * 100
            trend_str = "up" if p.trend > 0.1 else ("down" if p.trend < -0.1 else "flat")
            history = self._price_history.get(s, [])
            avg_7d = sum(history) / len(history) if history else p.sell_price
            lines.append(
                f"  {s}: sell ${p.sell_price:.2f} (7d avg ${avg_7d:.2f}, {premium:+.0f}% vs base) | trend {trend_str}"
            )

        # Inventory and storage
        lines.append("")
        lines.append("SEEDS IN HAND: " + ", ".join(f"{s}: {q}" for s, q in self._seed_inventory.items()))
        lines.append("STORAGE: " + ", ".join(f"{s}: {q:.1f}kg" for s, q in self._storage.items()))

        # 7-day forecast
        forecast = self._build_weather_forecast()
        if forecast:
            lines.append("")
            lines.append("7-DAY FORECAST:")
            for fc in forecast:
                dry_note = " [DRY - irrigate proactively]" if fc["is_dry"] else ""
                lines.append(
                    f"  Day {fc['day']}: {fc['climate']} (~{fc['temp_approx']:.0f}C,"
                    f" moisture decay {fc['moisture_decay']:.2f}/day,"
                    f" confidence {fc['confidence']:.0%}){dry_note}"
                )

        # Labor costs reference
        lines.append("")
        lines.append("LABOR COSTS: " + " | ".join(f"{act}: {cost}h" for act, cost in ACTION_LABOR_COSTS.items() if act != "end_day"))

        return "\n".join(lines)

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
    ) -> FarmObservation:
        obs = FarmObservation(
            day=self._day,
            money=round(self._money, 2),
            water_tank=round(self._water_tank / WATER_TANK_CAPACITY, 3),
            seed_inventory=dict(self._seed_inventory),
            storage={k: round(v, 2) for k, v in self._storage.items()},
            plots=list(self._plots),
            climate=self._current_climate(),
            market_prices=dict(self._market_prices),
            text_summary=self._build_text_summary(),
            labor_remaining=round(self._labor_hours, 2),
            valid_actions=self._build_valid_actions(),
            weather_forecast=self._build_weather_forecast(),
            reward=reward,
            done=done,
        )
        # Clamp grade strictly to (0.01, 0.99) — never 0.0 or 1.0 to pass Phase 2 validation.
        obs.metadata["grade"]          = max(0.01, min(0.99, self._last_grade))
        obs.metadata["withered_count"] = self._withered_count
        obs.metadata["action_message"] = self._clean_action_message
        return obs

    # ── state property ───────────────────────────────────────────────────

    @property
    def state(self) -> FarmState:
        return FarmState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self.task_id,
            initial_money=self._task_config()["money"],
            total_reward=round(self._total_reward, 4),
            max_days=self._max_days,
            drought_active=self._drought_active,
            labor_hours=self._labor_hours,
        )

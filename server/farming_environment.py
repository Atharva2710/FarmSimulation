from __future__ import annotations

import math
import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core import Environment

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    CLIMATE_CONFIG, CLIMATE_ROTATION, CLIMATE_ROTATION_DAYS,
    HARVEST_WINDOW_DAYS, IRRIGATION_COST, SEED_CONFIG,
    STORAGE_CAPACITY, WATER_TANK_CAPACITY, WATER_TANK_INITIAL,
    ClimateState, FarmAction, FarmObservation, FarmState, PlotState,
    MarketPrice,
)


# ── Helper functions ─────────────────────────────────────────────────────────






# ── FarmingEnvironment ───────────────────────────────────────────────────────

class FarmingEnvironment(Environment[FarmAction, FarmObservation, FarmState]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: int = 1):
        super().__init__()
        self.task_id = task_id

        # these will all be set properly by reset()
        self._day:            int              = 0
        self._money:          float            = 0.0
        self._water_tank:     float            = 0.0   # litres
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

    # ── config helpers ───────────────────────────────────────────────────

    def _task_config(self) -> Dict[str, Any]:
        configs = {
            1: {"money": 200.0, "max_days": 30, "drought": False},
            2: {"money": 150.0, "max_days": 45, "drought": False},
            3: {"money": 100.0, "max_days": 60, "drought": True},
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
            temperature=cfg["temp"],
            humidity=cfg["humidity"],
            precipitation=cfg["precip"],
        )

    # ── reset ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FarmObservation:

        if seed is not None:
            random.seed(seed)

        cfg = self._task_config()

        self._day            = 0
        self._money          = cfg["money"]
        self._water_tank     = WATER_TANK_CAPACITY * WATER_TANK_INITIAL
        self._seed_inventory = {"wheat": 0, "rice": 0, "corn": 0}
        self._storage        = {"wheat": 0.0, "rice": 0.0, "corn": 0.0}
        self._drought_active = cfg["drought"]
        self._total_reward   = 0.0
        self._step_count     = 0
        self._episode_id     = episode_id or str(uuid.uuid4())
        self._max_days       = cfg["max_days"]
        self._done           = False

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
        action: FarmAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FarmObservation:
        self._step_count += 1
        self._advance_day()
        obs = self._build_observation(reward=0.0, done=self._day >= self._max_days)
        self._total_reward += 0.0
        return obs

    # ── action handlers ──────────────────────────────────────────────────

    def _do_buy_seeds(self, action: FarmAction) -> tuple[float, str]:
        if action.seed_type is None or action.seed_type not in SEED_CONFIG:
            return -1.0, f"Invalid seed_type: {action.seed_type}"
        qty = action.quantity or 1
        price = self._market_prices[action.seed_type].buy_price * qty
        if price > self._money:
            return -0.5, f"Not enough money: need ${price:.2f}, have ${self._money:.2f}"
        self._money -= price
        self._seed_inventory[action.seed_type] += qty
        return 0.1, ""  # small positive reward for buying

    def _do_plant(self, action: FarmAction) -> tuple[float, str]:
        if action.plot_id is None:
            return -1.0, "plot_id is required for plant"
        if action.seed_type is None or action.seed_type not in SEED_CONFIG:
            return -1.0, f"Invalid seed_type: {action.seed_type}"
        if action.plot_id < 0 or action.plot_id > 3:
            return -1.0, f"Invalid plot_id: {action.plot_id}"

        plot = self._plots[action.plot_id]
        if plot.stage != "empty":
            return -0.5, f"Plot {action.plot_id} is not empty (stage={plot.stage})"

        if self._seed_inventory.get(action.seed_type, 0) <= 0:
            return -0.5, f"No {action.seed_type} seeds in inventory"

        self._seed_inventory[action.seed_type] -= 1
        self._plots[action.plot_id] = PlotState(
            plot_id=action.plot_id,
            crop_type=action.seed_type,
            stage="seedling",
            days_planted=0,
            soil_moisture=plot.soil_moisture,
            health=1.0,
            yield_estimate=0.0,
        )
        return 0.5, ""

    def _do_irrigate(self, action: FarmAction) -> tuple[float, str]:
        if action.plot_id is None:
            return -1.0, "plot_id is required for irrigate"
        if action.plot_id < 0 or action.plot_id > 3:
            return -1.0, f"Invalid plot_id: {action.plot_id}"

        plot = self._plots[action.plot_id]
        if plot.stage in ("empty", "withered"):
            return -0.5, f"Plot {action.plot_id} cannot be irrigated (stage={plot.stage})"

        if self._water_tank < IRRIGATION_COST:
            return -0.5, "Not enough water in tank"

        self._water_tank -= IRRIGATION_COST
        new_moisture = min(1.0, plot.soil_moisture + 0.3)
        self._plots[action.plot_id] = plot.model_copy(update={"soil_moisture": new_moisture})
        return 0.2, ""

    def _do_harvest(self, action: FarmAction) -> tuple[float, str]:
        if action.plot_id is None:
            return -1.0, "plot_id is required for harvest"
        if action.plot_id < 0 or action.plot_id > 3:
            return -1.0, f"Invalid plot_id: {action.plot_id}"

        plot = self._plots[action.plot_id]
        if plot.stage != "mature":
            return -0.5, f"Plot {action.plot_id} is not mature (stage={plot.stage})"

        crop = plot.crop_type
        if crop is None:
            return -1.0, "Internal error: mature plot has no crop_type"

        harvested_kg = _compute_yield_estimate(plot)

        # Check storage capacity
        total_stored = sum(self._storage.values())
        available = STORAGE_CAPACITY - total_stored
        if available <= 0:
            return -0.5, "Storage is full"

        actual_kg = min(harvested_kg, available)
        self._storage[crop] = self._storage.get(crop, 0.0) + actual_kg

        # Reset the plot
        self._plots[action.plot_id] = PlotState(
            plot_id=action.plot_id,
            soil_moisture=max(0.0, plot.soil_moisture - 0.1),
        )

        return 1.0 + actual_kg * 0.1, ""  # reward scales with yield

    def _do_sell(self, action: FarmAction) -> tuple[float, str]:
        seed_type = action.seed_type
        if seed_type is None:
            # Sell all storage
            total_revenue = 0.0
            for crop, kg in list(self._storage.items()):
                if kg > 0 and crop in self._market_prices:
                    revenue = kg * self._market_prices[crop].sell_price
                    total_revenue += revenue
                    self._storage[crop] = 0.0
            if total_revenue == 0:
                return -0.5, "Nothing to sell"
            self._money += total_revenue
            return total_revenue * 0.05, ""
        else:
            if seed_type not in SEED_CONFIG:
                return -1.0, f"Invalid seed_type: {seed_type}"
            qty_kg = float(action.quantity) if action.quantity else self._storage.get(seed_type, 0.0)
            available = self._storage.get(seed_type, 0.0)
            if available <= 0:
                return -0.5, f"No {seed_type} in storage"
            sell_kg = min(qty_kg, available)
            revenue = sell_kg * self._market_prices[seed_type].sell_price
            self._storage[seed_type] -= sell_kg
            self._money += revenue
            return revenue * 0.05, ""

    # ── market price tick ─────────────────────────────────────────────────

    # ── daily cycle ───────────────────────────────────────────────────────

    def _update_market_prices(self) -> None:
        for seed, cfg in SEED_CONFIG.items():
            old = self._market_prices[seed]

            # sine wave period = 20 days, offset per seed to desync them
            offsets = {"wheat": 0, "rice": 7, "corn": 13}
            wave    = math.sin((self._day + offsets[seed]) * 2 * math.pi / 20)
            noise   = random.uniform(-0.05, 0.05)

            # sell price swings ±20% around base
            new_sell  = cfg["base_sell"] * (1.0 + 0.2 * wave + noise)
            new_sell  = max(cfg["base_sell"] * 0.5, new_sell)   # floor at 50% of base

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
                trend=round(trend, 2),
            )

    def _advance_day(self) -> None:
        self._day += 1
        climate = self._current_climate()
        cfg     = CLIMATE_CONFIG[climate.climate_type]

        # 1. refill water tank from precipitation
        precip_litres    = climate.precipitation * 10   # mm → litres (assume 10L per mm)
        self._water_tank = min(
            WATER_TANK_CAPACITY,
            self._water_tank + precip_litres,
        )

        # 2. drought override (task 3): precipitation is 0 on drought days
        if self._drought_active and self._day % 5 == 0:
            # every 5th day is a drought day — no rain, extra moisture decay
            self._water_tank = max(0.0, self._water_tank - 15.0)

        # 3. update each plot: decay moisture, advance growth
        for plot in self._plots:
            if plot.stage == "empty":
                continue

            # moisture decay (climate-dependent)
            plot.soil_moisture = max(
                0.0,
                plot.soil_moisture - cfg["moisture_decay"],
            )

            # health degrades when moisture is critically low
            if plot.soil_moisture < 0.2:
                plot.health = max(0.0, plot.health - 0.1)

            # advance growth counter
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

        # 4. spoilage in storage
        spoilage_rate = cfg["spoilage_rate"]
        for crop in self._storage:
            loss              = self._storage[crop] * spoilage_rate
            self._storage[crop] = max(0.0, self._storage[crop] - loss)

        # 5. update market prices
        self._update_market_prices()

    # ── observation builder ──────────────────────────────────────────────

    def _build_valid_actions(self) -> List[str]:
        actions = ["wait"]
        actions.append("buy_seeds(seed_type=wheat/rice/corn, quantity=N)")

        for plot in self._plots:
            if plot.stage == "empty":
                actions.append(f"plant(plot_id={plot.plot_id}, seed_type=wheat/rice/corn)")
            elif plot.stage == "mature":
                actions.append(f"harvest(plot_id={plot.plot_id})")
            if plot.stage not in ("empty", "withered") and plot.soil_moisture < 0.7:
                actions.append(f"irrigate(plot_id={plot.plot_id})")

        for crop, qty in self._storage.items():
            if qty > 0:
                actions.append(f"sell(seed_type={crop}, quantity=N)")

        return actions

    def _build_text_summary(self) -> str:
        climate = self._current_climate()
        lines   = [
            f"Day {self._day} | Money: ${self._money:.2f} | "
            f"Climate: {climate.climate_type.upper()} "
            f"(temp={climate.temperature}°C, humidity={climate.humidity:.0%}, "
            f"precip={climate.precipitation}mm)",
            f"Water tank: {self._water_tank / WATER_TANK_CAPACITY:.0%} "
            f"({self._water_tank:.1f}L / {WATER_TANK_CAPACITY:.0f}L)",
            "",
            "PLOTS:",
        ]
        for plot in self._plots:
            if plot.stage == "empty":
                lines.append(f"  Plot {plot.plot_id}: empty")
            else:
                seed_cfg  = SEED_CONFIG[plot.crop_type]
                grow_days = int(seed_cfg["grow_days"])
                status    = "READY TO HARVEST" if plot.stage == "mature" else f"day {plot.days_planted}/{grow_days}"
                lines.append(
                    f"  Plot {plot.plot_id}: {plot.crop_type.upper()} | "
                    f"stage={plot.stage} | moisture={plot.soil_moisture:.2f} | "
                    f"health={plot.health:.2f} | {status}"
                )

        lines.append("")
        prices = " ".join(
            f"{s}=${p.sell_price:.2f}(trend={'up' if p.trend > 0.1 else 'dn' if p.trend < -0.1 else '--'})"
            for s, p in self._market_prices.items()
        )
        lines.append(f"MARKET SELL PRICES: {prices}")

        inv = " ".join(f"{s}={q}" for s, q in self._seed_inventory.items())
        lines.append(f"SEED INVENTORY: {inv}")

        store = " ".join(f"{s}={q:.1f}kg" for s, q in self._storage.items())
        lines.append(f"STORAGE: {store}")

        if self._drought_active:
            lines.append("WARNING: drought conditions active")

        return "\n".join(lines)

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
    ) -> FarmObservation:
        return FarmObservation(
            day=self._day,
            money=round(self._money, 2),
            water_tank=round(self._water_tank / WATER_TANK_CAPACITY, 3),
            seed_inventory=dict(self._seed_inventory),
            storage={k: round(v, 2) for k, v in self._storage.items()},
            plots=list(self._plots),
            climate=self._current_climate(),
            market_prices=dict(self._market_prices),
            text_summary=self._build_text_summary(),
            valid_actions=self._build_valid_actions(),
            reward=reward,
            done=done,
        )

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
        )



print("phase 2 completed")
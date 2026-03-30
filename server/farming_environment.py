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
from server.tasks import EpisodeRecord, grade_episode


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

        self._withered_count: int              = 0
        self._healthy_days:   int              = 0
        self._sell_events:    list             = []
        self._last_grade:     float            = 0.0
        self._withered_plots: set              = set()

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

        self._withered_count = 0
        self._healthy_days   = 0
        self._sell_events    = []
        self._last_grade     = 0.0
        self._withered_plots = set()

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
        reward = 0.0

        # validate action object exists
        if action is None:
            action = FarmAction(action_type="wait")

        # dispatch to handler, each returns a float reward delta
        act = action.action_type

        if act == "buy_seeds":
            reward += self._handle_buy_seeds(action)
        elif act == "plant":
            reward += self._handle_plant(action)
        elif act == "irrigate":
            reward += self._handle_irrigate(action)
        elif act == "harvest":
            reward += self._handle_harvest(action)
        elif act == "sell":
            reward += self._handle_sell(action)
        elif act == "wait":
            reward += self._handle_wait()
        else:
            reward += -1.0   # unknown action penalty

        # daily passive rewards/penalties before advancing the day
        reward += self._daily_passive_reward()

        # advance the world by one day
        self._advance_day()

        # count healthy days (at least 2 plots active and healthy)
        healthy_plots = sum(
            1 for p in self._plots
            if p.stage not in ("empty", "withered") and p.health >= 0.6
        )
        if healthy_plots >= 2:
            self._healthy_days += 1

        # post-advance penalties (spoilage already done inside _advance_day)
        reward += self._post_advance_penalties()

        # check done
        done = self._day >= self._max_days or self._money <= 0.0

        if done:
            storage_value = sum(
                self._storage[crop] * self._market_prices[crop].sell_price
                for crop in self._storage
            )
            record = EpisodeRecord(
                task_id=self.task_id,
                initial_money=self._task_config()["money"],
                final_money=self._money,
                storage_value=storage_value,
                total_reward=self._total_reward + reward,
                days_elapsed=self._day,
                max_days=self._max_days,
                withered_count=self._withered_count,
                drought_days=self._day if self._drought_active else 0,
                healthy_days=self._healthy_days,
                sell_events=self._sell_events,
            )
            self._last_grade = grade_episode(record)
            if self._money > 0.0:
                reward += self._terminal_bonus()

        self._total_reward += reward
        obs = self._build_observation(reward=round(reward, 4), done=done)
        return obs

    # ── action handlers ──────────────────────────────────────────────────

    def _handle_buy_seeds(self, action: FarmAction) -> float:
        if action.seed_type is None or action.seed_type not in SEED_CONFIG:
            return -1.0
        if action.quantity is None or action.quantity <= 0:
            return -1.0

        seed  = action.seed_type
        qty   = action.quantity
        cost  = self._market_prices[seed].buy_price * qty

        if cost > self._money:
            return -1.0   # cannot afford

        self._money -= cost
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
        plot.stage          = "seedling"
        plot.days_planted   = 0
        plot.soil_moisture  = 0.5
        plot.health         = 1.0
        plot.yield_estimate = SEED_CONFIG[seed]["yield_kg"]

        return 0.2   # small positive: agent committed to a plan

    def _handle_irrigate(self, action: FarmAction) -> float:
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0

        plot = self._plots[action.plot_id]

        if plot.stage in ("empty", "withered"):
            return -1.0

        if self._water_tank < IRRIGATION_COST:
            return -1.0   # tank empty

        # wasteful irrigation penalty
        if plot.soil_moisture > 0.8:
            self._water_tank -= IRRIGATION_COST
            return -0.5   # used water but didn't need to

        # good irrigation
        self._water_tank       -= IRRIGATION_COST
        plot.soil_moisture      = min(1.0, plot.soil_moisture + 0.3)

        # bonus if crop was in danger (moisture was critically low)
        if plot.soil_moisture < 0.25:
            return 0.5   # rescued a crop
        return 0.1

    def _handle_harvest(self, action: FarmAction) -> float:
        if action.plot_id is None or not (0 <= action.plot_id <= 3):
            return -1.0

        plot = self._plots[action.plot_id]

        if plot.stage != "mature":
            return -1.0   # nothing to harvest

        crop      = plot.crop_type
        yield_kg  = plot.yield_estimate   # already health-adjusted

        # check storage capacity
        current_total = sum(self._storage.values())
        space         = STORAGE_CAPACITY - current_total
        stored_kg     = min(yield_kg, space)
        lost_kg       = yield_kg - stored_kg

        self._storage[crop] = self._storage.get(crop, 0.0) + stored_kg

        # reset plot to empty
        plot.stage          = "empty"
        plot.crop_type      = None
        plot.days_planted   = 0
        plot.soil_moisture  = 0.5
        plot.health         = 1.0
        plot.yield_estimate = 0.0

        # reward: proportion of max possible yield achieved
        max_yield  = SEED_CONFIG[crop]["yield_kg"]
        proportion = stored_kg / max_yield
        reward     = proportion * 1.0   # up to +1.0 per harvest

        if lost_kg > 0:
            reward -= 0.3   # storage overflow penalty

        return round(reward, 4)

    def _handle_sell(self, action: FarmAction) -> float:
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

        price   = self._market_prices[crop].sell_price
        revenue = price * qty

        self._storage[crop] -= qty
        self._money         += revenue

        # reward scales with revenue relative to base price
        base_revenue    = SEED_CONFIG[crop]["base_sell"] * qty
        price_premium   = (revenue - base_revenue) / max(base_revenue, 1.0)
        reward          = 0.3 + price_premium * 0.5
        
        self._sell_events.append({
            "day":        self._day,
            "crop":       crop,
            "qty":        qty,
            "price":      price,
            "base_price": SEED_CONFIG[crop]["base_sell"],
        })

        # selling above base price gives up to +0.8, below gives as low as 0.05
        return round(max(0.05, reward), 4)

    def _handle_wait(self) -> float:
        return -0.05

    def _daily_passive_reward(self) -> float:
        reward = 0.0
        for plot in self._plots:
            if plot.stage in ("seedling", "growing", "mature"):
                reward += 0.1 * plot.health   # up to +0.1 per healthy plot per day
        return round(reward, 4)

    def _post_advance_penalties(self) -> float:
        penalty = 0.0
        for plot in self._plots:
            if plot.stage == "withered" and plot.plot_id not in self._withered_plots:
                penalty -= 5.0
                self._withered_count += 1
                self._withered_plots.add(plot.plot_id)
        return round(penalty, 4)

    def _terminal_bonus(self) -> float:
        cfg          = self._task_config()
        initial      = cfg["money"]

        # value storage at current market prices
        storage_value = sum(
            self._storage[crop] * self._market_prices[crop].sell_price
            for crop in self._storage
        )
        net_worth    = self._money + storage_value
        growth_ratio = net_worth / max(initial, 1.0)

        # bonus scales from 0 at break-even to +10 at 3× growth
        bonus = max(0.0, (growth_ratio - 1.0) * 5.0)
        return round(min(bonus, 10.0), 4)

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
            valid_actions=self._build_valid_actions(),
            reward=reward,
            done=done,
        )
        obs.metadata["grade"] = self._last_grade
        obs.metadata["withered_count"] = self._withered_count
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
        )



print("phase 2 completed")
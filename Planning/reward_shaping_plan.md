# Reward Shaping Plan — Layer 1 (PBRS) + Layer 2 (Market-Adaptive Sell)

## Context
Current reward system uses fixed scalars decided before training — same values on day 1 and day 29,
same regardless of market conditions. Goal: replace hard-coded scalars with formulas that respond to
episode phase and live market state, implementing PBRS and market-adaptive techniques from
`reward engg.md`.

Branch: `a/reward-shaping` off main. Files changed: `server/farming_environment.py` only.
Zero overlap with B's files (`notebooks/`, `README.md`).

---

## Layer 1 — PBRS Phase Multiplier

### Problem
`_daily_passive_reward()` gives flat 0.15/0.12/0.10 per healthy plot every day.
`_handle_wait()` gives flat 0.05 × active_plots patience bonus every day.
Same reward on day 1 (setup) and day 28 (harvest). Agent has no signal to change strategy
by phase.

### Design: `_phase_weights()` helper

```python
def _phase_weights(self) -> tuple[float, float, float]:
    t = self._day / max(self._max_days, 1)          # 0.0 → 1.0
    setup   = max(0.0, 1.0 - t / 0.3)              # 1.0 at day 0, 0 by 30%
    growth  = max(0.1, math.sin(math.pi * t))       # peaks at 50%, floor 0.1
    harvest = max(0.0, (t - 0.7) / 0.3)            # 0 until 70%, 1.0 at end
    return setup, growth, harvest
```

| Phase | Days (Task 1) | setup | growth | harvest |
|-------|--------------|-------|--------|---------|
| Setup | 0–9 | 1.0→0 | 0→0.87 | 0 |
| Growth | 10–21 | 0 | 0.87→1.0→0.87 | 0 |
| Harvest | 22–30 | 0 | 0.87→0 | 0→1.0 |

### Changes

**`_daily_passive_reward()`** — weight by `growth_weight`:
```python
_, growth, _ = self._phase_weights()
reward += daily_bonus * plot.health * growth
```
Effect: passive bonus near-zero at game start/end, peaks mid-game. Removes free
points for coasting at wrong phase.

**`_handle_wait()` patience bonus** — weight by `growth_weight`:
```python
_, growth, _ = self._phase_weights()
return round(0.05 * active_plots * growth * passive_multiplier, 4)
```
Effect: waiting at day 28 gives near-zero patience — agent must harvest/sell.

**`_handle_plant()` setup bonus** — multiply by `(1 + setup_weight)`:
```python
setup, _, _ = self._phase_weights()
return round(base_reward * (1.0 + setup), 4)
```
Effect: planting on day 1 gives ~2× reward of planting on day 25.

---

## Layer 2 — Market-Adaptive Sell Bonus

### Problem
`_handle_sell()` reward = raw revenue only. Selling at a 7-day price high gives
the same REWARD shape as selling at a low. GRPO has no gradient to learn market timing.

### Design: timing bonus on reward signal only (money unchanged)

`_price_history` already stores 7 days. `avg_7d` already on `MarketPrice`.

```python
avg_7d = self._market_prices[crop].avg_7d or execution_price
if len(self._price_history.get(crop, [])) >= 3:
    premium_ratio = (execution_price - avg_7d) / max(avg_7d, 0.01)
    timing_bonus = revenue * max(-0.3, min(0.4, premium_ratio))
else:
    timing_bonus = 0.0   # no signal until 3 days of history
sell_reward = revenue + timing_bonus
```

Clamped to [-30%, +40%] of revenue:
- Sell 20% above 7d avg → reward = 1.20 × revenue
- Sell at avg → reward = 1.00 × revenue
- Sell 15% below avg → reward = 0.85 × revenue

Money received stays `revenue` — only reward signal changes.

---

## Files Modified
- `server/farming_environment.py`
  - Add `_phase_weights()` helper (~8 lines)
  - Modify `_daily_passive_reward()` (~3 lines)
  - Modify `_handle_wait()` (~3 lines)
  - Modify `_handle_plant()` (~3 lines, find current plant reward return)
  - Modify `_handle_sell()` (~8 lines addition)

## Verification
1. `python3 verify_all.py` — all phases must pass
2. `python3 robustness_validation.py` — skill gradient must hold
3. Print `_phase_weights()` at day 0 / 15 / 28 — confirm intuitive shape
4. Sell same qty at high vs low price — confirm timing_bonus sign is correct

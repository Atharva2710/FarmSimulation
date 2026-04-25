# 🧮 FarmSimulation: Mathematical Foundation

This document details the scientific and economic models powering the `FarmSimulation` physics engine. It is designed to provide transparency and verification of the environment's rigor.

---

## 1. 🌍 Climate & Hydrology (FAO-56)

The simulation uses a simplified version of the **FAO-56 Penman-Monteith** methodology to calculate moisture loss and crop water requirements.

### Reference Evapotranspiration ($ET_o$)
The "demand" for water from the atmosphere is calculated daily based on dynamic temperature and humidity:
$$ET_o = \left( \frac{Temp}{100} \right) \cdot (1.1 - Humidity)$$

### Crop Evapotranspiration ($ET_c$)
Each crop has a specific **Crop Coefficient ($K_c$)** that reflects its biological water demand relative to the reference:
$$ET_c = ET_o \cdot K_c$$

| Crop | Wheat | Rice | Corn |
|------|-------|------|------|
| **$K_c$** | 0.80 | 1.10 | 1.20 |

### Soil Moisture Balance
The moisture state of each plot is updated using a mass-balance equation:
$$Moisture_{t+1} = \Phi\left( Moisture_t + \Delta R - ET_c - \omega \right)$$

Where:
- $\Delta R$: Precipitation benefit ($\approx 3\%$ of mm per day).
- $\omega$: Weed transpiration penalty ($0.05$ if weeds present).
- $\Phi$: Clamping function $[0, 1]$.

---

## 2. 💰 Market Dynamics (Almgren-Chriss)

The market model simulates price volatility and the **Permanent Market Impact** of agent actions.

### Price Trajectory (Harmonic Oscillator)
Market prices follow a desynchronized sine wave to simulate seasonal supply/demand cycles:
$$P_{sell}(t) = P_{base} \cdot \left( 1.0 + 0.2 \cdot \sin\left( \frac{2\pi(t + \text{offset})}{20} \right) + \epsilon \right)$$

Where:
- $\epsilon$: Stochastic noise based on task difficulty.
- **Period**: 20 simulation days.

### Market Impact (Elasticity)
Grounded in the **Almgren-Chriss (2000)** model for optimal execution, large sell orders negatively impact price liquidity:
$$\Delta P = \min\left( 0.50, \frac{Quantity}{10} \cdot 0.01 \right)$$
$$P_{new} = P_{old} \cdot (1.0 - \Delta P)$$

This forces agents to manage "Block Trades" carefully to avoid crashing their own revenue.

---

## 3. 🦠 Ecological Escalation

Pests and weeds follow non-linear growth patterns to simulate biological cascades.

### Exponential Pest Growth
If an infestation is not treated with `spray_pesticide`, it escalates following an exponential function:
$$Severity_{t+1} = \min\left( 1.0, (Severity_t + 0.1) \cdot 1.5 \right)$$

### Health Degradation
Crop $Health$ is a state variable representing physiological integrity $[0, 1]$. Damage is accumulated from multiple environmental stressors:
$$Damage = \sum [ \text{NPK Deficiency}, \text{Hydric Stress}, \text{Over-irrigation}, \text{Pest Severity} ]$$

---

## 🏗️ 5-Pass Simulation Order

Each step in the environment advances the world via five ordered logical passes:
1. **Hydrological Recharge**: Precipitation enters the aquifer and water tank.
2. **Pedological Decay**: Soil moisture and nutrients deplete based on $ET_c$ and drain.
3. **Biological Escalation**: Pests and weeds spawn or grow based on climate humidity.
4. **Physiological Update**: Crop health is recalculated based on cumulative stressors.
5. **Economic Tick**: Market prices update and interest/overhead is deducted.

---

*This mathematical rigor ensures that "clever" agents cannot exploit the environment by ignoring scientific constraints.*

import gradio as gr
import os

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');

:root {
    --bg-dark: #09090b;
    --panel-bg: rgba(24, 24, 27, 0.6);
    --border-color: rgba(255, 255, 255, 0.08);
    --accent: #10b981; /* Emerald Green */
    --accent-glow: rgba(16, 185, 129, 0.2);
    --text-main: #e4e4e7;
    --text-muted: #a1a1aa;
}

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: radial-gradient(circle at top, #1e1e24 0%, var(--bg-dark) 100%) !important;
    color: var(--text-main) !important;
}

/* Glassmorphic Section Panels */
.section-box {
    background: var(--panel-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 25px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.3s ease !important;
}

.section-box:hover {
    border-color: rgba(255,255,255,0.15) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6) !important;
}

/* Farm Plots - Interactive Cards */
.farm-plot {
    background: linear-gradient(145deg, rgba(39,39,42,0.8) 0%, rgba(24,24,27,0.95) 100%) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    text-align: center !important;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.05), 0 10px 20px rgba(0,0,0,0.5) !important;
    transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Shine effect on hover */
.farm-plot::before {
    content: '';
    position: absolute;
    top: 0; left: -150%; width: 50%; height: 100%;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.03), transparent);
    transform: skewX(-20deg);
    transition: left 0.7s ease;
}

.farm-plot:hover {
    transform: translateY(-6px) scale(1.02) !important;
    box-shadow: 0 15px 35px var(--accent-glow) !important;
    border-color: rgba(16, 185, 129, 0.5) !important;
}

.farm-plot:hover::before {
    left: 200%;
}

/* Floating animation for Plot Icons */
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

.plot-icon-wrapper {
    animation: float 4s ease-in-out infinite;
    display: inline-block;
}

/* Typography Enhancements */
h1, h2, h3, h4 {
    background: linear-gradient(135deg, #34d399, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    margin-bottom: 0.5rem !important;
}

hr {
    border-color: var(--border-color) !important;
    margin: 15px 0 !important;
}

/* Tables Styling */
table {
    width: 100%;
    border-collapse: separate !important;
    border-spacing: 0 8px !important;
    font-size: 0.9em;
}

th {
    text-align: left;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 1px;
    padding: 0 15px 10px 15px !important;
    border: none !important;
}

td {
    background: rgba(255,255,255,0.02) !important;
    padding: 12px 15px !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease;
}

tr td:first-child { border-radius: 8px 0 0 8px; font-weight: 600; color: #fff;}
tr td:last-child { border-radius: 0 8px 8px 0; }

tr:hover td {
    background: rgba(16, 185, 129, 0.08) !important;
    border-color: rgba(16, 185, 129, 0.2) !important;
    border-left: none !important;
    border-right: none !important;
    color: #fff !important;
}

/* Buttons */
button.gr-button {
    font-family: 'Outfit', sans-serif !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 12px 20px !important;
    text-transform: uppercase;
    font-size: 0.85rem !important;
}

button.gr-button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4) !important;
}

button.gr-button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
}

.gr-button-primary { 
    background: linear-gradient(135deg, #10b981, #059669) !important; 
    color: white !important; 
    box-shadow: 0 4px 15px rgba(16,185,129,0.3) !important;
}

.gr-button-secondary { 
    background: linear-gradient(135deg, #3f3f46, #27272a) !important; 
    color: white !important; 
    border: 1px solid rgba(255,255,255,0.1) !important;
}

.action-btn { 
    background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important; 
    color: white !important; 
    box-shadow: 0 4px 15px rgba(139,92,246,0.3) !important;
}

/* Inputs & Forms */
.gr-input, .gr-dropdown {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: white !important;
}

.gr-input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}
"""

def format_hud(obs, metadata):
    day = obs.day
    max_days = metadata.get('max_days', 30)
    money = obs.money
    water_pct = obs.water_tank * 100
    climate = obs.climate
    climate_type = getattr(climate, "climate_type", "UNKNOWN").upper()
    temp = getattr(climate, "temperature", 0)
    drought_active = metadata.get("drought_active", False)
    
    msg = f"## 📅 DAY {day}/{max_days} &nbsp;&nbsp;|&nbsp;&nbsp; 💰 FUNDS: ${money:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; 💧 TANK: {water_pct:.0f}% &nbsp;&nbsp;|&nbsp;&nbsp; 🌊 AQUIFER: {obs.aquifer:.0f}L\n"
    msg += f"**🌡️ CLIMATE**: <span style='color:#fbbf24'>{climate_type} ({temp}°C)</span>"
    if drought_active:
        msg += " 🔥 <strong style='color:#ef4444'>DROUGHT ACTIVE!</strong>"
    return msg

def format_plot(obs, idx):
    plot = obs.plots[idx]
    stage = getattr(plot, "stage", "empty")
    crop = getattr(plot, "crop_type", "DIRT")
    if not crop: crop = "DIRT"
    
    icon = "🟫"
    if stage == "seedling": icon = "🌱"
    elif stage == "growing": icon = "🌿"
    elif stage == "mature": icon = "🌾"
    elif stage == "withered": icon = "🥀"
    
    moisture = getattr(plot, "soil_moisture", 0) * 100
    health = getattr(plot, "health", 0) * 100
    nitrogen = getattr(plot, "nitrogen", 1.0) * 100
    phosphorus = getattr(plot, "phosphorus", 1.0) * 100
    potassium = getattr(plot, "potassium", 1.0) * 100
    has_weeds = getattr(plot, "has_weeds", False)
    has_pests = getattr(plot, "has_pests", False)
    pest_sev = getattr(plot, "pest_severity", 0.0) * 100
    
    warnings = ""
    if has_weeds:
        warnings += " 🌿"
    if has_pests:
        warnings += f" 🐛({pest_sev:.0f}%)"
    
    return f"""### PLOT {idx}{warnings}
    
<div class="plot-icon-wrapper" style="font-size: 3.5rem; margin:15px 0;">{icon}</div>

**{crop.upper()}**
<span style="color:#a1a1aa; font-size:0.85em; text-transform:uppercase;">{stage}</span>

<div style="margin-top: 15px; text-align: left; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
    <strong>💧 WATER</strong> &nbsp;&nbsp;&nbsp;&nbsp; {moisture:.0f}%<br>
    <strong>✨ HEALTH</strong> &nbsp;&nbsp;&nbsp; {health:.0f}%<br>
    <strong>🌱 N-P-K</strong> &nbsp;&nbsp;&nbsp;&nbsp; {nitrogen:.0f}-{phosphorus:.0f}-{potassium:.0f}
</div>
"""

def format_resources(obs, title, data_dict, unit=""):
    lines = [f"### {title}", "| ITEM | AMOUNT |", "|---|---|"]
    if not data_dict:
        lines.append("| (Empty) | - |")
    for k, v in data_dict.items():
        if isinstance(v, float):
            lines.append(f"| {k.title()} | **{v:.1f}{unit}** |")
        else:
            lines.append(f"| {k.title()} | **{v}{unit}** |")
    return "\n".join(lines)

def format_market(obs):
    lines = ["### 📈 MARKET", "| CROP | PRICE | TREND |", "|---|---|---|"]
    for k, v in obs.market_prices.items():
        trend = "<span style='color:#10b981'>▲ UP</span>" if v.trend > 0 else "<span style='color:#ef4444'>▼ DOWN</span>" if v.trend < 0 else "<span style='color:#a1a1aa'>─ FLAT</span>"
        lines.append(f"| {k.title()} | **${v.sell_price:.2f}** | {trend} |")
    return "\n".join(lines)

def prettify_observation_json(obs):
    """Convert observation to prettified JSON format."""
    import json
    
    # Convert observation to dict (assuming obs has a dict representation or attributes)
    obs_dict = {}
    
    # Extract all observable attributes
    if hasattr(obs, '__dict__'):
        for key, value in obs.__dict__.items():
            if not key.startswith('_'):
                # Handle special types
                if hasattr(value, '__dict__'):
                    # Nested object
                    obs_dict[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
                elif isinstance(value, (list, dict)):
                    obs_dict[key] = value
                else:
                    obs_dict[key] = value
    
    # Pretty print with indentation
    return json.dumps(obs_dict, indent=2, default=str)

def format_action_history(metadata):
    """Format action history as JSON object - organized by day with full state snapshots."""
    history = metadata.get("action_history", [])
    
    if not history:
        return {"message": "No actions taken yet"}
    
    # Organize actions by day (reverse order - newest first)
    history_by_day = {}
    for entry in reversed(history):
        day_key = f"Day {entry['day']}"
        # Include the full state snapshot for each day
        history_by_day[day_key] = entry
    
    # Return as dict with total count
    return {
        "total_days_recorded": len(history),
        **history_by_day  # Unpack day entries at top level
    }

def create_gradio_ui(env_factory):
    with gr.Blocks(title="Farming RL Dashboard", css=custom_css, theme=gr.themes.Monochrome()) as ui:
        gr.Markdown("# 🚜 Farming Intelligence Dashboard")

        with gr.Row():
            with gr.Column(scale=3):
                # Overview Section
                hud_md = gr.Markdown("Loading...", elem_classes=["section-box"])
                
                # Farm Plots Grid
                gr.Markdown("## 🌾 THE FARM", elem_classes=[])
                plot_mds = []
                for row_idx in range(2):
                    with gr.Row():
                        for col_idx in range(2):
                            plot_md = gr.Markdown("Loading plot...", elem_classes=["farm-plot"])
                            plot_mds.append(plot_md)
                
                # Resources Row
                gr.Markdown("## 📦 INVENTORY & ECONOMY", elem_classes=[])
                with gr.Row():
                    seeds_md = gr.Markdown("seeds", elem_classes=["section-box"])
                    storage_md = gr.Markdown("storage", elem_classes=["section-box"])
                    market_md = gr.Markdown("market", elem_classes=["section-box"])
                
                action_feed = gr.Markdown("> 🟢 **SYSTEM READY...** AWAITING COMMAND.", elem_classes=["section-box"])

                status_box = gr.Code(
                    label="📊 Detailed Observation JSON",
                    language="json",
                    lines=15,
                    interactive=False,
                    visible=False,
                )
                show_debug_btn = gr.Button("🔍 Toggle Debug", size="sm")

            # Control Panel
            with gr.Column(scale=1, elem_classes=["section-box"]):
                gr.Markdown("### 🕹️ COMMAND CENTER")
                
                with gr.Row():
                    task_id_input = gr.Dropdown(
                        choices=[("Easy", 1), ("Medium", 2), ("Hard", 3)],
                        value=1,
                        label="Difficulty"
                    )
                    reset_btn = gr.Button("♻️ RESET", variant="secondary")
                
                gr.Markdown("---")
                gr.Markdown("#### ⚡ QUICK ACTIONS")
                with gr.Row():
                    buy_btn = gr.Button("🛒 BUY SEEDS", elem_classes=["action-btn"])
                    wait_btn = gr.Button("⏰ WAIT", elem_classes=["action-btn"])
                with gr.Row():
                    pump_btn = gr.Button("⚙️ PUMP", elem_classes=["action-btn"])
                
                gr.Markdown("---")
                gr.Markdown("#### 🌱 PLOT OPERATIONS")
                plot_selector = gr.Radio(
                    choices=[0, 1, 2, 3],
                    value=0,
                    label="Select Target Plot"
                )
                
                with gr.Row():
                    plant_btn = gr.Button("🌱 PLANT", variant="primary")
                    irrigate_btn = gr.Button("💧 WATER", variant="primary")
                    harvest_btn = gr.Button("🌾 HARVEST", variant="primary")
                with gr.Row():
                    fertilize_btn = gr.Button("🧪 FERTILIZE", variant="primary")
                    spray_btn = gr.Button("🦟 SPRAY", variant="primary")
                    pull_weeds_btn = gr.Button("🤲 WEED", variant="primary")
                
                clear_btn = gr.Button("🧹 CLEAR DEAD", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("#### 📦 RESOURCES")
                seed_type = gr.Dropdown(
                    choices=["wheat", "rice", "corn"],
                    value="wheat",
                    label="Target Crop"
                )
                
                quantity = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Quantity (Buy/Sell)"
                )
                
                sell_btn = gr.Button("💰 SELL CROPS", variant="secondary")
                
                show_history_btn = gr.Button("📜 Toggle Action History", size="sm")
                history_display = gr.JSON(label="Action History (All Days)", visible=False)
                episode_stats = gr.JSON(label="Metadata", visible=False)

        # Output states matching in update function
        all_outputs = [hud_md] + plot_mds + [seeds_md, storage_md, market_md, action_feed, history_display, status_box, episode_stats]

        def get_status():
            env = env_factory()
            obs = env.get_observation()
            metadata = env.get_metadata()
            msg = getattr(env, "_action_message", "")
            
            out_hud = format_hud(obs, metadata)
            out_plots = [format_plot(obs, i) for i in range(4)]
            out_seeds = format_resources(obs, "🎒 SEEDS", obs.seed_inventory)
            out_storage = format_resources(obs, "🌾 STORAGE", obs.storage, "kg")
            out_market = format_market(obs)
            out_msg = f"> {msg}" if msg else "> AWAITING COMMAND..."
            out_history = format_action_history(metadata)
            out_json = prettify_observation_json(obs)
            
            return tuple([out_hud] + out_plots + [out_seeds, out_storage, out_market, out_msg, out_history, out_json, metadata])

        def handle_reset(tid):
            env = env_factory()
            os.environ["FARMING_TASK_ID"] = str(int(tid))
            env.reset(task_id=int(tid))
            return get_status()

        def handle_action(action_type, p_id, qty, s_type):
            env = env_factory()
            action = {"action_type": action_type}
            if action_type in ["plant", "irrigate", "harvest", "clear", "apply_fertilizer", "spray_pesticide", "pull_weeds"]:
                action["plot_id"] = int(p_id)
            if action_type in ["buy_seeds", "sell"]:
                action["seed_type"] = s_type
                action["quantity"] = int(qty)
            if action_type == "plant":
                action["seed_type"] = s_type
            
            env.step(action)
            return get_status()
        
        ui.load(get_status, outputs=all_outputs)
        reset_btn.click(handle_reset, inputs=[task_id_input], outputs=all_outputs)
        wait_btn.click(lambda p, q, s: handle_action("wait", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        buy_btn.click(lambda p, q, s: handle_action("buy_seeds", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        pump_btn.click(lambda p, q, s: handle_action("pump_water", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        plant_btn.click(lambda p, q, s: handle_action("plant", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        irrigate_btn.click(lambda p, q, s: handle_action("irrigate", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        harvest_btn.click(lambda p, q, s: handle_action("harvest", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        clear_btn.click(lambda p, q, s: handle_action("clear", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        fertilize_btn.click(lambda p, q, s: handle_action("apply_fertilizer", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        spray_btn.click(lambda p, q, s: handle_action("spray_pesticide", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        pull_weeds_btn.click(lambda p, q, s: handle_action("pull_weeds", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        sell_btn.click(lambda p, q, s: handle_action("sell", p, q, s), inputs=[plot_selector, quantity, seed_type], outputs=all_outputs)
        show_debug_btn.click(lambda: gr.update(visible=not status_box.visible), outputs=[status_box])
        show_history_btn.click(lambda: gr.update(visible=not history_display.visible), outputs=[history_display])

    return ui

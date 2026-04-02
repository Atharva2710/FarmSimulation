import gradio as gr
import os

def create_gradio_ui(env_factory):
    """
    Creates a Gradio UI for the farming environment.
    Uses the provided factory function to ensure we share the same 
    singleton environment as the FastAPI app.
    """
    
    with gr.Blocks(title="Farming RL Dashboard") as ui:
        gr.Markdown("# 🚜 Farming RL Environment Dashboard")
        gr.Markdown(
            "🎮 **Interactive Farm Simulator** - Manage your farm, grow crops, and maximize profits! "
            "Perfect for testing RL agents or manual gameplay."
        )

        with gr.Row():
            with gr.Column(scale=2):
                summary_md = gr.Markdown("### 🌾 Farm Status", elem_classes=["farm-display"])
                status_box = gr.Textbox(
                    label="📊 Detailed Observation (for debugging)",
                    lines=10,
                    interactive=False,
                    visible=False,  # Hide by default for cleaner UI
                )
                show_debug_btn = gr.Button("🔍 Show/Hide Debug Info", size="sm")
            
            with gr.Column(scale=1, elem_classes=["controls-panel"]):
                gr.Markdown("### 🕹️ Controls")
                
                with gr.Row():
                    reset_btn = gr.Button("♻️ Reset", variant="secondary", scale=2)
                    task_id_input = gr.Dropdown(
                        choices=[("Easy", 1), ("Medium", 2), ("Hard", 3)],
                        value=1,
                        label="Task Difficulty",
                        scale=1
                    )
                
                gr.Markdown("#### 🎯 Quick Actions")
                with gr.Row():
                    wait_btn = gr.Button("⏰ Wait", elem_classes=["action-btn"])
                    buy_btn = gr.Button("🛒 Buy Seeds", elem_classes=["action-btn"])
                
                gr.Markdown("#### 🌱 Plot Actions")
                plot_selector = gr.Radio(
                    choices=[0, 1, 2, 3],
                    value=0,
                    label="Select Plot",
                    type="value"
                )
                
                with gr.Row():
                    plant_btn = gr.Button("🌱 Plant", variant="primary")
                    irrigate_btn = gr.Button("💧 Irrigate", variant="primary")
                    harvest_btn = gr.Button("🌾 Harvest", variant="primary")
                
                clear_btn = gr.Button("🧹 Clear Withered", variant="secondary")

                seed_type = gr.Dropdown(
                    choices=["wheat", "rice", "corn"],
                    value="wheat",
                    label="🌾 Seed Type"
                )
                
                quantity = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Quantity (for buy/sell)"
                )
                
                sell_btn = gr.Button("💰 Sell Crops", variant="secondary")
                
                gr.Markdown("#### 📈 Stats")
                episode_stats = gr.JSON(label="Metadata", visible=False)

        def get_status():
            env = env_factory()
            obs = env.get_observation()
            metadata = env.get_metadata()
            return obs.text_summary, str(obs), metadata

        def handle_reset(tid):
            env = env_factory()
            os.environ["FARMING_TASK_ID"] = str(int(tid))
            # Force recreation of the environment if we are changing task_id
            # In a real singleton this might need adjustment, but here reset handles it.
            env.reset(task_id=int(tid))
            return get_status()

        def handle_action(action_type, p_id, qty, s_type):
            """Handle any action with appropriate parameters."""
            env = env_factory()
            action = {"action_type": action_type}
            if action_type in ["plant", "irrigate", "harvest", "clear"]:
                action["plot_id"] = int(p_id)
            if action_type in ["buy_seeds", "sell"]:
                action["seed_type"] = s_type
                action["quantity"] = int(qty)
            if action_type == "plant":
                action["seed_type"] = s_type
            
            env.step(action)
            return get_status()
        
        def toggle_debug():
            """Toggle debug info visibility."""
            return gr.update(visible=not status_box.visible)

        # Connect event handlers
        ui.load(get_status, outputs=[summary_md, status_box, episode_stats])
        reset_btn.click(handle_reset, inputs=[task_id_input], outputs=[summary_md, status_box, episode_stats])
        
        # Quick action buttons
        wait_btn.click(
            lambda p, q, s: handle_action("wait", p, q, s),
            inputs=[plot_selector, quantity, seed_type],
            outputs=[summary_md, status_box, episode_stats]
        )
        buy_btn.click(
            lambda p, q, s: handle_action("buy_seeds", p, q, s),
            inputs=[plot_selector, quantity, seed_type],
            outputs=[summary_md, status_box, episode_stats]
        )
        
        # Plot action buttons
        plant_btn.click(
            lambda p, q, s: handle_action("plant", p, q, s),
            inputs=[plot_selector, quantity, seed_type],
            outputs=[summary_md, status_box, episode_stats]
        )
        irrigate_btn.click(
            lambda p, q, s: handle_action("irrigate", p, q, s),
            inputs=[plot_selector, quantity, seed_type],
            outputs=[summary_md, status_box, episode_stats]
        )
        harvest_btn.click(
            lambda p, q, s: handle_action("harvest", p, q, s),
            inputs=[plot_selector, quantity, seed_type],
            outputs=[summary_md, status_box, episode_stats]
        )
        clear_btn.click(
            lambda p, q, s: handle_action("clear", p, q, s),
            inputs=[plot_selector, quantity, seed_type],
            outputs=[summary_md, status_box, episode_stats]
        )
        
        # Sell button
        sell_btn.click(
            lambda p, q, s: handle_action("sell", p, q, s),
            inputs=[plot_selector, quantity, seed_type],
            outputs=[summary_md, status_box, episode_stats]
        )
        
        # Debug toggle
        show_debug_btn.click(
            lambda: gr.update(visible=not status_box.visible),
            outputs=[status_box]
        )

    return ui

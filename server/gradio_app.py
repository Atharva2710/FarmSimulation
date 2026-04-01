import gradio as gr
import os

def create_gradio_ui(env_factory):
    """
    Creates a Gradio UI for the farming environment.
    Uses the provided factory function to ensure we share the same 
    singleton environment as the FastAPI app.
    """
    
    with gr.Blocks(title="Farming RL Dashboard", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🚜 Farming RL Environment Dashboard")
        gr.Markdown(
            "This dashboard allows you to monitor and interact with the reinforcement "
            "learning environment manually. The same instance is available via API "
            "for RL agents."
        )

        with gr.Row():
            with gr.Column(scale=2):
                summary_md = gr.Markdown("### Current Farm Status")
                status_box = gr.Textbox(
                    label="Internal Observation",
                    lines=15,
                    interactive=False,
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🕹️ Manual Controls")
                with gr.Row():
                    reset_btn = gr.Button("♻️ Reset Environment", variant="secondary")
                    task_id_input = gr.Number(value=1, label="Task ID", precision=0)
                
                gr.Markdown("#### Take Action")
                action_type = gr.Dropdown(
                    choices=["wait", "buy_seeds", "plant", "irrigate", "harvest", "sell"],
                    value="wait",
                    label="Action Type"
                )
                
                with gr.Row():
                    plot_id = gr.Number(value=0, label="Plot ID (0-3)", precision=0)
                    quantity = gr.Number(value=1, label="Quantity", precision=0)
                    seed_type = gr.Dropdown(
                        choices=["wheat", "rice", "corn"],
                        value="wheat",
                        label="Seed Type"
                    )
                
                step_btn = gr.Button("➡️ Perform Action", variant="primary")
                
                gr.Markdown("#### Episode Stats")
                episode_stats = gr.JSON(label="Last Step Metadata")

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
            env.reset()
            return get_status()

        def handle_step(a_type, p_id, qty, s_type):
            env = env_factory()
            action = {"action_type": a_type}
            if a_type in ["plant", "irrigate", "harvest"]:
                action["plot_id"] = int(p_id)
            if a_type in ["buy_seeds", "sell"]:
                action["seed_type"] = s_type
                action["quantity"] = int(qty)
            if a_type == "plant":
                action["seed_type"] = s_type
            
            env.step(action)
            return get_status()

        # Connect event handlers
        ui.load(get_status, outputs=[summary_md, status_box, episode_stats])
        reset_btn.click(handle_reset, inputs=[task_id_input], outputs=[summary_md, status_box, episode_stats])
        step_btn.click(handle_step, inputs=[action_type, plot_id, quantity, seed_type], outputs=[summary_md, status_box, episode_stats])

    return ui

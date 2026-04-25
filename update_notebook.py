import json

with open("notebooks/train_grpo_unsloth.ipynb", "r") as f:
    nb = json.load(f)

# Update cell 6: add write_journal to _VALID_ACTION_TYPES
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "Cell 6: Action Parser" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "buy_plot\", \"clear\", \"end_day\"," in line:
                new_source.append("    \"buy_plot\", \"clear\", \"write_journal\", \"end_day\",\n")
            else:
                new_source.append(line)
        cell["source"] = new_source

# Update cell 7: add write_journal to SYSTEM_PROMPT
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "Cell 7: Reward Functions" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "pull_weeds, buy_plot, clear, end_day.\"\"\"" in line:
                new_source.append(line.replace("clear, end_day.", "clear, write_journal, end_day."))
            else:
                new_source.append(line)
        cell["source"] = new_source

# Update cell 9: ensure max_steps is 50
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "Cell 9: GRPOConfig" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "max_steps=" in line:
                new_source.append("    max_steps=50,\n")
            elif "output_dir=" in line:
                new_source.append("    output_dir=\"grpo_farm_qwen_0_5b_gen2\",\n")
            else:
                new_source.append(line)
        cell["source"] = new_source

# Update cell 12: Gen 2 saving and pushing
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "Cell 12: Save and Push to Hub" in "".join(cell["source"]):
        cell["source"] = [
            "# Cell 12: Save and Push to Hub\n",
            "HF_USERNAME = \"your_hf_username\"\n",
            "REPO_ID = f\"{HF_USERNAME}/qwen-0.5b-farmsim-grpo-gen2\"\n",
            "\n",
            "print(\"Saving locally to grpo_farm_qwen_0_5b_gen2_final...\")\n",
            "model.save_pretrained(\"grpo_farm_qwen_0_5b_gen2_final\")\n",
            "tokenizer.save_pretrained(\"grpo_farm_qwen_0_5b_gen2_final\")\n",
            "\n",
            "try:\n",
            "    model.push_to_hub(REPO_ID, use_auth_token=True)\n",
            "    tokenizer.push_to_hub(REPO_ID, use_auth_token=True)\n",
            "    print(f\"Successfully pushed Gen 2 adapter to Hub! URL: https://huggingface.co/{REPO_ID}\")\n",
            "except Exception as e:\n",
            "    print(\"Push failed:\", e)\n"
        ]

with open("notebooks/train_grpo_unsloth.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

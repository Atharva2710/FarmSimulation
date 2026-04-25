# Live Status (last updated H+2 by A)

## Person A — Ashdeep
- Branch: a/ws2
- DONE: WS1.1 narrative observation layer (green verification)
- DONE: WS1.2 narrative content (4 components: weather, market, logs, gossip)
- DONE: WS1.3 Robustness validation (skill gradient PASSED) + audit verify fix + MCP name check
- HANDOFF #1: merged to main ✅
- DONE: WS2 economy hardening (patience cap, Almgren-Chriss slippage, binary competence gate)
- DONE: WS3 RubricComposer (Dimension/Gate/RubricComposer classes, 3 per-task rubrics, grade_episode_detailed(), openenv.yaml composite grader blocks)
- HANDOFF #2 pending: need a/ws2 + a/ws3 PR merged → main
- Next: HANDOFF #2+#3 → then re-baseline pivoted env

## Person B — Vivek
- Branch: b/ws4-notebook (rebased on main HEAD `0c701e1`)
- DONE: H+0–1 — cells 1-4 green on Colab T4. Pin matrix locked: unsloth 2026.4.x · trl 0.22.2 · transformers 4.56.2 · huggingface_hub 1.12.0 · torch 2.10.0+cu128. lora_dropout=0.05 (plan §20.3, accepts ~30-50% Unsloth fast-path penalty).
- In progress: H+1 drafting cells 6 (parse_action) + 8 (prompt_dataset) + 9 (GRPOConfig) + 10 (GRPOTrainer scaffold). All independent of WS2/WS3.
- Blocked on: HANDOFF #2 (A's WS2 merge) for cells 5 (FarmEnvClient) and 7 (reward functions).
- Next: post-WS2, fill cells 5/7 with live env URL + reward fn bodies; post-WS3, wire RubricComposer into reward functions.

## Reward Engineering Reference (from IMPLEMENTATION_PLAN.md §21.5)
- Source: Planning/reward engg.md (9 named techniques + future-work roadmap)
- USE NOW: Hybrid Multi-Objective (Exec+Sim+Pref), Curriculum Schedule, Gated Thinking
- STRETCH (H 22-24 slack only): PBRS via Φ(state) = optimal_sell_value - current_inventory_value
- DEFER (cite in README future work): PRMs, EXPLORS, BiPaRS, RUNE, IRD, Text2Reward
- Rule: any reward edit must map to a named technique; cite in PR description

## H-0 Promises (both)
1. "I will not push to main directly. PR + merge only."
2. "I will rebase on main at the start of every session and after every PR I see merged."
3. "I will post a status ping every hour."

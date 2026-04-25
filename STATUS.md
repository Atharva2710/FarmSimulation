# Live Status (last updated H+0 by A)

## Person A — Ashdeep
- Branch: `a/ws1`
- In progress: WS1.1 narrative observation layer scaffold
- Blocked on: nothing
- Next: WS1.2 narrative content (4 components: journal, weather report with 25% lie rate, neighbor email, market gossip)

## Person B — Vivek
- Branch: `b/ws4-notebook`
- In progress: TBD (Vivek to update)
- Blocked on: A's HANDOFF #1 at H+5
- Next: notebook cells 1-4 (Unsloth model load + LoRA wiring per `IMPLEMENTATION_PLAN.md` §20.3)

## Reward Engineering Reference (from `IMPLEMENTATION_PLAN.md` §21.5)
- Source: `Planning/reward engg.md` (9 named techniques + future-work roadmap)
- USE NOW: Hybrid Multi-Objective (Exec+Sim+Pref), Curriculum Schedule, Gated Thinking
- STRETCH (H 22-24 slack only): PBRS via `Φ(state) = optimal_sell_value − current_inventory_value`
- DEFER (cite in README future work): PRMs, EXPLORS, BiPaRS, RUNE, IRD, Text2Reward
- Rule: any reward edit must map to a named technique; cite in PR description.

## H-0 Promises (both)
1. I will not push to `main` directly. PR + merge only (after this STATUS.md commit).
2. I will rebase on `main` at the start of every Claude Code session and after every PR merged.
3. I will post a status ping every hour in the team channel.

## HANDOFF Sequence (`IMPLEMENTATION_PLAN.md` §4)
- HANDOFF #1 at H+5  — A's `a/ws1` merged → B can use real `narrative_text`
- HANDOFF #2 at H+7  — A's `a/ws2` merged → B trains on hardened economy
- HANDOFF #3 at H+9  — A's `a/ws3` merged → B uses RubricComposer scores
- HANDOFF #4 at H+22 — B's training done → A re-runs robustness with trained tier

## Pre-flight notes
- Local working tree had 80 line-ending-only modifications (CRLF↔LF churn). Stashed pre-H0 (`git stash list` to recover; no content was changed).
- `core.autocrlf=input` configured to prevent recurrence on this WSL/Windows-drive setup.

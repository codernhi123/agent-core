# TB2 split shared by both arms (reef Meta-Harness vs agent-core RSI)

Terminal-Bench 2 @ 69671fbaac6d67a7ef0dfec016cc38a64ef7a77c, 89 tasks.

| File | Tasks | Use |
|---|---|---|
| search_tasks_20.txt | 20 | **Search set actually used** by both optimizers |
| heldout_tasks.txt | 59 | Held-out scoring only (never seen during search) |
| search_tasks.txt | 30 | Original search pool (seed 20260928, stratified by difficulty); the 10 not in search_tasks_20 are unused |

search_tasks_20 = category-diverse subset of search_tasks (all 11 categories kept; near-duplicates and 60-min tasks dropped).

## Matched settings (both arms)
- Harbor 0.20.0, docker env, task.toml timeouts/resources unchanged
- Qwen 27B via llama.cpp, one server per arm, identical flags: --parallel 4 --ctx-size 393216 (98,304 tokens/slot)
  - reef: GPU 6, port 8081 — agent-core: GPU 7, port 8082
- Thinking off, temperature 0.7, 4 concurrent episodes per arm
- Edit surface: prompt/rules + skills only
- Search budget: ≤ 492 target episodes (reef: 20 tasks × 2 repeats × 2 harnesses × 6 iterations = 480 gate + ≤ 12 feedback rollouts; agent-core: hard cap 492)
- Held-out: 59 tasks × (seed, final) × 2 trials = 236 episodes
- Episode = Harbor trial where the agent started; missing reward = 0
- Held-out CSV: arm,harness,task,trial,reward,exception_type,agent_started

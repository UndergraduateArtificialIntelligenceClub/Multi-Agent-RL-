# 2026-11-10: Independent PPO (IPPO) baseline on Pursuit (Willy)

**Setup:** `sisl/pursuit-v5` defaults (8 pursuers, 30 evaders, 7×7 local obs, 500 cycles).
8 *separate* actor-critic CNNs, one per pursuer, each with its own Adam optimizer and trained
only on its own trajectories. No parameter sharing and no centralized critic.

**Code:** `src/phase1_pursuit/ippo.py` (train), `src/phase1_pursuit/eval_ippo.py` (eval + plots)

```
python -m phase1_pursuit.ippo --total-steps 1500000 --run-name ippo-baseline
python -m phase1_pursuit.eval_ippo --run-name ippo-baseline --episodes 20
```

**Key hyperparameters:** 16 envs × 128 steps per update, 1.5M env steps (732 updates),
lr 2.5e-4 (linear decay), γ 0.99, GAE λ 0.95, clip 0.2, 4 epochs × 4 minibatches,
entropy 0.01, value coef 0.5, grad-norm 0.5, seed 1. Ran on an RTX 3050 Ti (CUDA) at
~650 env steps/s, 38 min wall-clock.

## Results (20 eval episodes, seeds 0–19, same seeds as `random_baseline`)

| Policy            | Team reward / episode | Evaders captured (of 30) |
|-------------------|-----------------------|--------------------------|
| Random            | −363.1 ± 13.7         | 1.3 ± 1.3                |
| IPPO (sampled)    | −76.9 ± 44.9          | 21.5 ± 2.6               |
| IPPO (greedy)     | **−24.9 ± 36.4**      | **22.1 ± 2.3**           |

- The learning curve rises fast in the first ~200k steps (0 → ~14 captures), then improves
  slowly and noisily. It **had not plateaued** at 1.5M steps.
- The training curve is noisy (for example a dip around 950k steps). That's expected when
  each agent's environment includes 7 other learners whose policies keep changing
  (non-stationarity).
- With the default config, all pursuers received identical episode returns, so the reward
  is effectively shared. Each agent gets the same signal no matter what it contributed,
  which is the credit-assignment problem.

Plots (local, gitignored): `outputs/ippo-baseline/learning_curve.png`, `eval_comparison.png`.
Checkpoint (local, gitignored): `checkpoints/phase1_ippo-ippo-baseline_2026-11-10.pt`.
W&B: not used for this run.

**Takeaway:** fully independent PPO beats random by a wide margin (≈17× more captures). It
gives us a concrete number for parameter-sharing PPO and MAPPO to beat at an equal step
budget.

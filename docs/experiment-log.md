# Experiment Log

Running record of what we tried and what happened. Newest entries at the top. Short is fine.

Template:

```
## YYYY-MM-DD: short title (your name)
- Tried:
- Why:
- Result:
- Next:
```

---

## 2026-11-10: Independent PPO baseline (Willy)
- Tried: IPPO, 8 separate PPO networks (one per pursuer, no parameter sharing), 1.5M env steps.
- Why: the independent-learner side of the independent vs. centralized-training comparison, and a first trained baseline.
- Result: greedy policy 22.1 / 30 evaders captured per episode vs 1.3 for random; team reward −24.9 vs −363.1. Still improving at the end. Details: `experiments/2026-11-10-ippo-baseline.md`.
- Next: same budget with parameter-sharing PPO (SB3 + SuperSuit) and MAPPO, multiple seeds.

# Phase 9 Benchmark Report

## Protocol
- World-model data budget (steps): 64
- Baselines trained with fixed timesteps per seed
- Metrics aggregated across seeds

## Results
- CEM: mean cumulative reward=-483.651097, std=0.000000, mean reward/step=-7.55704839
- MCTS: mean cumulative reward=-484.074761, std=0.000000, mean reward/step=-7.56366813
- PPO: mean cumulative reward=-810.247231, std=0.000000, mean reward/step=-0.81024723
- SAC: mean cumulative reward=-919.348322, std=0.000000, mean reward/step=-0.91934832

## Notes
- Higher cumulative reward is better.
- Reward/step provides a sample-efficiency style comparison.
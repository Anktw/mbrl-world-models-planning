# Phase 9 Benchmark Report

## Protocol
- World-model data budget (steps): 64
- Baselines trained with fixed timesteps per seed
- Metrics aggregated across seeds

## Results
- CEM: mean cumulative reward=-3555.659525, std=162.325921, mean reward/step=-55.55718008
- MCTS: mean cumulative reward=-3883.630039, std=117.380167, mean reward/step=-60.68171935
- PPO: mean cumulative reward=-3563.969210, std=795.903417, mean reward/step=-0.71279384
- SAC: mean cumulative reward=-409.235431, std=143.698910, mean reward/step=-0.08184709

## Notes
- Higher cumulative reward is better.
- Reward/step provides a sample-efficiency style comparison.
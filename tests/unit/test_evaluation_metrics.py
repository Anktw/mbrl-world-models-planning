from mbrl.evaluation.metrics import cumulative_reward, sample_efficiency


def test_cumulative_reward() -> None:
    rewards = [1.0, -0.5, 2.5]
    assert cumulative_reward(rewards) == 3.0


def test_sample_efficiency_curve() -> None:
    metrics = sample_efficiency(collected_steps=[10, 20, 30], executed_rewards=[1.0, 2.0, 3.0])
    assert metrics.cumulative_collected_steps == [10, 30, 60]
    assert metrics.cumulative_executed_rewards == [1.0, 3.0, 6.0]
    assert len(metrics.reward_per_step_curve) == 3
    assert metrics.reward_per_step_curve[-1] == 0.1

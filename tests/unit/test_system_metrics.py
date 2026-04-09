from pathlib import Path

from mbrl.training.system_trainer import CycleMetrics, FullSystemTrainer


def test_metrics_file_init_and_append(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    FullSystemTrainer._init_metrics_file(metrics_path)

    metrics = CycleMetrics(
        cycle=1,
        collected_steps=64,
        average_collection_reward=-1.0,
        vae_loss=0.1,
        rssm_loss=0.2,
        reward_loss=0.3,
        cem_predicted_return=1.0,
        mcts_predicted_return=0.5,
        executed_reward=-0.2,
    )
    FullSystemTrainer._append_metrics(metrics_path, metrics)

    content = metrics_path.read_text(encoding="utf-8")
    assert "cycle,collected_steps" in content
    assert "1,64" in content

from pathlib import Path

from mbrl.config.base import load_config


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
env:
  name: Pendulum-v1
  obs_shape: [3]
  action_dim: 1
training:
  seed: 11
  batch_size: 8
data:
  capacity: 100
  min_size: 8
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.env.name == "Pendulum-v1"
    assert config.env.obs_shape == (3,)
    assert config.training.seed == 11
    assert config.training.batch_size == 8
    assert config.data.capacity == 100

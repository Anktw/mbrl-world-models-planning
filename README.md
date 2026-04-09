# mbrl-world-models-planning

Production-grade Model-Based Reinforcement Learning system with learned world models (VAE + RSSM) and planning (MCTS, CEM) for sample-efficient decision making in continuous control environments.

## Engineering Principles

- Modular code only (no notebook-centric pipeline logic)
- Strict separation between models, training, inference/planning, and evaluation
- PyTorch-first implementation
- Phase-wise delivery with validation before moving forward

## Phase Workflow

Each phase includes:

1. Brief architecture explanation
2. Modular implementation
3. Validation tests and runnable checks
4. Expected outputs

## Current Status

### Phase 1: Foundation (Completed)

Implemented in this phase:

- Clean project/package structure under `src/mbrl`
- Typed config stack (Pydantic + YAML)
- Replay buffer and dataset utilities
- Reproducibility and device/logging utilities
- Smoke runner to validate end-to-end wiring
- CI quality gates with Ruff, MyPy, and PyTest

### Phase 2: Data Collection (Completed)

Implemented in this phase:

- Gymnasium environment wrapper with `reset()` and `step()`
- Random policy for bootstrap collection
- Random rollout collector that fills the replay buffer
- Replay buffer persistence to disk via `.npz`
- Validation script that prints sample transitions and tensor shapes

Run the data collection pipeline:

```powershell
python -m scripts.collect_data --env Pendulum-v1 --steps 128 --output artifacts/replay_buffer.npz
```

Expected output includes:

- Sample transition prints
- Observation/action/reward shapes
- Collection counts
- Saved replay buffer path

### Phase 3: VAE (Completed)

Implemented in this phase:

- CNN encoder that outputs latent mean and log-variance
- CNN decoder that reconstructs RGB observations
- Reparameterization trick
- Reconstruction and KL loss
- Replay-buffer-driven VAE trainer
- Reconstruction artifact output for inspection

Run the VAE pipeline:

```powershell
python -m scripts.collect_vae_data --env Pendulum-v1 --steps 24 --capacity 64 --image-size 64 --output artifacts/vae_buffer.npz
python -m scripts.train_vae --buffer artifacts/vae_buffer.npz --epochs 3 --batch-size 8 --steps-per-epoch 4 --checkpoint artifacts/vae_model.pt --reconstructions artifacts/vae_reconstructions.png
```

Expected output includes:

- Sample pixel transition prints
- Epoch loss values decreasing
- Saved VAE checkpoint
- Saved reconstruction image
- Latent vector shape printout

### Phase 4: RSSM (Completed)

Implemented in this phase:

- GRU-based recurrent state model with deterministic and stochastic state
- Latent rollout prediction from current latent and action sequences
- Sequence sampler from the replay buffer
- RSSM trainer that compares predicted vs actual latent trajectories
- Latent comparison artifact for validation

Run the RSSM pipeline:

```powershell
python -m scripts.train_rssm --buffer artifacts/vae_buffer.npz --vae-checkpoint artifacts/vae_model.pt --epochs 3 --batch-size 8 --sequence-length 8 --steps-per-epoch 4 --checkpoint artifacts/rssm_model.pt --comparison artifacts/rssm_latent_comparison.png
```

Expected output includes:

- Epoch loss values decreasing
- Saved RSSM checkpoint
- Saved latent comparison plot
- Predicted latent shape and actual latent shape printout

### Phase 5: Reward Predictor (Completed)

Implemented in this phase:

- Lightweight MLP reward predictor that maps latent z to scalar reward
- Supervised training loop with MSE loss
- Reward validation plot comparing predicted vs actual rewards

Run reward predictor training:

```powershell
python -m scripts.train_reward --buffer artifacts/vae_buffer.npz --vae-checkpoint artifacts/vae_model.pt --epochs 5 --batch-size 16 --steps-per-epoch 8 --checkpoint artifacts/reward_model.pt --comparison artifacts/reward_comparison.png
```

Expected output includes:

- Epoch reward loss values
- Saved reward checkpoint
- Saved prediction-vs-actual plot
- Printed predicted vs actual reward samples

### Phase 6: Planning Module (Completed)

Implemented in this phase:

- CEM planner that samples latent action sequences, evaluates predicted outcomes, and refits elites
- MCTS planner with latent rollouts and reward-guided tree search
- Planner comparison script printing chosen actions and predicted returns

Run planner comparison:

```powershell
python -m scripts.plan --buffer artifacts/vae_buffer.npz --vae-checkpoint artifacts/vae_model.pt --rssm-checkpoint artifacts/rssm_model.pt --reward-checkpoint artifacts/reward_model.pt --horizon 8 --cem-samples 128 --cem-iterations 5 --mcts-simulations 100
```

Expected output includes:

- Chosen action from CEM
- Chosen action from MCTS
- Predicted return for both planners
- Winner based on predicted return

### Phase 7: Full Training Loop (Completed)

Implemented in this phase:

- Config-driven end-to-end loop: collect -> train VAE -> train RSSM -> train reward -> plan
- Periodic cycles with optional buffer refresh interval
- Metrics logging to CSV with losses and reward signals per cycle

Run full system training:

```powershell
python -m scripts.train_system --config configs/default.yaml --cycles 2 --collect-steps 32 --bootstrap-steps 64 --steps-per-epoch 4 --artifacts-dir artifacts/full_loop
```

Expected output includes:

- Per-cycle logs for VAE, RSSM, reward losses
- Planner predicted returns and executed rewards
- Final cycle summary line
- Metrics file at artifacts/full_loop/system_metrics.csv

### Phase 8: Evaluation (Completed)

Implemented in this phase:

- Clear evaluation metrics: cumulative reward, sample efficiency, prediction error
- Planner comparison: CEM vs MCTS
- Visualization suite: planner rewards, sample efficiency, prediction errors, latent space
- Markdown evaluation report generation
- Optional PPO/SAC note pathway in report

Run evaluation:

```powershell
python -m scripts.evaluate --config configs/default.yaml --buffer artifacts/vae_buffer.npz --vae-checkpoint artifacts/vae_model.pt --rssm-checkpoint artifacts/rssm_model.pt --reward-checkpoint artifacts/reward_model.pt --metrics-csv artifacts/full_loop/system_metrics.csv --artifacts-dir artifacts/evaluation --episode-horizon 200
```

Expected output includes:

- Evaluation summary with planner rewards and prediction errors
- Graph files in artifacts/evaluation
- Evaluation report at artifacts/evaluation/evaluation_report.md

### Phase 9: PPO/SAC Benchmarking (Completed)

Implemented in this phase:

- Reproducible benchmark protocol across multiple seeds
- Direct method comparison: CEM, MCTS, PPO, SAC
- Metrics: cumulative reward and reward-per-step efficiency
- Benchmark outputs: CSV, comparison plot, markdown report

Install optional baseline dependencies:

```powershell
pip install -e .[baseline]
```

Run baseline benchmark:

```powershell
python -m scripts.benchmark_baselines --config configs/default.yaml --buffer artifacts/vae_buffer.npz --vae-checkpoint artifacts/vae_model.pt --rssm-checkpoint artifacts/rssm_model.pt --reward-checkpoint artifacts/reward_model.pt --metrics-csv artifacts/full_loop/system_metrics.csv --seeds 7,13,23 --eval-episodes 3 --episode-horizon 200 --baseline-timesteps 5000 --artifacts-dir artifacts/baseline_benchmark
```

Expected output includes:

- Per-method aggregate reward summary
- Benchmark CSV at artifacts/baseline_benchmark/baseline_benchmark_runs.csv
- Comparison plot at artifacts/baseline_benchmark/baseline_benchmark.png
- Markdown report at artifacts/baseline_benchmark/baseline_benchmark_report.md

## Project Structure

```text
src/mbrl/
	config/
	data/
	models/
	planners/
	training/
	inference/
	evaluation/
	utils/
scripts/
	smoke.py
configs/
	default.yaml
tests/
	unit/
	integration/
```

## Step-by-Step Run

### 1. Install dependencies

```powershell
pip install -e .[dev]
```

### 2. Run tests

```powershell
pytest
```

### 3. Run lint and type checks

```powershell
ruff check .
mypy src
```

### 4. Run Phase 1 smoke validation

```powershell
python -m scripts.smoke --config configs/default.yaml
```

Expected smoke output includes:

- Loaded config line with environment and device
- Sampled batch shapes
- `PHASE_1_SMOKE_SUCCESS`

## Next Phase

Phase 10 can target hyperparameter sweeps, ablations, and production deployment hardening.

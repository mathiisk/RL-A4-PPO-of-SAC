## Dependecies
```bash
pip install -r requirements.txt
```

## Files
```
Config.py      — hyperparameters (PPOConfig dataclass)
PPOAgent.py    — RolloutBuffer, PPOAgent, train_PPO
Networks.py    — PolicyNetwork, ValueNetwork, QValueNetwork
Experiment.py  — experiment definitions and training loop
Helpers.py     — LearningCurvePlot, smooth
Plot.py        — standalone plot script for saved JSON results
```

## Run an experiment
```bash
python Experiment.py <experiment_name>
```

Available experiments:
```
exp_ppo             — baseline PPO run
abl_ppo_clip        — clip epsilon ablation (0.1, 0.2, 0.3)
abl_ppo_epochs      — update epochs ablation (2, 4, 8)
abl_ppo_lr          — learning rate ablation (1e-4, 3e-4, 1e-3)
abl_ppo_nsteps      — rollout length ablation (64, 128, 256)
abl_ppo_entropy     — entropy coefficient ablation (0.0, 0.01, 0.05)
ppo_recovers_a2c    — PPO vs A2C-like variants
all_ablations       — runs all ablations above
```

Ablation experiments use `total_steps=500_000` and `num_rep=3`. The main `exp_ppo` uses the full config defaults (1M steps, 5 reps).

## Results
Each run saves to `results/`:
- `<name>.png` — learning curve plot
- `<name>_results.json` — mean/std returns per eval step, plus params

## Re-plot from saved JSON
```bash
python Plot.py --files results/ablation_ppo_clip_results.json \
               --title "Clip Ablation" \
               --out clip_plot.png \
               --smooth 11
```

Optional flags: `--labels` to select a subset of curves, `--ylim 0 520`.

## Config
Edit `Config.py` to change defaults (lr, n_steps, clip_eps, etc.). Ablation overrides are defined inline in `Experiment.py` via the `ABLATION` dict.

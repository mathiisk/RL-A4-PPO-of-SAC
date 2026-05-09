import os 
import numpy as np
import torch
import time
import json
import argparse

from Config import PPOConfig
from Helpers import smooth, LearningCurvePlot
from PPOAgent import train_PPO



# shared ablation overrides, less reps and steps
ABLATION = {"total_steps": 500_000, "num_rep": 3}

# dir to store json results for plotting
RESULTS_DIR = "results"



def train_one_run(agent_name, params, device):
    if agent_name == "PPO":
        return train_PPO(params, device)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def average_returns(agent_name, params, device):
    all_runs = []
    start_time = time.time()

    for i in range(params.num_rep):
        print(f"  Repetition {i+1}/{params.num_rep}")
        run_returns = train_one_run(agent_name, params, device)
        all_runs.append(run_returns)

    min_len = min(len(r) for r in all_runs)
    all_runs = np.array([r[:min_len] for r in all_runs])

    mean_curve = np.mean(all_runs, axis=0)
    std_curve  = np.std(all_runs,  axis=0)

    if params.smoothing_window is not None and min_len > params.smoothing_window:
        mean_curve = smooth(mean_curve, params.smoothing_window)
        std_curve  = smooth(std_curve,  params.smoothing_window)

    print(f"  Took {(time.time() - start_time) / 60:.1f} minutes")
    return mean_curve, std_curve


def run_experiment(experiments, base_params, device, title, save_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    results = []
    plot = LearningCurvePlot(title=title)
    plot.set_ylim(0, 520)
 
    for exp in experiments:
        label      = exp["label"]
        agent_name = exp["agent_name"]
        overrides  = exp["params"]
 
        print(f"\nRunning: {label}")
        params_dict = base_params.__dict__.copy()
        params_dict.update(overrides)
        params = PPOConfig(**params_dict)
 
        mean_curve, std_curve = average_returns(agent_name, params, device)
        timesteps = list(range(
            params.evaluate_every,
            params.total_steps + 1,
            params.evaluate_every,
        ))[:len(mean_curve)]
 
        plot.add_curve(timesteps, mean_curve, std=std_curve,
                       label=f"{label} (±{std_curve[-1]:.1f})")
        results.append({
            "label":        label,
            "params":       params_dict,
            "mean_returns": mean_curve.tolist(),
            "std_returns":  std_curve.tolist(),
        })
 
    plot.add_hline(500, label="Optimal (500)")
 
    plot_path = os.path.join(RESULTS_DIR, f"{save_name}.png")
    json_path = os.path.join(RESULTS_DIR, f"{save_name}_results.json")
 
    plot.save(plot_path)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {plot_path}  |  {json_path}")


# Main experiments
def exp_ppo(base_params, device):
    run_experiment(
        [{"label": "PPO", "agent_name": "PPO", "params": {}}],
        base_params, device,
        title="PPO on CartPole-v1",
        save_name="ppo",
    )


def exp_ablation_ppo_clip(base_params, device):
    run_experiment([
        {"label": "clip=0.1", "agent_name": "PPO", "params": {**ABLATION, "clip_eps": 0.1}},
        {"label": "clip=0.2", "agent_name": "PPO", "params": {**ABLATION, "clip_eps": 0.2}},
        {"label": "clip=0.3", "agent_name": "PPO", "params": {**ABLATION, "clip_eps": 0.3}},
    ], base_params, device,
       title="PPO — Clip Epsilon Ablation",
       save_name="ablation_ppo_clip")
 
 
def exp_ablation_ppo_epochs(base_params, device):
    run_experiment([
        {"label": "epochs=2", "agent_name": "PPO", "params": {**ABLATION, "n_epochs": 2}},
        {"label": "epochs=4", "agent_name": "PPO", "params": {**ABLATION, "n_epochs": 4}},
        {"label": "epochs=8", "agent_name": "PPO", "params": {**ABLATION, "n_epochs": 8}},
    ], base_params, device,
       title="PPO — Update Epochs Ablation",
       save_name="ablation_ppo_epochs")
 
 
def exp_ablation_ppo_lr(base_params, device):
    run_experiment([
        {"label": "lr=1e-4", "agent_name": "PPO", "params": {**ABLATION, "lr": 1e-4}},
        {"label": "lr=3e-4", "agent_name": "PPO", "params": {**ABLATION, "lr": 3e-4}},
        {"label": "lr=1e-3", "agent_name": "PPO", "params": {**ABLATION, "lr": 1e-3}},
    ], base_params, device,
       title="PPO — Learning Rate Ablation",
       save_name="ablation_ppo_lr")
 
 
def exp_ablation_ppo_nsteps(base_params, device):
    run_experiment([
        {"label": "n_steps=64",  "agent_name": "PPO", "params": {**ABLATION, "n_steps": 64}},
        {"label": "n_steps=128", "agent_name": "PPO", "params": {**ABLATION, "n_steps": 128}},
        {"label": "n_steps=256", "agent_name": "PPO", "params": {**ABLATION, "n_steps": 256}},
    ], base_params, device,
       title="PPO — Rollout Length Ablation",
       save_name="ablation_ppo_nsteps")
 
 
def exp_all_ablations(base_params, device):
    exp_ablation_ppo_clip(base_params, device)
    exp_ablation_ppo_epochs(base_params, device)
    exp_ablation_ppo_lr(base_params, device)
    exp_ablation_ppo_nsteps(base_params, device)



EXPERIMENTS = {
    # Main
    "exp_ppo":          exp_ppo,
    "abl_ppo_clip":     exp_ablation_ppo_clip,
    "abl_ppo_epochs":   exp_ablation_ppo_epochs,
    "abl_ppo_lr":       exp_ablation_ppo_lr,
    "abl_ppo_nsteps":   exp_ablation_ppo_nsteps,
    "all_ablations":    exp_all_ablations,
}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO experiments on CartPole-v1")
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENTS.keys()),
        help=(
            "Which experiment to run:\n"
            "  Main       : ppo\n"
            "  Ablations  : abl_ppo_clip | abl_ppo_epochs | abl_ppo_lr | abl_ppo_nsteps\n"
            "  All        : all_ablations\n"
        ),
    )
    args = parser.parse_args()
 
    torch.set_float32_matmul_precision("high")
    device = torch.device("cpu")
 
    base_params = PPOConfig()
    EXPERIMENTS[args.experiment](base_params, device)
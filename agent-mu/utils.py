# Evaluating, Loading and Saving functions
# 2024 - Veer Pareek

import json
import torch

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Union

from config import MuZeroWeights, TransformerWeights, ModelParams, MuZeroConfig
from encoder import freqs_cis
from environment import Environment, Action
from mcts import MCTSConfig, simulate, get_action_policy

def save_checkpoint(muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, muzero_opt: torch.optim.Optimizer, encoder_opt: torch.optim.Optimizer, step: int, avg_losses: Dict[str, float], metrics: Dict[str, float], checkpoint_dir: Union[str, Path], is_best: bool = False) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"checkpoint_step_{step}_{timestamp}"
    if is_best:
        checkpoint_name = "best_" + checkpoint_name

    weights_dict = {
        "muzero": {
            "dynamics": muzero_weights.dynamics.__dict__,
            "prediction": muzero_weights.prediction.__dict__
        },
        "encoder": {
            "input_projection": encoder_weights.input_projection,
            "norm": encoder_weights.norm,
            "layer_weights": [layer.__dict__ for layer in encoder_weights.layer_weights]
        }
    }

    optimizer_dict = {
        "muzero": muzero_opt.state_dict(),
        "encoder": encoder_opt.state_dict()
    }

    training_state = {
        "step": step,
        "losses": avg_losses,
        "metrics": metrics,
        "timestamp": timestamp
    }

    checkpoint = {
        "weights": weights_dict,
        "optimizers": optimizer_dict,
        "training_state": training_state
    }

    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
    torch.save(checkpoint, checkpoint_path)

    metrics_path = checkpoint_dir / f"{checkpoint_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "step": step,
            "losses": avg_losses,
            "metrics": metrics,
            "timestamp": timestamp
        }, f, indent=2)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        if best_path.exists():
            best_path.unlink()
        best_path.symlink_to(checkpoint_path)

def load_checkpoint(checkpoint_path: str, muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, muzero_opt: Optional[torch.optim.Optimizer] = None, encoder_opt: Optional[torch.optim.Optimizer] = None) -> Dict:
    checkpoint = torch.load(checkpoint_path)

    muzero_dict = checkpoint["weights"]["muzero"]
    muzero_weights.dynamics.__dict__.update(muzero_dict["dynamics"])
    muzero_weights.prediction.__dict__.update(muzero_dict["prediction"])

    encoder_dict = checkpoint["weights"]["encoder"]
    encoder_weights.input_projection = encoder_dict["input_projection"]
    encoder_weights.norm = encoder_dict["norm"]
    for layer, layer_dict in zip(encoder_weights.layer_weights, encoder_dict["layer_weights"]):
        layer.__dict__.update(layer_dict)

    if muzero_opt is not None and encoder_opt is not None:
        muzero_opt.load_state_dict(checkpoint["optimizers"]["muzero"])
        encoder_opt.load_state_dict(checkpoint["optimizers"]["encoder"])

    return checkpoint["training_state"]

def evaluate_model(env: Environment, muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, model_params: ModelParams, muzero_config: MuZeroConfig, mcts_config: MCTSConfig, num_episodes: int = 10) -> Dict[str, float]:
    total_reward = 0
    total_portfolio_value = 0
    wins = 0

    device = next(iter(muzero_weights.dynamics.__dict__.values())).device
    freqs_cis_cache = freqs_cis(model_params.head_dim, model_params.n_layers, device=device)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            root = simulate(state.market_state, muzero_weights, encoder_weights, model_params, muzero_config, mcts_config, freqs_cis_cache)

            action_stats = get_action_policy(root, temperature=0)
            action = torch.argmax(action_stats.search_policy).item()

            state, reward, done = env.step(Action(action))
            episode_reward += reward

        total_reward += episode_reward
        total_portfolio_value += state.portfolio_value
        if state.portfolio_value > env.config.starting_capital:
            wins += 1

    metrics = {
        "avg_reward": total_reward / num_episodes,
        "avg_portfolio_value": total_portfolio_value / num_episodes,
        "win_rate": wins / num_episodes
    }

    return metrics

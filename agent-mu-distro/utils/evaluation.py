import torch
from typing import Dict

from config import MuZeroWeights, TransformerWeights, ModelParams, MuZeroConfig
from encoder import freqs_cis
from environment import Environment, Action
from mcts import MCTSConfig, simulate, get_action_policy

def evaluate_model(
    env: Environment,
    muzero_weights: MuZeroWeights,
    encoder_weights: TransformerWeights,
    model_params: ModelParams,
    muzero_config: MuZeroConfig,
    mcts_config: MCTSConfig,
    rank: int = 0,
    num_episodes: int = 10
) -> Dict[str, float]:
    """Evaluate model performance (only rank 0 evaluates)."""
    if rank != 0:
        return {}

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

    return {
        "avg_reward": total_reward / num_episodes,
        "avg_portfolio_value": total_portfolio_value / num_episodes,
        "win_rate": wins / num_episodes
    }

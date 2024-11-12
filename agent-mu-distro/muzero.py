# MuZero Reinforcement Learning Algorithm
# 2024 - Veer Pareek

import torch
from typing import Tuple, Dict, Optional, List

from config import DynamicsWeights, PredictionWeights, MuZeroConfig, MuZeroWeights, TransformerWeights, ModelParams
from encoder import rms_norm, encoder

def dynamics_network(state: torch.Tensor, action: torch.Tensor, weights: DynamicsWeights, config: MuZeroConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamics network with shared computation."""
    x = torch.cat([state, action], dim=-1)
    state_hidden = torch.nn.functional.relu(torch.nn.functional.linear(x, weights.state_net_1))
    state_hidden = rms_norm(state_hidden, weights.norm_1)

    next_state = torch.nn.functional.linear(state_hidden, weights.state_net_2)
    next_state = rms_norm(next_state, weights.norm_2)
    reward = torch.nn.functional.linear(state_hidden, weights.reward_net_2)

    return next_state, reward

def prediction_network(state: torch.Tensor, weights: PredictionWeights, config: MuZeroConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.nn.functional.relu(torch.nn.functional.linear(state, weights.shared_net_1))
    hidden = rms_norm(hidden, weights.norm_1)
    hidden = torch.nn.functional.relu(torch.nn.functional.linear(hidden, weights.shared_net_2))
    hidden = rms_norm(hidden, weights.norm_2)

    policy_logits = torch.nn.functional.linear(hidden, weights.policy_net)
    policy = torch.nn.functional.softmax(policy_logits, dim=-1)

    value = torch.nn.functional.linear(hidden, weights.value_net)

    return policy, value

def initial_inference(market_state: torch.Tensor, encoder_weights: TransformerWeights, prediction_weights: PredictionWeights, encoder_params: ModelParams, muzero_config: MuZeroConfig, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded_state = encoder(market_state, encoder_weights, encoder_params, freqs_cis)
    policy, value = prediction_network(encoded_state[:, -1], prediction_weights, muzero_config)

    return encoded_state, policy, value

def recurrent_inference(encoded_state: torch.Tensor, action: torch.Tensor, dynamics_weights: DynamicsWeights, prediction_weights: PredictionWeights, muzero_config: MuZeroConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_state, reward = dynamics_network(encoded_state, action, dynamics_weights, muzero_config)
    policy, value = prediction_network(next_state, prediction_weights, muzero_config)

    return next_state, reward, policy, value

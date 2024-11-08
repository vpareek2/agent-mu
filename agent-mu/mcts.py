# Monte Carlo Tree Search
# 2024 - Veer Pareek

import torch
from typing import Tuple, List

from config import MCTSConfig, ModelParams, MuZeroConfig, MuZeroWeights, NodeStats, ActionStats, MinMaxStats, TransformerWeights
from muzero import initial_inference, recurrent_inference

def get_action_policy(root: NodeStats, temperature: float = 1.0) -> ActionStats:
    visits = torch.Tensor([root.children_stats[action].visit_count if action in root.children_stats else 0 for action in range(3)])
    if temperature == 0:
        action_probs = torch.zeros_like(visits, dtype=torch.float).index_fill_(0, torch.argmax(visits).unsqueeze(0), 1.0)
    else:
        action_probs = (visits ** (1 / temperature)) / ((visits ** (1 / temperature)).sum())

    return ActionStats(search_policy=action_probs, visit_counts=visits)

def compute_ucb_score(node: NodeStats, child: NodeStats, min_max_stats: MinMaxStats, config: MCTSConfig) -> torch.Tensor:
    pb_c = torch.log((node.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c = pb_c * torch.sqrt(node.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.visit_count + 1)

    return prior_score + value_score

def add_dirichlet_noise(policy: torch.Tensor, alpha: float = 0.3, noise_fraction: float = 0.25) -> torch.Tensor:
    noise = torch.distributions.Dirichlet(torch.Tensor([alpha] * len(policy))).sample()

    return policy * (1 - noise_fraction) + noise * noise_fraction

def select_action(node: NodeStats, min_max_stats: MinMaxStats, config: MCTSConfig) -> Tuple[int, List[NodeStats]]:
    search_path = [node]

    while node.children_stats:
        max_ucb = float('-inf')
        best_action = -1

        for action, child in node.children_stats.items():
            ucb = compute_ucb_score(node, child, min_max_stats, config)

            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action

        if best_action == -1:
            best_action = 0
        node = node.children_stats[best_action]
        search_path.append(node)

    if len(node.children_stats) == 0:
        return 0, search_path
    return best_action, search_path # type: ignore

def expand(node: NodeStats, state: torch.Tensor, reward: torch.Tensor, policy_logits: torch.Tensor, value: torch.Tensor) -> None:
    node.state = state
    node.reward = reward
    policy = torch.softmax(policy_logits, dim=-1)

    if len(node.children_stats) == 0:
        policy = add_dirichlet_noise(policy)

    for action in range(len(policy)):
        node.children_stats[action] = NodeStats(visit_count=torch.zeros(1), value_sum=torch.zeros(1), prior=policy[action], children_stats={}, state=None, reward=None)

def backpropagate(search_path: List[NodeStats], value: torch.Tensor, config: MCTSConfig) -> None:
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1

        if node.reward is not None:
            value = node.reward + config.discount * value

def simulate(root_state: torch.Tensor, muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, model_params: ModelParams, muzero_config: MuZeroConfig, mcts_config: MCTSConfig, freqs_cis: torch.Tensor) -> NodeStats:
    root = NodeStats(visit_count=torch.zeros(1), value_sum=torch.zeros(1), prior=torch.ones(muzero_config.action_space) / muzero_config.action_space, children_stats={}, state=None, reward=None)
    encoded_state, policy, value = initial_inference(root_state, encoder_weights, muzero_weights.prediction, model_params, muzero_config, freqs_cis)
    root.state = encoded_state
    expand(root, encoded_state, torch.zeros(1), policy, value)
    min_max_stats = MinMaxStats()

    for _ in range(mcts_config.num_simulations):
        action, search_path = select_action(root, min_max_stats, mcts_config)
        parent = search_path[-2]
        assert parent.state is not None, "Parent state should not be None"
        action_one_hot = torch.zeros(mcts_config.action_space)
        action_one_hot[action] = 1.0
        next_state, reward, policy, value = recurrent_inference(encoded_state, action_one_hot, muzero_weights.dynamics, muzero_weights.prediction, muzero_config)
        leaf = search_path[-1]
        expand(leaf, next_state, reward, policy, value)
        backpropagate(search_path, value, mcts_config)
        min_max_stats.update(value.item())

    return root

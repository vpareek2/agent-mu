# Monte Carlo Tree Search
# 2024 - Veer Pareek

import torch
from typing import Tuple, List

from config import MCTSConfig, ModelParams, MuZeroConfig, MuZeroWeights, NodeStats, ActionStats, MinMaxStats, TransformerWeights
from muzero import initial_inference, recurrent_inference

def get_action_policy(root: NodeStats, temperature: float = 1.0) -> ActionStats:
    batch_size = root.visit_count.shape[0]
    visits = torch.zeros((batch_size, 3), device=root.visit_count.device)

    for action in range(3):
        if action in root.children_stats:
            visits[:, action] = root.children_stats[action].visit_count.squeeze(-1)

    if temperature == 0:
        action_probs = torch.zeros_like(visits, dtype=torch.float)
        action_probs.scatter_(1, visits.argmax(dim=1, keepdim=True), 1.0)
    else:
        visits_temp = (visits ** (1 / temperature))
        action_probs = visits_temp / visits_temp.sum(dim=1, keepdim=True)

    return ActionStats(search_policy=action_probs, visit_counts=visits)

def compute_ucb_score(node: NodeStats, min_max_stats: MinMaxStats, config: MCTSConfig) -> torch.Tensor:
    batch_size = node.visit_count.shape[0]
    scores = torch.full((batch_size, config.action_space), float('-inf'), device=node.visit_count.device)

    for action, child in node.children_stats.items():
        pb_c = torch.log((node.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
        pb_c_score = pb_c * torch.sqrt(node.visit_count) / (child.visit_count + 1)
        prior_score = pb_c_score * child.prior[:, action].unsqueeze(-1)
        value_score = min_max_stats.normalize(child.value_sum / (child.visit_count + 1))
        scores[:, action] = (prior_score + value_score).squeeze(-1)

    return scores

def add_dirichlet_noise(policy: torch.Tensor, config: MCTSConfig) -> torch.Tensor:
    batch_size = policy.shape[0]
    noise = torch.distributions.Dirichlet(
        torch.full((batch_size, config.action_space), config.root_dirichlet_alpha, device=policy.device)
    ).sample()

    return policy * (1 - config.root_exploration_fraction) + noise * config.root_exploration_fraction

def select_action(node: NodeStats, min_max_stats: MinMaxStats, config: MCTSConfig) -> Tuple[torch.Tensor, List[List[NodeStats]]]:
    batch_size = node.visit_count.shape[0]
    device = node.visit_count.device

    search_paths = [[node] for _ in range(batch_size)]
    current_nodes = node
    active = torch.ones(batch_size, dtype=torch.bool, device=device)

    while active.any():
        if not current_nodes.children_stats:
            active.fill_(False)
            continue

        ucb_scores = compute_ucb_score(current_nodes, min_max_stats, config)
        best_actions = ucb_scores.argmax(dim=1)

        for b in range(batch_size):
            if active[b]:
                action = best_actions[b].item()
                if action in current_nodes.children_stats:
                    current_nodes = current_nodes.children_stats[action]
                    search_paths[b].append(current_nodes)
                else:
                    active[b] = False

    return best_actions, search_paths

def expand(node: NodeStats, states: torch.Tensor, rewards: torch.Tensor, policies: torch.Tensor, values: torch.Tensor, config: MCTSConfig) -> None:
    node.state = states
    node.reward = rewards

    policies = torch.softmax(policies, dim=-1)
    if len(node.children_stats) == 0:
        policies = add_dirichlet_noise(policies, config)

    for action in range(config.action_space):
        node.children_stats[action] = NodeStats(
            visit_count=torch.zeros_like(node.visit_count),
            value_sum=torch.zeros_like(node.value_sum),
            prior=policies[:, action].unsqueeze(-1),
            children_stats={},
            state=None,
            reward=None
        )

def backpropagate(search_paths: List[List[NodeStats]], values: torch.Tensor, config: MCTSConfig) -> None:
    batch_size = values.shape[0]

    for b in range(batch_size):
        value = values[b]
        path = search_paths[b]

        for node in reversed(path):
            node.value_sum[b] += value
            node.visit_count[b] += 1

            if node.reward is not None:
                value = node.reward[b] + config.discount * value

def simulate(root_state: torch.Tensor, muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights,
            model_params: ModelParams, muzero_config: MuZeroConfig, mcts_config: MCTSConfig,
            freqs_cis: torch.Tensor) -> NodeStats:
    batch_size = root_state.shape[0]
    device = root_state.device

    root = NodeStats(
        visit_count=torch.zeros((batch_size, 1), device=device),
        value_sum=torch.zeros((batch_size, 1), device=device),
        prior=torch.ones((batch_size, muzero_config.action_space), device=device) / muzero_config.action_space,
        children_stats={},
        state=None,
        reward=None
    )

    encoded_states, policies, values = initial_inference(
        root_state, encoder_weights, muzero_weights.prediction,
        model_params, muzero_config, freqs_cis
    )

    expand(root, encoded_states, torch.zeros((batch_size, 1), device=device),
           policies, values, mcts_config)

    min_max_stats = MinMaxStats(batch_size=batch_size, device=device)

    for _ in range(mcts_config.num_simulations):
        actions, search_paths = select_action(root, min_max_stats, mcts_config)

        parent_states = [path[-2].state for path in search_paths]
        if any(state is None for state in parent_states):
            raise ValueError("Parent states cannot be None during MCTS simulation")
        parents = torch.stack(parent_states)  # type: ignore[arg-type]

        action_one_hot = torch.zeros((batch_size, mcts_config.action_space), device=device)
        action_one_hot.scatter_(1, actions.unsqueeze(1), 1.0)

        next_states, rewards, policies, values = recurrent_inference(
            parents, action_one_hot, muzero_weights.dynamics,
            muzero_weights.prediction, muzero_config
        )

        leaves = [path[-1] for path in search_paths]
        for leaf, next_state, reward, policy, value in zip(leaves, next_states, rewards, policies, values):
            expand(leaf, next_state.unsqueeze(0), reward.unsqueeze(0),
                  policy.unsqueeze(0), value.unsqueeze(0), mcts_config)

        backpropagate(search_paths, values, mcts_config)
        min_max_stats.update(values)

    return root

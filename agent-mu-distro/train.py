# Training file (distributed)
# 2024 - Veer Pareek

import logging
import torch

from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from buffer import ReplayBuffer
from config import MuZeroConfig, ModelParams, MuZeroWeights, TransformerWeights, GameHistory, TradingState, Action, MCTSConfig, TrainingConfig, ReplayBufferConfig
from encoder import freqs_cis
from environment import Environment
from mcts import simulate, get_action_policy
from muzero import initial_inference, recurrent_inference
from optimizers import soap, muon, adopt
from utils.checkpoint import save_checkpoint
from utils.evaluation import evaluate_model
def create_distributed_optimizers(muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, config: TrainingConfig) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    muzero_params = [
        {"params": muzero_weights.dynamics.__dict__.values()},
        {"params": muzero_weights.prediction.__dict__.values()}
    ]

    encoder_params = [
        {"params": encoder_weights.input_projection},
        {"params": [w for layer in encoder_weights.layer_weights for w in layer.__dict__.values()]},
        {"params": encoder_weights.norm}
    ]

    muzero_optimizer = torch.optim.Adam(muzero_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # adopt
    # muzero_optimizer = adopt.ADOPT(muzero_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    # encoder_optimizer = adopt.ADOPT(encoder_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # soap
    # muzero_optimizer = soap.SOAP(muzero_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    # encoder_optimizer = soap.SOAP(encoder_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # muon
    # muzero_optimizer = muon.Muon(muzero_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    # encoder_optimizer = muon.Muon(encoder_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    for opt in [muzero_optimizer, encoder_optimizer]:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * torch.distributed.get_world_size()

    return muzero_optimizer, encoder_optimizer

def compute_losses(states: torch.Tensor, actions: torch.Tensor, target_values: torch.Tensor, target_rewards: torch.Tensor, target_policies: torch.Tensor, predicted_states: torch.Tensor, predicted_values: torch.Tensor, predicted_rewards: torch.Tensor, predicted_policies: torch.Tensor, config: TrainingConfig) -> Dict[str, torch.Tensor]:
    value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)
    policy_loss = -(target_policies * torch.log(predicted_policies + 1e-8)).sum(-1).mean()
    reward_loss = torch.nn.functional.mse_loss(predicted_rewards, target_rewards)
    state_loss = torch.nn.functional.mse_loss(predicted_states, states)
    total_loss = (config.value_loss_weight * value_loss + config.policy_loss_weight * policy_loss + config.reward_loss_weight * reward_loss)

    return {
        "value_loss": value_loss,
        "policy_loss": policy_loss,
        "reward_loss": reward_loss,
        "state_loss": state_loss,
        "total_loss": total_loss
    }

def process_game_history(game: GameHistory, muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, model_params: ModelParams, muzero_config: MuZeroConfig, freqs_cis_cache: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    encoded_states = []
    target_values = []
    target_rewards = []
    target_policies = []
    device = next(iter(muzero_weights.dynamics.__dict__.values())).device

    for idx, (state, action, reward, root_value, policy) in enumerate(zip(game.states, game.actions, game.rewards, game.root_values, game.search_policies)):
        encoded_state, _, _ = initial_inference(state.to(device), encoder_weights, muzero_weights.prediction, model_params, muzero_config, freqs_cis_cache)
        encoded_states.append(encoded_state)
        target_values.append(root_value)
        target_rewards.append(reward)
        target_policies.append(policy)

    return (
        torch.stack(encoded_states),
        torch.tensor(game.actions, device=device),
        torch.tensor(target_values, device=device),
        torch.tensor(target_rewards, device=device),
        torch.stack(target_policies).to(device)
    )

def self_play_game(env: Environment, muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, model_params: ModelParams, muzero_config: MuZeroConfig, mcts_config: MCTSConfig, temperature: float, freqs_cis_cache: torch.Tensor) -> GameHistory:
    states: List[torch.Tensor] = []
    actions: List[Action] = []
    rewards: List[float] = []
    root_values: List[float] = []
    policies: List[torch.Tensor] = []

    device = next(iter(muzero_weights.dynamics.__dict__.values())).device
    state = env.reset()
    done = False

    while not done:
        root = simulate(state.market_state.to(device), muzero_weights, encoder_weights, model_params, muzero_config, mcts_config, freqs_cis_cache)
        action_stats = get_action_policy(root, temperature)
        action = torch.multinomial(action_stats.search_policy, 1).item()
        states.append(state.market_state)
        actions.append(Action(action))
        policies.append(action_stats.search_policy)
        root_values.append(float(root.value_sum / root.visit_count))
        state, reward, done = env.step(Action(action))
        rewards.append(reward)

    game_priority = max(abs(reward) for reward in rewards)
    return GameHistory(states=states, actions=actions, rewards=rewards, root_values=root_values, search_policies=policies, game_priority=game_priority)

def synchronize_buffers(replay_buffer: ReplayBuffer, rank: int, world_size: int) -> None:
    if not replay_buffer.config.prioritized:
        return

    if replay_buffer.priorities is None:
        return
    priorities_list = list(replay_buffer.priorities)
    if not priorities_list:
        return
    local_priorities = torch.tensor(priorities_list, device=torch.device(f'cuda:{rank}'))
    gathered_priorities = [torch.zeros_like(local_priorities) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_priorities, local_priorities)

    all_priorities = torch.cat(gathered_priorities)
    replay_buffer.priorities = all_priorities.cpu().numpy().tolist()

def train(env: Environment, muzero_weights: MuZeroWeights, encoder_weights: TransformerWeights, model_params: ModelParams, muzero_config: MuZeroConfig, mcts_config: MCTSConfig, train_config: TrainingConfig, rank: int, world_size: int) -> None:
    logger = logging.getLogger()
    device = torch.device(f'cuda:{rank}')

    muzero_ddp = torch.nn.parallel.DistributedDataParallel(muzero_weights, device_ids=[rank])
    encoder_ddp = torch.nn.parallel.DistributedDataParallel(encoder_weights, device_ids=[rank])

    replay_buffer = ReplayBuffer(config=ReplayBufferConfig(capacity=10000 // world_size, prioritized=True, alpha=0.6, beta=0.4))
    muzero_opt, encoder_opt = create_distributed_optimizers(muzero_weights, encoder_weights, train_config)
    freqs_cis_cache = freqs_cis(model_params.head_dim, model_params.n_layers, device=device)

    best_portfolio_value = float('-inf')
    local_batch_size = train_config.batch_size // world_size

    for step in tqdm(range(train_config.num_training_steps), disable=rank != 0):
        torch.distributed.barrier()

        if step % train_config.num_self_play_games == 0:
            assert train_config.temperature_schedule is not None, "temperature_schedule should not be None"
            temperature = next(t for s,t in sorted(train_config.temperature_schedule.items()) if step >= s)
            game_history = self_play_game(env, muzero_weights, encoder_weights, model_params, muzero_config, mcts_config, temperature, freqs_cis_cache)
            replay_buffer.store_game(game_history)
            synchronize_buffers(replay_buffer, rank, world_size)

        if not replay_buffer.is_ready(local_batch_size):
            continue

        muzero_opt.zero_grad()
        encoder_opt.zero_grad()

        batch, weights, indices = replay_buffer.sample_batch(local_batch_size)
        batch_losses = []
        all_losses = {
            "value_loss": [],
            "policy_loss": [],
            "reward_loss": [],
            "state_loss": []
        }

        for game in batch:
            encoded_states, actions, target_values, target_rewards, target_policies = process_game_history(game, muzero_weights, encoder_weights, model_params, muzero_config, freqs_cis_cache)
            states, policies, values = initial_inference(encoded_states, encoder_ddp.module, muzero_ddp.module.prediction, model_params, muzero_config, freqs_cis_cache)
            next_states, rewards, next_policies, next_values = recurrent_inference(states, actions, muzero_ddp.module.dynamics, muzero_ddp.module.prediction, muzero_config)

            losses = compute_losses(states=states, actions=actions, target_values=target_values, target_rewards=target_rewards, target_policies=target_policies, predicted_states=next_states, predicted_values=values, predicted_rewards=rewards, predicted_policies=policies, config=train_config)

            batch_losses.append(losses["total_loss"])
            for k, v in losses.items():
                if k != "total_loss":
                    all_losses[k].append(v.detach())

        batch_loss = torch.stack(batch_losses).mean()
        torch.distributed.all_reduce(batch_loss)
        batch_loss = batch_loss / world_size

        batch_loss.backward()

        for model in [muzero_ddp, encoder_ddp]:
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(train_config.gradient_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)

        muzero_opt.step()
        encoder_opt.step()

        if replay_buffer.config.prioritized and indices is not None:
            priorities = [batch_loss.detach().item()] * len(indices)
            replay_buffer.update_priorities(indices, priorities)

        avg_losses = {k: torch.stack(v).mean().item() for k, v in all_losses.items()}

        if rank == 0 and step % 100 == 0:
            logger.info(
                f"Step {step}: Value Loss: {avg_losses['value_loss']:.4f}, "
                f"Policy Loss: {avg_losses['policy_loss']:.4f}, "
                f"Reward Loss: {avg_losses['reward_loss']:.4f}, "
                f"State Loss: {avg_losses['state_loss']:.4f}"
            )

        if rank == 0 and step % train_config.checkpoint_interval == 0:
            eval_metrics = evaluate_model(env, muzero_weights, encoder_weights, model_params, muzero_config, mcts_config)

            is_best = eval_metrics["avg_portfolio_value"] > best_portfolio_value
            if is_best:
                best_portfolio_value = eval_metrics["avg_portfolio_value"]

            save_checkpoint(muzero_weights=muzero_weights, encoder_weights=encoder_weights, muzero_opt=muzero_opt, encoder_opt=encoder_opt, step=step, avg_losses=avg_losses, metrics=eval_metrics, checkpoint_dir=train_config.checkpoint_dir, is_best=is_best)

            logger.info(
                f"Step {step} Evaluation: "
                f"Avg Reward: {eval_metrics['avg_reward']:.2f}, "
                f"Avg Portfolio Value: {eval_metrics['avg_portfolio_value']:.2f}, "
                f"Win Rate: {eval_metrics['win_rate']:.2%}"
            )

        torch.distributed.barrier()

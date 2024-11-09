# Vectorized Market Environment
# 2024 - Veer Pareek

import torch

from typing import List, Optional, Tuple
from dataclasses import dataclass

from torch.distributions import Transform
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from config import Position, Action, TradingState, EnvironmentConfig, MarketFeatures

@dataclass
class BatchTradingState:
    market_state: torch.Tensor
    positions: Optional[torch.Tensor]
    portfolio_values: torch.Tensor
    timestamps: torch.Tensor
    done: torch.Tensor

class VectorizedEnvironment:
    def __init__(self, market_features: MarketFeatures, num_envs: int, config: Optional[EnvironmentConfig] = None, device: Optional[torch.device] = None):
        self.config = config or EnvironmentConfig()
        self.device = device or torch.device('cpu')
        self.num_envs = num_envs

        self.market_features = market_features
        self.seq_len = market_features.ohlvc.shape[0]

        if self.seq_len < self.config.window_size:
            raise ValueError(f"Market data length ({self.seq_len}) must be greaater than window size ({self.config.window_size})")

        self.portfolio_values =  torch.full((num_envs,), self.config.window_size, device=self.device, dtype=torch.long)
        self.positions = None
        self.timestamps = torch.full((num_envs,), self.config.window_size, device=self.device, dtype=torch.long)

    def _get_market_window(self) -> torch.Tensor:
        starts = self.timestamps - self.config.window_size
        ends = self.timestamps

        batch_indices = torch.arange(self.num_envs, device=self.device)

        features = [
            self.market_features.ohlvc[starts:ends],
            self.market_features.returns[starts:ends],
            self.market_features.sma[starts:ends],
            self.market_features.ema[starts:ends],
            self.market_features.rsi[starts:ends],
            self.market_features.macd[starts:ends],
            self.market_features.reddit_sentiment[starts:ends],
            self.market_features.news_sentiment[starts:ends],
            self.market_features.combined_sentiment[starts:ends],
            self.market_features.day[starts:ends],
            self.market_features.month[starts:ends]
        ]

        return torch.cat(features, dim=-1)

    def _get_current_prices(self) -> torch.Tensor:
        return self.market_features.ohlvc[self.timestamps, 3]

    def get_valid_actions_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.num_envs, len(Action)), dtype=torch.bool, device=self.device)

        mask[:, Action.HOLD] = True

        if self.positions is None:
            mask[:, [Action.LONG, Action.SHORT]] = True
        else:
            long_positions = self.positions[:, 0] > 0
            short_positions = self.positions[:, 0] < 0
            no_positions = self.positions[:, 0] == 0

            mask[long_positions | no_positions, Action.SHORT] = True
            mask[short_positions | no_positions, Action.LONG] = True

        return mask

    def _update_positions_batch(self, actions: torch.Tensor, current_prices: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        transaction_costs = torch.zeros(self.num_envs, device=self.device)

        if self.positions is None:
           self.positions = torch.zeros((self.num_envs, 4), device=self.device)

        position_sizes = self.portfolio_values * self.config.max_position_size
        long_mask = actions == Action.LONG
        short_mask = actions == Action.SHORT

        self.positions[long_mask] = torch.stack([
            position_sizes[long_mask],
            current_prices[long_mask],
            self.timestamps[long_mask].float(),
            torch.zeros_like(position_sizes[long_mask])
        ], dim=1)

        self.positions[short_mask] = torch.stack([
            -position_sizes[short_mask],
            current_prices[short_mask],
            self.timestamps[short_mask].float(),
            torch.zeros_like(position_sizes[short_mask])
        ], dim=1)

        transaction_mask = (actions != Action.HOLD)
        transaction_costs[transaction_mask] = (abs(self.positions[transaction_mask, 0] * current_prices[transaction_mask] * self.config.transaction_cost))

        return self.positions, transaction_costs

    def step(self, actions: torch.Tensor) -> Tuple[BatchTradingState, torch.Tensor, torch.Tensor]:
        valid_actions_mask = self.get_valid_actions_mask()
        if not torch.all(valid_actions_mask.gather(1, actions.unsqueeze(1))):
            raise ValueError("Invalid actions for current states")

        market_state = self._get_market_window()
        current_prices = self._get_current_prices()

        old_portfolio_values = self.portfolio_values.clone()

        self.positions, transaction_costs = self._update_positions_batch(actions, current_prices)

        if self.positions is not None:
            pnl = self.positions[:, 0] * (current_prices - self.positions[:, 1])
            self.positions[:, 3] = pnl
            self.portfolio_values += pnl - transaction_costs

        self.timestamps += 1

        done = torch.logical_or(self.portfolio_values <= 0,self.timestamps >= torch.min(torch.tensor([self.seq_len, self.config.max_timesteps], device=self.device)))

        rewards = self.portfolio_values - old_portfolio_values
        if self.positions is not None:
            rewards -= abs(self.positions[:, 0]) * self.config.transaction_cost

        new_state = BatchTradingState(
            market_state=market_state,
            positions=self.positions,
            portfolio_values=self.portfolio_values,
            timestamps=self.timestamps,
            done=done
        )

        return new_state, rewards, done

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> BatchTradingState:
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.portfolio_values[env_mask] = self.config.starting_capital
        self.timestamps[env_mask] = self.config.window_size
        if self.positions is not None:
            self.positions[env_mask] = 0

        return BatchTradingState(
            market_state=self._get_market_window(),
            positions=self.positions,
            portfolio_values=self.portfolio_values,
            timestamps=self.timestamps,
            done=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

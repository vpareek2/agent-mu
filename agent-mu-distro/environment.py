# Market Environment
# 2024 - Veer Pareek

import pandas as pd
import torch

from typing import List, Optional, Tuple

from config import Position, Action, TradingState, EnvironmentConfig, MarketFeatures

class Environment:

    def __init__(self, market_features: MarketFeatures, config: Optional[EnvironmentConfig] = None):
        self.config: EnvironmentConfig = config or EnvironmentConfig()
        self.market_features: MarketFeatures = market_features
        self.seq_len: int = market_features.ohlvc.shape[0]
        if self.seq_len < self.config.window_size: raise ValueError(f"Market data length ({self.seq_len}) must be greater than window size ({self.config.window_size})")
        self.portfolio_value: float = self.config.starting_capital
        self.position: Optional[Position] = None
        self.timestamp: int = self.config.window_size

    def _get_market_window(self) -> torch.Tensor:
        start_idx = self.timestamp - self.config.window_size
        end_idx = self.timestamp
        features = [
            self.market_features.ohlvc[start_idx:end_idx],
            self.market_features.returns[start_idx:end_idx],
            self.market_features.sma[start_idx:end_idx],
            self.market_features.ema[start_idx:end_idx],
            self.market_features.rsi[start_idx:end_idx],
            self.market_features.macd[start_idx:end_idx],
            self.market_features.reddit_sentiment[start_idx:end_idx],
            self.market_features.news_sentiment[start_idx:end_idx],
            self.market_features.combined_sentiment[start_idx:end_idx],
            self.market_features.day[start_idx:end_idx],
            self.market_features.month[start_idx:end_idx]
        ]
        return torch.cat(features, dim=-1)

    def _get_current_price(self) -> float:
        return float(self.market_features.ohlvc[self.timestamp, 3])

    def reset(self) -> TradingState:
        self.portfolio_value = self.config.starting_capital
        self.position = None
        self.timestamp = self.config.window_size

        return TradingState(
            market_state=self._get_market_window(),
            position=None,
            portfolio_value=self.portfolio_value,
            timestamp=self.timestamp,
            done=False
        )

    def get_valid_actions(self) -> List[Action]:
        valid_actions = [Action.HOLD]
        if self.position is None:
            valid_actions.extend([Action.LONG, Action.SHORT])
        elif self.position.size > 0:
            valid_actions.append(Action.SHORT)
        else:
            valid_actions.append(Action.LONG)

        return valid_actions

    def calculate_reward(self, old_state: TradingState, action: Action, new_state: TradingState) -> float:
        reward = new_state.portfolio_value - old_state.portfolio_value
        if self.position is not None:
            reward -= abs(self.position.size) * self.config.transaction_cost

        return reward

    def _update_position(self, action: Action, current_price: float) -> Tuple[Optional[Position], float]:
        transaction_cost = 0.0

        if self.position is not None:

            if (self.position.size > 0 and action == Action.SHORT) or (self.position.size < 0 and action == Action.LONG):
                transaction_cost = abs(self.position.size * current_price * self.config.transaction_cost)
                self.position = None

        if action != Action.HOLD and self.position is None:
            position_size = self.portfolio_value * self.config.max_position_size

            if action == Action.SHORT:
                position_size = -position_size

            transaction_cost = abs(position_size * current_price * self.config.transaction_cost)
            self.position = Position(size=position_size, entry_price=current_price, entry_time=self.timestamp, pnl=0.0)

        return self.position, transaction_cost

    def step(self, action: Action) -> Tuple[TradingState, float, bool]:
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid action {action} for current state")

        old_state = TradingState(market_state=self._get_market_window(), position=self.position, portfolio_value=self.portfolio_value, timestamp=self.timestamp, done=False)
        current_price = self._get_current_price()
        self.position, transaction_cost = self._update_position(action, current_price)

        if self.position:
            pnl = self.position.update_pnl(current_price)
            self.portfolio_value += pnl - transaction_cost

        self.timestamp += 1
        done = (self.portfolio_value <= 0 or self.timestamp >= min(self.seq_len, self.config.max_timesteps))

        new_state = TradingState(market_state=self._get_market_window(), position=self.position, portfolio_value=self.portfolio_value, timestamp=self.timestamp, done=done)
        reward = self.calculate_reward(old_state, action, new_state)

        return new_state, reward, done

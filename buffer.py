import torch
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

from config import GameHistory, ReplayBufferConfig, Action

class ReplayBuffer:

    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.buffer = deque(maxlen=config.capacity)
        self.priorities = deque(maxlen=config.capacity) if config.prioritized else None

    def store_game(self, game: GameHistory) -> None:
        self.buffer.append(game)
        if self.config.prioritized and self.priorities is not None:
            self.priorities.append(game.game_priority)

    def sample_batch(self, batch_size: int) -> Tuple[List[GameHistory], Optional[torch.Tensor], Optional[List[int]]]:
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough games in buffer ({len(self.buffer)} < {batch_size})")

        if not self.config.prioritized:
            indices = np.random.choice(len(self.buffer), batch_size).tolist()
            games = [self.buffer[idx] for idx in indices]
            return games, None, None

        probs = np.array(self.priorities) ** self.config.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs).tolist()
        games = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.config.beta)
        weights = weights / weights.max()

        return games, torch.FloatTensor(weights), indices

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        if not self.config.prioritized or not self.priorities:
            return

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

from typing import List, Tuple, Optional, Dict, cast
import torch.cuda
from torch import Tensor
from numbers import Number
from config import ReplayBufferConfig, GameHistory
from dataclasses import dataclass

@dataclass
class BufferStats:
    total_games: int
    local_games: int
    global_priorities_sum: float
    local_priorities_sum: float

class DistributedReplayBuffer:
    def __init__(
        self,
        config: ReplayBufferConfig,
        rank: int,
        world_size: int,
        device: torch.device
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # Adjust capacity for sharding
        self.local_capacity = config.capacity // world_size
        if rank < config.capacity % world_size:
            self.local_capacity += 1

        # Initialize storage as tensors for efficiency
        self.buffer: List[GameHistory] = []
        self.priorities: Optional[Tensor] = (
            torch.zeros(self.local_capacity, device=device)
            if config.prioritized else None
        )

        # Track indices for circular buffer
        self.current_idx = 0
        self.size = 0

        # Create events for async operations
        self.priority_sync_event = torch.cuda.Event(enable_timing=False, blocking=False)
        # Create a dedicated stream for priority updates
        self.priority_stream = torch.cuda.Stream(device=device)

    def _get_global_stats(self) -> BufferStats:
        """Gather global buffer statistics efficiently."""
        local_size = torch.tensor([len(self.buffer)], device=self.device)
        global_size = torch.zeros_like(local_size)
        torch.distributed.all_reduce(local_size, op=torch.distributed.ReduceOp.SUM)

        if self.priorities is not None:
            local_sum = self.priorities[:self.size].sum()
            global_sum = torch.zeros_like(local_sum)
            torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        else:
            local_sum = global_sum = torch.tensor(0.0, device=self.device)

        return BufferStats(
            total_games=int(global_size.item()),  # Explicit cast to int
            local_games=len(self.buffer),
            global_priorities_sum=float(global_sum.item()),  # Explicit cast to float
            local_priorities_sum=float(local_sum.item())  # Explicit cast to float
        )

    def store_game(self, game: GameHistory) -> None:
        """Store game in local buffer shard."""
        if len(self.buffer) < self.local_capacity:
            self.buffer.append(game)
            if self.priorities is not None:
                self.priorities[self.size] = game.game_priority
        else:
            self.buffer[self.current_idx] = game
            if self.priorities is not None:
                self.priorities[self.current_idx] = game.game_priority

        self.size = min(self.size + 1, self.local_capacity)
        self.current_idx = (self.current_idx + 1) % self.local_capacity

    def _compute_sampling_probs(self) -> torch.Tensor:
        """Compute sampling probabilities efficiently."""
        if not self.config.prioritized or self.priorities is None:
            return torch.ones(self.size, device=self.device) / self.size

        local_probs = self.priorities[:self.size] ** self.config.alpha

        # Get global sum for normalization
        local_sum = local_probs.sum()
        global_sum = torch.zeros_like(local_sum)
        torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)

        return local_probs / global_sum

    def sample_batch(self, batch_size: int) -> Tuple[List[GameHistory], Optional[torch.Tensor], Optional[List[int]]]:
        stats = self._get_global_stats()

        if stats.total_games < batch_size:
            raise ValueError(
                f"Not enough games in buffer "
                f"({stats.total_games} < {batch_size})"
            )

        local_batch_size = batch_size // self.world_size
        if self.rank < batch_size % self.world_size:
            local_batch_size += 1

        if not self.config.prioritized:
            indices = torch.randint(
                0, self.size, (local_batch_size,),
                device=self.device
            )
            games = [self.buffer[idx] for idx in indices]
            return games, None, indices.tolist()

        probs = self._compute_sampling_probs()

        indices = torch.multinomial(
            probs, local_batch_size,
            replacement=True
        )

        games = [self.buffer[idx] for idx in indices]

        weights = (
            (stats.total_games * probs[indices])
            ** (-self.config.beta)
        )
        weights = weights / weights.max()

        return games, weights, indices.tolist()

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities with async operations."""
        if not self.config.prioritized or self.priorities is None:
            return

        stream = cast(torch._C.Stream, self.priority_stream)
        with torch.cuda.stream(stream):
            priority_tensor = torch.tensor(
                priorities,
                device=self.device
            )
            index_tensor = torch.tensor(
                indices,
                device=self.device,
                dtype=torch.long
            )

            self.priorities.index_copy_(
                0, index_tensor, priority_tensor
            )

            self.priority_sync_event.record(stream)

    def synchronize(self) -> None:
        """Synchronize async operations."""
        if self.config.prioritized:
            self.priority_sync_event.synchronize()

    def __len__(self) -> int:
        return self._get_global_stats().total_games

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size

# Configs and Data Structures
# 2024 - Veer Pareek

import torch

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple, Optional, List, Dict, NamedTuple

# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    symbol: str = "AAPL"
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    timeframe: str = "1d"

    reddit_credentials: Dict[str, str] = field(default_factory=lambda: {
        'client_id': 'jOPlAtel1Z6Ndm5gQPaE8g',
        'client_secret': 'VLk8nhZs6OBTAm_29iLlf6XKp26p5w',
        'user_agent': '"Muzerotrader/1.0 by /u/Interesting-Iron8269"'
    })
    newsapi_key: str = 'a26bd92479c44f9e94fa13953ef29b9c'

    price_windows: List[int] = field(default_factory=lambda:  [5, 10, 20])
    use_log_returns: bool = True

    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14

    sentiment_window: int = 1
    reddit_weight: float = 0.5

    use_day_of_week: bool = True
    use_month: bool = True

@dataclass
class MarketFeatures:
    ohlvc: torch.Tensor
    returns: torch.Tensor
    sma: torch.Tensor
    ema: torch.Tensor
    rsi: torch.Tensor
    macd: torch.Tensor
    reddit_sentiment: torch.Tensor
    news_sentiment: torch.Tensor
    combined_sentiment: torch.Tensor
    day: torch.Tensor
    month: torch.Tensor

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class Position:
    size: float
    entry_price: float
    entry_time: int
    pnl: float

    def update_pnl(self, current_price: float) -> float:
        if self.size > 0:
            self.pnl = self.size * (current_price - self.entry_price)
        else:
            self.pnl = -self.size * (self.entry_price - current_price)
        return self.pnl

class Action(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2

@dataclass
class TradingState:
    market_state: torch.Tensor
    position: Optional[Position]
    portfolio_value: float
    timestamp: int
    done: bool

@dataclass
class EnvironmentConfig:
    window_size: int = 50
    starting_capital: float = 100_000.0
    transaction_cost: float = 0.001
    max_position_size: float = 1.0
    max_timesteps: int = 1000

# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    n_layers: int = 4
    n_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024
    head_dim: int = 32
    n_market_features: int = 19

@dataclass
class LayerWeights:
    attention_norm: torch.Tensor
    ffn_norm: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor

@dataclass
class TransformerWeights:
    input_projection: torch.Tensor
    layer_weights: List[LayerWeights]
    norm: torch.Tensor

# ---------------------------------------------------------------------------
# MuZero
# ---------------------------------------------------------------------------

@dataclass
class DynamicsWeights:
    state_net_1: torch.Tensor
    state_net_2: torch.Tensor
    reward_net_1: torch.Tensor
    reward_net_2: torch.Tensor
    norm_1: torch.Tensor
    norm_2: torch.Tensor

@dataclass
class PredictionWeights:
    shared_net_1: torch.Tensor
    shared_net_2: torch.Tensor
    policy_net: torch.Tensor
    value_net: torch.Tensor
    norm_1: torch.Tensor
    norm_2: torch.Tensor

@dataclass
class MuZeroWeights:
    dynamics: DynamicsWeights
    prediction: PredictionWeights

@dataclass
class MuZeroConfig:
    state_dim: int = 256
    action_space: int = 3
    hidden_dim: int = 128
    reward_dim: int = 1

# ---------------------------------------------------------------------------
# Monte Carlo Tree Search
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    num_simulations: int = 50
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    discount: float = 0.997
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    action_space: int = 3

@dataclass
class NodeStats:
    visit_count: torch.Tensor
    value_sum: torch.Tensor
    prior: torch.Tensor
    children_stats: Dict[int, 'NodeStats']
    state: torch.Tensor
    reward: Optional[torch.Tensor] = None

@dataclass
class ActionStats(NamedTuple):
    search_policy: torch.Tensor
    visit_counts: torch.Tensor

@dataclass
class MinMaxStats:
    minimum: float = float('inf')
    maximum: float = float('-inf')

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = min(self.maximum, value)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class GameHistory:
    states: List[torch.Tensor]
    actions: List[Action]
    rewards: List[float]
    root_values: List[float]
    search_policies: List[torch.Tensor]
    game_priority: float

@dataclass
class ReplayBufferConfig:
    capacity: int = 10000
    prioritized: bool = True
    alpha: float = 0.6
    beta: float = 0.4

# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    num_training_steps: int = 1000000
    num_epochs: int = 100
    batch_size: int = 128
    checkpoint_interval: int = 1000
    num_self_play_games: int = 100
    temperature_schedule: Optional[Dict[int, float]] = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        if self.temperature_schedule is None:
            self.temperature_schedule = {
                0: 1.0,
                50000: 0.5,
                200000: 0.25
            }

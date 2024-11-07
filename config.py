import torch
from dataclasses import dataclass, field

from typing import Tuple, Optional, List, Dict, NamedTuple
from enum import IntEnum

# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

"""
Configuration parameters for the transformer encoder
"""
@dataclass
class ModelParams:
    n_layers: int = 4           # Number of transformer layers
    n_heads: int = 8            # Number of attention heads per layer
    d_model: int = 256          # Model Dimension
    d_ff: int = 1024            # Dimension of feed forward network layer
    head_dim: int = 32          # Dimension of attention head
    n_market_features: int = 19 # Number of market features in the input data

"""
Weights for a single transformer encoder layer
"""
@dataclass
class LayerWeights:
    attention_norm: torch.Tensor
    ffn_norm: torch.Tensor
    wq: torch.Tensor    # Query weights
    wk: torch.Tensor    # Key weights
    wv: torch.Tensor    # Value weights
    wo: torch.Tensor    # Output projection
    w1: torch.Tensor    # First ffn layer
    w2: torch.Tensor    # Second ffn layer
    w3: torch.Tensor    # Gate weights for SwiGLU

"""
Full transformer weights
"""
@dataclass
class TransformerWeights:
    input_projection: torch.Tensor     # Projects input features to model dimension
    layer_weights: List[LayerWeights]  # Weights for each transformer layer
    norm: torch.Tensor                 # Final layer normalization

# ---------------------------------------------------------------------------
# MuZero
# ---------------------------------------------------------------------------

"""
Weights for the Dynamics Network
"""
@dataclass
class DynamicsWeights:
    state_net_1: torch.Tensor
    state_net_2: torch.Tensor
    reward_net_1: torch.Tensor
    reward_net_2: torch.Tensor
    norm_1: torch.Tensor
    norm_2: torch.Tensor

"""
Weights for Policy & Value prediction
"""
@dataclass
class PredictionWeights:
    shared_net_1: torch.Tensor
    shared_net_2: torch.Tensor
    policy_net: torch.Tensor
    value_net: torch.Tensor
    norm_1: torch.Tensor
    norm_2: torch.Tensor

"""
Full MuZero weights
"""
@dataclass
class MuZeroWeights:
    dynamics: DynamicsWeights
    prediction: PredictionWeights

"""
Configuration parameters for MuZero
"""
@dataclass
class MuZeroConfig:
    state_dim: int = 256       # Dimension of encoded state
    action_space: int = 3      # Number of possible trading actions (HOLD, LONG, SHORT)
    hidden_dim: int = 128      # Hidden dimension for MuZero networks
    reward_dim: int = 1        # Dimension of reward prediction (usually 1)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

"""
Configuration parameters for MCTS
"""
@dataclass
class MCTSConfig:
    num_simulations: int = 50
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    discount: float = 0.997
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    action_space: int = 3

"""
Stats for each node in the tree
"""
@dataclass
class NodeStats:
    visit_count: torch.Tensor
    value_sum: torch.Tensor
    prior: torch.Tensor
    children_stats: Dict[int, 'NodeStats']
    state: torch.Tensor
    reward: Optional[torch.Tensor] = None

"""
Stats for action solution
"""
@dataclass
class ActionStats(NamedTuple):
    search_policy: torch.Tensor
    visit_counts: torch.Tensor

"""
Track value bounds for normalization
"""
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
# Environment
# ---------------------------------------------------------------------------

"""
Track position state
"""
@dataclass
class Position:
    size: float
    entry_price: float
    entry_time: int
    pnl: float = 0.0

    def update_pnl(self, current_price: float) -> float:
        if self.size > 0:   # Long position
            self.pnl = self.size * (current_price - self.entry_price)
        else:   # Short position
            self.pnl = -self.size * (self.entry_price - current_price)
        return self.pnl

"""
Trading Actions
"""
class Action(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2

"""
Complete state of the trading environment
"""
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
# Replay Buffer
# ---------------------------------------------------------------------------

"""
Single game trajectory with MCTS info
"""
@dataclass
class GameHistory:
    states: List[torch.Tensor]
    actions: List[Action]
    rewards: List[float]
    root_values: List[float]
    search_policies: List[torch.Tensor]
    game_priority: float

"""
Configuration for replay buffer
"""
@dataclass
class ReplayBufferConfig:
    capacity: int = 10000
    prioritized: bool = True
    alpha: float = 0.6
    beta: float = 0.4

# ---------------------------------------------------------------------------
# Dataset Collection
# ---------------------------------------------------------------------------

"""Configuration for Feature Engineering"""
@dataclass
class FeatureConfig:

    # Data source parameters
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


    # Price Features
    price_windows: List[int] = field(default_factory=lambda:  [5, 10, 20])
    use_log_returns: bool = True

    # Technical Features
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14

    # Sentiment Features
    sentiment_window: int = 1   # Days to aggregate sentiment
    reddit_weight: float = 0.5  # WEight for combining sentiment scores

    # Calendar features
    use_day_of_week: bool = True
    use_month: bool = True

"""Organized feature groups"""
@dataclass
class MarketFeatures:

    # Price Features (5)
    ohlvc: torch.Tensor     # [bsz, seq_len, 5]

    # Returns (1)
    returns: torch.Tensor   # [bsz, seq_len, 1]

    # Technical (7)
    sma: torch.Tensor       # [bsz, seq_len, 3] (5, 10, 20)
    ema: torch.Tensor       # [bsz, seq_len, 2] (12, 26)
    rsi: torch.Tensor       # [bsz, seq_len, 1]
    macd: torch.Tensor      # [bsz, seq_len, 1]

    # Sentiment (3)
    reddit_sentiment: torch.Tensor      # [bsz, seq_len, 1]
    news_sentiment: torch.Tensor        # [bsz, seq_len, 1]
    combined_sentiment: torch.Tensor    # [bsz, seq_len, 1]

    # Calendar (2)
    day: torch.Tensor   # [bsz, seq_len, 1]
    month: torch.Tensor # [bsz, seq_len, 1]

# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    num_training_steps: int = 1000000
    num_epochs: int = 100
    batch_size: int = 128
    checkpoint_interval: int = 1000

    # Self-play parameters
    num_self_play_games: int = 100
    temperature_schedule: Dict[int, float] = None

    # Optimization parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # Loss weights
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        if self.temperature_schedule is None:
            self.temperature_schedule = {
                0: 1.0,         # Early exploration
                50000: 0.5,     # Mid-training exploration
                200000: 0.25    # Lower temperature later
            }

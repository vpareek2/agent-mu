import logging
import os
import pickle
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from torchvision.ops.misc import Tuple

from config import (FeatureConfig, ModelParams, MuZeroConfig, MCTSConfig, EnvironmentConfig, ReplayBufferConfig, TrainingConfig, DynamicsWeights, PredictionWeights, LayerWeights, MuZeroWeights, TransformerWeights)
from data import pipeline
from environment import Environment
from train import train
from encoder import freqs_cis

def setup_logging(log_dir: str = "logs") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[ logging.FileHandler(f"{log_dir}/muzero_{timestamp}.log"), logging.StreamHandler()])

def init_weights(model_params: ModelParams, muzero_config: MuZeroConfig, device: torch.device) -> Tuple[MuZeroWeights, TransformerWeights]:
    layer_weights = []
    for _ in range(model_params.n_layers):
        layer = LayerWeights(
            attention_norm=torch.nn.Parameter(torch.ones(model_params.d_model)).to(device),
            ffn_norm=torch.nn.Parameter(torch.ones(model_params.d_model)).to(device),
            wq=torch.nn.Parameter(torch.randn(model_params.d_model, model_params.n_heads * model_params.head_dim) * 0.02).to(device),
            wk=torch.nn.Parameter(torch.randn(model_params.d_model, model_params.n_heads * model_params.head_dim) * 0.02).to(device),
            wv=torch.nn.Parameter(torch.randn(model_params.d_model, model_params.n_heads * model_params.head_dim) * 0.02).to(device),
            wo=torch.nn.Parameter(torch.randn(model_params.n_heads * model_params.head_dim, model_params.d_model) * 0.02).to(device),
            w1=torch.nn.Parameter(torch.randn(model_params.d_model, model_params.d_ff) * 0.02).to(device),
            w2=torch.nn.Parameter(torch.randn(model_params.d_ff, model_params.d_model) * 0.02).to(device),
            w3=torch.nn.Parameter(torch.randn(model_params.d_model, model_params.d_ff) * 0.02).to(device),
        )
        layer_weights.append(layer)

        encoder_weights = TransformerWeights(
            input_projection=torch.nn.Parameter(torch.randn(model_params.n_market_features, model_params.d_model) * 0.02).to(device),
            layer_weights=layer_weights,
            norm=torch.nn.Parameter(torch.ones(model_params.d_model)).to(device)
        )

        dynamics_weights = DynamicsWeights(
            state_net_1=torch.nn.Parameter(torch.randn(muzero_config.state_dim + muzero_config.action_space, muzero_config.hidden_dim) * 0.02).to(device),
            state_net_2=torch.nn.Parameter(torch.randn(muzero_config.hidden_dim, muzero_config.state_dim) * 0.02).to(device),
            reward_net_1=torch.nn.Parameter(torch.randn(muzero_config.state_dim + muzero_config.action_space, muzero_config.hidden_dim) * 0.02).to(device),
            reward_net_2=torch.nn.Parameter(torch.randn(muzero_config.hidden_dim, muzero_config.reward_dim) * 0.02).to(device),
            norm_1=torch.nn.Parameter(torch.ones(muzero_config.hidden_dim)).to(device),
            norm_2=torch.nn.Parameter(torch.ones(muzero_config.state_dim)).to(device)
        )

        prediction_weights = PredictionWeights(
            shared_net_1=torch.nn.Parameter(torch.randn(muzero_config.state_dim, muzero_config.hidden_dim) * 0.02).to(device),
            shared_net_2=torch.nn.Parameter(torch.randn(muzero_config.hidden_dim, muzero_config.hidden_dim) * 0.02).to(device),
            policy_net=torch.nn.Parameter(torch.randn(muzero_config.hidden_dim, muzero_config.action_space) * 0.02).to(device),
            value_net=torch.nn.Parameter(torch.randn(muzero_config.hidden_dim, 1) * 0.02).to(device),
            norm_1=torch.nn.Parameter(torch.ones(muzero_config.hidden_dim)).to(device),
            norm_2=torch.nn.Parameter(torch.ones(muzero_config.hidden_dim)).to(device)
        )

        muzero_weights = MuZeroWeights(
            dynamics=dynamics_weights,
            prediction=prediction_weights
        )

    return muzero_weights, encoder_weights

def load_features(feature_config: FeatureConfig, cache_file: str = "features_cache.pkl") -> MarketFeatures:
    if os.path.exists(cache_file):
        logging.info(f"Loading cached features from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    logging.info("Fetching new market features...")
    features = pipeline(symbol=feature_config.symbol, start_date=feature_config.start_date, end_date=feature_config.end_date, config=feature_config, reddit_credentials=feature_config.reddit_credentials, newsapi_key=feature_config.newsapi_key)

    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    logging.info(f"Features cached to {cache_file}")

    return features

def main():
    setup_logging()
    logging.info("Initializing Agent")

    feature_config = FeatureConfig(symbol="APPL", start_date="2023-01-01", end_date="2024-01-01")
    model_params = ModelParams()
    muzero_config = MuZeroConfig()
    mcts_config = MCTSConfig()
    env_config = EnvironmentConfig()
    training_config = TrainingConfig()

    device = torch.device(training_config.device)
    logging.info(f"Using device: {device}")

    reddit_credentials = {
        'client_id': os.getenv('REDDIT_CLIENT_ID'),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'user_agent': os.getenv('REDDIT_USER_AGENT')
    }
    newsapi_key = os.getenv('NEWSAPI_KEY')

    features = load_features(feature_config)

    env = Environment(features, env_config)
    logging.info("Environment initialized")

    muzero_weights, encoder_weights = init_weights(model_params, muzero_config, device)
    logging.info("Model weights initialized")

    freqs_cis_cache = freqs_cis(model_params.head_dim, model_params.n_layers, device=device)
    logging.info("Starting training...")
    try:
        train(env=env, muzero_weights=muzero_weights, encoder_weights=encoder_weights, model_params=model_params, muzero_config=muzero_config, mcts_config=mcts_config, train_config=training_config)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

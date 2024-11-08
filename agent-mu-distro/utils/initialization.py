import logging
import os
import pickle
import torch
from typing import Dict, Optional, Tuple

from config import (
    FeatureConfig, ModelParams, MuZeroConfig,
    MuZeroWeights, TransformerWeights, MarketFeatures,
    DynamicsWeights, PredictionWeights, LayerWeights
)
from data import pipeline

def init_weights(model_params: ModelParams, muzero_config: MuZeroConfig, device: torch.device) -> Tuple[MuZeroWeights, TransformerWeights]:
    """Initialize model weights with proper scaling."""
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
    """Load or fetch market features with caching."""
    if os.path.exists(cache_file):
        logging.info(f"Loading cached features from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    logging.info("Fetching new market features...")
    features = pipeline(
        symbol=feature_config.symbol,
        start_date=feature_config.start_date,
        end_date=feature_config.end_date,
        config=feature_config,
        reddit_credentials=feature_config.reddit_credentials,
        newsapi_key=feature_config.newsapi_key
    )

    with open(cache_file, 'wb') as f:
        pickle.dump(features, f)
    logging.info(f"Features cached to {cache_file}")

    return features

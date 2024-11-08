# Main file (distributed)
# 2024 - Veer Pareek

import logging
import torch

from config import FeatureConfig, ModelParams, MuZeroConfig, MCTSConfig, EnvironmentConfig, TrainingConfig
from encoder import freqs_cis
from environment import Environment
from feature_manager import DistributedFeatureManager
from train import train

from utils.distributed import setup_distributed, cleanup_distributed
from utils.logging import setup_logging, rank_logging_context
from utils.initialization import init_weights, load_features

def train_process(rank: int, world_size: int, feature_config: FeatureConfig, model_params: ModelParams, muzero_config: MuZeroConfig, mcts_config: MCTSConfig, env_config: EnvironmentConfig, training_config: TrainingConfig) -> None:
    try:
        setup_distributed(rank, world_size)
        with rank_logging_context(rank):
            setup_logging(rank=rank)

            device = torch.device(f'cuda:{rank}')
            logging.info(f"Process {rank}/{world_size} using device: {device}")

            feature_manager = DistributedFeatureManager(world_size, rank)
            features = feature_manager.distribute_features(load_features(feature_config) if rank == 0 else None)

            env = Environment(features, env_config)
            muzero_weights, encoder_weights = init_weights(model_params, muzero_config, device)
            freqs_cis_cache = freqs_cis(model_params.head_dim, model_params.n_layers, device=device)

            logging.info("Starting training...")
            train(env=env, muzero_weights=muzero_weights, encoder_weights=encoder_weights, model_params=model_params, muzero_config=muzero_config, mcts_config=mcts_config, train_config=training_config, rank=rank, world_size=world_size)
            logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        feature_manager.cleanup()
        cleanup_distributed()

def main():
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")

    setup_logging()
    logging.info(f"Initializing distributed training across {world_size} GPUs")

    configs = {
        'feature': FeatureConfig(symbol="AAPL", start_date="2023-01-01", end_date="2024-01-01"),
        'model': ModelParams(),
        'muzero': MuZeroConfig(),
        'mcts': MCTSConfig(),
        'env': EnvironmentConfig(),
        'training': TrainingConfig()
    }

    try:
        torch.multiprocessing.spawn(train_process, args=(world_size, configs['feature'], configs['model'], configs['muzero'], configs['mcts'], configs['env'], configs['training']), nprocs=world_size, join=True)
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

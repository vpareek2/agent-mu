import os
import torch

def setup_distributed(rank: int, world_size: int) -> None:
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training environment."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

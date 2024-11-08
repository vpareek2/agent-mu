import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from config import MuZeroWeights, TransformerWeights

def save_checkpoint(
    muzero_weights: MuZeroWeights,
    encoder_weights: TransformerWeights,
    muzero_opt: torch.optim.Optimizer,
    encoder_opt: torch.optim.Optimizer,
    step: int,
    avg_losses: Dict[str, float],
    metrics: Dict[str, float],
    checkpoint_dir: Union[str, Path],
    rank: int = 0,
    is_best: bool = False
) -> None:
    """Save checkpoint (only rank 0 saves)."""
    if rank != 0:
        return

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"checkpoint_step_{step}_{timestamp}"
    if is_best:
        checkpoint_name = "best_" + checkpoint_name

    weights_dict = {
        "muzero": {
            "dynamics": muzero_weights.dynamics.__dict__,
            "prediction": muzero_weights.prediction.__dict__
        },
        "encoder": {
            "input_projection": encoder_weights.input_projection,
            "norm": encoder_weights.norm,
            "layer_weights": [layer.__dict__ for layer in encoder_weights.layer_weights]
        }
    }

    optimizer_dict = {
        "muzero": muzero_opt.state_dict(),
        "encoder": encoder_opt.state_dict()
    }

    training_state = {
        "step": step,
        "losses": avg_losses,
        "metrics": metrics,
        "timestamp": timestamp
    }

    checkpoint = {
        "weights": weights_dict,
        "optimizers": optimizer_dict,
        "training_state": training_state
    }

    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
    torch.save(checkpoint, checkpoint_path)

    metrics_path = checkpoint_dir / f"{checkpoint_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "step": step,
            "losses": avg_losses,
            "metrics": metrics,
            "timestamp": timestamp
        }, f, indent=2)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        if best_path.exists():
            best_path.unlink()
        best_path.symlink_to(checkpoint_path)

def load_checkpoint(
    checkpoint_path: str,
    muzero_weights: MuZeroWeights,
    encoder_weights: TransformerWeights,
    muzero_opt: Optional[torch.optim.Optimizer] = None,
    encoder_opt: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_path)

    muzero_dict = checkpoint["weights"]["muzero"]
    muzero_weights.dynamics.__dict__.update(muzero_dict["dynamics"])
    muzero_weights.prediction.__dict__.update(muzero_dict["prediction"])

    encoder_dict = checkpoint["weights"]["encoder"]
    encoder_weights.input_projection = encoder_dict["input_projection"]
    encoder_weights.norm = encoder_dict["norm"]
    for layer, layer_dict in zip(encoder_weights.layer_weights, encoder_dict["layer_weights"]):
        layer.__dict__.update(layer_dict)

    if muzero_opt is not None and encoder_opt is not None:
        muzero_opt.load_state_dict(checkpoint["optimizers"]["muzero"])
        encoder_opt.load_state_dict(checkpoint["optimizers"]["encoder"])

    return checkpoint["training_state"]

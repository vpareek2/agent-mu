import math
import torch
from typing import Tuple, Optional, List

from config import ModelParams, LayerWeights, TransformerWeights

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def freqs_cis(dim: int, end: int, theta: float = 10000.0, device: Optional[torch.device] = None) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.to(dtype), xk_out.to(dtype)

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params: ModelParams, freqs_cis: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, _ = x.shape

    xq = torch.nn.functional.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_heads, model_params.head_dim)
    xk = torch.nn.functional.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_heads, model_params.head_dim)
    xv = torch.nn.functional.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_heads, model_params.head_dim)

    xq, xk = rope(xq, xk, freqs_cis)

    xq = xq.reshape(bsz, seqlen, model_params.n_heads, model_params.head_dim).transpose(1 ,2)
    xk = xk.reshape(bsz, seqlen, model_params.n_heads, model_params.head_dim).transpose(1 ,2)
    xv = xv.transpose(1, 2)

    scale = 1.0 / math.sqrt(model_params.head_dim)
    scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale

    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, xv)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

    return torch.nn.functional.linear(output, layer_weights.wo)

def ffn(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x, layer_weights.w1)) * torch.nn.functional.linear(x, layer_weights.w3), layer_weights.w2)

def layer(x: torch.Tensor, layer_weights: LayerWeights, model_params: ModelParams, freqs_cis: torch.Tensor) -> torch.Tensor:
    norm_x = rms_norm(x, layer_weights.attention_norm)
    x = x + attention(norm_x, layer_weights, model_params, freqs_cis)
    norm_x = rms_norm(x, layer_weights.ffn_norm)
    x = x + ffn(norm_x, layer_weights)

    return x

def encoder(market_state: torch.Tensor, weights: TransformerWeights, model_params: ModelParams, freqs_cis: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.linear(market_state, weights.input_projection)
    for layer_idx in range(model_params.n_layers):
        x = layer(x, weights.layer_weights[layer_idx], model_params, freqs_cis)

    return rms_norm(x, weights.norm)

# Transformer Encoder
# 2024 - Veer Pareek

import math
import torch

from typing import Tuple, Optional, List, Dict

from config import ModelParams, LayerWeights, TransformerWeights

class RoPECache:
    _cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    @classmethod
    def freqs_cis(cls, dim: int, end: int, theta: float = 10000.0, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        device = device or torch.device('cpu')
        key = (dim, end, device)

        if key in cls._cache:
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype)[:(dim // 2)] / dim))
            t = torch.arange(end, device=device, dtype=dtype)
            freqs = torch.outer(t, freqs)
            cls._cache[key] = torch.polar(torch.ones_like(freqs), freqs)

        return cls._cache[key].to(dtype)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = torch.pow(x, 2).mean(-1, keepdim=True)
    variance = variance.clamp(min=eps * eps)

    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = xq.dtype
    xq_shape = xq.shape[:-1] + (-1, 2)
    xk_shape = xk.shape[:-1] + (-1, 2)

    reshape_xq = xq.float().reshape(xq_shape)
    reshape_xk = xk.float().reshape(xk_shape)

    xq_out = torch.view_as_complex(reshape_xq) * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = torch.view_as_complex(reshape_xk) * freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_out).reshape(xq.shape)
    xk_out = torch.view_as_real(xk_out).reshape(xk.shape)

    return xq_out.to(dtype), xk_out.to(dtype)

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params: ModelParams, freqs_cis: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, _ = x.shape
    head_dim = model_params.head_dim
    n_heads = model_params.n_heads

    qkv = torch.stack([torch.nn.functional.linear(x, w) for w in [layer_weights.wq, layer_weights.wk, layer_weights.wv]], dim=0)

    qkv = qkv.reshape(3, bsz, seqlen, n_heads, head_dim)
    xq, xk, xv = qkv[0], qkv[1], qkv[2]

    xq, xk = rope(xq, xk, freqs_cis)

    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    scale = head_dim ** -0.5
    scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale

    attention_weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)

    output = torch.matmul(attention_weights, xv)
    output = output.transpose(1, 2).contiguous()
    output = output.reshape(bsz, seqlen, n_heads, * head_dim)

    return torch.nn.functional.linear(output, layer_weights.wo)

def ffn(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(x, layer_weights.w1))
    gate = torch.nn.functional.linear(x, layer_weights.w3)

    return torch.nn.functional.linear(hidden * gate, layer_weights.w2)

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

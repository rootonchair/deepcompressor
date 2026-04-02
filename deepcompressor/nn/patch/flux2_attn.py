# -*- coding: utf-8 -*-
"""Patched Flux2 parallel self-attention module."""

import torch
import torch.nn as nn
from diffusers.models.embeddings import apply_rotary_emb

from .sdpa import ScaleDotProductAttention

__all__ = ["PatchedFlux2ParallelSelfAttention"]


class PatchedFlux2ParallelSelfAttention(nn.Module):
    """Replaces Flux2ParallelSelfAttention by splitting the fused to_qkv_mlp_proj
    into separate to_q, to_k, to_v, mlp_proj linear layers.

    The original Flux2ParallelSelfAttention fuses Q/K/V projections and MLP input
    projection into a single linear layer (to_qkv_mlp_proj). This patch class splits
    it by rows into four separate nn.Linear modules, enabling the struct system to
    treat them independently for quantization.

    Expects to_out to already be a ConcatLinear (from replace_fused_linear_with_concat_linear).
    """

    def __init__(self, orig) -> None:
        super().__init__()
        # Copy scalar attributes needed by AttentionStruct
        self.heads = orig.heads
        self.head_dim = orig.head_dim
        self.inner_dim = orig.inner_dim
        self.query_dim = orig.query_dim
        self.out_dim = orig.out_dim

        # Split to_qkv_mlp_proj (rows) into separate linears
        inner_dim = orig.inner_dim
        w = orig.to_qkv_mlp_proj.weight.data  # [3*inner + mlp*mult, query_dim]
        b = orig.to_qkv_mlp_proj.bias  # may be None
        device, dtype = w.device, w.dtype

        self.to_q = nn.Linear(orig.query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.to_k = nn.Linear(orig.query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.to_v = nn.Linear(orig.query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        mlp_out_features = orig.mlp_hidden_dim * orig.mlp_mult_factor
        has_bias = b is not None
        self.mlp_proj = nn.Linear(orig.query_dim, mlp_out_features, bias=has_bias, device=device, dtype=dtype)

        # Copy weights by row slices
        self.to_q.weight.data.copy_(w[:inner_dim])
        self.to_k.weight.data.copy_(w[inner_dim : 2 * inner_dim])
        self.to_v.weight.data.copy_(w[2 * inner_dim : 3 * inner_dim])
        self.mlp_proj.weight.data.copy_(w[3 * inner_dim :])
        if has_bias:
            bias_data = b.data
            self.mlp_proj.bias.data.copy_(bias_data[3 * inner_dim :])

        # Move sub-modules from orig (shared references, not copies)
        self.norm_q = orig.norm_q
        self.norm_k = orig.norm_k
        self.mlp_act_fn = orig.mlp_act_fn
        self.to_out = orig.to_out  # already ConcatLinear from prior patch

        # Self-contained SDPA (no processor indirection)
        self.sdpa = ScaleDotProductAttention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # Separate Q/K/V and MLP projections
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        mlp_hidden_states = self.mlp_proj(hidden_states)

        # Reshape to multi-head: (batch, seq, heads, head_dim)
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        # QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Transpose to (batch, heads, seq, head_dim) for SDPA
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = self.sdpa(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # MLP activation (SwiGLU)
        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)

        # Concatenate and output projection (ConcatLinear splits attn/MLP)
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states

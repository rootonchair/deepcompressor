# FLUX.2-klein-4B Integration Notes

## Overview

Integrated `black-forest-labs/FLUX.2-klein-4B` into the deepcompressor quantization pipeline. FLUX.2-klein uses entirely new diffusers classes that are **not subclasses** of the Flux 1 classes, requiring explicit support across the codebase.

## Key Architectural Differences from Flux 1

| Aspect | Flux 1 | Flux 2 Klein |
|--------|--------|--------------|
| Pipeline class | `FluxPipeline` | `Flux2KleinPipeline` |
| Transformer class | `FluxTransformer2DModel` | `Flux2Transformer2DModel` |
| Attention (double blocks) | `Attention` (diffusers base) | `Flux2Attention` (not a subclass of `Attention`) |
| Attention (single blocks) | `Attention` with `pre_only=True` | `Flux2ParallelSelfAttention` (fused QKV+MLP proj) |
| Single block class | `FluxSingleTransformerBlock` | `Flux2SingleTransformerBlock` |
| Double block class | `FluxTransformerBlock` | `Flux2TransformerBlock` |
| FFN class | `FeedForward` (diffusers) | `Flux2FeedForward` (uses `linear_in`/`linear_out` + `Flux2SwiGLU`) |
| Time embedding | `time_text_embed` (`CombinedTimestepGuidanceTextProjEmbeddings`) | `time_guidance_embed` (`Flux2TimestepGuidanceEmbeddings`) |
| Modulation | Per-block `norm.linear` / `norm1.linear` | Shared at model level: `double_stream_modulation_img/txt`, `single_stream_modulation` |
| Single block output | `proj_out` (fused attn+MLP output) | `attn.to_out` (fused attn+MLP output, plain `nn.Linear`) |
| Single block input proj | Separate `attn.to_q/k/v` + `proj_mlp` | Fused `attn.to_qkv_mlp_proj` (Q+K+V+MLP_gate+MLP_up) |
| Text encoder | T5 / CLIP | Qwen3 (`Qwen3ForCausalLM`) |
| Block counts | e.g. 19 double + 38 single (dev) | 5 double + 20 single |
| Forward params | `hidden_states, encoder_hidden_states, pooled_projections, timestep, ...` | `hidden_states, encoder_hidden_states, timestep, ...` (no `pooled_projections`) |

## Files Modified

### 1. `deepcompressor/app/diffusion/pipeline/config.py`

**Changes:**
- Imported `Flux2KleinPipeline` from `diffusers.pipelines.flux2`
- Added `"flux.2-klein"` -> `"black-forest-labs/FLUX.2-klein-4B"` path mapping in `_default_build()`
- Added `Flux2KleinPipeline.from_pretrained(...)` loading branch (cannot use `AutoPipelineForText2Image`)
- Added `_flux2_extract_text_encoders()` static method and registered it for `"flux.2-klein"` to handle the Qwen3 text encoder (bypasses the default `T5EncoderModel` filter)

**Why a custom text extractor:** The default `_default_extract_text_encoders` filters by `supported=(T5EncoderModel,)`, which would exclude `Qwen3ForCausalLM`. The custom extractor accepts the text encoder unconditionally. Note: the default extractor also has a pre-existing bug (`vars.__dict__.keys()` iterates over the builtin `vars` function, not the pipeline object) which was left unfixed to avoid unrelated changes.

### 2. `deepcompressor/app/diffusion/nn/struct.py`

**Changes:**
- Imported all Flux2 classes: `Flux2Transformer2DModel`, `Flux2TransformerBlock`, `Flux2SingleTransformerBlock`, `Flux2Attention`, `Flux2ParallelSelfAttention`, `Flux2FeedForward`, `Flux2KleinPipeline`
- Added Flux2 types to unions: `DIT_BLOCK_CLS`, `DIT_CLS`, `DIT_PIPELINE_CLS`
- **`DiffusionAttentionStruct._default_construct`**: Extended to handle `Flux2Attention` (guarded `getattr` for missing attributes) and `PatchedFlux2ParallelSelfAttention` (to_q/to_k/to_v from split, o_proj from to_out.linears[0], self-contained SDPA)
- **`DiffusionFeedForwardStruct._default_construct`**: Added branches for `Flux2SingleTransformerBlock` (up_proj=attn.mlp_proj, down_proj=attn.to_out.linears[1], act=swish_glu) and `Flux2FeedForward` (linear_in/linear_out)
- **`DiffusionTransformerBlockStruct._default_construct`**: Added branches for `Flux2SingleTransformerBlock` (parallel, standard attns/ffn path after patching) and `Flux2TransformerBlock` (non-parallel, layer_norm type)
- **`DiTStruct._default_construct`**: Routes `Flux2Transformer2DModel` to `Flux2KleinStruct.construct()`
- **`Flux2KleinStruct(FluxStruct)`**: New struct class mapping Flux2 module names (`x_embedder` -> `input_embed`, `time_guidance_embed` -> `time_embed`, `context_embedder` -> `text_embed`, `norm_out`, `proj_out`, `transformer_blocks`, `single_transformer_blocks`)
- Registered factories for `Flux2KleinStruct`, `Flux2Attention`, `Flux2ParallelSelfAttention`, `Flux2SingleTransformerBlock`, `Flux2FeedForward`

### 3. `deepcompressor/app/diffusion/nn/attention.py`

**Changes:**
- Imported `Flux2ParallelSelfAttention`
- **`DiffusionAttentionProcessor.__call__`**: Guarded attribute assertions with `getattr` so `Flux2Attention` (which lacks `spatial_norm`, `group_norm`, `norm_cross`, `residual_connection`, `rescale_output_factor`, `scale`, `inner_kv_dim`) doesn't crash. The existing `Flux` processor name check (`startswith("Flux")`) covers `Flux2AttnProcessor`.
- **`Flux2ParallelAttentionProcessor`** (new class): Handles `Flux2ParallelSelfAttention` after ConcatLinear splitting. Runs the fused `to_qkv_mlp_proj` projection, splits Q/K/V from MLP, applies QK-norm + RoPE, runs SDPA via `self.sdpa` (enabling quantization hooks), runs MLP activation, concatenates attn+MLP outputs, and runs fused `to_out` projection.

### 4. `deepcompressor/app/diffusion/nn/patch.py`

**Changes:**
- Imported `Flux2Attention`, `Flux2ParallelSelfAttention`, `Flux2SingleTransformerBlock`, `Flux2ParallelAttentionProcessor`
- **`replace_fused_linear_with_concat_linear()`**: Added `Flux2SingleTransformerBlock` handling — splits `attn.to_out` into ConcatLinear with split point `[inner_dim]` (attn_out + mlp_out)
- **`replace_flux2_parallel_attn()`** (NEW): Replaces `Flux2ParallelSelfAttention` with `PatchedFlux2ParallelSelfAttention`, splitting fused `to_qkv_mlp_proj` into separate to_q/to_k/to_v/mlp_proj
- **`replace_attn_processor()`**: Added `isinstance` check for `Flux2Attention` (uses `DiffusionAttentionProcessor`). Patched Flux2 single blocks use self-contained SDPA, no processor needed.

### 5. `deepcompressor/app/diffusion/dataset/collect/utils.py`

**Changes:**
- Imported `Flux2Transformer2DModel`
- Added to the `isinstance` check alongside `FluxTransformer2DModel` (same handling: pop `hidden_states` from input kwargs)

## ConcatLinear Split Strategy

`ConcatLinear` splits along the **input features** dimension (columns of the weight matrix). It decomposes `y = W @ [x1|x2]` into `y = W1 @ x1 + W2 @ x2`.

**`to_qkv_mlp_proj` is NOT split.** It maps FROM `query_dim` TO `3*inner_dim + mlp_hidden_dim*2` — the fused dimension is on the output side, which ConcatLinear cannot decompose. It remains a single `nn.Linear` and is quantized as one unit.

**`to_out` IS split** (same pattern as Flux 1's `proj_out`). It takes concatenated `[attn_output | mlp_output]` as input:

```
to_out (out_features = out_dim, in_features = inner_dim + mlp_hidden_dim)
  -> ConcatLinear.from_linear(..., split_points=[inner_dim])
  -> linears[0]: Attention output  (out_dim x inner_dim)
     linears[1]: MLP output        (out_dim x mlp_hidden_dim)
```

### PatchedFlux2ParallelSelfAttention

`Flux2ParallelSelfAttention` has a fused `to_qkv_mlp_proj` that can't be decomposed by `ConcatLinear` (wrong axis) or fit into `AttentionStruct` (shape constraints). The solution: `PatchedFlux2ParallelSelfAttention` (`deepcompressor/nn/patch/flux2_attn.py`) replaces the original module entirely by splitting `to_qkv_mlp_proj` weight rows into separate `to_q`, `to_k`, `to_v`, `mlp_proj` linears:

```
to_qkv_mlp_proj weight [27648, 3072]:
  rows [0:3072]      → to_q.weight
  rows [3072:6144]   → to_k.weight
  rows [6144:9216]   → to_v.weight
  rows [9216:27648]  → mlp_proj.weight  (gate+up for SwiGLU)
```

The patch class has a self-contained `forward()` with `ScaleDotProductAttention` as a sub-module — no processor indirection. `replace_attn_processor` skips it.

Post-patch layout:
```
Flux2SingleTransformerBlock
├── norm: LayerNorm
└── attn: PatchedFlux2ParallelSelfAttention
    ├── to_q, to_k, to_v: nn.Linear         ← split from to_qkv_mlp_proj
    ├── mlp_proj: nn.Linear                  ← split from to_qkv_mlp_proj
    ├── norm_q, norm_k: RMSNorm
    ├── mlp_act_fn: Flux2SwiGLU
    ├── sdpa: ScaleDotProductAttention       ← built-in SDPA
    └── to_out: ConcatLinear                 ← from replace_fused_linear_with_concat_linear
        ├── linears[0]: attn output
        └── linears[1]: MLP output
```

This mirrors Flux1's pattern: standard `AttentionStruct` (q/k/v/o_proj) + `FeedForwardStruct` (up=mlp_proj, down=to_out.linears[1]).

## Config YAML

`examples/diffusion/configs/model/flux.2-klein.yaml` was pre-existing. It uses 4 inference steps, no guidance (matches schnell), bfloat16, and the same skip patterns as other Flux models.

## Not Implemented (Deferred)

- **Nunchaku backend conversion** (`backend/nunchaku/convert.py`): Flux2 blocks have different internal module names and structures. New conversion functions would be needed for Nunchaku deployment.
- **LoRA conversion** (`backend/nunchaku/convert_lora.py`): Same as above.
- **Pre-existing bug**: `_default_extract_text_encoders` uses `vars.__dict__.keys()` which iterates over the builtin `vars` function, not the pipeline object. This doesn't affect Flux2 (which uses a custom extractor) but may affect other models if text encoder quantization is enabled.

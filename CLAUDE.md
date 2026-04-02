# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepCompressor is a post-training quantization (PTQ) toolbox for large language models and diffusion models. It implements algorithms including AWQ, GPTQ, SmoothQuant, QoQ (W4A8KV4), and SVDQuant, with deployment backends for QServe, TinyChat, and Nunchaku.

## Setup and Installation

```bash
conda env create -f environment.yml
poetry install
```

Requires Python >= 3.10, PyTorch >= 2.5.0, Transformers >= 4.46.0.

## Running Quantization

LLM quantization entry point:
```bash
python -m deepcompressor.app.llm.ptq configs/<algorithm>.yaml --model-name <model>
```

Diffusion quantization entry point:
```bash
python -m deepcompressor.app.diffusion.ptq configs/model/<model>.yaml configs/svdquant/<config>.yaml
```

Example configs are in `examples/llm/configs/` and `examples/diffusion/configs/`.

## Linting

Uses Ruff (line-length 120, target py310, rules: B, C, E, F, I, W). F401/F403 are ignored in `__init__.py` files.

```bash
ruff check deepcompressor/
ruff format deepcompressor/
```

## No Test Suite

There are no automated tests in this repository.

## Architecture

### Layered Design

```
Application Layer (app/llm/, app/diffusion/)  — algorithm orchestration, model loading, evaluation
Quantizer Layer (quantizer/)                   — core quantization logic, kernels (RTN, GPTQ)
NN Patching Layer (nn/)                        — non-invasive model modification via hooks/patches
Data Layer (data/)                             — QuantTensor, QuantDataType, QuantScale, DynamicRange
Backend Layer (backend/)                       — inference engine converters (QServe, TinyChat, Nunchaku)
```

### Key Abstractions

- **`Quantizer`** (`quantizer/processor.py`): Main quantization processor, extends `QuantizerImpl` + `BaseTensorProcessor`. Manages scale, zero-point, and dynamic range state.
- **`QuantizerImpl`** (`quantizer/impl/`): Pluggable quantization implementations (simple, scale-based, STE, info-tracking).
- **`QuantTensor`** / **`QuantDataType`** / **`QuantScale`** (`data/`): Core data structures for quantized tensors, data types (INT4, INT8, FP4_E2M1, etc.), and scale management.
- **Configuration classes** (`utils/config/`, `quantizer/config/`): Uses `@configclass` decorator from `omniconfig` for hierarchical, YAML-driven configuration with CLI overrides.
- **Hook system** (`utils/hooks/`): PyTorch forward hooks (`ProcessHook`, `BaseTensorProcessor`) for non-invasive quantization injection.
- **NN patches** (`nn/patch/`): `LowRankBranch` (SVD decomposition), `ConcatLinear`, `ShiftedLinear`, `SDPAPatch` for transparent model modification.

### Configuration System

All quantization is configured via YAML files processed by `omniconfig`. Configs support:
- Per-key enablement and skip patterns (`KeyEnableConfig`, `SkipBasedConfig`)
- Group-wise granularity via `group_shapes` (e.g., `[[1, -1]]` for channel-wise, `[[1, 128]]` for group-128)
- Multi-step/decomposed quantization (`DecomposedQuantizerConfig`)
- Runtime CLI overrides of any config parameter

### Quantization Techniques

Implemented in `app/llm/quant/` and `app/diffusion/quant/`:
- **Smoothing**: Shift outliers from activations to weights
- **Rotation**: Apply rotation matrices for better quantization ranges
- **Reordering**: Compute-aware weight reordering
- **Low-rank decomposition**: SVD-based outlier absorption (SVDQuant)

### Code Style

- Type hints throughout (Python 3.10+ syntax with `|` unions)
- Dataclass-heavy for configuration and data structures
- Explicit memory management with `gc.collect()` and `torch.cuda.empty_cache()`
- Quote style: double quotes, indent: 4 spaces

## Integrating a New Diffusion Model

### Step 1: Model Config YAML

Create `examples/diffusion/configs/model/<new-model>.yaml` specifying pipeline name/dtype, eval parameters (num_steps, guidance_scale, protocol), and quant settings (batch sizes, skip patterns). Reference existing configs (flux, pixart, sana) for architecture-appropriate values.

### Step 2: Pipeline Loading (`app/diffusion/pipeline/config.py`)

In `_default_build()`:
- Add HuggingFace path mapping (e.g., `elif name == "new-model": path = "org/repo"`)
- Add pipeline class conditional if the model needs a non-standard diffusers pipeline (not `AutoPipelineForText2Image`)
- Add task override in `__post_init__` if the model isn't text-to-image

### Step 3: Model Struct (`app/diffusion/nn/struct.py`)

The most involved step. `DiffusionModelStruct.construct()` must recognize the model's transformer/unet class.

- **DiT variants**: Add a branch in `DiTStruct._default_construct()` mapping the model's layer names (input_embed, time_embed, text_embed, norm_out, proj_out, transformer_blocks) to canonical struct fields.
- **Substantially different architecture**: Create a new struct subclass (e.g., `NewModelStruct(DiTStruct)`) with custom `_get_default_key_map()`.

The struct generates the **key map** — short keys like `"embed"`, `"transformer_norm"` mapping to actual module paths, used for skip patterns in quantization configs.

### Step 4: Attention Processor (`app/diffusion/nn/attention.py`)

`DiffusionAttentionProcessor` wraps attention for quantization tracking. If the model uses a custom attention processor (not `AttnProcessor2_0`, `JointAttnProcessor2_0`, or Flux/Sana variants), add support in `__init__` and `__call__`.

If the model uses a fundamentally different attention class (not a subclass of `diffusers.Attention`), consider creating a **patch class** in `nn/patch/` that replaces the original module with one exposing standard `to_q/to_k/to_v/to_out` attributes and a self-contained `forward()` with `ScaleDotProductAttention`. See `PatchedFlux2ParallelSelfAttention` in `nn/patch/flux2_attn.py` for an example.

### Step 5: NN Patches (`app/diffusion/nn/patch.py`)

Patches are applied in `_default_build()` in this order:
1. `replace_fused_linear_with_concat_linear()` — splits fused output projections along input features using `ConcatLinear`. Note: `ConcatLinear` splits the **input** dimension (weight columns), so it only works for layers where the fused dimension is on the input side (e.g., `proj_out` taking `[attn_out | mlp_out]` as input). It does NOT work for output-dimension fusion (e.g., `to_qkv_mlp_proj`).
2. `replace_flux2_parallel_attn()` — replaces `Flux2ParallelSelfAttention` with `PatchedFlux2ParallelSelfAttention`, splitting fused `to_qkv_mlp_proj` by weight rows into separate `to_q/to_k/to_v/mlp_proj` linears.
3. `replace_up_block_conv_with_concat_conv()` — UNet only.
4. `shift_input_activations()` — optional.

### Step 6: Generate Reference Baseline

```bash
python -m deepcompressor.app.diffusion.ptq configs/model/<new-model>.yaml --output-dirname reference
```

Runs the unquantized model to produce reference images (quantization is skipped when `--output-dirname reference`). These are used by future quantized runs for computing similarity metrics (PSNR, LPIPS, SSIM).

### Step 7: Run Quantization

```bash
python -m deepcompressor.app.diffusion.ptq configs/model/<new-model>.yaml configs/svdquant/<config>.yaml
```

### Existing Model Reference

| Model | Struct Class | Pipeline Class | Attention | Fused Linear Patch |
|-------|-------------|----------------|-----------|-------------------|
| SDXL | `UNetStruct` | Auto | Standard | No |
| PixArt | `DiTStruct` | Auto | Standard | No |
| Flux 1 | `FluxStruct` | Auto | RoPE + dual blocks | Yes (`FluxSingleTransformerBlock.proj_out`) |
| Flux 2 Klein | `Flux2KleinStruct` | `Flux2KleinPipeline` | Patch class + self-contained SDPA | Yes (patch splits `to_qkv_mlp_proj` rows; ConcatLinear splits `to_out`) |
| Sana | `DiTStruct` | `SanaPipeline` | Linear attention | No |

For detailed integration notes on Flux 2 Klein (architecture differences, patch class design, ConcatLinear strategy), see [`examples/diffusion/flux2-klein-integration.md`](examples/diffusion/flux2-klein-integration.md).

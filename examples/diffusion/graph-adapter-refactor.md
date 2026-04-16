# Diffusion Graph Adapter Refactor

This document explains the diffusion-side refactor that decouples top-level quantization flows from `DiffusionModelStruct`, and shows how to bring a new architecture into the new adapter path.

## Why This Refactor Exists

Before this change, `deepcompressor/app/diffusion/nn/struct.py` handled too many responsibilities at once:

- model traversal
- activation planning
- key alias generation
- module grouping for weight and activation quantization
- architecture-specific naming rules

That made new architectures expensive to add and hard to maintain. The refactor introduces a smaller `graph/` layer so diffusion PTQ code can depend on a stable adapter interface instead of a specific struct hierarchy.

## What Changed

The refactor landed in five phases.

### Phase 1: Compatibility seam

Added `deepcompressor/app/diffusion/graph/` with:

- [`adapter.py`](../../deepcompressor/app/diffusion/graph/adapter.py): `DiffusionModelAdapter`
- [`types.py`](../../deepcompressor/app/diffusion/graph/types.py): `QuantNode`, `QuantGroup`, `ActivationPlan`
- [`compat_struct.py`](../../deepcompressor/app/diffusion/graph/compat_struct.py): `StructModelAdapter`, `ensure_model_adapter`, `get_default_key_map`

This kept existing behavior unchanged while introducing a new abstraction boundary.

### Phase 2: Read-only consumers

Moved read-only traversal consumers onto the adapter seam so code can work with either a legacy struct-backed adapter or a future native adapter.

### Phase 3: Grouping policy extraction

Pulled block grouping logic into [`groups.py`](../../deepcompressor/app/diffusion/graph/groups.py). This isolates activation and low-rank grouping rules from the old struct traversal code.

### Phase 4: Activation-plan migration

Calibration iteration now consumes `ActivationPlan` from the adapter instead of directly calling struct traversal helpers. This is the main decoupling point for cache collection and layer scheduling.

### Phase 5: Native adapter path

Added:

- [`native.py`](../../deepcompressor/app/diffusion/graph/native.py): `NativeDiffusionAdapter`, `attach_adapter`
- [`registry.py`](../../deepcompressor/app/diffusion/graph/registry.py): adapter factory registry

Top-level diffusion entry points such as:

- [`quant/activation.py`](../../deepcompressor/app/diffusion/quant/activation.py)
- [`quant/weight.py`](../../deepcompressor/app/diffusion/quant/weight.py)
- [`quant/smooth.py`](../../deepcompressor/app/diffusion/quant/smooth.py)
- [`dataset/calib.py`](../../deepcompressor/app/diffusion/dataset/calib.py)

now start from `ensure_model_adapter(...)` instead of assuming `DiffusionModelStruct`.

## Current Architecture

The adapter interface is intentionally small:

- `get_root_module()`: returns the actual `nn.Module`
- `iter_nodes()`: returns quantizable modules as `QuantNode`
- `iter_groups()`: returns grouped quantization targets as `QuantGroup`
- `get_activation_plan()`: returns ordered calibration layers
- `get_named_layers()`: returns named pre/block/post layers
- `get_prev_keys()` and `get_post_keys()`: expose skip boundaries
- `get_key_map()`: exposes config alias matching

There are now two main adapter implementations:

- `StructModelAdapter`: wraps the existing `DiffusionModelStruct`
- `NativeDiffusionAdapter`: lets you describe a model directly without creating a new `*Struct`

## What Still Uses Struct Semantics

This refactor decouples the top-level orchestration layer first. It does **not** eliminate all struct-oriented logic yet.

The group planners in [`groups.py`](../../deepcompressor/app/diffusion/graph/groups.py) and the block-level quantization helpers still expect diffusion block/module semantics such as attention projections and FFN layout. In practice:

- existing supported models continue to work through `StructModelAdapter`
- a brand-new architecture can now enter the pipeline through a native adapter
- if its block internals differ from current diffusion layouts, you may still need to extend grouping logic before full PTQ works end-to-end

That boundary is intentional. The refactor reduces coupling at the orchestration layer first, then makes block-policy changes more local.

## Usage Tutorial

### Use the default compatibility path

If your model is already supported by the current struct system, you do not need to do anything. The existing entry points still work:

```bash
python -m deepcompressor.app.diffusion.ptq \
    examples/diffusion/configs/model/flux.1-schnell.yaml \
    examples/diffusion/configs/svdquant/int4.yaml
```

Internally, the PTQ flow now wraps the model with `StructModelAdapter`.

### Attach a native adapter to a new model

Use this path when you want to experiment with a new architecture without creating a new `*Struct` subclass up front.

```python
from collections import OrderedDict
import torch.nn as nn

from deepcompressor.app.diffusion.graph import (
    NativeDiffusionAdapter,
    attach_adapter,
)


class ToyVideoDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4096, 4096)
        self.block0 = nn.Sequential(nn.Linear(4096, 4096), nn.Linear(4096, 4096))
        self.head = nn.Conv3d(16, 16, 1)


model = ToyVideoDiT()
adapter = NativeDiffusionAdapter.from_named_layers(
    model,
    named_layers=OrderedDict([
        ("embed", model.embed),
        ("block0", model.block0),
        ("head", model.head),
    ]),
    prev_keys=("embed",),
    post_keys=("head",),
    pre_layer_names=("embed",),
    post_layer_names=("head",),
)
attach_adapter(model, adapter)
```

After `attach_adapter(model, adapter)`, any diffusion code path that calls `ensure_model_adapter(model)` will pick up the attached adapter.

### Register an adapter factory

If you do not want to attach adapters manually, register a resolver:

```python
from deepcompressor.app.diffusion.graph import register_model_adapter


@register_model_adapter
def resolve_toy_adapter(model):
    if not isinstance(model, ToyVideoDiT):
        return None
    return adapter
```

This allows the PTQ flow to resolve an adapter directly from the model instance.

### Build a useful native adapter

For a real model, define these pieces carefully:

1. `named_layers`
   Use stable names in execution order. These drive activation planning and progress reporting.
2. `prev_keys` and `post_keys`
   Mark modules that belong to pre-processing and post-processing boundaries for skip logic.
3. `key_map`
   Add aliases if your config should treat multiple names as the same semantic target.
4. `groups`
   If your architecture has grouped projections, define `QuantGroup` entries instead of relying on single-module defaults.

### Run validation

At minimum, validate the adapter before attempting a full quantization run:

```python
from deepcompressor.app.diffusion.graph import ensure_model_adapter

resolved = ensure_model_adapter(model)
print(type(resolved).__name__)
print([node.name for node in resolved.iter_nodes()])
print(resolved.get_activation_plan(
    skip_pre_modules=False,
    skip_post_modules=False,
).layer_names)
```

Then run the narrowest diffusion PTQ or calibration command that exercises the path you changed.

## Recommended Extension Strategy

For a new image-editing or video model, start in this order:

1. Build a `NativeDiffusionAdapter` and validate nodes plus activation order.
2. Run calibration iteration through `dataset/calib.py`.
3. If PTQ fails inside block quantization, extend `groups.py` for the new block pattern.
4. Only add a new struct hierarchy if the architecture truly needs deep struct-specific semantics.

That keeps most new work localized to the adapter and grouping layers instead of `nn/struct.py`.

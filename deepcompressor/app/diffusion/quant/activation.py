# -*- coding: utf-8 -*-
"""Diffusion model activation quantization calibration module."""

import gc
import typing as tp

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.data.cache import IOTensorsCache
from deepcompressor.data.common import TensorType
from deepcompressor.utils import tools

from ..graph import ensure_model_adapter, iter_activation_group_specs
from ..nn.struct import (
    DiffusionBlockStruct,
    DiffusionModelStruct,
    DiffusionModuleStruct,
)
from .config import DiffusionQuantConfig
from .quantizer import DiffusionActivationQuantizer
from .utils import get_needs_inputs_fn, get_needs_outputs_fn

__all__ = ["quantize_diffusion_activations"]


@torch.inference_mode()
def quantize_diffusion_block_activations(  # noqa: C901
    layer: DiffusionBlockStruct | DiffusionModuleStruct,
    config: DiffusionQuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, DiffusionActivationQuantizer]:
    """Quantize the activations of a diffusion model block.

    Args:
        layer (`DiffusionBlockStruct` or `DiffusionModuleStruct`):
            The diffusion model block.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        quantizer_state_dict (`dict[str, dict[str, torch.Tensor | float | None]]`):
            The activation quantizers state dict cache.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*):
            The layer cache.
        layer_kwargs (`dict[str, Any]`, *optional*):
            The layer keyword arguments.
        orig_state_dict (`dict[str, torch.Tensor]`, *optional*):
            The original state dictionary.

    Returns:
        `dict[str, DiffusionActivationQuantizer]`:
            The activation quantizers.
    """
    logger = tools.logging.getLogger(f"{__name__}.ActivationQuant")
    logger.debug("- Quantizing layer %s", layer.name)
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    orig_state_dict = orig_state_dict or {}
    args_caches: list[
        tuple[
            str,  # key
            TensorType,
            list[nn.Linear],  # modules
            list[str],  # module names
            nn.Module,  # eval module
            str,  # eval name
            dict[str, tp.Any],  # eval kwargs
            list[tuple[nn.Parameter, torch.Tensor]],  # original wgts
        ]
    ] = []
    In, Out = TensorType.Inputs, TensorType.Outputs  # noqa: F841

    for spec in iter_activation_group_specs(layer, layer_kwargs=layer_kwargs):
        orig_wgts = None
        if orig_state_dict:
            orig_wgts = [(weight, orig_state_dict[f"{name}.weight"]) for weight, name in spec.orig_weight_refs]
        args_caches.append(
            (
                spec.key,
                In,
                list(spec.modules),
                list(spec.module_names),
                spec.eval_module,
                spec.eval_name,
                spec.eval_kwargs,
                orig_wgts,
            )
        )
    # endregion
    quantizers: dict[str, DiffusionActivationQuantizer] = {}
    tools.logging.Formatter.indent_inc()
    for module_key, tensor_type, modules, module_names, eval_module, eval_name, eval_kwargs, orig_wgts in args_caches:
        if isinstance(modules[0], nn.Linear):
            channels_dim = -1
            assert all(isinstance(m, nn.Linear) for m in modules)
        elif isinstance(modules[0], nn.Conv2d):
            channels_dim = 1
            assert all(isinstance(m, nn.Conv2d) for m in modules)
        else:
            raise ValueError(f"Unknown module type: {type(modules[0])}")
        if tensor_type == TensorType.Inputs:
            cache_keys = [f"{name}.input" for name in module_names]
            quantizer_config = config.unsigned_ipts if getattr(modules[0], "unsigned", False) else config.ipts
            activations = layer_cache.get(module_names[0], IOTensorsCache()).inputs
        else:
            cache_keys = [f"{name}.output" for name in module_names]
            quantizer_config = config.opts
            activations = layer_cache.get(module_names[0], IOTensorsCache()).outputs
        quantizer = DiffusionActivationQuantizer(
            quantizer_config,
            channels_dim=channels_dim,
            develop_dtype=config.develop_dtype,
            key=module_key,
            tensor_type=tensor_type,
        )
        if quantizer.is_enabled():
            if cache_keys[0] not in quantizer_state_dict:
                logger.debug("- Calibrating %s", ", ".join(cache_keys))
                quantizer.calibrate_dynamic_range(
                    modules=modules,
                    activations=activations,
                    eval_module=eval_module,
                    eval_inputs=layer_cache[eval_name].inputs if layer_cache else None,
                    eval_kwargs=eval_kwargs,
                    orig_weights=orig_wgts,
                )
                quantizer_state_dict[cache_keys[0]] = quantizer.state_dict()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                quantizer.load_state_dict(quantizer_state_dict[cache_keys[0]], device=modules[0].weight.device)
            for cache_key in cache_keys:
                quantizers[cache_key] = quantizer
        del quantizer
    tools.logging.Formatter.indent_dec()
    return quantizers


@torch.inference_mode()
def quantize_diffusion_activations(
    model: nn.Module | DiffusionModelStruct,
    config: DiffusionQuantConfig,
    quantizer_state_dict: dict[str, dict[str, torch.Tensor | float | None]] | None = None,
    orig_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor | float | None]]:
    """Quantize the activations of a diffusion model.

    Args:
        model (`nn.Module` or `DiffusionModelStruct`):
            The diffusion model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        quantizer_state_dict (`dict[str, dict[str, torch.Tensor | float | None]]`, *optional*, defaults to `None`):
            The activation quantizers state dict cache.
        orig_state_dict (`dict[str, torch.Tensor]`, *optional*, defaults to `None`):
            The original state dictionary.

    Returns:
        `dict[str, dict[str, torch.Tensor | float | None]]`:
            The activation quantizers state dict cache.
    """
    logger = tools.logging.getLogger(f"{__name__}.ActivationQuant")
    adapter = ensure_model_adapter(model)
    model = adapter.struct
    quantizer_state_dict = quantizer_state_dict or {}
    quantizers: dict[str, DiffusionActivationQuantizer] = {}
    skip_pre_modules = all(key in config.ipts.skips for key in adapter.get_prev_keys())
    skip_post_modules = all(key in config.ipts.skips for key in adapter.get_post_keys())
    if not quantizer_state_dict and config.needs_acts_quantizer_cache:
        activation_plan = adapter.get_activation_plan(
            skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
        )
        with tools.logging.redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader().iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model, config=config),
                    needs_outputs_fn=get_needs_outputs_fn(model, config=config),
                    skip_pre_modules=skip_pre_modules,
                    skip_post_modules=skip_post_modules,
                ),
                desc="quantizing activations",
                leave=False,
                total=len(activation_plan.layers),
                dynamic_ncols=True,
            ):
                block_quantizers = quantize_diffusion_block_activations(
                    layer=layer,
                    config=config,
                    quantizer_state_dict=quantizer_state_dict,
                    layer_cache=layer_cache,
                    layer_kwargs=layer_kwargs,
                    orig_state_dict=orig_state_dict,
                )
                quantizers.update(block_quantizers)
    else:
        for _, layer in adapter.get_named_layers(
            skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
        ).items():
            block_quantizers = quantize_diffusion_block_activations(
                layer=layer,
                config=config,
                quantizer_state_dict=quantizer_state_dict,
                orig_state_dict=orig_state_dict,
            )
            quantizers.update(block_quantizers)
    for _, module_name, module, _, _ in model.named_key_modules():
        ipts_quantizer = quantizers.get(f"{module_name}.input", None)
        opts_quantizer = quantizers.get(f"{module_name}.output", None)
        needs_quant_ipts = ipts_quantizer is not None and ipts_quantizer.is_enabled()
        needs_quant_opts = opts_quantizer is not None and opts_quantizer.is_enabled()
        if needs_quant_ipts or needs_quant_opts:
            logger.debug(
                "- Quantizing %s (%s)",
                module_name,
                ("inputs" if needs_quant_ipts else "")
                + (" and " if needs_quant_ipts and needs_quant_opts else "")
                + ("outputs" if needs_quant_opts else ""),
            )
            if needs_quant_ipts:
                ipts_quantizer.as_hook(is_output=False).register(module)
            if needs_quant_opts:
                opts_quantizer.as_hook(is_output=True).register(module)
    return quantizer_state_dict

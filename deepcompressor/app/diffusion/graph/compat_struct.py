# -*- coding: utf-8 -*-

from __future__ import annotations

import torch.nn as nn

from ..nn.struct import DiffusionModelStruct
from .adapter import DiffusionModelAdapter
from .registry import resolve_model_adapter
from .types import ActivationLayer, ActivationPlan, QuantGroup, QuantNode

__all__ = ["StructModelAdapter", "ensure_model_adapter", "get_default_key_map"]


def get_default_key_map() -> dict[str, set[str]]:
    return DiffusionModelStruct._get_default_key_map()


def ensure_model_adapter(model: nn.Module | DiffusionModelStruct | DiffusionModelAdapter) -> DiffusionModelAdapter:
    if isinstance(model, DiffusionModelAdapter):
        return model
    if isinstance(model, DiffusionModelStruct):
        return StructModelAdapter(model)
    adapter = resolve_model_adapter(model)
    return adapter if adapter is not None else StructModelAdapter(model)


class StructModelAdapter(DiffusionModelAdapter):
    def __init__(self, model: nn.Module | DiffusionModelStruct) -> None:
        self.struct = model if isinstance(model, DiffusionModelStruct) else DiffusionModelStruct.construct(model)

    def get_root_module(self) -> nn.Module:
        return self.struct.module

    def iter_nodes(self):
        for key, name, module, parent, field_name in self.struct.named_key_modules():
            if isinstance(module, nn.Linear):
                kind, channels_dim = "linear", -1
            elif isinstance(module, nn.Conv2d):
                kind, channels_dim = "conv2d", 1
            elif isinstance(module, nn.Conv3d):
                kind, channels_dim = "conv3d", 1
            else:
                kind, channels_dim = "other", None
            yield QuantNode(
                key=key,
                name=name,
                module=module,
                parent=parent,
                parent_name=parent.name,
                field_name=field_name,
                kind=kind,
                channels_dim=channels_dim,
            )

    def iter_groups(self):
        for node in self.iter_nodes():
            yield QuantGroup(key=node.key, name=node.name, kind="single", nodes=(node,), eval_name=node.name)

    def get_activation_plan(self, *, skip_pre_modules: bool, skip_post_modules: bool) -> ActivationPlan:
        layers, structs, recomputes, uses = self.struct.get_iter_layer_activations_args(
            skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules
        )
        entries = tuple(
            ActivationLayer(
                name=layer_struct.name,
                module=layer,
                needs_recompute=recompute,
                use_prev_layer_outputs=use_prev_layer_outputs,
                ref=layer_struct,
            )
            for layer, layer_struct, recompute, use_prev_layer_outputs in zip(
                layers, structs, recomputes, uses, strict=True
            )
        )
        return ActivationPlan(layers=entries, layer_names=tuple(entry.name for entry in entries))

    def get_named_layers(
        self, *, skip_pre_modules: bool, skip_post_modules: bool, skip_blocks: bool = False
    ) -> dict[str, object]:
        return self.struct.get_named_layers(
            skip_pre_modules=skip_pre_modules, skip_post_modules=skip_post_modules, skip_blocks=skip_blocks
        )

    def get_prev_keys(self) -> tuple[str, ...]:
        return self.struct.get_prev_module_keys()

    def get_post_keys(self) -> tuple[str, ...]:
        return self.struct.get_post_module_keys()

    def get_key_map(self) -> dict[str, set[str]]:
        return type(self.struct)._get_default_key_map()

    def iter_transformer_block_structs(self):
        return self.struct.iter_transformer_block_structs()

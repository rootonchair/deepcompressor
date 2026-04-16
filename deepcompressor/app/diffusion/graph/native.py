# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch.nn as nn

from .adapter import DiffusionModelAdapter
from .registry import register_model_adapter
from .types import ActivationLayer, ActivationPlan, QuantGroup, QuantNode

__all__ = ["NativeDiffusionAdapter", "attach_adapter"]


@dataclass
class NativeDiffusionAdapter(DiffusionModelAdapter):
    root_module: nn.Module
    nodes: tuple[QuantNode, ...]
    activation_layers: tuple[ActivationLayer, ...]
    named_layers: OrderedDict[str, object]
    prev_keys: tuple[str, ...] = ()
    post_keys: tuple[str, ...] = ()
    key_map: dict[str, set[str]] = field(default_factory=dict)
    groups: tuple[QuantGroup, ...] = ()
    transformer_blocks: tuple[object, ...] = ()
    pre_layer_names: tuple[str, ...] = ()
    post_layer_names: tuple[str, ...] = ()

    def get_root_module(self) -> nn.Module:
        return self.root_module

    def iter_nodes(self):
        return iter(self.nodes)

    def iter_groups(self):
        if self.groups:
            return iter(self.groups)
        return iter(
            QuantGroup(key=node.key, name=node.name, kind="single", nodes=(node,), eval_name=node.name)
            for node in self.nodes
        )

    def get_activation_plan(self, *, skip_pre_modules: bool, skip_post_modules: bool) -> ActivationPlan:
        layers = []
        for layer in self.activation_layers:
            if skip_pre_modules and layer.name in self.pre_layer_names:
                continue
            if skip_post_modules and layer.name in self.post_layer_names:
                continue
            layers.append(layer)
        layers = tuple(layers)
        return ActivationPlan(layers=layers, layer_names=tuple(layer.name for layer in layers))

    def get_named_layers(
        self, *, skip_pre_modules: bool, skip_post_modules: bool, skip_blocks: bool = False
    ) -> dict[str, object]:
        named_layers: OrderedDict[str, object] = OrderedDict()
        for name, layer in self.named_layers.items():
            if skip_pre_modules and name in self.pre_layer_names:
                continue
            if skip_post_modules and name in self.post_layer_names:
                continue
            if skip_blocks and name not in self.pre_layer_names and name not in self.post_layer_names:
                continue
            named_layers[name] = layer
        return named_layers

    def get_prev_keys(self) -> tuple[str, ...]:
        return self.prev_keys

    def get_post_keys(self) -> tuple[str, ...]:
        return self.post_keys

    def get_key_map(self) -> dict[str, set[str]]:
        return self.key_map

    def iter_transformer_block_structs(self):
        return iter(self.transformer_blocks)

    @classmethod
    def from_named_layers(
        cls,
        root_module: nn.Module,
        *,
        named_layers: OrderedDict[str, object],
        prev_keys: tuple[str, ...] = (),
        post_keys: tuple[str, ...] = (),
        key_map: dict[str, set[str]] | None = None,
        pre_layer_names: tuple[str, ...] = (),
        post_layer_names: tuple[str, ...] = (),
    ) -> "NativeDiffusionAdapter":
        activation_layers = tuple(
            ActivationLayer(name=name, module=layer, needs_recompute=False, use_prev_layer_outputs=False, ref=layer)
            for name, layer in named_layers.items()
            if isinstance(layer, nn.Module)
        )
        nodes: list[QuantNode] = []
        for layer_name, layer in named_layers.items():
            if not isinstance(layer, nn.Module):
                continue
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                modules = [("", layer)]
            else:
                modules = list(layer.named_modules())
            for name, module in modules:
                if not name and module is not layer:
                    continue
                if isinstance(module, nn.Linear):
                    kind, channels_dim = "linear", -1
                elif isinstance(module, nn.Conv2d):
                    kind, channels_dim = "conv2d", 1
                elif isinstance(module, nn.Conv3d):
                    kind, channels_dim = "conv3d", 1
                else:
                    continue
                module_name = layer_name if not name else f"{layer_name}.{name}"
                nodes.append(
                    QuantNode(
                        key=module_name,
                        name=module_name,
                        module=module,
                        parent=layer,
                        parent_name=layer_name,
                        field_name=name or layer_name,
                        kind=kind,
                        channels_dim=channels_dim,
                    )
                )
        return cls(
            root_module=root_module,
            nodes=tuple(nodes),
            activation_layers=activation_layers,
            named_layers=named_layers,
            prev_keys=prev_keys,
            post_keys=post_keys,
            key_map=key_map or {node.key: {node.key} for node in nodes},
            pre_layer_names=pre_layer_names,
            post_layer_names=post_layer_names,
        )


def attach_adapter(model: nn.Module, adapter: DiffusionModelAdapter) -> nn.Module:
    model._deepcompressor_adapter = adapter
    return model


@register_model_adapter
def _resolve_attached_adapter(model: nn.Module) -> DiffusionModelAdapter | None:
    adapter = getattr(model, "_deepcompressor_adapter", None)
    return adapter if isinstance(adapter, DiffusionModelAdapter) else None

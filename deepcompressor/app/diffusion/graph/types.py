# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

NodeKind = Literal["linear", "conv2d", "conv3d", "attention", "ffn", "module", "other"]
GroupKind = Literal["single", "qkv", "add_qkv", "ffn_up", "ffn_down", "parallel_block", "layer"]


@dataclass(frozen=True)
class QuantNode:
    key: str
    name: str
    module: nn.Module
    parent: object
    parent_name: str
    field_name: str
    kind: NodeKind
    channels_dim: int | None
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class QuantGroup:
    key: str
    name: str
    kind: GroupKind
    nodes: tuple[QuantNode, ...]
    eval_name: str
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ActivationLayer:
    name: str
    module: nn.Module
    needs_recompute: bool
    use_prev_layer_outputs: bool
    ref: object | None = None


@dataclass(frozen=True)
class ActivationPlan:
    layers: tuple[ActivationLayer, ...]
    layer_names: tuple[str, ...]

# -*- coding: utf-8 -*-

from .adapter import DiffusionModelAdapter
from .compat_struct import StructModelAdapter, ensure_model_adapter, get_default_key_map
from .groups import (
    ActivationGroupSpec,
    LowRankGroupSpec,
    iter_activation_group_specs,
    iter_low_rank_group_specs,
    resolve_eval_scope,
)
from .native import NativeDiffusionAdapter, attach_adapter
from .registry import register_model_adapter, resolve_model_adapter
from .types import ActivationLayer, ActivationPlan, QuantGroup, QuantNode

__all__ = [
    "ActivationLayer",
    "ActivationPlan",
    "ActivationGroupSpec",
    "DiffusionModelAdapter",
    "LowRankGroupSpec",
    "NativeDiffusionAdapter",
    "QuantGroup",
    "QuantNode",
    "StructModelAdapter",
    "attach_adapter",
    "ensure_model_adapter",
    "get_default_key_map",
    "iter_activation_group_specs",
    "iter_low_rank_group_specs",
    "register_model_adapter",
    "resolve_eval_scope",
    "resolve_model_adapter",
]

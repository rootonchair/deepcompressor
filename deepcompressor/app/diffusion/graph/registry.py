# -*- coding: utf-8 -*-

from __future__ import annotations

import typing as tp

import torch.nn as nn

from .adapter import DiffusionModelAdapter

__all__ = ["register_model_adapter", "resolve_model_adapter"]

AdapterFactory = tp.Callable[[nn.Module], DiffusionModelAdapter | None]

_ADAPTER_FACTORIES: list[AdapterFactory] = []


def register_model_adapter(factory: AdapterFactory) -> AdapterFactory:
    _ADAPTER_FACTORIES.append(factory)
    return factory


def resolve_model_adapter(model: nn.Module) -> DiffusionModelAdapter | None:
    for factory in _ADAPTER_FACTORIES:
        adapter = factory(model)
        if adapter is not None:
            return adapter
    return None

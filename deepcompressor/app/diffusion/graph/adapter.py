# -*- coding: utf-8 -*-

from typing import Any, Iterable, Protocol, runtime_checkable

import torch.nn as nn

from .types import ActivationPlan, QuantGroup, QuantNode


@runtime_checkable
class DiffusionModelAdapter(Protocol):
    def get_root_module(self) -> nn.Module: ...

    def iter_nodes(self) -> Iterable[QuantNode]: ...

    def iter_groups(self) -> Iterable[QuantGroup]: ...

    def get_activation_plan(self, *, skip_pre_modules: bool, skip_post_modules: bool) -> ActivationPlan: ...

    def get_named_layers(
        self, *, skip_pre_modules: bool, skip_post_modules: bool, skip_blocks: bool = False
    ) -> dict[str, Any]: ...

    def get_prev_keys(self) -> tuple[str, ...]: ...

    def get_post_keys(self) -> tuple[str, ...]: ...

    def get_key_map(self) -> dict[str, set[str]]: ...

    def iter_transformer_block_structs(self) -> Iterable[Any]: ...

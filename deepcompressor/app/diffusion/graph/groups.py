# -*- coding: utf-8 -*-

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..nn.struct import (
    DiffusionAttentionStruct,
    DiffusionBlockStruct,
    DiffusionModuleStruct,
    DiffusionTransformerBlockStruct,
)

__all__ = [
    "ActivationGroupSpec",
    "LowRankGroupSpec",
    "resolve_eval_scope",
    "iter_activation_group_specs",
    "iter_low_rank_group_specs",
]


def _wrap_joint_attn(attn: nn.Module, /, *, indexes: int | tuple[int, ...] = 1) -> tp.Callable:
    if isinstance(indexes, int):

        def eval(*args, **kwargs) -> torch.Tensor:
            return attn(*args, **kwargs)[indexes]

    else:

        def eval(*args, **kwargs) -> tuple[torch.Tensor, ...]:
            tensors = attn(*args, **kwargs)
            result = torch.concat([tensors[i] for i in indexes], dim=-2)
            return result

    return eval


@dataclass(frozen=True)
class ActivationGroupSpec:
    key: str
    module_names: tuple[str, ...]
    modules: tuple[nn.Linear | nn.Conv2d, ...]
    eval_module: nn.Module
    eval_name: str
    eval_kwargs: dict[str, tp.Any] | None
    orig_weight_refs: tuple[tuple[nn.Parameter, str], ...]


@dataclass(frozen=True)
class LowRankGroupSpec:
    key: str
    module_names: tuple[str, ...]
    modules: tuple[nn.Linear | nn.Conv2d, ...]
    eval_module: nn.Module
    eval_name: str
    eval_kwargs: dict[str, tp.Any] | None


def resolve_eval_scope(
    *,
    module: nn.Module,
    module_name: str,
    parent: object,
    field_name: str,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> tuple[nn.Module, str, dict[str, tp.Any] | None]:
    if field_name.endswith(("q_proj", "k_proj")):
        assert isinstance(parent, DiffusionAttentionStruct)
        if parent.parent.parallel and parent.idx == 0:
            eval_module = parent.parent.module
            eval_name = parent.parent.name
            eval_kwargs = layer_kwargs
        else:
            eval_module = parent.module
            eval_name = parent.name
            eval_kwargs = parent.filter_kwargs(layer_kwargs) if layer_kwargs else {}
        if parent.is_joint_attn() and "add_" in field_name:
            eval_module = _wrap_joint_attn(eval_module, indexes=1)
    else:
        eval_module, eval_name, eval_kwargs = module, module_name, None
    return eval_module, eval_name, eval_kwargs


def iter_activation_group_specs(
    layer: DiffusionBlockStruct | DiffusionModuleStruct,
    *,
    layer_kwargs: dict[str, tp.Any] | None = None,
) -> tp.Generator[ActivationGroupSpec, None, None]:
    used_modules: set[nn.Module] = set()
    layer_kwargs = layer_kwargs or {}
    for module_key, module_name, module, parent, field_name in layer.named_key_modules():
        modules: tuple[nn.Linear | nn.Conv2d, ...] | None = None
        module_names: tuple[str, ...] = ()
        orig_weight_refs: tuple[tuple[nn.Parameter, str], ...] = ()
        if field_name in ("k_proj", "v_proj", "add_q_proj", "add_v_proj"):
            continue
        if field_name in ("q_proj", "add_k_proj", "up_proj"):
            grandparent = parent.parent
            assert isinstance(grandparent, DiffusionTransformerBlockStruct)
            if grandparent.parallel and parent.idx == 0:
                orig_weight_refs = tuple(
                    (proj_module.weight, proj_name)
                    for _, proj_name, proj_module, _, _ in grandparent.named_key_modules()
                )
                if field_name == "q_proj":
                    assert isinstance(parent, DiffusionAttentionStruct)
                    assert module_name == parent.q_proj_name
                    modules = tuple(parent.qkv_proj)
                    module_names = tuple(parent.qkv_proj_names)
                    if grandparent.ffn_struct is not None:
                        modules = (*modules, grandparent.ffn_struct.up_proj)
                        module_names = (*module_names, grandparent.ffn_struct.up_proj_name)
                elif field_name == "add_k_proj":
                    assert isinstance(parent, DiffusionAttentionStruct)
                    assert module_name == parent.add_k_proj_name
                    modules = tuple(parent.add_qkv_proj)
                    module_names = tuple(parent.add_qkv_proj_names)
                    if grandparent.add_ffn_struct is not None:
                        modules = (*modules, grandparent.add_ffn_struct.up_proj)
                        module_names = (*module_names, grandparent.add_ffn_struct.up_proj_name)
                else:
                    assert field_name == "up_proj"
                    if module in used_modules:
                        continue
                    assert module_name == grandparent.add_ffn_struct.up_proj_name
                    assert grandparent.attn_structs[0].is_self_attn()
                eval_module, eval_name, eval_kwargs = grandparent.module, grandparent.name, layer_kwargs
            elif isinstance(parent, DiffusionAttentionStruct):
                eval_module, eval_name, eval_kwargs = resolve_eval_scope(
                    module=module,
                    module_name=module_name,
                    parent=parent,
                    field_name=field_name,
                    layer_kwargs=layer_kwargs,
                )
                orig_weight_refs = tuple(
                    (proj_module.weight, proj_name) for _, proj_name, proj_module, _, _ in parent.named_key_modules()
                )
                if field_name == "q_proj":
                    assert module_name == parent.q_proj_name
                    modules = tuple(parent.qkv_proj)
                    module_names = tuple(parent.qkv_proj_names)
                else:
                    assert field_name == "add_k_proj"
                    assert module_name == parent.add_k_proj_name
                    modules = tuple(parent.add_qkv_proj)
                    module_names = tuple(parent.add_qkv_proj_names)
        if modules is None:
            assert module not in used_modules
            used_modules.add(module)
            assert isinstance(module, (nn.Linear, nn.Conv2d))
            yield ActivationGroupSpec(
                key=module_key,
                module_names=(module_name,),
                modules=(module,),
                eval_module=module,
                eval_name=module_name,
                eval_kwargs=None,
                orig_weight_refs=((module.weight, module_name),),
            )
        else:
            for grouped_module in modules:
                assert grouped_module not in used_modules
                used_modules.add(grouped_module)
            yield ActivationGroupSpec(
                key=module_key,
                module_names=module_names,
                modules=modules,
                eval_module=eval_module,
                eval_name=eval_name,
                eval_kwargs=eval_kwargs,
                orig_weight_refs=orig_weight_refs,
            )


def iter_low_rank_group_specs(
    layer: DiffusionBlockStruct | DiffusionModuleStruct,
    *,
    layer_kwargs: dict[str, tp.Any] | None = None,
    exclusive: bool = False,
) -> tp.Generator[LowRankGroupSpec, None, None]:
    layer_kwargs = layer_kwargs or {}
    for module_key, module_name, module, parent, field_name in layer.named_key_modules():
        modules: tuple[nn.Linear | nn.Conv2d, ...] = (module,)
        module_names: tuple[str, ...] = (module_name,)
        if not exclusive and field_name.endswith(("q_proj", "k_proj", "v_proj")):
            assert isinstance(parent, DiffusionAttentionStruct)
            if parent.is_self_attn():
                if field_name == "q_proj":
                    modules = tuple(parent.qkv_proj)
                    module_names = tuple(parent.qkv_proj_names)
                else:
                    continue
            elif parent.is_cross_attn():
                if field_name == "add_k_proj":
                    modules = (module, parent.add_v_proj)
                    module_names = (module_name, parent.add_v_proj_name)
                elif field_name != "q_proj":
                    continue
            else:
                assert parent.is_joint_attn()
                if field_name == "q_proj":
                    modules = tuple(parent.qkv_proj)
                    module_names = tuple(parent.qkv_proj_names)
                elif field_name == "add_k_proj":
                    modules = tuple(parent.add_qkv_proj)
                    module_names = tuple(parent.add_qkv_proj_names)
                else:
                    continue
        eval_module, eval_name, eval_kwargs = resolve_eval_scope(
            module=module,
            module_name=module_name,
            parent=parent,
            field_name=field_name,
            layer_kwargs=layer_kwargs,
        )
        yield LowRankGroupSpec(
            key=module_key,
            module_names=module_names,
            modules=modules,
            eval_module=eval_module,
            eval_name=eval_name,
            eval_kwargs=eval_kwargs,
        )

import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2SingleTransformerBlock,
)

from deepcompressor.nn.patch.conv import ConcatConv2d, ShiftedConv2d
from deepcompressor.nn.patch.flux2_attn import PatchedFlux2ParallelSelfAttention
from deepcompressor.nn.patch.linear import ConcatLinear, ShiftedLinear
from deepcompressor.utils import patch, tools

from .attention import DiffusionAttentionProcessor
from .struct import DiffusionFeedForwardStruct, DiffusionModelStruct, DiffusionResnetStruct, UNetStruct

__all__ = [
    "replace_up_block_conv_with_concat_conv",
    "replace_fused_linear_with_concat_linear",
    "replace_flux2_parallel_attn",
    "replace_attn_processor",
    "shift_input_activations",
]


def replace_up_block_conv_with_concat_conv(model: nn.Module) -> None:
    """Replace up_block convolutions in UNet with ConcatConv."""
    model_struct = DiffusionModelStruct.construct(model)
    if not isinstance(model_struct, UNetStruct):
        return
    logger = tools.logging.getLogger(__name__)
    logger.info("Replacing up_block convolutions with ConcatConv.")
    tools.logging.Formatter.indent_inc()
    parents_map = patch.get_module_parents_map(model)
    for up_block in model_struct.up_block_structs:
        logger.info(f"+ Replacing convolutions in up_block {up_block.name}")
        tools.logging.Formatter.indent_inc()
        for resnet in up_block.resnet_structs:
            assert len(resnet.convs[0]) == 1
            conv, conv_name = resnet.convs[0][0], resnet.conv_names[0][0]
            logger.info(f"- Replacing {conv_name} in resnet {resnet.name}")
            tools.logging.Formatter.indent_inc()
            if resnet.idx == 0:
                if up_block.idx == 0:
                    prev_block = model_struct.mid_block_struct
                else:
                    prev_block = model_struct.up_block_structs[up_block.idx - 1]
                logger.info(f"+ using previous block {prev_block.name}")
                prev_channels = prev_block.resnet_structs[-1].convs[-1][-1].out_channels
            else:
                prev_channels = up_block.resnet_structs[resnet.idx - 1].convs[-1][-1].out_channels
            logger.info(f"+ conv_in_channels = {prev_channels}/{conv.in_channels}")
            logger.info(f"+ conv_out_channels = {conv.out_channels}")
            concat_conv = ConcatConv2d.from_conv2d(conv, [prev_channels])
            for parent_name, parent_module, child_name in parents_map[conv]:
                logger.info(f"+ replacing {child_name} in {parent_name}")
                setattr(parent_module, child_name, concat_conv)
            tools.logging.Formatter.indent_dec()
        tools.logging.Formatter.indent_dec()
    tools.logging.Formatter.indent_dec()


def replace_fused_linear_with_concat_linear(model: nn.Module) -> None:
    """Replace fused Linear in FluxSingleTransformerBlock / Flux2SingleTransformerBlock with ConcatLinear."""
    logger = tools.logging.getLogger(__name__)
    logger.info("Replacing fused Linear with ConcatLinear.")
    tools.logging.Formatter.indent_inc()
    for name, module in model.named_modules():
        if isinstance(module, FluxSingleTransformerBlock):
            logger.info(f"+ Replacing fused Linear in {name} with ConcatLinear.")
            tools.logging.Formatter.indent_inc()
            logger.info(f"- in_features = {module.proj_out.out_features}/{module.proj_out.in_features}")
            logger.info(f"- out_features = {module.proj_out.out_features}")
            tools.logging.Formatter.indent_dec()
            module.proj_out = ConcatLinear.from_linear(module.proj_out, [module.proj_out.out_features])
        elif isinstance(module, Flux2SingleTransformerBlock):
            attn = module.attn
            inner_dim = attn.inner_dim
            mlp_hidden_dim = attn.mlp_hidden_dim
            # Split to_out along input features into [attn_out, mlp_out]
            # to_out takes concatenated [attn_output | mlp_output] as input
            logger.info(f"+ Replacing fused to_out in {name}.attn with ConcatLinear.")
            tools.logging.Formatter.indent_inc()
            logger.info(f"- in_features = [{inner_dim}, {mlp_hidden_dim}]")
            logger.info(f"- out_features = {attn.to_out.out_features}")
            tools.logging.Formatter.indent_dec()
            attn.to_out = ConcatLinear.from_linear(attn.to_out, [inner_dim])
    tools.logging.Formatter.indent_dec()


def replace_flux2_parallel_attn(model: nn.Module) -> None:
    """Replace Flux2ParallelSelfAttention with PatchedFlux2ParallelSelfAttention.

    Splits the fused to_qkv_mlp_proj into separate to_q, to_k, to_v, mlp_proj linears.
    Must be called after replace_fused_linear_with_concat_linear (which splits to_out).
    """
    logger = tools.logging.getLogger(__name__)
    logger.info("Replacing Flux2ParallelSelfAttention with patched version.")
    tools.logging.Formatter.indent_inc()
    for name, module in model.named_modules():
        if isinstance(module, Flux2SingleTransformerBlock):
            logger.info(f"+ Patching {name}.attn")
            module.attn = PatchedFlux2ParallelSelfAttention(module.attn)
    tools.logging.Formatter.indent_dec()


def shift_input_activations(model: nn.Module) -> None:
    """Shift input activations of convolutions and linear layers if their lowerbound is negative.

    Args:
        model (nn.Module): model to shift input activations.
    """
    logger = tools.logging.getLogger(__name__)
    model_struct = DiffusionModelStruct.construct(model)
    module_parents_map = patch.get_module_parents_map(model)
    logger.info("- Shifting input activations.")
    tools.logging.Formatter.indent_inc()
    for _, module_name, module, parent, field_name in model_struct.named_key_modules():
        lowerbound = None
        if isinstance(parent, DiffusionResnetStruct) and field_name.startswith("conv"):
            lowerbound = parent.config.intermediate_lowerbound
        elif isinstance(parent, DiffusionFeedForwardStruct) and field_name.startswith("down_proj"):
            lowerbound = parent.config.intermediate_lowerbound
        if lowerbound is not None and lowerbound < 0:
            shift = -lowerbound
            logger.info(f"+ Shifting input activations of {module_name} by {shift}")
            tools.logging.Formatter.indent_inc()
            if isinstance(module, nn.Linear):
                shifted = ShiftedLinear.from_linear(module, shift=shift)
                shifted.linear.unsigned = True
            elif isinstance(module, nn.Conv2d):
                shifted = ShiftedConv2d.from_conv2d(module, shift=shift)
                shifted.conv.unsigned = True
            else:
                raise NotImplementedError(f"Unsupported module type {type(module)}")
            for parent_name, parent_module, child_name in module_parents_map[module]:
                logger.info(f"+ Replacing {child_name} in {parent_name}")
                setattr(parent_module, child_name, shifted)
            tools.logging.Formatter.indent_dec()
    tools.logging.Formatter.indent_dec()


def replace_attn_processor(model: nn.Module) -> None:
    """Replace Attention processor with DiffusionAttentionProcessor."""
    logger = tools.logging.getLogger(__name__)
    logger.info("Replacing Attention processors.")
    tools.logging.Formatter.indent_inc()
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            logger.info(f"+ Replacing {name} processor with DiffusionAttentionProcessor.")
            module.set_processor(DiffusionAttentionProcessor(module.processor))
        elif isinstance(module, Flux2Attention):
            logger.info(f"+ Replacing {name} processor with DiffusionAttentionProcessor.")
            module.set_processor(DiffusionAttentionProcessor(module.processor))
    tools.logging.Formatter.indent_dec()

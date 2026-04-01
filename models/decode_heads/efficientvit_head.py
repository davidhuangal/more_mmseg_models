# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
# Source: https://github.com/mit-han-lab/efficientvit

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS

from models.backbones.efficientvit import (
    ConvLayer,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
    list_sum,
)

__all__ = ["EfficientViTHead"]


@MODELS.register_module()
class EfficientViTHead(BaseDecodeHead):
    def __init__(
        self,
        in_channels: Sequence[int],
        stride_list: List[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: int | None,
        norm: str = "bn2d",
        act_func: str = "hswish",
        merge: str = "add",
        **kwargs,
    ) -> None:
        super().__init__(
            input_transform="multiple_select",
            in_channels=in_channels,
            channels=head_width * (final_expand or 1),
            **kwargs,
        )

        index_list = self._normalize_in_index(self.in_index)
        if len(index_list) != len(in_channels):
            raise ValueError(
                "Length of in_channels must match number of selected indices"
            )
        if len(index_list) != len(stride_list):
            raise ValueError(
                "Length of stride_list must match number of selected indices"
            )

        self.selected_indices = index_list
        input_modules: List[nn.Module] = []
        for in_channel, stride in zip(in_channels, stride_list):
            factor = stride // head_stride
            if factor == 1:
                module = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                module = OpSequential(
                    [
                        ConvLayer(
                            in_channel,
                            head_width,
                            1,
                            norm=norm,
                            act_func=None,
                        ),
                        UpSampleLayer(factor=factor),
                    ]
                )
            input_modules.append(module)

        middle_blocks = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError(f"Unsupported middle_op: {middle_op}")
            middle_blocks.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle_blocks)

        self.input_ops = nn.ModuleList(input_modules)
        self.merge = merge
        self.post_input = None

        self.middle = middle

        self.output_ops = nn.ModuleList(  # single head projection
            [
                OpSequential(
                    [
                        None
                        if final_expand is None
                        else ConvLayer(
                            head_width,
                            head_width * final_expand,
                            1,
                            norm=norm,
                            act_func=act_func,
                        ),
                    ]
                )
            ]
        )

    @staticmethod
    def _normalize_in_index(in_index: int | Sequence[int]) -> List[int]:
        if isinstance(in_index, Sequence) and not isinstance(in_index, (str, bytes)):
            index_list = list(in_index)
        else:
            index_list = [int(in_index)]
        index_list = [int(idx) for idx in index_list]
        if not index_list:
            raise ValueError("in_index must select at least one feature map")
        if min(index_list) < 0:
            raise ValueError("in_index values must be non-negative")
        return index_list

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        feat = [
            op(features[idx]) for idx, op in zip(self.selected_indices, self.input_ops)
        ]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError(f"Unsupported merge mode: {self.merge}")
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        out = self.output_ops[0](feat)
        return self.cls_seg(out)

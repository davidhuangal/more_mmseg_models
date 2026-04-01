# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
# Source: https://github.com/mit-han-lab/efficientvit

from __future__ import annotations

import os
from functools import partial
from inspect import signature
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from mmseg.registry import MODELS
from mmengine.model import BaseModule


#################################################################################
#                                  Utilities                                    #
#################################################################################


def build_kwargs_from_config(
    config: dict[str, Any], target_func: Any
) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    return {key: config[key] for key in config if key in valid_keys}


def list_sum(x: Sequence[Any]) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: Sequence[Any]) -> Any:
    return list_sum(x) / len(x)


def weighted_list_sum(x: Sequence[Any], weights: Sequence[float]) -> Any:
    assert len(x) == len(weights)
    return (
        x[0] * weights[0]
        if len(x) == 1
        else x[0] * weights[0] + weighted_list_sum(x[1:], weights[1:])
    )


def list_join(x: Sequence[Any], sep: str = "\t", format_str: str = "%s") -> str:
    return sep.join([format_str % val for val in x])


def val2list(x: Any | Sequence[Any], repeat_time: int = 1) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(
    x: Any | Sequence[Any], min_len: int = 1, idx_repeat: int = -1
) -> tuple[Any, ...]:
    x_list = val2list(x)
    if len(x_list) > 0:
        x_list[idx_repeat:idx_repeat] = [
            x_list[idx_repeat] for _ in range(min_len - len(x_list))
        ]
    return tuple(x_list)


def squeeze_list(x: list[Any] | None) -> Any:
    if x is not None and len(x) == 1:
        return x[0]
    return x


def is_parallel(model: nn.Module) -> bool:
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def get_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_same_padding(kernel_size: int | Sequence[int]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple(get_same_padding(ks) for ks in kernel_size)
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: Sequence[int] | None = None,
    scale_factor: Sequence[float] | None = None,
    mode: str = "bicubic",
    align_corners: bool | None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    if mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def load_state_dict_from_file(
    file: str, only_state_dict: bool = True
) -> dict[str, torch.Tensor]:
    file_path = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file_path, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def torch_randint(low: int, high: int, generator: torch.Generator | None = None) -> int:
    if low == high:
        return low
    assert low < high
    return int(torch.randint(low=low, high=high, generator=generator, size=(1,)))


def torch_random(generator: torch.Generator | None = None) -> float:
    return float(torch.rand(1, generator=generator))


def torch_shuffle(
    src_list: Sequence[Any], generator: torch.Generator | None = None
) -> list[Any]:
    rand_indexes = torch.randperm(len(src_list), generator=generator).tolist()
    return [src_list[i] for i in rand_indexes]


def torch_uniform(
    low: float, high: float, generator: torch.Generator | None = None
) -> float:
    rand_val = torch_random(generator)
    return (high - low) * rand_val + low


def torch_random_choices(
    src_list: Sequence[Any],
    generator: torch.Generator | None = None,
    k: int = 1,
    weight_list: Sequence[float] | None = None,
) -> Any:
    if weight_list is None:
        rand_idx = torch.randint(
            low=0, high=len(src_list), generator=generator, size=(k,)
        )
        out_list = [src_list[i] for i in rand_idx]
    else:
        assert len(weight_list) == len(src_list)
        accumulate_weight_list = np.cumsum(weight_list)
        out_list = []
        for _ in range(k):
            val = torch_uniform(0, float(accumulate_weight_list[-1]), generator)
            active_id = 0
            for i, weight_val in enumerate(accumulate_weight_list):
                active_id = i
                if weight_val > val:
                    break
            out_list.append(src_list[active_id])
    return out_list[0] if k == 1 else out_list


#################################################################################
#                               Activations & Norms                             #
#################################################################################


REGISTERED_ACT_DICT: dict[str, Any] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str | None, **kwargs) -> nn.Module | None:
    if not name:
        return None
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    return None


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


REGISTERED_NORM_DICT: dict[str, Any] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}


def build_norm(
    name: str = "bn2d", num_features: int | None = None, **kwargs
) -> nn.Module | None:
    if name in {"ln", "ln2d"}:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    return None


def set_norm_eps(model: nn.Module, eps: float | None = None) -> None:
    for module in model.modules():
        if isinstance(
            module, (nn.GroupNorm, nn.LayerNorm, nn.modules.batchnorm._BatchNorm)
        ):
            if eps is not None:
                module.eps = eps


#################################################################################
#                                 Core Layers                                   #
#################################################################################


class OpSequential(nn.Module):
    def __init__(self, module_list: Iterable[nn.Module | None]):
        super().__init__()
        self.module_list = nn.ModuleList(
            [module for module in module_list if module is not None]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.module_list:
            x = module(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool | Sequence[bool] = False,
        dropout: float = 0,
        norm: str | None = "bn2d",
        act_func: str | None = "relu",
    ):
        super().__init__()
        padding = get_same_padding(kernel_size)
        padding = (
            padding * dilation
            if isinstance(padding, int)
            else tuple(p * dilation for p in padding)
        )
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias if isinstance(use_bias, bool) else False,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode: str = "bicubic",
        size: Sequence[int] | None = None,
        factor: int = 2,
        align_corners: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.size is not None and tuple(x.shape[-2:]) == tuple(self.size)
        ) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dropout: float = 0,
        norm: str | None = None,
        act_func: str | None = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    @staticmethod
    def _try_squeeze(x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=1) if x.dim() > 2 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool | Sequence[bool] = False,
        norm: str | Sequence[str] = ("bn2d", "bn2d"),
        act_func: str | Sequence[str | None] = ("relu6", None),
    ):
        super().__init__()
        use_bias_tuple = val2tuple(use_bias, 2)
        norm_tuple = val2tuple(norm, 2)
        act_tuple = val2tuple(act_func, 2)
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm_tuple[0],
            act_func=act_tuple[0],
            use_bias=use_bias_tuple[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm_tuple[1],
            act_func=act_tuple[1],
            use_bias=use_bias_tuple[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: int | None = None,
        expand_ratio: float = 6,
        use_bias: bool | Sequence[bool] = False,
        norm: str | Sequence[str] = ("bn2d", "bn2d", "bn2d"),
        act_func: Sequence[str | None] = ("relu6", "relu6", None),
    ):
        super().__init__()
        use_bias_tuple = val2tuple(use_bias, 3)
        norm_tuple = val2tuple(norm, 3)
        act_tuple = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm_tuple[0],
            act_func=act_tuple[0],
            use_bias=use_bias_tuple[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm_tuple[1],
            act_func=act_tuple[1],
            use_bias=use_bias_tuple[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm_tuple[2],
            act_func=act_tuple[2],
            use_bias=use_bias_tuple[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: int | None = None,
        expand_ratio: float = 6,
        groups: int = 1,
        use_bias: bool | Sequence[bool] = False,
        norm: str | Sequence[str] = ("bn2d", "bn2d"),
        act_func: Sequence[str | None] = ("relu6", None),
    ):
        super().__init__()
        use_bias_tuple = val2tuple(use_bias, 2)
        norm_tuple = val2tuple(norm, 2)
        act_tuple = val2tuple(act_func, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias_tuple[0],
            norm=norm_tuple[0],
            act_func=act_tuple[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias_tuple[1],
            norm=norm_tuple[1],
            act_func=act_tuple[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: int | None = None,
        expand_ratio: float = 1,
        use_bias: bool | Sequence[bool] = False,
        norm: str | Sequence[str] = ("bn2d", "bn2d"),
        act_func: Sequence[str | None] = ("relu6", None),
    ):
        super().__init__()
        use_bias_tuple = val2tuple(use_bias, 2)
        norm_tuple = val2tuple(norm, 2)
        act_tuple = val2tuple(act_func, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias_tuple[0],
            norm=norm_tuple[0],
            act_func=act_tuple[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias_tuple[1],
            norm=norm_tuple[1],
            act_func=act_tuple[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 8,
        use_bias: bool | Sequence[bool] = False,
        norm: Sequence[str | None] = (None, "bn2d"),
        act_func: Sequence[str | None] = (None, None),
        kernel_func: str = "relu",
        scales: tuple[int, ...] = (5,),
        eps: float = 1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias_tuple = val2tuple(use_bias, 2)
        norm_tuple = val2tuple(norm, 2)
        act_tuple = val2tuple(act_func, 2)
        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias_tuple[0],
            norm=norm_tuple[0],
            act_func=act_tuple[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias_tuple[0],
                    ),
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        1,
                        groups=3 * heads,
                        bias=use_bias_tuple[0],
                    ),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias_tuple[1],
            norm=norm_tuple[1],
            act_func=act_tuple[1],
        )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = qkv.size()
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        qkv = torch.reshape(qkv, (batch, -1, 3 * self.dim, height * width))
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (batch, -1, height, width))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)
        return out


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim: int = 32,
        expand_ratio: float = 4,
        scales: Sequence[int] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=tuple(scales),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module | None,
        shortcut: nn.Module | None,
        post_act: str | None = None,
        pre_norm: nn.Module | None = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


#################################################################################
#                                   Backbones                                   #
#################################################################################


@MODELS.register_module()
class EfficientViTBackbone(BaseModule):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels: int = 3,
        dim: int = 32,
        expand_ratio: float = 4,
        norm: str = "bn2d",
        act_func: str = "hswish",
        init_cfg: dict = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.width_list: list[int] = []
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        stem_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(stem_channels)

        self.stages: list[nn.Module] = []
        in_channels_stage = stem_channels
        for width, depth in zip(width_list[1:3], depth_list[1:3]):
            stage_modules = []
            for idx in range(depth):
                stride = 2 if idx == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels_stage,
                    out_channels=width,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage_modules.append(block)
                in_channels_stage = width
            self.stages.append(OpSequential(stage_modules))
            self.width_list.append(in_channels_stage)

        for width, depth in zip(width_list[3:], depth_list[3:]):
            stage_modules = []
            block = self.build_local_block(
                in_channels=in_channels_stage,
                out_channels=width,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage_modules.append(ResidualBlock(block, None))
            in_channels_stage = width
            for _ in range(depth):
                stage_modules.append(
                    EfficientViTBlock(
                        in_channels=in_channels_stage,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage_modules))
            self.width_list.append(in_channels_stage)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            return DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        return MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False) if fewer_norm else False,
            norm=(None, None, norm) if fewer_norm else norm,
            act_func=(act_func, act_func, None),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.input_stem(x)
        outputs: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


@MODELS.register_module()
class EfficientViTLargeBackbone(BaseModule):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: list[str] | None = None,
        expand_list: list[float] | None = None,
        fewer_norm_list: list[bool] | None = None,
        in_channels: int = 3,
        qkv_dim: int = 32,
        norm: str = "bn2d",
        act_func: str = "gelu",
        init_cfg: dict = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        block_list = block_list or ["res", "fmb", "fmb", "mb", "att"]
        expand_list = expand_list or [1, 4, 4, 4, 6]
        fewer_norm_list = fewer_norm_list or [False, False, False, True, True]

        self.width_list: list[int] = []
        self.stages: list[nn.Module] = []

        stage0_modules = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block_type=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0_modules.append(ResidualBlock(block, IdentityLayer()))
        in_channels_stage = width_list[0]
        self.stages.append(OpSequential(stage0_modules))
        self.width_list.append(in_channels_stage)

        for stage_id, (width, depth) in enumerate(
            zip(width_list[1:], depth_list[1:]), start=1
        ):
            stage_modules = []
            block = self.build_local_block(
                block_type=(
                    "mb"
                    if block_list[stage_id] not in {"mb", "fmb"}
                    else block_list[stage_id]
                ),
                in_channels=in_channels_stage,
                out_channels=width,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage_modules.append(ResidualBlock(block, None))
            in_channels_stage = width

            for _ in range(depth):
                if block_list[stage_id].startswith("att"):
                    stage_modules.append(
                        EfficientViTBlock(
                            in_channels=in_channels_stage,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    local_block = self.build_local_block(
                        block_type=block_list[stage_id],
                        in_channels=in_channels_stage,
                        out_channels=in_channels_stage,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    stage_modules.append(ResidualBlock(local_block, IdentityLayer()))
            self.stages.append(OpSequential(stage_modules))
            self.width_list.append(in_channels_stage)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block_type: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if block_type == "res":
            return ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        if block_type == "fmb":
            return FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        if block_type == "mb":
            return MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        raise ValueError(block_type)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for stage_id, stage in enumerate(self.stages):
            x = stage(x)
            if stage_id == 0:
                continue
            outputs.append(x)
        return outputs

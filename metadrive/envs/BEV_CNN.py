import math
from typing import List, Tuple, Optional, Literal

try:
    import gymnasium as gym
except ImportError:
    import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# =========================
# Utils
# =========================
def layer_orth_init(layer: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def conv_out_size(size: int, conv: Tuple[int, int, int, int, int]) -> int:
    _, _, k, s, p = conv
    return (size + 2 * p - k) // s + 1


def pool_out_size(size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    return (size + 2 * padding - kernel_size) // stride + 1


# ==========================================
# CNN encoder (与你的 ImageNet 同风格)
# ==========================================
class ImageNet(nn.Sequential):
    def __init__(
        self,
        conv_arch: List[Tuple[int, int, int, int, int]],
        in_dim: int,
        out_dim: int = 256,
        layer_norm: bool = False,
        pool_first: bool = False,
        pool_last: bool = False,
        use_tanh: bool = False,
    ):
        assert len(conv_arch) > 0, "conv_arch 不能为空"

        network = nn.Sequential()
        size = in_dim

        for i, (in_c, out_c, k, s, p) in enumerate(conv_arch):
            network.add_module(f"conv{i}", layer_orth_init(nn.Conv2d(in_c, out_c, k, s, p)))
            size = conv_out_size(size=size, conv=(in_c, out_c, k, s, p))

            if pool_first and i == 0:
                network.add_module("pool_first", nn.MaxPool2d(kernel_size=3, stride=2))
                size = pool_out_size(size, kernel_size=3, stride=2, padding=0)

            network.add_module(f"relu{i}", nn.ReLU())

            if layer_norm:
                network.add_module(f"ln{i}", nn.LayerNorm([out_c, size, size]))

        if pool_last:
            network.add_module("pool_last", nn.MaxPool2d(kernel_size=3, stride=2))
            size = pool_out_size(size, kernel_size=3, stride=2, padding=0)

        network.add_module("flatten", nn.Flatten())
        flat_dim = size * size * out_c
        network.add_module("fc", layer_orth_init(nn.Linear(flat_dim, out_dim), gain=1.0))

        if use_tanh:
            network.add_module("tanh_flat", nn.Tanh())
        if layer_norm:
            network.add_module("ln_flat", nn.LayerNorm(out_dim))

        super().__init__(network)


# ==========================================
# 更鲁棒的 SB3 FeaturesExtractor
# - 动态处理输入通道 C
# - 兼容 HWC / CHW
# ==========================================
class RobustBEVCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        conv_arch: Optional[List[Tuple[int, int, int, int, int]]] = None,
        layer_norm: bool = False,
        pool_first: bool = False,
        pool_last: bool = False,
        use_tanh: bool = False,
        channel_layout: Literal["auto", "channels_last", "channels_first"] = "auto",
        normalize_uint8: bool = True,
        auto_fix_conv_in_channels: bool = True,
    ):
        super().__init__(observation_space, features_dim)

        assert isinstance(observation_space, gym.spaces.Box), "仅支持 Box 观测"
        assert len(observation_space.shape) == 3, f"期望 3D image-like obs，但拿到 {observation_space.shape}"

        H, W, C = None, None, None
        shape = observation_space.shape  # could be (H,W,C) or (C,H,W)

        # ---- 1) 推断通道布局 ----
        if channel_layout == "channels_last":
            H, W, C = shape
            self._channels_last = True
        elif channel_layout == "channels_first":
            C, H, W = shape
            self._channels_last = False
        else:
            # auto: 用一个实用启发式
            # 常见情况：H,W较大(>=32)，C较小(<=16)
            a, b, c = shape
            if c <= 16 and a >= 32 and b >= 32:
                H, W, C = a, b, c
                self._channels_last = True
            elif a <= 16 and b >= 32 and c >= 32:
                C, H, W = a, b, c
                self._channels_last = False
            else:
                # 兜底：默认 channels_last（更常见）
                H, W, C = a, b, c
                self._channels_last = True

        assert H == W, f"当前实现假设输入是方形(H==W)，但得到 H={H}, W={W}"
        self.H = H
        self.C = C
        self.normalize_uint8 = normalize_uint8

        # ---- 2) 构造 / 修正 conv_arch ----
        if conv_arch is None:
            # 默认 Nature-CNN 风格，但第一层 in_channels 动态设为 C
            conv_arch = [
                (C, 32, 8, 4, 0),
                (32, 64, 4, 2, 0),
                (64, 64, 3, 1, 0),
            ]
        else:
            # 如果用户给的 conv_arch 第一层 in_channels 不匹配，默认自动修正更鲁棒
            first_in = conv_arch[0][0]
            if first_in != C:
                if not auto_fix_conv_in_channels:
                    raise AssertionError(f"conv_arch 第一层 in_channels={first_in} 但输入 C={C}")
                # 自动修正：只改第一层 in_channels
                conv_arch = [(C, conv_arch[0][1], conv_arch[0][2], conv_arch[0][3], conv_arch[0][4])] + list(conv_arch[1:])

        self.encoder = ImageNet(
            conv_arch=conv_arch,
            in_dim=H,
            out_dim=features_dim,
            layer_norm=layer_norm,
            pool_first=pool_first,
            pool_last=pool_last,
            use_tanh=use_tanh,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations:
          - channels_last: (B,H,W,C)
          - channels_first: (B,C,H,W)
        """
        x = observations

        # uint8 图像常见，需要归一化
        if self.normalize_uint8 and x.dtype == torch.uint8:
            x = x.float() / 255.0

        # 统一成 (B,C,H,W)
        if self._channels_last:
            x = x.permute(0, 3, 1, 2).contiguous()

        return self.encoder(x)

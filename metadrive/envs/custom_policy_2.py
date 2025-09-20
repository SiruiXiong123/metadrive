import math
from typing import Any, Callable, Dict, Tuple, Union
import gymnasium as gym
import gymnasium.spaces
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.optim
from gymnasium.spaces import Box
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu",)

        image_shape: Box = observation_space.spaces["image"]
        chanels = math.prod(image_shape.shape[-2:])

        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dinov2 = self.dinov2.to(device[0])

        self.embedding_compression_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=64, kernel_size=1, stride=1, bias=False
            ),
            nn.LayerNorm((64, 16, 16)),
            nn.ELU(),
        )

        self.compression_2_and_linear = nn.Sequential(
            nn.Conv2d(
                in_channels=64 * 4, out_channels=32, kernel_size=3, stride=1, bias=False
            ),
            nn.LayerNorm((32, 14, 14)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(in_features=6272, out_features=256, bias=False),
            nn.LayerNorm((256)),
            nn.ELU(),
        )
        self.vector_extractor = nn.Sequential(
            nn.Linear(observation_space.spaces["state"].shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self._features_dim = 256+64

    def forward(self, observations) -> torch.Tensor:
        images: torch.Tensor = observations["image"]
        images = observations["image"]
        chanels_third = images.permute((0, 4, 3, 1, 2))
        shape = chanels_third.shape
        stacked_frames = images.reshape(shape=tuple([shape[0] * shape[1], *shape[2:]]))
        device = next(self.parameters()).device

        with torch.no_grad():
            result = self.dinov2.forward_features(stacked_frames)
            patch_embedings: torch.Tensor = result["x_norm_patchtokens"]


        separated_patches = patch_embedings.reshape(shape=tuple([-1, 16, 16, 384]))
        chanels_second = separated_patches.permute((0, 3, 1, 2))

        res = self.embedding_compression_1(chanels_second)


        res = res.reshape((shape[0], shape[1] * 64, 16, 16))

        res = self.compression_2_and_linear(res)

        vector_features = self.vector_extractor(state)
        combined_features = torch.cat([res, vector_features], dim=-1)


        return combined_features



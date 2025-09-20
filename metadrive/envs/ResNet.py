import math
from typing import Any, Callable, Dict, Tuple, Union

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models, transforms

resnet_model = None  # 全局共享

class CustomResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=128)

        # 正确设置并保存为 self.device
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        image_shape: gym.spaces.Box = observation_space.spaces["image"]
        # 假设 image_shape.shape = (H, W, C, F), F=4

        global resnet_model
        if resnet_model is None:
            pretrained_resnet = models.resnet34(weights='DEFAULT')
            resnet_model = nn.Sequential(*list(pretrained_resnet.children())[:-1])  # 移除fc层，输出512维
            resnet_model = resnet_model.to(self.device)
            resnet_model.eval()  # 冻结，不训练

        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.embeddings_compression = nn.Sequential(
            nn.Linear(in_features=512*4, out_features=64, bias=False),
            nn.LayerNorm((64)),
            nn.GELU(),
        ).to(self.device)

        self.vector_extractor = nn.Sequential(
            nn.Linear(observation_space.spaces["state"].shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        ).to(self.device)

        self._features_dim = 128

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        images: torch.Tensor = observations["image"].to(self.device)  # (B, H, W, C, F=4)
        state: torch.Tensor = observations["state"].to(self.device)

        # Permute to (B, F, C, H, W)
        images = images.permute(0, 4, 3, 1, 2)  # (B, F, C, H, W)

        # Reshape to (B*F, C, H, W)
        B, F, C, H, W = images.shape
        stacked_frames = images.reshape(B * F, C, H, W)

        # Apply preprocessing
        stacked_frames = self.preprocess(stacked_frames)

        # Extract features with frozen ResNet
        with torch.no_grad():
            resnet_features = resnet_model(stacked_frames)  # (B*F, 512, 1, 1)
            resnet_features = resnet_features.squeeze(-1).squeeze(-1)  # (B*F, 512)

        # Reshape back to (B, F*512)
        resnet_features = resnet_features.reshape(B, -1)
        vector_features = self.vector_extractor(state)
        # Compress
        result = self.embeddings_compression(resnet_features)
        result = torch.cat([result, vector_features], dim=-1)
        return result
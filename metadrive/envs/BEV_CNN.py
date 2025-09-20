import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# class CustomBEVCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box):
#         # 输入为 (84, 84, 5)，输出 features_dim=256
#         super().__init__(observation_space, features_dim=256)
#         n_input_channels = observation_space.shape[2]  # 应该是 5

#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # 自动计算 flatten 后的特征维度
#         with torch.no_grad():
#             sample_input = torch.zeros((1, n_input_channels, 84, 84))
#             n_flatten = self.cnn(sample_input).shape[1]

#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, 256),
#             nn.ReLU()
#         )

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         # 将 (B, H, W, C) 转为 (B, C, H, W)
#         x = observations.permute(0, 3, 1, 2)
#         return self.linear(self.cnn(x))
#------------------------------------直接拼接------------------
import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomBEVCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.H, self.W, self.C = observation_space.shape
        assert self.C == 3, f"期望输入 3 通道，但拿到 {self.C}"

        # 定义一个小 CNN 分支（单通道输入）
        def make_branch():
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        # 三个分支
        self.branch1 = make_branch()
        self.branch2 = make_branch()
        self.branch3 = make_branch()

        # 自动计算每个分支的展平维度
        with torch.no_grad():
            sample_input = torch.zeros((1, 1, self.H, self.W))
            n_flatten = self.branch1(sample_input).shape[1]

        # 三个分支拼接 → 全连接层
        self.linear = nn.Sequential(
            nn.Linear(3 * n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) → (B, C, H, W)
        x = observations.permute(0, 3, 1, 2)

        # 拆出三个单通道 (B,1,H,W)
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]
        x3 = x[:, 2:3, :, :]

        # 三个分支
        f1 = self.branch1(x1)
        f2 = self.branch2(x2)
        f3 = self.branch3(x3)

        # 拼接
        features = torch.cat([f1, f2, f3], dim=1)
        return self.linear(features)
#-----------------注意力权重融合-----------------
# import torch
# import torch.nn as nn
# import gym
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# class CustomBEVCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super().__init__(observation_space, features_dim)

#         self.H, self.W, self.C = observation_space.shape
#         assert self.C == 3, f"期望输入 3 通道，但拿到 {self.C}"

#         # 定义单通道 CNN 分支
#         def make_branch():
#             return nn.Sequential(
#                 nn.Conv2d(1, 16, kernel_size=8, stride=4),
#                 nn.ReLU(),
#                 nn.Conv2d(16, 32, kernel_size=4, stride=2),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 32, kernel_size=3, stride=1),
#                 nn.ReLU(),
#                 nn.Flatten(),
#             )

#         # 三个分支
#         self.branch1 = make_branch()
#         self.branch2 = make_branch()
#         self.branch3 = make_branch()

#         # 自动计算每个分支的展平维度
#         with torch.no_grad():
#             sample_input = torch.zeros((1, 1, self.H, self.W))
#             n_flatten = self.branch1(sample_input).shape[1]

#         self.feature_dim = n_flatten

#         # 注意力模块（Squeeze-Excitation 风格）
#         self.attn = nn.Sequential(
#             nn.Linear(n_flatten, n_flatten // 16),
#             nn.ReLU(),
#             nn.Linear(n_flatten // 16, n_flatten),
#             nn.Sigmoid()
#         )

#         # 融合后的全连接映射
#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU()
#         )

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         # (B, H, W, C) → (B, C, H, W)
#         x = observations.permute(0, 3, 1, 2)

#         # 拆成三个单通道 (B,1,H,W)
#         x1 = x[:, 0:1, :, :]
#         x2 = x[:, 1:2, :, :]
#         x3 = x[:, 2:3, :, :]

#         # 三个分支提特征
#         f1 = self.branch1(x1)  # (B, feature_dim)
#         f2 = self.branch2(x2)
#         f3 = self.branch3(x3)

#         # 堆成 (B, 3, feature_dim)
#         feats = torch.stack([f1, f2, f3], dim=1)

#         # 注意力加权
#         weights = self.attn(feats.mean(dim=1))  # (B, feature_dim)
#         weights = weights.unsqueeze(1)          # (B,1,feature_dim)
#         fused = (feats * weights).sum(dim=1)    # (B, feature_dim)

#         # 最终映射
#         return self.linear(fused)

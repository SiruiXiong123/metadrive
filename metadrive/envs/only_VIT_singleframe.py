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
    def __init__(self, observation_space, frames: int = 1, temporal_pool: Union[None, str] = None):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        device = (torch.device("cuda:0") if torch.cuda.is_available() else "cpu",)
        # torch.Size([B, 84, 84, 3, 4])
        # image shape: batch, {x, y}, chanels, time stack 4
        # observation_space can be either a Dict-like with a 'spaces' attribute or a Box-like
        if hasattr(observation_space, 'spaces'):
            image_space = None
            # Try to get 'image' key first
            try:
                image_space = observation_space.spaces.get("image")
            except Exception:
                image_space = None
            if image_space is None:
                # fallback: pick the first Box-like in the dict
                for v in getattr(observation_space, 'spaces', {}).values():
                    if hasattr(v, 'shape'):
                        image_space = v
                        break
                if image_space is None:
                    raise ValueError("No Box-like image space found in observation_space Dict")
        elif hasattr(observation_space, 'shape'):
            image_space = observation_space
        else:
            raise ValueError(f"Unsupported observation_space type: {type(observation_space)}")

        image_shape: Box = image_space
        chanels = math.prod(image_shape.shape[-2:])

        # detect optional state vector in observation_space (Dict case)
        self.state_dim = 0
        try:
            if hasattr(observation_space, 'spaces'):
                state_space = observation_space.spaces.get("state")
                if state_space is not None and hasattr(state_space, 'shape'):
                    # flatten any trailing dims
                    sd = int(math.prod(state_space.shape))
                    self.state_dim = sd
        except Exception:
            self.state_dim = 0

        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dinov2 = self.dinov2.to(device[0])
        # remember device and move subsequent submodules there as well
        self.device = device[0]
        # how many frames we expect to concatenate (used when temporal_pool is None)
        self.frames = int(frames)
        # temporal_pool: if 'mean', average across time and use a single-frame pipeline
        # if None or 'concat', keep original concatenation behavior
        self.temporal_pool = temporal_pool

        self.embedding_compression_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=64, kernel_size=1, stride=1, bias=False
            ),
            nn.LayerNorm((64, 16, 16)),
            nn.ELU(),
        )
        # ensure module on same device as dinov2
        self.embedding_compression_1 = self.embedding_compression_1.to(self.device)
        # self.embedding_compression_1 = nn.Conv2d(
        #     in_channels=384, out_channels=64, kernel_size=1, stride=1, bias=False
        # )
        # self.norm_1 = nn.LayerNorm((64, 16, 16))
        # self.activate = nn.ELU()

        # Build the second compression block depending on temporal handling.
        # If using concatenation across time (default behavior), the expected
        # in_channels is 64 * frames. If using temporal pooling (mean), then
        # in_channels is 64.
        if self.temporal_pool == 'mean':
            comp_in_channels = 64
        else:
            comp_in_channels = 64 * max(1, self.frames)

        self.compression_2_and_linear = nn.Sequential(
            nn.Conv2d(
                in_channels=comp_in_channels, out_channels=32, kernel_size=3, stride=1, bias=False
            ),
            nn.LayerNorm((32, 14, 14)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(in_features=6272, out_features=256, bias=False),
            nn.LayerNorm((256)),
            nn.ELU(),
        )
        self.compression_2_and_linear = self.compression_2_and_linear.to(self.device)
        # Update the features dim manually (ViT features 256 + optional state dim)
        self._features_dim = 256 + max(0, int(self.state_dim))
        # self.embedding_compression_2 = nn.Conv2d(
        #     in_channels=64 * 4, out_channels=32, kernel_size=3, stride=1, bias=False
        # )
        # self.norm_2 = nn.LayerNorm((32, 14, 14))
        # self.flatten = nn.Flatten()
        # self.last_compression = nn.Linear(
        #     in_features=6272, out_features=256, bias=False
        # )
        # self.norm_3 = nn.LayerNorm((256))

    

    def forward(self, observations) -> torch.Tensor:
        """
        Expect flexible input formats from the environment. Supported formats:
        - H x W x C (single image)
        - B x H x W x C (batch of images)
        - B x H x W x C x T (batch with time stack)
        - H x W x C x T (single sample with time stack)

        This function will normalize/reshape inputs into the form expected by
        dinov2: stacked frames of shape (B*T, C, H, W). It also fixes the
        previous bug where reshape was applied to the un-permuted tensor.
        """
        # Stable-Baselines may pass a dict with an 'image' key or a raw tensor/array.
        # Try dict access first, otherwise fall back to treating observations as the image tensor.
        try:
            images = observations["image"]
        except Exception:
            images = observations

        # extract state if provided in the dict (keep raw for later batching)
        states_raw = None
        try:
            # observations may be a dict-like
            states_raw = observations.get("state") if isinstance(observations, dict) else None
        except Exception:
            states_raw = None

        # Convert to torch tensor if needed
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)

        # Ensure float
        images = images.float()

        # Bring to a common 5D shape: B x H x W x C x T
        if images.dim() == 3:
            # H x W x C -> add batch and time dims
            images = images.unsqueeze(0).unsqueeze(-1)  # 1 x H x W x C x 1
        elif images.dim() == 4:
            # Could be B x H x W x C  OR H x W x C x T (if first dim small like 224)
            # Heuristic: if last dim is channels (1,3,4), assume B x H x W x C
            if images.shape[-1] in (1, 3, 4):
                images = images.unsqueeze(-1)  # B x H x W x C x 1
            else:
                # Treat as H x W x C x T -> add batch dim
                images = images.unsqueeze(0)
        elif images.dim() == 5:
            # assume already B x H x W x C x T
            pass
        else:
            raise ValueError(f"Unsupported image tensor dims: {images.dim()}")

        # Now images is B x H x W x C x T
        B, H, W, C, T = images.shape

        # Make sure channel is last before permute; if channels not in {1,3,4}, try
        if C not in (1, 3, 4):
            # maybe the input is B x C x H x W x T (unlikely) -> try to move
            images = images.permute(0, 3, 4, 1, 2)  # attempt fallback
            B, H, W, C, T = images.shape

    # Prepare for dinov2: B,T,C,H,W -> stack to (B*T, C, H, W)
        chanels_third = images.permute(0, 4, 3, 1, 2).contiguous()  # B, T, C, H, W
        Bp, Tp, Cp, Hp, Wp = chanels_third.shape
        stacked_frames = chanels_third.reshape(Bp * Tp, Cp, Hp, Wp)

        # prepare state tensor if present
        state_tensor = None
        if states_raw is not None:
            # convert to tensor and ensure batch dim matches Bp
            if not isinstance(states_raw, torch.Tensor):
                try:
                    states_raw = torch.as_tensor(states_raw)
                except Exception:
                    states_raw = None
            if isinstance(states_raw, torch.Tensor):
                # ensure float
                states_raw = states_raw.float()
                # If state is given per-sample with shape (B, D) or (D,), handle accordingly
                if states_raw.dim() == 1:
                    # single sample, expand to batch
                    states_raw = states_raw.unsqueeze(0)
                # If batch size mismatches, try to broadcast a single state to all batch items
                if states_raw.shape[0] != Bp:
                    if states_raw.shape[0] == 1:
                        states_raw = states_raw.repeat(Bp, *([1] * (states_raw.dim() - 1)))
                    else:
                        # leave as-is; mismatch will be caught later
                        pass
                state_tensor = states_raw

        device = next(self.dinov2.parameters()).device if any(True for _ in self.dinov2.parameters()) else torch.device('cpu')
        stacked_frames = stacked_frames.to(device)

        with torch.no_grad():
            result = self.dinov2.forward_features(stacked_frames)
            patch_embedings: torch.Tensor = result["x_norm_patchtokens"]

        # patch_embedings is expected to have shape (N, num_patches, dim)
        # reshape to (N, 16, 16, 384) then permute to (N, 384, 16, 16)
        separated_patches = patch_embedings.reshape(-1, 16, 16, 384)
        chanels_second = separated_patches.permute((0, 3, 1, 2)).contiguous()

        # Run the first compression (per-frame)
        res = self.embedding_compression_1(chanels_second)  # (B*T, 64, 16, 16)

        # Combine time frames back into batch dimension: (B, T*64, 16, 16)
        # Reshape to (B, T, 64, 16, 16)
        res = res.reshape(Bp, Tp, 64, 16, 16)  # B x T x 64 x 16 x 16

        if self.temporal_pool == 'mean':
            # Average across time dimension -> (B, 64, 16, 16)
            res = res.mean(dim=1)  # B x 64 x 16 x 16
            # compression_2_and_linear expects in_channels=64 in this mode
            res = self.compression_2_and_linear(res)
            # attach state if present (same logic as after final compression)
            if state_tensor is not None:
                try:
                    state_tensor = state_tensor.to(res.device)
                    state_tensor = state_tensor.reshape(res.shape[0], -1)
                except Exception:
                    state_tensor = state_tensor.cpu().reshape(res.shape[0], -1).to(res.device)
                if state_tensor.shape[0] == res.shape[0] and state_tensor.dim() == 2:
                    res = torch.cat([res, state_tensor], dim=1)
            return res

        # Default behavior: concatenation across time. Pad/truncate to configured frames.
        desired_T = max(1, int(self.frames))
        if Tp < desired_T:
            # repeat last frame as many times as needed
            last = res[:, -1:, ...]  # B x 1 x 64 x 16 x 16
            repeats = desired_T - Tp
            pad = last.repeat(1, repeats, 1, 1, 1)
            res = torch.cat([res, pad], dim=1)
            Tp = desired_T
        elif Tp > desired_T:
            # keep only the last desired_T frames
            res = res[:, -desired_T:, ...]
            Tp = desired_T

        res = res.reshape(Bp, Tp * 64, 16, 16)

        # Final compression to feature vector
        res = self.compression_2_and_linear(res)

        # res shape: (Bp, 256)
        if state_tensor is not None:
            try:
                # move state to same device and flatten trailing dims
                state_tensor = state_tensor.to(res.device)
                state_tensor = state_tensor.reshape(res.shape[0], -1)
            except Exception:
                # fallback: try converting via cpu then to device
                state_tensor = state_tensor.cpu().reshape(res.shape[0], -1).to(res.device)

            # If dimension matches, concatenate; otherwise ignore state to avoid crashes
            if state_tensor.shape[0] == res.shape[0] and state_tensor.dim() == 2:
                res = torch.cat([res, state_tensor], dim=1)

        return res

#--------------------------two layer
# class CustomNetwork(nn.Module):
#    def __init__(self, feature_dim: int, action_dim: int = 2):  # 明确动作维度
#        super().__init__()
#        self.latent_dim_pi = action_dim  # 用于高斯分布的 μ 输出
#        self.latent_dim_vf = 1           # 用于 V(s) 输出

#         #改进后的 policy_net：两层 MLP，输出动作均值
#        self.policy_net = nn.Sequential(
#            nn.Linear(feature_dim, 128),
#            nn.LayerNorm(128),
#            nn.GELU(),
#            nn.Linear(128, 64),
#            nn.GELU(),
#            nn.Linear(64, action_dim)
#        )

#        # 改进后的 value_net：输出标量
#        self.value_net = nn.Sequential(
#            nn.Linear(feature_dim, 128),
#            nn.LayerNorm(128),
#            nn.GELU(),
#            nn.Linear(128, 64),
#            nn.GELU(),
#            nn.Linear(64, 1)
#        )

#    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#        return self.forward_actor(features), self.forward_critic(features)

#    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
#        return self.policy_net(features)

#    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
#        return self.value_net(features)

#onelayer---------------------
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CustomViTPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            ortho_init=False,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs={},
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={},
            *args,
            **kwargs,
        )
        parameters_to_train = []
        for module_name, module in self.named_children():
            if module_name == "features_extractor":
                extractor_parts = dict(module.named_children())
                parameters_to_train += extractor_parts[
                    "embedding_compression_1"
                ].parameters()
                parameters_to_train += extractor_parts[
                    "compression_2_and_linear"
                ].parameters()
            else:
                parameters_to_train += module.parameters()
        self.optimizer = torch.optim.AdamW(
            params=parameters_to_train, lr=lr_schedule(1), **self.optimizer_kwargs
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
    
    # def _build_mlp_extractor(self) -> None:
    #     action_dim = self.action_space.shape[0]  # 读取 Box 动作维度
    #     self.mlp_extractor = CustomNetwork(self.features_dim, action_dim=action_dim)
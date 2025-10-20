import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import sys
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)




class ImageNetBEVCNN(BaseFeaturesExtractor):
    """
    A single-path RGB BEV CNN for image feature extraction.
    If concat_state=True and observation_space contains 'state', 
    the extracted image features will be concatenated with the state vector.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, concat_state: bool = False):
        super().__init__(observation_space, features_dim)
        self.concat_state = concat_state

        if hasattr(observation_space, 'spaces') and isinstance(observation_space.spaces, dict):
            image_space = observation_space.spaces.get('image', list(observation_space.spaces.values())[0])
            self.state_dim = 0
            if concat_state and 'state' in observation_space.spaces:
                self.state_dim = int(np.prod(observation_space.spaces['state'].shape))
        else:
            image_space = observation_space
            self.state_dim = 0

        shape = image_space.shape
        if len(shape) == 3:
            if shape[0] in (1, 3):
                C, H, W = shape
            else:
                H, W, C = shape
        else:
            raise AssertionError(f"Unexpected image shape {shape}")

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            sample = torch.zeros((1, C, H, W))
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.Tanh(),
        )


        self._features_dim = features_dim + (self.state_dim if self.concat_state else 0)

    def forward(self, observations):
        if isinstance(observations, dict):
            img = observations.get('image', list(observations.values())[0])
            state = observations.get('state', None)
        else:
            img = observations
            state = None

        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)
        img = img.float()
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.shape[-1] in (1, 3):
            img = img.permute(0, 3, 1, 2)  # HWC -> CHW

        img = img.to(next(self.parameters()).device)
        features = self.linear(self.cnn(img))

        if self.concat_state and state is not None:
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            state = state.to(features.device)
            state = state.view(state.shape[0], -1) if state.ndim > 1 else state.unsqueeze(0)
            features = torch.cat([features, state], dim=1)
        # print(f"[DEBUG CNN] image_features={features.shape}, "f"concat_state={self.concat_state}, state_dim={self.state_dim}", flush=True)

        return features


import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImageNetBEVCNN(BaseFeaturesExtractor):
    """A single-path RGB BEV CNN matching the architecture you described.

    Expected input: (B, H, W, C) with C==3 (RGB) and typically H==W==128.
    Architecture:
      Conv2d(3,32,k=8,s=4) -> ReLU
      Conv2d(32,64,k=4,s=2) -> ReLU
      Conv2d(64,64,k=3,s=2) -> ReLU
      Flatten -> Linear(n_flatten, features_dim) -> Tanh

    This produces a (B, features_dim) tensor suitable for an LSTM input.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Pass final desired features_dim to BaseFeaturesExtractor so policy expects correct size
        super().__init__(observation_space, features_dim)

        # Support Dict observation_space that contains an 'image' key (our env returns {'image', 'state'})
        self.state_dim = 0
        if hasattr(observation_space, 'spaces') and isinstance(observation_space.spaces, dict):
            # prefer 'image' key
            if 'image' in observation_space.spaces:
                image_space = observation_space.spaces['image']
            else:
                # fallback: take the first subspace
                image_space = list(observation_space.spaces.values())[0]
            # detect state dim if provided
            if 'state' in observation_space.spaces:
                s = observation_space.spaces['state']
                try:
                    self.state_dim = int(getattr(s, 'shape', (0,))[0])
                except Exception:
                    self.state_dim = 0
            shape = getattr(image_space, 'shape', None)
        else:
            shape = getattr(observation_space, "shape", None)
        if shape is None:
            raise AssertionError(f"observation_space has no 'shape' attribute: {observation_space}")

        if len(shape) == 3:
            # shape can be (H,W,C) or (C,H,W). Detect common patterns:
            # - channels-first (C,H,W) when shape[0] in {1,3}
            # - channels-last  (H,W,C) when shape[2] in {1,3}
            if int(shape[0]) in (1, 3):
                # (C,H,W)
                C, H, W = int(shape[0]), int(shape[1]), int(shape[2])
            else:
                # default to (H,W,C)
                H, W, C = int(shape[0]), int(shape[1]), int(shape[2])
        elif len(shape) == 2:
            H, W, C = int(shape[0]), int(shape[1]), 1
        else:
            raise AssertionError(f"Unexpected observation shape {tuple(shape)}; expected (H,W,C) or (H,W)")

        if C != 3:
            # allow both grayscale and different channels but note the extractor expects RGB.
            # We'll accept C==1 (grayscale) and repeat channels in forward if needed.
            pass

        # single-path CNN using the specified kernels/strides
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # infer flatten size
        with torch.no_grad():
            sample = torch.zeros((1, C, H, W))
            n_flatten = self.cnn(sample).shape[1]

        # ensure this matches the 64*6*6 = 2304 expectation for 128x128 input
        self.n_flatten = n_flatten

        # allocate image features dims as total_output_dim - state_dim
        total_output_dim = int(features_dim)
        image_features_dim = total_output_dim - int(self.state_dim or 0)
        if image_features_dim <= 0:
            raise AssertionError(f"features_dim={features_dim} is too small for state_dim={self.state_dim}")

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, image_features_dim),
            nn.Tanh(),
        )
        # expose dims for downstream use
        self.image_features_dim = image_features_dim
        self.output_dim = total_output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Support dict observations: extract image and optional state
        orig_state = None
        if isinstance(observations, dict):
            orig_image = observations.get('image', None)
            orig_state = observations.get('state', None)
            if orig_image is None:
                # fallback to first value
                orig_image = list(observations.values())[0]
            observations = orig_image

        # Convert to tensor if needed
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations)

        device = next(self.parameters()).device
        observations = observations.to(device)

        # normalize
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        else:
            try:
                maxv = float(observations.max().item())
            except Exception:
                maxv = 1.0
            if maxv > 2.0:
                observations = observations.float() / 255.0
            else:
                observations = observations.float()
        observations = observations.clamp(0.0, 1.0)

        # Ensure we have a batch dimension: accept (H,W,C), (C,H,W) or batched variants
        if observations.ndim == 3:
            # add batch dim
            observations = observations.unsqueeze(0)

        # Now observations is expected to be (B,H,W,C) or (B,C,H,W)
        if observations.ndim == 4:
            # channels-first case: (B,C,H,W)
            if observations.shape[1] == self.cnn[0].in_channels:
                x = observations
            # channels-last case: (B,H,W,C)
            elif observations.shape[-1] == self.cnn[0].in_channels:
                x = observations.permute(0, 3, 1, 2)
            # single-channel -> repeat to 3 channels if cnn expects 3
            elif observations.shape[1] == 1 and self.cnn[0].in_channels == 3:
                x = observations.repeat(1, 3, 1, 1)
            elif observations.shape[-1] == 1 and self.cnn[0].in_channels == 3:
                x = observations.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
            else:
                # fallback: try permute last dim
                try:
                    x = observations.permute(0, 3, 1, 2)
                except Exception:
                    x = observations
        else:
            # if we somehow still don't have 4 dims, raise informative error
            raise ValueError(f"Unexpected image tensor ndim={observations.ndim}; expected 3 or 4")

        feats = self.cnn(x)
        out = self.linear(feats)

        # If original dict contained a state vector, concat it
        if orig_state is not None:
            st = torch.as_tensor(orig_state).to(device)
            if st.ndim == 1:
                st = st.unsqueeze(0)
            # repeat single-state across batch if needed
            if st.shape[0] != out.shape[0]:
                if st.shape[0] == 1:
                    st = st.repeat(out.shape[0], 1)
                else:
                    raise ValueError(f"Batch size mismatch between image features ({out.shape[0]}) and state ({st.shape[0]})")
            # (diagnostic removed)
            st = st.float().to(device)
            combined = torch.cat([out, st], dim=1)
            return combined

        return out

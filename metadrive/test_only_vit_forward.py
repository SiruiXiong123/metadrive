import torch
from envs.only_VIT import CustomCombinedExtractor
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

# Build a dummy observation_space compatible with TopDownMultiChannel output
H=W=224
obs_space = gym.spaces.Dict({
    'image': Box(low=0.0, high=1.0, shape=(H, W, 3), dtype=np.float32)
})

print('Creating extractor...')
extractor = CustomCombinedExtractor(obs_space)
print('Extractor created.\nTesting forward with single image HWC...')

# Single image HWC -> expected to be handled
img = np.random.rand(H, W, 3).astype(np.float32)
obs = {'image': img}
res = extractor(obs)
print('Output shape (single HWC):', res.shape)

print('\nTesting forward with stacked T=4 frames BxHxWxCxT...')
imgs = np.stack([img]*4, axis=-1)  # H W C T
imgs = np.expand_dims(imgs, 0)    # B H W C T
obs2 = {'image': torch.from_numpy(imgs)}
res2 = extractor(obs2)
print('Output shape (BHT):', res2.shape)

import os
import time
import numpy as np
import gymnasium as gym
from PIL import Image

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    # Dummy fallback: create a simple wrapper compatible interface
    DummyVecEnv = None

from metadrive.envs.top_down_env import TopDownMetaDrive

# Match cfg from LSTM(CNN).py so the observation the agent saw is similar
cfg = {
    "num_scenarios": 500,
    "start_seed": 123,
    "random_lane_width": True,
    "random_lane_num": False,
    "use_render": False,
    "traffic_density": 0.0,
    "traffic_mode": "hybrid",
    "manual_control": False,
    "controller": "keyboard",
    "vehicle_config": {
        "show_navi_mark": True,
        "show_line_to_dest": False,
        "show_line_to_navi_mark": True,
    },
    "distance": 40,
    "resolution_size": 224,
}


class ImageOnlyWrapper(gym.ObservationWrapper):
    """If env returns a Dict observation with key 'image', return only that image."""
    def __init__(self, env):
        super().__init__(env)
        obs_space = getattr(env, 'observation_space', None)
        try:
            image_space = obs_space.spaces['image']
        except Exception:
            image_space = obs_space
        self.observation_space = image_space

    def observation(self, observation):
        if isinstance(observation, dict):
            return observation.get('image', observation)
        return observation


class ResetCompatWrapper(gym.Wrapper):
    """Wrap env so reset(...) works with or without options kwarg and normalize return."""
    def reset(self, seed=None, options=None, **kwargs):
        try:
            res = self.env.reset(seed=seed, options=options, **kwargs)
        except TypeError:
            try:
                res = self.env.reset(seed=seed, **kwargs)
            except TypeError:
                res = self.env.reset(**kwargs)

        if isinstance(res, tuple) and len(res) == 2:
            return res
        return res, {}


def make_env():
    def _init():
        env = TopDownMetaDrive(cfg)
        env = ResetCompatWrapper(env)
        try:
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'spaces') and 'image' in env.observation_space.spaces:
                env = ImageOnlyWrapper(env)
        except Exception:
            pass
        return env

    if DummyVecEnv is not None:
        return DummyVecEnv([_init])
    # Minimal fallback: return single env instance that mimics VecEnv reset/step signatures
    return _init()


def save_one_frame(out_path: str = None):
    if out_path is None:
        out_path = os.path.join(os.getcwd(), 'agent_view_frame.png')

    env = make_env()

    # For DummyVecEnv, reset() returns array (n_envs, H, W, C) or (obs, info)
    try:
        res = env.reset()
    except Exception as e:
        print('env.reset() failed:', e)
        try:
            env.close()
        except Exception:
            pass
        return

    # Normalize the reset return
    if isinstance(res, tuple) and len(res) == 2:
        obs = res[0]
    else:
        obs = res

    # If vectorized, take first env
    arr = np.array(obs)
    if arr.ndim == 4:
        arr = arr[0]

    # If observation is dict-like wrapped into numpy object, try to extract 'image'
    if arr.dtype == object:
        # likely a numpy array of dicts
        try:
            maybe = obs[0] if hasattr(obs, '__len__') else obs
            if isinstance(maybe, dict) and 'image' in maybe:
                arr = np.array(maybe['image'])
        except Exception:
            pass

    # Ensure numeric dtype and convert float->uint8
    try:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    except Exception:
        arr = np.asarray(arr, dtype=np.uint8)

    # If grayscale single channel
    if arr.ndim == 2:
        img = Image.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        img = Image.fromarray(arr[:, :, 0])
    else:
        img = Image.fromarray(arr)

    img.save(out_path)
    print('Saved frame to', out_path)

    try:
        env.close()
    except Exception:
        pass


if __name__ == '__main__':
    save_one_frame()
